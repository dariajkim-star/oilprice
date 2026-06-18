# OilPriceValuation — KFBI 금융시장분석팀 내부용

> **Operation Roaring Lion (2026.02.28~)** 호르무즈 봉쇄 상황에서  
> WTI/Brent 적정가 산출 및 시장 리스크 자동 모니터링 시스템

---

## 프로젝트 구조

```
oilpricevaluation/
│
├── hormuz_monitor.py          # 호르무즈 실시간 모니터링 데몬
├── five_house_weighted.py     # 5사 가중치 시나리오 분석 + Monte Carlo
├── oil_fair_value_model.py    # MS Kalman + GS Bridge + RF 통합 모델
├── market_sentiment_pipeline.py  # 뉴스/SNS 감성 분석 파이프라인
├── nasdaq_mc_v2.py            # Nasdaq Monte Carlo (Merton JDM + 레짐 스위칭)
└── README.md
```

---

## 전체 데이터 흐름

```
IMF PortWatch (선박수)
EIA API (WTI, 재고, SPR)          ┐
Yahoo Finance (WTI 실시간)        ├──▶ hormuz_monitor.py ──▶ 알림 출력
                                  ┘

뉴스/SNS 크롤링 데이터 (JSON/CSV)
        │
        ▼
market_sentiment_pipeline.py
        │  fear_index (0~100)
        ▼
nasdaq_mc_v2.py ──▶ Nasdaq 경로 시뮬레이션 + VaR

IMF PortWatch + EIA + 시뮬 데이터
        │
        ▼
oil_fair_value_model.py  (MS Kalman + GS Bridge + RF)
        │
        ▼
five_house_weighted.py   (5사 가중 시나리오 EV)
```

---

## 파일별 상세

### 1. `hormuz_monitor.py` — 실시간 모니터링 데몬

**역할:** 호르무즈 선박 통과 수 + WTI 현재가를 수집하여 GS house view 기준 과/저평가 알림

**핵심 파라미터:**

| 파라미터 | 값 | 설명 |
|---|---|---|
| `GS_MAR23_WTI` | $105.0 | GS Mar23 house view (결정 기준) |
| `GS_BASE_WTI` | $84.0 | GS 원래 base (참고용) |
| `HORMUZ_NORMAL_VES` | 138척 | 정상 7일 MA 선박수 |
| `PIPELINE_OFFSET` | 3.5 mb/d | 파이프라인 우회분 |
| `SPR_RELEASE` | 2.0 mb/d | SPR 방출분 |
| `OVER_THRESHOLD` | +$10 | 과도평가 경보 기준 |
| `UNDER_THRESHOLD` | -$20 | 저평가 경보 기준 |

**데이터 소스 우선순위:**

호르무즈 선박수:
1. IMF PortWatch ArcGIS REST API (`services9.arcgis.com`, `Daily_Chokepoints_Data`, `chokepoint6`)
2. GeoJSON Hub
3. CSV 직접 다운로드 → 로컬 캐시 저장
4. 로컬 캐시 (`portwatch_cache.csv`)
5. 하드코딩 fallback (6척)

WTI 현재가:
1. **EIA API v2** (주간 현물가, 시리즈 `RWTC`)
2. Yahoo Finance (`CL=F`, 15분 딜레이)
3. fallback 98.34

**알림 로직:**

```
현재가 - GS Mar23 $105 > +$10  →  🔴 과도평가 경보
현재가 - GS Mar23 $105 < -$20  →  🔴 극단 저평가 경보
두 GS 버전이 방향 상충           →  ⚠️ 방향 불일치, 판단 유보
그 외                            →  ✅ 정상 범위
```

**스케줄 (데몬 모드):** KST 08:00, 10:00 고정 + 12:00, 14:00, 16:00, 18:00, 20:00, 22:00

**실행:**
```bash
python hormuz_monitor.py           # 1회 즉시 실행
python hormuz_monitor.py --daemon  # 스케줄 데몬 모드
```

---

### 2. `five_house_weighted.py` — 5사 가중치 시나리오 분석

**역할:** GS/MS/Citi/JPM/Bloomberg 5사 목표가 × 4개 시나리오 → 가중평균 EV + MC 10만 경로

**하우스 가중치:**

| 하우스 | 가중치 | 선정 이유 |
|---|---|---|
| Goldman Sachs | 28% | 공급 산술 메인 |
| Morgan Stanley | 27% | Forward Curve 메인 |
| Citigroup | 18% | 보조 |
| JPMorgan | 18% | 보조 |
| Bloomberg | 9% | 전시 물리적 수급 왜곡 감안 하향 |

**시나리오:**

| 시나리오 | 확률 | 내용 |
|---|---|---|
| S1 Agreement | 10% | 협상 합의, 봉쇄 해제 |
| S2 Strike | 40% | 에너지 인프라 타격 |
| S3 Prolonged | 30% | 6주 이상 장기화 |
| S4 Withdrawal | 20% | 봉쇄 + 미국 철군 |

**파라미터 수정 위치:** 파일 상단 `WTI_SPOT`, `HOUSE_WEIGHTS`, `TARGETS`, `SCENARIO_PROBS`

**출력:** 터미널 수치 + `five_house_weighted.png` (6패널 다크 차트)

```bash
python five_house_weighted.py
```

---

### 3. `oil_fair_value_model.py` — MS Kalman + GS Bridge + RF 통합 모델

**역할:** 3레이어 아키텍처로 WTI 적정가 산출 및 오버슈트 분석

**모델 아키텍처:**

```
Layer 0:  고빈도 선박 데이터 (vessel_count, speed, tonnage, tanker_ratio)
               │
               ▼  RandomForestRegressor (TimeSeriesSplit CV)
          Supply Shock Index (0~1 정규화)

Layer 1:  WTI 수준 시계열
               │  ADF 단위근 검정 → 정상성 확인
               ▼  Kalman Filter Local Level Model (Q/R Grid Search)
          잠재 펀더멘털 가격 + 불확실성 밴드 (MS 방식)

Layer 2a: 정상 변수 (supply_gap, ΔSPR, ΔDXY, EIA재고)
               │
               ▼  OLS Bridge Equation
          Stage A 적정가 (수급 기반)

Layer 2b: (d_WTI - Stage A) 잔차
               │  shock_index, OVX, WTI-Brent spread, futures slope
               ▼  OLS
          Stage B 지정학 프리미엄

최종:    GS Fair Value = Stage A + Stage B 누적
         오버슈트 = WTI 실제가 - GS Fair Value
```

**실제 데이터 교체 포인트 (현재 시뮬레이션):**

| 변수 | 교체 소스 |
|---|---|
| `wti_spot` | EIA API `RWTC` 또는 Yahoo `CL=F` |
| `supply_gap` | OPEC/IEA Monthly |
| `eia_stock` | EIA API `WCESTUS1` |
| `spr_level` | EIA API `W_EPC0_SAX_YCUOK_MBBL` |
| `dxy` | FRED `DTWEXBGS` |
| `ovx` | Yahoo `^OVX` |
| `vessel_count` | IMF PortWatch ArcGIS |

**출력:** 터미널 수치 + `oil_fair_value_model.png` (6패널 다크 차트)

```bash
python oil_fair_value_model.py
```

---

### 4. `market_sentiment_pipeline.py` — 감성 분석 파이프라인

**역할:** 뉴스/SNS 크롤링 결과 → 공포지수 0~100 산출 → Plotly 대시보드

**입력:** JSON 또는 CSV (`text`, `source`, `timestamp` 컬럼)

**파이프라인 단계:**

1. **데이터 로드** — JSON/CSV 자동 감지
2. **워드클라우드** — 금융 키워드 가중치 부여 (폭락×3, 전쟁×3 등)
3. **감성 분석** — 두 모드 선택
   - `lexicon`: 사전 기반, 빠름, 한국어 지원
   - `transformer`: FinBERT, 영문 금융 텍스트 정확도 높음
4. **유가 수집** — yfinance `CL=F`
5. **괴리율** — 공포지수 vs 유가 정규화 차이 (양수 = 시장이 유가 리스크 무시 중)
6. **대시보드** — Plotly 4분할 HTML
7. **트레이더 사고 흐름** — 공포지수 구간별 포지션 바이어스 출력

**실행:**
```bash
python market_sentiment_pipeline.py
# 또는 코드에서
result = run_pipeline("crawled_data.json", sentiment_mode="lexicon")
```

**출력:** `dashboard.html`, `wordcloud.png`, 공포지수 딕셔너리

---

### 5. `nasdaq_mc_v2.py` — Nasdaq Monte Carlo (IB급)

**역할:** 공포지수를 입력받아 Nasdaq 향후 N일 경로 시뮬레이션 + 리스크 지표

**v1 대비 개선점:**

| 항목 | v1 | v2 |
|---|---|---|
| 변동성 소스 | 과거 실현변동성 × 공포지수 배수 | VIX 직접 사용 (내재변동성) |
| 점프 모델 | 없음 | Merton JDM (포아송 λ 캘리브레이션) |
| 레짐 | 단일 | 저변동/고변동/위기 3개 마르코프 체인 |
| VaR | 정규분포 | Cornish-Fisher + Historical + CVaR |

**레짐 전환:**
- 공포지수가 변동성이 아닌 **레짐 전환 확률**에만 반영
- 위기 레짐 진입 확률 = `fear_index / 200`

**pipeline 연결:**
```python
from market_sentiment_pipeline import run_pipeline
from nasdaq_mc_v2 import run

sentiment = run_pipeline("crawled_data.json")
result = run(fear_index=sentiment["fear_index"])
```

**출력:** `nasdaq_mc_v2.html` (Plotly 인터랙티브 대시보드)

---

## API 키 및 소스 정리

| API | 키 / 방법 | 비용 | 용도 |
|---|---|---|---|
| EIA API v2 | `sshh6mSdDhyaEtl5XGdOAvFiQDQDOSnSRdvvKnrV` | 무료 | WTI, 재고, SPR |
| IMF PortWatch | 키 불필요 (ArcGIS REST) | 무료 | 호르무즈 선박수 |
| Yahoo Finance | 키 불필요 | 무료 | WTI 실시간, VIX, OVX |
| FRED | 키 불필요 (공개 endpoint) | 무료 | DXY 백업 |
| MarineTraffic | 별도 계약 필요 | **유료** | 실시간 AIS (미연결) |
| Anthropic API | console.anthropic.com 발급 | 유료 | kfbi_agent React 앱 |

**EIA API 주요 시리즈 ID:**

```
WTI Cushing 현물가 (주간)  : petroleum/pri/spt  →  facets[series][]=RWTC
Brent 현물가 (주간)        : petroleum/pri/spt  →  facets[series][]=RBRTE
미국 상업 원유 재고        : petroleum/stoc/wstk →  facets[series][]=WCESTUS1
쿠싱 원유 재고             : petroleum/stoc/wstk →  facets[series][]=WTTSTUS1
SPR 잔여                   : petroleum/stoc/wstk →  facets[series][]=W_EPC0_SAX_YCUOK_MBBL
정제소 가동률              : petroleum/pnp/cap   →  facets[series][]=WCRFPUS2
```

**브라우저 테스트 URL:**
```
https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key=sshh6mSdDhyaEtl5XGdOAvFiQDQDOSnSRdvvKnrV&frequency=weekly&data[0]=value&facets[series][]=RWTC&sort[0][column]=period&sort[0][direction]=desc&length=5
```

---

## 설치 (한 번만)

```bash
# Python 패키지 전체
pip install requests pandas numpy scipy matplotlib scikit-learn schedule yfinance plotly wordcloud transformers torch

# transformers/torch는 FinBERT 사용 시만 필요 (용량 큼, 선택)
pip install transformers torch
```

---

## 전체 실행 순서

```bash
# Step 1: 호르무즈 모니터 (현재 상황 확인)
python hormuz_monitor.py

# Step 2: 데몬 모드 (백그라운드 상시 모니터링)
python hormuz_monitor.py --daemon

# Step 3: 5사 시나리오 분석 차트
python five_house_weighted.py

# Step 4: 통합 공정가 모델
python oil_fair_value_model.py

# Step 5: 감성 분석 (크롤링 데이터 있을 때)
python market_sentiment_pipeline.py

# Step 6: Nasdaq MC (감성 파이프라인 연결 후)
python nasdaq_mc_v2.py
```

---

## VS Code 설정

**추천 확장:**
- Python (Microsoft)
- Pylance
- Prettier

**Task 등록** (`.vscode/tasks.json`):
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Hormuz Daemon",
      "type": "shell",
      "command": "python hormuz_monitor.py --daemon",
      "isBackground": true,
      "problemMatcher": []
    },
    {
      "label": "5사 시나리오 분석",
      "type": "shell",
      "command": "python five_house_weighted.py"
    },
    {
      "label": "통합 공정가 모델",
      "type": "shell",
      "command": "python oil_fair_value_model.py"
    }
  ]
}
```

`Ctrl+Shift+P` → `Run Task` → 원하는 작업 선택

---

## 주요 제약 및 주의사항

- **IMF PortWatch 래그:** 매주 화요일 오전 9시 ET 업데이트, 실제 데이터 기준 2~4일 래그 존재
- **GPS 재밍/AIS 스푸핑:** 현재 호르무즈 분쟁 지역 특성상 실제 통과 선박보다 수치가 낮게 잡힐 수 있음 (IMF 자체 경고)
- **EIA WTI는 주간 데이터:** 당일 실시간 가격이 아님 → 장중 실시간은 Yahoo Finance 사용
- **`oil_fair_value_model.py`:** 현재 시뮬레이션 데이터 사용 중. 실제 운용 시 코드 내 "실제 교체 포인트" 주석 위치에 API 연결 필요
- **GS house view 업데이트:** `GS_MAR23_WTI = 105.0` — 새 리포트 발표 시 수동 업데이트 필요
- **FinBERT:** GPU 없으면 느림. CPU에서도 동작하나 lexicon 모드 권장

---

## 분석 컨텍스트 (2026.04 기준)

- **배경:** 이란-이스라엘 전쟁 Operation Roaring Lion (2026.02.28 개시), 호르무즈 봉쇄율 ~95.7%
- **기준 데이터:** WTI $98.34, Brent $99.81, 호르무즈 7dMA 6척 (정상 138척)
- **GS 결정 기준:** Mar23 house view $105 (오버슈트 -$6.66 → 저평가 범위)
- **가중평균 EV:** $112.1 (5사 × 4 시나리오, GS28%/MS27%)
- **Polymarket:** 4월 30일 정상화 19% (당시 기준)
