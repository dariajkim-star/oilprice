# OilPriceValuation — KFBI 금융시장분석팀

> **Operation Roaring Lion (2026.02.28~)** 호르무즈 봉쇄 상황에서  
> WTI/Brent 적정가 산출 및 시장 리스크 자동 모니터링 시스템
>
> **최종 파라미터 업데이트: 2026-06-18**  
> 미국-이란 60일 MOU 체결, 호르무즈 재개방 진행, IEA 공급 과잉 경고 반영

---

## 프로젝트 구조

```
oilpricevaluation/
│
├── hormuz_monitor.py          # 호르무즈 실시간 모니터링 데몬
├── five_house_weighted.py     # 5사 가중치 시나리오 분석 + Monte Carlo
├── oil_fair_value_model.py    # MS Kalman + GS Bridge + RF 통합 모델 (Streamlit)
├── app.py                     # Flask 통합 대시보드 (전체 기능 통합)
├── market_sentiment_pipeline.py  # 뉴스/SNS 감성 분석 파이프라인
├── nasdaq_mc_v2.py            # Nasdaq Monte Carlo (Merton JDM + 레짐 스위칭)
└── README.md
```

---

## 전체 데이터 흐름

```
Yahoo Finance (WTI CL=F, 15분 딜레이 실시간)
FRED DCOILWTICO (전일 확정, API 키 불필요)
EIA API v2 (주간 확정, 최대 5일 래그 — 참고용)    ┐
IMF PortWatch ArcGIS (선박수/속도/톤수)             ├──▶ hormuz_monitor.py ──▶ 알림 출력
                                                   ┘

뉴스/SNS 크롤링 데이터 (JSON/CSV)
        │
        ▼
market_sentiment_pipeline.py
        │  fear_index (0~100)
        ▼
nasdaq_mc_v2.py ──▶ Nasdaq 경로 시뮬레이션 + VaR

Yahoo Finance + EIA + IMF PortWatch + FRED + 매크로
        │
        ▼
app.py  (Flask 통합 대시보드)
   ├── oil_fair_value  (MS Kalman + GS Bridge + RF)
   ├── 5사 시나리오 EV
   ├── MC 가격 예측 (30/60/90일)
   └── 탱커 수 예측

app.py / oil_fair_value_model.py 결과
        │
        ▼
five_house_weighted.py  (5사 가중 시나리오 EV 독립 실행)
```

---

## 파일별 상세

### 1. `hormuz_monitor.py` — 실시간 모니터링 데몬

**역할:** 호르무즈 선박 통과 수 + WTI 현재가를 수집하여 GS house view 기준 과/저평가 알림

**핵심 파라미터 (2026-06-18 업데이트):**

| 파라미터 | 이전값 | 현재값 | 변경 근거 |
|---|---|---|---|
| `GS_MAR23_WTI` | $105.0 | **$80.0** | MOU 체결·재개방, IEA 과잉 경고 반영 |
| `GS_BASE_WTI` | $84.0 | **$75.54** | 2026-06-18 실제 WTI 기준 |
| `OVER_THRESHOLD` | +$10 | **+$8** | 재개방 변동성 확대 대응 |
| `UNDER_THRESHOLD` | -$20 | **-$12** | 새 house view $80 기준 재조정 |
| `HORMUZ_NORMAL_VES` | 138척 | 138척 | 정상 기준 유지 |
| `PIPELINE_OFFSET` | 3.5 mb/d | 3.5 mb/d | 변동 없음 |
| `SPR_RELEASE` | 2.0 mb/d | 2.0 mb/d | 변동 없음 |

**WTI 수집 우선순위 (코드 기준):**

> **결정: Yahoo Finance 1순위 유지** (README 구버전의 EIA 우선 명세를 코드에 맞게 수정)
>
> EIA v2 RWTC는 주간 확정값으로 최대 5일 래그 발생 → 실시간 알림 판단 부적합.
> Yahoo Finance(15분 딜레이)가 호르무즈 모니터의 즉시성 요구에 가장 적합.

1. **Yahoo Finance** `CL=F` — 15분 딜레이 실시간 선물가
2. **FRED** `DCOILWTICO` — 전일 확정, API 키 불필요 (공개 CSV)
3. **EIA v2** `RWTC` — 주간 확정, 래그 최대 5일 (참고용 fallback)
4. Fallback `$75.54` (2026-06-18 실제가)

**알림 로직 (2026-06-18 기준):**

```
현재가 - GS $80 > +$8   →  🔴 과도평가 경보 (현재가 > $88)
현재가 - GS $80 < -$12  →  🔴 저평가 경보   (현재가 < $68)
두 GS 버전이 방향 상충   →  ⚠️ 방향 불일치, 판단 유보
그 외                    →  ✅ 정상 범위
```

**ArcGIS URL (app.py 기준 통일):**
```
https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query
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

**하우스 가중치:** 변동 없음

| 하우스 | 가중치 | 선정 이유 |
|---|---|---|
| Goldman Sachs | 28% | 공급 산술 메인 |
| Morgan Stanley | 27% | Forward Curve 메인 |
| Citigroup | 18% | 보조 |
| JPMorgan | 18% | 보조 |
| Bloomberg | 9% | 전시 물리적 수급 왜곡 감안 하향 |

**시나리오 (2026-06-18 업데이트):**

| 시나리오 | 이전 확률 | 현재 확률 | 5사 WTI 목표 범위 | 내용 |
|---|---|---|---|---|
| S1 Agreement | 10% | **60%** | $68~$74 | 합의·호르무즈 재개방 현실화 |
| S2 Strike | 40% | **15%** | $100~$110 | 재점화·에너지 인프라 타격 |
| S3 Prolonged | 30% | **15%** | $110~$145 | 협상 결렬·장기화 |
| S4 Withdrawal | 20% | **10%** | $87~$98 | 봉쇄+미국 철군 극단 시나리오 |

**WTI_SPOT:** `$98.34` → **`$75.54`** (2026-06-18 실제가)

**파라미터 수정 위치:** 파일 상단 `WTI_SPOT`, `SCENARIO_PROBS`, `TARGETS`, `SIGMA`

**출력:** 터미널 수치 + `five_house_weighted.png` (6패널 다크 차트)

```bash
python five_house_weighted.py
```

---

### 3. `oil_fair_value_model.py` — MS Kalman + GS Bridge + RF 통합 모델

**역할:** 3레이어 아키텍처로 WTI 적정가 산출 및 오버슈트 분석

> ⚠️ **실행 방식**: `streamlit run oil_fair_value_model.py` (Streamlit 앱)
> 동일 로직은 `app.py` (Flask) 에도 통합되어 있음.

**모델 아키텍처:** 변동 없음 (MS Kalman → GS Bridge → RF Shock Index)

**시뮬레이션 파라미터 (2026-06-18 업데이트):**

| 변수 | 이전 기준값 | 현재 기준값 | 변경 근거 |
|---|---|---|---|
| `wti_spot` 초기값 | $80 | **$75.54** | 2026-06-18 실제가 |
| `supply_gap` 평균 | 0 mb/d | **+0.8 mb/d** | IEA 과잉 +6mb/d 경고 |
| `eia_stock` 평균 | -0.5 Mbbl | **-1.5 Mbbl** | 최근 주간 -8.3 Mbbl 반영 |
| `spr_level` 초기값 | 350 Mbbl | **320 Mbbl** | 봉쇄 기간 중 방출 |
| `dxy` 초기값 | 100 | **103** | 달러 강세 |
| `vessel_count` | 138 지수감소 | **6→80 점진 회복** | 재개방 진행 현실 반영 |
| 오버슈트 알림선 | ±10 / -20 | **+8 / -12** | house view $80 기준 재조정 |

**실제 데이터 교체 포인트:** 변동 없음 (코드 내 주석 위치 동일)

```bash
streamlit run oil_fair_value_model.py
```

---

### 4. `app.py` — Flask 통합 대시보드

**역할:** 전체 기능을 하나의 웹 대시보드로 통합 (실시간 크롤링 + 모든 모델)

**5사 시나리오 파라미터:** `five_house_weighted.py`와 동일하게 2026-06-18 업데이트 완료

**추가 데이터 소스 (README 구버전 미기재):**

| 소스 | 데이터 | 비고 |
|---|---|---|
| FRED DCOILWTICO / DCOILBRENTEU | WTI·Brent 일별 확정가 | 키 불필요 |
| Yahoo Finance USO / BNO ETF | 원유 ETF 가격 | WTI·Brent 프록시 |
| Yahoo Finance ^TNX, ^GSPC, URTH, GC=F, HG=F, ^VIX, TIP, XLI, FXI, MCHI | 매크로 10종 | GS Bridge Stage B 보강 |
| EIA WPSR 가솔린·중간유·수입량 | 정제·수요 수급 | RF 피처 확장 |
| straits.live | 호르무즈 현황 스냅샷 | 보조 지표 |
| WTI 선물 커브 M1~M6 | 콘탱고/백워데이션 | Stage B 피처 |

**실행:**
```bash
pip install flask yfinance requests pandas numpy scikit-learn matplotlib
python app.py
# → http://localhost:5000
```

---

### 5. `market_sentiment_pipeline.py` — 감성 분석 파이프라인

**역할:** 뉴스/SNS 크롤링 결과 → 공포지수 0~100 산출 → Plotly 4분할 대시보드

**입력:** JSON 또는 CSV (`text`, `source`, `timestamp` 컬럼)

**파이프라인 단계:**

1. **데이터 로드** — JSON/CSV 자동 감지, 컬럼 자동 정규화
2. **워드클라우드** — 금융 키워드 가중치 (폭락×3, 전쟁×3, ceasefire×3 등)
3. **감성 분석** — 두 모드 선택
   - `lexicon`: 사전 기반, 빠름, 한국어 지원 (기본값)
   - `transformer`: FinBERT(`ProsusAI/finbert`), 영문 금융 텍스트 정확도 높음
4. **공포지수 산출** — 약세비중 70% + 강도 30% 가중합산 → 0~100 정규화
5. **유가 수집** — Yahoo `CL=F` → FRED `DCOILWTICO` → EIA v2 fallback
6. **괴리율** — 공포지수 vs 유가 정규화 차이 (양수 = 시장이 유가 리스크 무시 중)
7. **대시보드** — Plotly 4분할 HTML (`dashboard.html`)
8. **트레이더 코멘트** — 공포지수 구간별 포지션 바이어스 출력

**실행:**
```bash
python market_sentiment_pipeline.py                          # 샘플 데이터 자동 사용
python market_sentiment_pipeline.py --input data.json        # 실제 JSON 데이터
python market_sentiment_pipeline.py --mode transformer       # FinBERT 모드
python market_sentiment_pipeline.py --input d.csv --mode lexicon --out dash.html
```

**출력:** `dashboard.html`, `wordcloud.png`, fear_index 딕셔너리

**공포지수 구간별 포지션 바이어스:**

| 구간 | Zone | 바이어스 |
|---|---|---|
| 75~100 | 극단적 공포 | 역추세 매수 (Contrarian Long) |
| 55~75 | 공포 우세 | 중립 → 소량 Long 탐색 |
| 45~55 | 균형 | 관망 |
| 25~45 | 낙관 우세 | Short 탐색 또는 헤지 |
| 0~25 | 극단적 낙관 | 역추세 Short (Contrarian Short) |

---

### 6. `nasdaq_mc_v2.py` — Nasdaq Monte Carlo (IB급)

**역할:** 공포지수를 입력받아 Nasdaq 향후 N일 경로 시뮬레이션 + 리스크 지표

**v1 대비 개선점:**

| 항목 | v1 | v2 |
|---|---|---|
| 변동성 소스 | 과거 실현변동성 × 공포지수 배수 | VIX 직접 사용 (내재변동성) |
| 점프 모델 | 없음 | **Merton JDM** (포아송 λ 캘리브레이션) |
| 레짐 | 단일 | **저변동/고변동/위기 3개 마르코프 체인** |
| VaR | 정규분포 | **Cornish-Fisher + Historical + CVaR** |

**레짐 파라미터:**

| 레짐 | μ(연) | σ(연) | λ(점프/년) |
|---|---|---|---|
| 저변동 | +15% | 12% | 2.0 |
| 고변동 | +5% | 25% | 6.0 |
| 위기 | -20% | 55% | 15.0 |

**레짐 전환과 공포지수:**
- 공포지수는 변동성에 직접 곱하지 않음 (v1 방식 폐기)
- **레짐 전환 확률에만 반영**: 위기 레짐 진입 확률 추가분 = `fear_index / 200`

**pipeline 연결:**
```python
from market_sentiment_pipeline import run_pipeline
from nasdaq_mc_v2 import run

sentiment = run_pipeline("crawled_data.json")
result = run(fear_index=sentiment["fear_index"])
```

**실행:**
```bash
python nasdaq_mc_v2.py                         # 기본 (fear_index=40)
python nasdaq_mc_v2.py --fear 72 --days 90    # 공포지수·기간 지정
python nasdaq_mc_v2.py --pipeline data.json   # 감성 파이프라인 연동
```

**출력:** `nasdaq_mc_v2.html` (Plotly 인터랙티브 대시보드)

---

## API 키 및 소스 정리

| API | 키 / 방법 | 비용 | 용도 |
|---|---|---|---|
| EIA API v2 | `sshh6mSdDhyaEtl5XGdOAvFiQDQDOSnSRdvvKnrV` | 무료 | WTI(3순위·참고), 재고, SPR |
| Yahoo Finance | 키 불필요 | 무료 | WTI 실시간(1순위), VIX, OVX, 매크로 |
| FRED | 키 불필요 (공개 CSV) | 무료 | WTI(2순위), Brent, DXY |
| IMF PortWatch | 키 불필요 (ArcGIS REST) | 무료 | 호르무즈 선박수 |
| straits.live | 키 불필요 (스크래핑) | 무료 | 호르무즈 현황 보조 |
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

---

## 설치 (한 번만)

```bash
# 기본 (hormuz_monitor + five_house_weighted + oil_fair_value_model)
pip install requests pandas numpy scipy matplotlib schedule yfinance

# Streamlit (oil_fair_value_model.py)
pip install streamlit scikit-learn

# Flask 통합 대시보드 (app.py)
pip install flask

# 감성 파이프라인 (market_sentiment_pipeline.py)
pip install plotly wordcloud

# FinBERT (transformer 모드, 선택)
pip install transformers torch

# Nasdaq MC v2
pip install plotly scipy
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

# Step 4: 통합 공정가 모델 (Streamlit)
streamlit run oil_fair_value_model.py

# Step 5: Flask 통합 대시보드 (전체 기능)
python app.py
# → http://localhost:5000

# Step 6: 감성 분석 (크롤링 데이터 있을 때)
python market_sentiment_pipeline.py --input crawled_data.json

# Step 7: Nasdaq MC (감성 파이프라인 연결 후)
python nasdaq_mc_v2.py --pipeline crawled_data.json

# Step 8: 감성+Nasdaq 통합 실행
python -c "
from market_sentiment_pipeline import run_pipeline
from nasdaq_mc_v2 import run
s = run_pipeline('crawled_data.json')
run(fear_index=s['fear_index'])
"
```

---

## 알려진 갭 / 주의사항

| 항목 | 내용 |
|---|---|
| `oil_fair_value_model.py` 실행 | `streamlit run` 필요 (CLI `python` 직접 실행 불가) |
| IMF PortWatch 래그 | 매주 화요일 오전 9시 ET 업데이트, 실제 2~4일 래그 |
| AIS 스푸핑 | 분쟁 지역 특성상 실제 통과 선박보다 수치 낮을 수 있음 |
| EIA WTI | 주간 데이터 — 알림 판단에 Yahoo Finance 사용, EIA는 재고·SPR 용도만 권장 |
| GS house view | `GS_MAR23_WTI = 80.0` — 신규 리포트 시 수동 업데이트 |
| FinBERT | GPU 없으면 느림. CPU 동작하나 lexicon 모드 권장 |
| straits.live | HTML 구조 변경 시 스크래핑 패턴 수동 업데이트 필요 |

---

## 분석 컨텍스트 (2026-06-18 기준)

| 항목 | 값 |
|---|---|
| 배경 | 이란-이스라엘 전쟁 Operation Roaring Lion (2026.02.28 개시) |
| **현재 상황** | **미국-이란 60일 MOU 체결, 호르무즈 재개방 진행 중** |
| WTI Spot | **$75.54** (전월 대비 -27.5%, 3개월 최저) |
| Brent Spot | ~$76~78 |
| 호르무즈 7dMA | 6척 (정상 138척) → 재배치 중 |
| GS house view | **$80** (구 $105 → 2026-06 재조정) |
| 5사 가중평균 EV | **$78.5** (구 $112.1 → 시나리오 현실화 반영) |
| IEA 2027 전망 | 공급 +8 mb/d vs 수요 +2 mb/d → **연간 +6 mb/d 과잉** 경고 |
| MC P5~P95 | $68~$95 (구 $91~$136) |
| 억류 선박 | 100척+ 해제 전망 → 단기 공급 급증 리스크 |
