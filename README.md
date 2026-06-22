# OilPriceValuation — KFBI 금융시장분석팀

> **Operation Roaring Lion (2026.02.28~)** 호르무즈 봉쇄 상황에서  
> WTI/Brent 적정가 산출 및 시장 리스크 자동 모니터링 시스템
>
> **최종 파라미터 업데이트: 2026-06-18**  
> 미국-이란 60일 MOU 체결, 호르무즈 재개방 진행, IEA 공급 과잉 경고 반영  
> **아키텍처 업데이트: 2026-06-22**  
> 크롤링 모듈 분리(`data_crawler.py`), ML 파이프라인 분류→회귀 2단계 전환

---

## 프로젝트 구조

```
oilpricevaluation/
│
├── data_crawler.py            # ★ 크롤링 전용 모듈 (API 과사용 방지, 3시간 데몬)
├── app.py                     # Flask 통합 대시보드 (크롤링은 data_crawler 위임)
├── hormuz_monitor.py          # 호르무즈 실시간 모니터링 데몬
├── five_house_weighted.py     # 5사 가중치 시나리오 분석 + Monte Carlo
├── oil_fair_value_model.py    # MS Kalman + GS Bridge + RF 통합 모델 (Streamlit)
├── market_sentiment_pipeline.py  # 뉴스/SNS 감성 분석 파이프라인
├── nasdaq_mc_v2.py            # Nasdaq Monte Carlo (Merton JDM + 레짐 스위칭)
├── data/                      # ★ 크롤링 결과 CSV 중앙 저장소
│   ├── yahoo_finance_wti.csv
│   ├── yahoo_finance_brent.csv
│   ├── yahoo_finance_dxy.csv
│   ├── yahoo_finance_ovx.csv
│   ├── yahoo_finance_uso.csv
│   ├── yahoo_finance_bno.csv
│   ├── yahoo_finance_futures_curve.csv
│   ├── brent_wti_spread.csv
│   ├── eia_commercial_inventory.csv
│   ├── eia_spr.csv
│   ├── eia_wpsr_gasoline.csv
│   ├── eia_wpsr_distillate.csv
│   ├── macro_snapshot.csv
│   ├── arcgis_hormuz.csv
│   ├── straits_live_snapshot.csv
│   └── portwatch_cache.csv    # hormuz_monitor 캐시 (구: 루트 저장 → data/ 이동)
└── README.md
```

---

## 전체 데이터 흐름

```
Yahoo Finance (WTI/Brent/DXY/OVX/USO/BNO + 선물커브 M1~M6)
EIA API v2    (상업재고·SPR·WPSR 가솔린·중간유)
IMF PortWatch ArcGIS (호르무즈 일별 선박 365일)
straits.live  (호르무즈 실시간 스냅샷)
매크로 10종   (TNX·SP500·Gold·Copper·VIX·TIP·XLI·FXI·MCHI·MSCI)
        │
        ▼
  data_crawler.py   ←── python data_crawler.py --daemon  (3시간 간격 자동 실행)
        │  CSV → data/
        ▼
  data/ (중앙 저장소)
        │
        ├──▶ app.py          (Flask 대시보드, data_crawler에서 임포트)
        │      ├── MS Kalman Filter (잠재 펀더멘털)
        │      ├── GS Two-Stage Bridge (수급 → 지정학 프리미엄)
        │      ├── RF Supply Shock Index
        │      ├── ML 예측: RF 분류(방향) → RF 회귀(크기) 2단계  ← ★ 2026-06-22 변경
        │      ├── Monte Carlo GBM (30/60/90일)
        │      └── 5사 가중 시나리오 EV
        │
        └──▶ hormuz_monitor.py  (portwatch_cache.csv → data/ 저장)
                │
                ▼
             알림 출력

뉴스/SNS 크롤링 데이터 (JSON/CSV)
        │
        ▼
market_sentiment_pipeline.py
        │  fear_index (0~100)
        ▼
nasdaq_mc_v2.py ──▶ Nasdaq 경로 시뮬레이션 + VaR

app.py / oil_fair_value_model.py 결과
        │
        ▼
five_house_weighted.py  (5사 가중 시나리오 EV 독립 실행)
```

---

## 파일별 상세

### 0. `data_crawler.py` — 크롤링 전용 모듈 ★ 2026-06-22 신규

**역할:** 모든 API 호출을 `app.py`에서 분리하여 API 과사용 방지 및 독립 실행 가능

**수집 소스 (1회 실행 기준):**

| 소스 | 데이터 | 저장 파일 |
|---|---|---|
| Yahoo Finance | WTI·Brent·DXY·OVX·USO·BNO (1년 일별) | `yahoo_finance_*.csv` |
| Yahoo Finance | WTI 선물 커브 M1~M6 | `yahoo_finance_futures_curve.csv` |
| Yahoo Finance | 매크로 10종 (TNX·SP500·Gold·Copper·VIX 등) | `macro_snapshot.csv` |
| EIA API v2 | 상업재고 60주, SPR 60주 | `eia_commercial_inventory.csv`, `eia_spr.csv` |
| EIA WPSR | 가솔린·중간유 재고 60주 | `eia_wpsr_gasoline.csv`, `eia_wpsr_distillate.csv` |
| ArcGIS FeatureServer | 호르무즈 일별 선박 365일 | `arcgis_hormuz.csv` |
| straits.live | 호르무즈 현황 스냅샷 | `straits_live_snapshot.csv` |
| 계산값 | Brent-WTI 스프레드 | `brent_wti_spread.csv` |

**캐시 TTL:** 3600초 (1시간) — 앱 재시작마다 재크롤링 방지

**실행:**
```bash
python data_crawler.py                      # 1회 실행 (TTL 내 캐시 재사용)
python data_crawler.py --force              # 강제 재크롤링
python data_crawler.py --daemon             # 3시간 간격 무한 반복
python data_crawler.py --daemon --interval 90  # 90분 간격으로 변경
```

**`app.py` 연동:**
```python
from data_crawler import fetch_all_data, CSV_DIR, EIA_KEY, EIA_BASE, _cache
```
`app.py`는 직접 API를 호출하지 않고 `data_crawler.fetch_all_data()`만 호출함.

---

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

**캐시 저장 경로 (2026-06-22 변경):** `portwatch_cache.csv` (프로젝트 루트) → `data/portwatch_cache.csv`

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

**역할:** 전체 기능을 하나의 웹 대시보드로 통합. 크롤링은 `data_crawler.py`에 위임.

**2026-06-22 아키텍처 변경:**
- 크롤링 코드 전량 `data_crawler.py`로 이동 (`app.py`는 `fetch_all_data()` 호출만 함)
- ML 파이프라인: **순수 회귀 → 분류(방향)→회귀(크기) 2단계**로 전환

**ML 2단계 파이프라인 상세:**

| 단계 | 모델 | 목표 | 하이퍼파라미터 튜닝 |
|---|---|---|---|
| Stage 1 | RF Classifier | 방향 분류 (-1=하락 / 0=중립 / +1=상승) | `scoring="balanced_accuracy"`, `class_weight="balanced"` |
| Stage 2 | RF Regressor | 수익률 크기 예측 | `scoring="neg_mean_absolute_error"` |
| 결합 | — | 방향 일치 → 원값 / 불일치 → 절댓값×0.5 / 중립 → ×0.2 | — |

- 방향 기준: `DIRECTION_THRESHOLD = ±0.5%` (이내는 중립 0)
- 공통: `TimeSeriesSplit(n_splits=5)` + `RandomizedSearchCV(n_iter=20)`
- 출력 추가: `dir_label`(상승▲/중립─/하락▼), `dir_acc`(방향 정확도 %), `dir_proba`(클래스별 확률)

**데이터 소스 (data_crawler.py 경유):**

| 소스 | 데이터 | 비고 |
|---|---|---|
| Yahoo Finance USO / BNO ETF | 원유 ETF 가격 | WTI·Brent 프록시 |
| Yahoo Finance ^TNX, ^GSPC, URTH, GC=F, HG=F, ^VIX, TIP, XLI, FXI, MCHI | 매크로 10종 | GS Bridge Stage B 보강 |
| EIA WPSR 가솔린·중간유 재고 | 정제·수요 수급 | RF 피처 확장 |
| straits.live | 호르무즈 현황 스냅샷 | 보조 지표 |
| WTI 선물 커브 M1~M6 | 콘탱고/백워데이션 | Stage B 피처 |

**실행:**
```bash
# 크롤링 먼저 (최초 1회 또는 데몬 상시 실행)
python data_crawler.py --daemon

# 대시보드 실행
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
# 크롤러 + Flask 대시보드 (data_crawler.py + app.py)
pip install requests pandas numpy scipy matplotlib yfinance scikit-learn flask beautifulsoup4

# 호르무즈 모니터 + 5사 시나리오
pip install schedule

# Streamlit (oil_fair_value_model.py)
pip install streamlit

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
# Step 0: 데이터 크롤링 (최초 1회 필수 — data/ 폴더에 CSV 저장)
python data_crawler.py --force

# Step 0-daemon: 3시간 간격 자동 갱신 (백그라운드 상시 실행 권장)
python data_crawler.py --daemon

# Step 1: Flask 통합 대시보드 (크롤링 후 실행)
python app.py
# → http://localhost:5000

# Step 2: 호르무즈 모니터 1회 확인
python hormuz_monitor.py

# Step 3: 호르무즈 모니터 데몬
python hormuz_monitor.py --daemon

# Step 4: 5사 시나리오 분석 차트
python five_house_weighted.py

# Step 5: 통합 공정가 모델 (Streamlit)
streamlit run oil_fair_value_model.py

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
| EIA 생산량·원유수입 | DEMO_KEY rate limit 또는 facet 파라미터 문제로 간헐적 SIM 전환 |
| FRED WTI/Brent | 공개 CSV 엔드포인트 간헐적 차단 — yfinance 데이터로 대체됨 |
| GS house view | `GS_MAR23_WTI = 80.0` — 신규 리포트 시 수동 업데이트 |
| FinBERT | GPU 없으면 느림. CPU 동작하나 lexicon 모드 권장 |
| straits.live | HTML 구조 변경 시 스크래핑 패턴 수동 업데이트 필요 |
| data/ CSV 누적 | 현재 최신 스냅샷 1개만 유지 (덮어쓰기 방식) — 히스토리 필요 시 파일명에 날짜 추가 필요 |
| data_crawler.py 터미널 출력 | Windows CP949 환경에서 한글 깨짐 (데이터 저장은 UTF-8-BOM으로 정상) |

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
