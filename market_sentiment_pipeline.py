"""
market_sentiment_pipeline.py
=============================
뉴스/SNS 크롤링 데이터 → 공포지수(0~100) + Plotly 4분할 대시보드

파이프라인 단계:
  1. 데이터 로드    — JSON / CSV 자동 감지 (text, source, timestamp 컬럼)
  2. 워드클라우드   — 금융 키워드 가중치 (폭락×3, 전쟁×3 등)
  3. 감성 분석     — lexicon(사전, 빠름·한국어) / transformer(FinBERT, 영문)
  4. 유가 수집     — yfinance CL=F → FRED DCOILWTICO fallback
  5. 괴리율        — 공포지수 vs 유가 정규화 차이
  6. 대시보드      — Plotly 4분할 HTML 저장
  7. 트레이더 코멘트 — 공포지수 구간별 포지션 바이어스

실행:
  python market_sentiment_pipeline.py                       # 샘플 데이터 자동 생성
  python market_sentiment_pipeline.py --input data.json     # 실제 데이터
  python market_sentiment_pipeline.py --mode transformer    # FinBERT 모드

pip install pandas numpy plotly wordcloud yfinance requests
pip install transformers torch   # transformer 모드만 필요
"""

import os
import json
import argparse
import datetime
import warnings
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[WARN] plotly 미설치 — pip install plotly")

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ─────────────────────────────────────────
# 상수
# ─────────────────────────────────────────
BULLISH_WORDS = {
    "반등": 2, "상승": 2, "급등": 3, "강세": 2, "매수": 2,
    "회복": 2, "낙관": 2, "호재": 3, "공급감소": 2, "수요증가": 2,
    "rally": 2, "surge": 3, "bullish": 2, "rebound": 2, "recovery": 2,
    "supply cut": 2, "demand rise": 2, "sanctions": 2,
}
BEARISH_WORDS = {
    "폭락": 3, "하락": 2, "급락": 3, "약세": 2, "매도": 2,
    "공포": 3, "전쟁": 3, "봉쇄": 3, "위기": 2, "충격": 2,
    "ceasefire": 3, "deal": 2, "oversupply": 3, "recession": 3,
    "crash": 3, "collapse": 3, "bearish": 2, "sell-off": 3,
    "glut": 3, "surplus": 3,
}
FEAR_KEYWORDS_WEIGHT = {
    **{k: v * 1.5 for k, v in BEARISH_WORDS.items()},
    **{k: v for k, v in BULLISH_WORDS.items()},
}

FINBERT_MODEL = "ProsusAI/finbert"


# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """JSON 또는 CSV 자동 감지 로드. text / source / timestamp 컬럼 정규화."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw if isinstance(raw, list) else raw.get("articles", raw.get("data", [])))
    else:
        df = pd.read_csv(path, encoding="utf-8")

    # 컬럼 정규화
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("text", "content", "body", "headline", "title"):
            col_map[c] = "text"
        elif lc in ("source", "src", "domain", "publisher"):
            col_map[c] = "source"
        elif lc in ("timestamp", "date", "time", "created_at", "published_at"):
            col_map[c] = "timestamp"
    df = df.rename(columns=col_map)

    if "text" not in df.columns:
        raise ValueError("'text' 컬럼(또는 content/body/headline)이 없습니다.")
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "timestamp" not in df.columns:
        df["timestamp"] = datetime.datetime.now().isoformat()

    df["text"] = df["text"].fillna("").astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.reset_index(drop=True)


def _sample_data() -> pd.DataFrame:
    """실제 데이터 없을 때 사용할 샘플 생성 (2026-04 ~ 2026-06 호르무즈 상황)."""
    texts_bear = [
        "호르무즈 봉쇄 95% 지속, 원유 선박 통과 6척으로 급감",
        "Iran blockade tightens: oil tankers rerouted, supply shock imminent",
        "IEA 경고: 글로벌 원유 공급 위기 심화, 유가 폭락 가능성",
        "전쟁 리스크 상승, 원유 선물 급등 후 급락, 투자자 혼란",
        "Hormuz closure fears surge as tanker dark AIS signals spike",
        "OPEC 생산 차질 우려, 원유 수급 충격 경보",
        "Oil prices crash on oversupply fears, IEA projects 8mbpd glut by 2027",
        "미국-이란 협상 결렬 위기, 호르무즈 봉쇄 장기화 가능성",
    ]
    texts_bull = [
        "US-Iran ceasefire talks progress, Hormuz reopening expected",
        "미국·이란 MOU 60일 연장 합의, 호르무즈 재개방 전망",
        "Oil rebounds on Trump warning over Iran, uncertainty lingers",
        "Tanker operators reposition to Gulf as Hormuz deal near",
        "원유 재고 감소세, 수요 회복 기대감 상승",
    ]
    now = datetime.datetime.now()
    rows = []
    for i, t in enumerate(texts_bear):
        rows.append({"text": t, "source": "reuters" if i % 2 == 0 else "bloomberg",
                     "timestamp": now - datetime.timedelta(days=i * 3)})
    for i, t in enumerate(texts_bull):
        rows.append({"text": t, "source": "ft" if i % 2 == 0 else "wsj",
                     "timestamp": now - datetime.timedelta(days=i * 2 + 1)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 2. 워드클라우드
# ─────────────────────────────────────────
def build_wordcloud(df: pd.DataFrame, out_path: str = "wordcloud.png") -> str:
    """키워드 가중치 적용 워드클라우드 생성. 반환: 저장 경로."""
    if not HAS_WC:
        print("[SKIP] wordcloud 미설치")
        return ""

    all_text = " ".join(df["text"].tolist())

    # 가중치 적용: 고빈도 금융 키워드를 반복 삽입
    weighted_tokens = []
    for word, weight in FEAR_KEYWORDS_WEIGHT.items():
        if word.lower() in all_text.lower():
            weighted_tokens.extend([word] * int(weight * 3))

    weighted_tokens.extend(all_text.split())
    text_for_wc = " ".join(weighted_tokens)

    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        max_words=120,
        colormap="RdYlGn_r",
        collocations=False,
    ).generate(text_for_wc)

    wc.to_file(out_path)
    print(f"[WordCloud] 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────
# 3. 감성 분석
# ─────────────────────────────────────────
def _lexicon_sentiment(text: str) -> float:
    """
    사전 기반 감성 점수 (-1 ~ +1).
    양수 = 강세(bullish), 음수 = 약세(bearish).
    """
    text_l = text.lower()
    score = 0.0
    for w, v in BULLISH_WORDS.items():
        if w.lower() in text_l:
            score += v
    for w, v in BEARISH_WORDS.items():
        if w.lower() in text_l:
            score -= v
    # 정규화: 최대 가능 점수로 나누기
    return float(np.clip(score / 10.0, -1.0, 1.0))


def _transformer_sentiment(texts: list) -> list:
    """FinBERT 기반 감성 분석. 반환: [-1, +1] 리스트."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            truncation=True,
            max_length=512,
        )
        results = pipe(texts, batch_size=8)
        scores = []
        for r in results:
            label = r["label"].lower()
            conf = float(r["score"])
            if label == "positive":
                scores.append(conf)
            elif label == "negative":
                scores.append(-conf)
            else:
                scores.append(0.0)
        return scores
    except Exception as e:
        print(f"[FinBERT] 실패 ({e}), lexicon fallback")
        return [_lexicon_sentiment(t) for t in texts]


def run_sentiment(df: pd.DataFrame, mode: str = "lexicon") -> pd.DataFrame:
    """감성 점수 컬럼 추가. mode: 'lexicon' | 'transformer'."""
    print(f"[Sentiment] 모드: {mode}, 건수: {len(df)}")
    if mode == "transformer":
        scores = _transformer_sentiment(df["text"].tolist())
    else:
        scores = [_lexicon_sentiment(t) for t in df["text"]]

    df = df.copy()
    df["sentiment_score"] = scores
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda s: "bearish" if s < -0.1 else ("bullish" if s > 0.1 else "neutral")
    )
    return df


# ─────────────────────────────────────────
# 4. 공포지수 산출
# ─────────────────────────────────────────
def compute_fear_index(df: pd.DataFrame) -> dict:
    """
    공포지수 (0~100).
    0  = 극단적 낙관, 100 = 극단적 공포.
    산출: 약세 비중 × 가중치 합산 + 부정 강도 반영.
    """
    if df.empty or "sentiment_score" not in df.columns:
        return {"fear_index": 50.0, "bearish_pct": 0.5, "bullish_pct": 0.5, "neutral_pct": 0.0}

    scores = df["sentiment_score"].values
    bearish_pct = float((scores < -0.1).mean())
    bullish_pct = float((scores > 0.1).mean())
    neutral_pct = float(np.abs(scores <= 0.1).mean())

    # 평균 강도 (-1~+1 → 공포 기여)
    mean_score = float(scores.mean())
    intensity = (-mean_score + 1) / 2  # 양수=공포 기여 높음

    # 가중 공포지수: 약세 비중 70% + 강도 30%
    fear_raw = 0.70 * bearish_pct + 0.30 * intensity
    fear_index = float(np.clip(fear_raw * 100, 0, 100))

    return {
        "fear_index":   round(fear_index, 1),
        "bearish_pct":  round(bearish_pct * 100, 1),
        "bullish_pct":  round(bullish_pct * 100, 1),
        "neutral_pct":  round(neutral_pct * 100, 1),
        "mean_score":   round(mean_score, 4),
        "n_articles":   len(df),
    }


# ─────────────────────────────────────────
# 5. 유가 수집 (Yahoo → FRED)
# ─────────────────────────────────────────
def fetch_oil_price() -> dict:
    """
    WTI 현재가 수집.
    우선순위: Yahoo Finance CL=F (15분 딜레이 실시간)
              → FRED DCOILWTICO (전일 확정, API키 불필요)
              → fallback 75.54 (2026-06-18 기준)
    ※ EIA v2 주간 데이터는 알림 판단에 래그가 크므로 3순위로 밀고
      실시간에 가장 근접한 Yahoo를 1순위로 사용.
    """
    # 1순위: Yahoo Finance
    if HAS_YF:
        try:
            import yfinance as yf
            df = yf.download("CL=F", period="5d", interval="1d",
                             progress=False, auto_adjust=True)
            arr = df["Close"].dropna()
            if not arr.empty:
                price = float(arr.iloc[-1])
                series = arr.values.astype(float).flatten()
                date_str = str(arr.index[-1])[:10]
                print(f"[WTI] Yahoo Finance: ${price:.2f} ({date_str})")
                return {"price": price, "series": series, "source": "Yahoo Finance (CL=F)",
                        "date": date_str}
        except Exception as e:
            print(f"[WTI] Yahoo 실패: {e}")

    # 2순위: FRED DCOILWTICO
    try:
        from io import StringIO
        r = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO",
            timeout=12, headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        df2 = pd.read_csv(StringIO(r.text))
        df2.columns = ["date", "value"]
        df2["value"] = pd.to_numeric(df2["value"], errors="coerce")
        df2 = df2.dropna()
        if not df2.empty:
            price = float(df2["value"].iloc[-1])
            series = df2["value"].values.astype(float)
            date_str = str(df2["date"].iloc[-1])
            print(f"[WTI] FRED: ${price:.2f} ({date_str})")
            return {"price": price, "series": series, "source": "FRED (DCOILWTICO)",
                    "date": date_str}
    except Exception as e:
        print(f"[WTI] FRED 실패: {e}")

    # 3순위: EIA v2 RWTC (주간, 래그 있음)
    try:
        EIA_KEY = os.getenv("EIA_API_KEY", "sshh6mSdDhyaEtl5XGdOAvFiQDQDOSnSRdvvKnrV")
        r = requests.get(
            "https://api.eia.gov/v2/petroleum/pri/spt/data/",
            params={
                "api_key": EIA_KEY, "frequency": "weekly",
                "data[0]": "value", "facets[series][]": "RWTC",
                "sort[0][column]": "period", "sort[0][direction]": "desc",
                "length": 10,
            }, timeout=12,
        )
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        if data:
            price = float(data[0]["value"])
            series = np.array([float(d["value"]) for d in reversed(data)])
            date_str = str(data[0]["period"])
            print(f"[WTI] EIA v2 RWTC (주간): ${price:.2f} ({date_str}) ← 래그 있음")
            return {"price": price, "series": series, "source": "EIA v2 RWTC (weekly)",
                    "date": date_str}
    except Exception as e:
        print(f"[WTI] EIA 실패: {e}")

    print("[WTI] 모든 소스 실패 → fallback $75.54")
    return {"price": 75.54, "series": np.array([75.54]), "source": "fallback", "date": "2026-06-18"}


# ─────────────────────────────────────────
# 6. 괴리율
# ─────────────────────────────────────────
def compute_divergence(fear_index: float, oil_price: float,
                       fear_baseline: float = 50.0,
                       oil_baseline: float = 75.54) -> dict:
    """
    공포지수 vs 유가 정규화 괴리율.
    양수 = 시장이 유가 리스크를 무시 중 (fear↑ but oil↓)
    음수 = 유가가 공포보다 과도하게 반영 중
    """
    fear_norm = (fear_index - fear_baseline) / 100.0
    oil_norm  = (oil_price  - oil_baseline) / oil_baseline
    divergence = float(fear_norm - oil_norm)

    return {
        "divergence":    round(divergence * 100, 2),
        "fear_norm":     round(fear_norm * 100, 2),
        "oil_norm":      round(oil_norm * 100, 2),
        "interpretation": (
            "시장이 유가 리스크를 과소반영 중 (공포 대비 유가 저평가)"
            if divergence > 0.05 else (
                "유가가 공포보다 과도하게 반영 중 (유가 상방 리스크 과대)"
                if divergence < -0.05 else "공포지수와 유가 수준 균형"
            )
        ),
    }


# ─────────────────────────────────────────
# 7. 트레이더 코멘트
# ─────────────────────────────────────────
def trader_comment(fear_index: float) -> dict:
    """공포지수 구간별 포지션 바이어스 출력."""
    if fear_index >= 75:
        return {
            "zone": "극단적 공포",
            "bias": "역추세 매수 고려 (Contrarian Long)",
            "rationale": "공포 극대화 → 저평가 가능성. 지정학 리스크 해소 시 빠른 반등 기대.",
            "stop_hint": "GS Mar23 house view $78 하단 이탈 시 손절",
        }
    elif fear_index >= 55:
        return {
            "zone": "공포 우세",
            "bias": "중립 → 소량 Long 탐색",
            "rationale": "공포 우위이나 극단 미달. 수급 데이터 확인 후 진입.",
            "stop_hint": "호르무즈 선박수 회복 속도 체크",
        }
    elif fear_index >= 45:
        return {
            "zone": "균형",
            "bias": "관망",
            "rationale": "뚜렷한 방향성 없음. 호르무즈 뉴스·EIA 재고 발표 대기.",
            "stop_hint": "없음 — 포지션 보유 불필요",
        }
    elif fear_index >= 25:
        return {
            "zone": "낙관 우세",
            "bias": "Short 탐색 또는 헤지",
            "rationale": "시장 낙관 과도 가능. IEA 공급 과잉 경고 감안 시 하방 리스크.",
            "stop_hint": "미국-이란 협상 결렬 뉴스 시 즉시 청산",
        }
    else:
        return {
            "zone": "극단적 낙관",
            "bias": "역추세 Short (Contrarian Short)",
            "rationale": "낙관 극대화 → 과매수 가능. 공급 과잉 현실화 시 급락 리스크.",
            "stop_hint": "새로운 지정학 충격 발생 시 손절",
        }


# ─────────────────────────────────────────
# 8. Plotly 4분할 대시보드
# ─────────────────────────────────────────
def build_dashboard(df: pd.DataFrame, fear_info: dict, oil_info: dict,
                    div_info: dict, out_path: str = "dashboard.html") -> str:
    """Plotly 4분할 대시보드 HTML 저장."""
    if not HAS_PLOTLY:
        print("[SKIP] plotly 미설치 — dashboard 생성 불가")
        return ""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "감성 분포 (Sentiment Distribution)",
            f"공포지수 게이지 (Fear Index: {fear_info['fear_index']:.1f})",
            "시간별 감성 추이 (Sentiment Over Time)",
            "공포지수 vs 유가 괴리 (Divergence)",
        ),
        specs=[[{"type": "pie"}, {"type": "indicator"}],
               [{"type": "scatter"}, {"type": "bar"}]],
    )

    # 패널 1: 감성 파이 차트
    pie_labels = ["Bearish", "Bullish", "Neutral"]
    pie_values = [fear_info["bearish_pct"], fear_info["bullish_pct"], fear_info["neutral_pct"]]
    fig.add_trace(go.Pie(
        labels=pie_labels, values=pie_values,
        marker_colors=["#E24B4A", "#3FB950", "#8B949E"],
        textinfo="label+percent", hole=0.35,
    ), row=1, col=1)

    # 패널 2: 공포지수 게이지
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=fear_info["fear_index"],
        delta={"reference": 50, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#E24B4A" if fear_info["fear_index"] > 55 else
                             "#3FB950" if fear_info["fear_index"] < 45 else "#D29922"},
            "steps": [
                {"range": [0, 25],   "color": "#c0dd97"},
                {"range": [25, 45],  "color": "#fac775"},
                {"range": [45, 55],  "color": "#f0f0f0"},
                {"range": [55, 75],  "color": "#f7c1c1"},
                {"range": [75, 100], "color": "#f09595"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": fear_info["fear_index"]},
        },
        number={"suffix": " / 100"},
    ), row=1, col=2)

    # 패널 3: 시간별 감성 추이
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df_ts = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        fig.add_trace(go.Scatter(
            x=df_ts["timestamp"], y=df_ts["sentiment_score"],
            mode="markers+lines",
            marker=dict(
                color=df_ts["sentiment_score"].apply(
                    lambda s: "#E24B4A" if s < -0.1 else "#3FB950" if s > 0.1 else "#8B949E"
                ),
                size=8,
            ),
            line=dict(color="#58A6FF", width=1),
            name="Sentiment Score",
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=0.1, line_dash="dot", line_color="#3FB950", row=2, col=1)
        fig.add_hline(y=-0.1, line_dash="dot", line_color="#E24B4A", row=2, col=1)

    # 패널 4: 공포지수 vs 유가 괴리 바 차트
    bar_labels = ["공포지수\n정규화 (%)", "유가 변화율\n(%)", "괴리율\n(공포-유가)"]
    bar_values = [div_info["fear_norm"], div_info["oil_norm"], div_info["divergence"]]
    bar_colors = [
        "#E24B4A" if v > 0 else "#3FB950"
        for v in bar_values
    ]
    fig.add_trace(go.Bar(
        x=bar_labels, y=bar_values,
        marker_color=bar_colors,
        text=[f"{v:+.1f}%" for v in bar_values],
        textposition="outside",
        name="Divergence",
    ), row=2, col=2)
    fig.add_hline(y=0, line_color="gray", row=2, col=2)

    # 레이아웃
    fig.update_layout(
        title=dict(
            text=(
                f"OilPriceValuation — Market Sentiment Pipeline | "
                f"WTI ${oil_info['price']:.2f} ({oil_info['source']}) | "
                f"Fear Index {fear_info['fear_index']:.1f} | "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M KST')}"
            ),
            font=dict(size=13),
        ),
        height=700,
        showlegend=False,
        template="plotly_white",
    )

    fig.write_html(out_path)
    print(f"[Dashboard] 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────
# 9. 메인 파이프라인
# ─────────────────────────────────────────
def run_pipeline(input_path: str = None,
                 sentiment_mode: str = "lexicon",
                 dashboard_out: str = "dashboard.html",
                 wordcloud_out: str = "wordcloud.png") -> dict:
    """
    전체 파이프라인 실행.
    반환: {
        "fear_index": float,       # 0~100
        "bearish_pct": float,
        "bullish_pct": float,
        "neutral_pct": float,
        "mean_score": float,
        "n_articles": int,
        "wti_price": float,
        "divergence": float,
        "trader_comment": dict,
        "dashboard_path": str,
        "wordcloud_path": str,
    }
    """
    print("\n" + "=" * 60)
    print("  Market Sentiment Pipeline  |  OilPriceValuation KFBI")
    print("=" * 60)

    # Step 1: 데이터 로드
    if input_path and os.path.exists(input_path):
        df = load_data(input_path)
        print(f"[Load] {len(df)}건 로드: {input_path}")
    else:
        print("[Load] 입력 파일 없음 → 샘플 데이터 사용")
        df = _sample_data()

    # Step 2: 워드클라우드
    wc_path = build_wordcloud(df, wordcloud_out)

    # Step 3: 감성 분석
    df = run_sentiment(df, mode=sentiment_mode)

    # Step 4: 공포지수
    fear_info = compute_fear_index(df)
    print(f"[Fear Index] {fear_info['fear_index']:.1f} "
          f"(Bearish {fear_info['bearish_pct']:.0f}% / "
          f"Bullish {fear_info['bullish_pct']:.0f}% / "
          f"Neutral {fear_info['neutral_pct']:.0f}%)")

    # Step 5: 유가 수집
    oil_info = fetch_oil_price()

    # Step 6: 괴리율
    div_info = compute_divergence(fear_info["fear_index"], oil_info["price"])
    print(f"[Divergence] {div_info['divergence']:+.1f}%  → {div_info['interpretation']}")

    # Step 7: 트레이더 코멘트
    comment = trader_comment(fear_info["fear_index"])
    print(f"\n[Trader] Zone: {comment['zone']}")
    print(f"         Bias: {comment['bias']}")
    print(f"         {comment['rationale']}")
    print(f"         Stop: {comment['stop_hint']}")

    # Step 8: 대시보드
    dash_path = build_dashboard(df, fear_info, oil_info, div_info, dashboard_out)

    print("\n" + "=" * 60)

    return {
        **fear_info,
        "wti_price":      oil_info["price"],
        "wti_source":     oil_info["source"],
        "divergence":     div_info["divergence"],
        "interpretation": div_info["interpretation"],
        "trader_comment": comment,
        "dashboard_path": dash_path,
        "wordcloud_path": wc_path,
    }


# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Sentiment Pipeline")
    parser.add_argument("--input",   default=None,      help="입력 파일 경로 (JSON/CSV)")
    parser.add_argument("--mode",    default="lexicon",  help="감성 분석 모드: lexicon | transformer")
    parser.add_argument("--out",     default="dashboard.html", help="대시보드 출력 경로")
    parser.add_argument("--wc",      default="wordcloud.png",  help="워드클라우드 출력 경로")
    args = parser.parse_args()

    result = run_pipeline(
        input_path=args.input,
        sentiment_mode=args.mode,
        dashboard_out=args.out,
        wordcloud_out=args.wc,
    )

    print(f"\n최종 결과: fear_index={result['fear_index']}, "
          f"WTI=${result['wti_price']:.2f}, "
          f"divergence={result['divergence']:+.1f}%")
