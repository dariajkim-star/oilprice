"""
nasdaq_mc_v2.py
===============
Nasdaq Monte Carlo — IB급 가격 경로 시뮬레이터

v1 대비 개선:
  - 변동성 소스: 과거 실현변동성 × 공포지수 배수 → VIX 내재변동성 직접 사용
  - 점프 모델:  없음 → Merton JDM (포아송 λ 캘리브레이션)
  - 레짐:       단일 → 저변동/고변동/위기 3개 마르코프 체인
  - VaR:        정규분포 → Cornish-Fisher + Historical + CVaR

공포지수와 레짐의 관계:
  - 공포지수는 변동성에 직접 곱하지 않음
  - 레짐 전환 확률에만 반영 → 위기 레짐 진입 확률 = fear_index / 200

pipeline 연결:
  from market_sentiment_pipeline import run_pipeline
  from nasdaq_mc_v2 import run

  sentiment = run_pipeline("crawled_data.json")
  result = run(fear_index=sentiment["fear_index"])

실행:
  python nasdaq_mc_v2.py                        # 기본 (fear_index=40)
  python nasdaq_mc_v2.py --fear 72 --days 90   # 공포지수·기간 지정
  python nasdaq_mc_v2.py --pipeline data.json  # 감성 파이프라인 연동

pip install numpy scipy plotly yfinance requests pandas
"""

import os
import argparse
import datetime
import warnings
import numpy as np
import requests
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
    from scipy.stats import norm, kurtosis, skew
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ─────────────────────────────────────────
# 상수
# ─────────────────────────────────────────
NASDAQ_TICKER = "^IXIC"    # Nasdaq Composite
VIX_TICKER    = "^VIX"

# 레짐 파라미터 (연율화 수익률, 변동성)
REGIMES = {
    "low_vol": {
        "mu_annual":    0.15,   # 연간 드리프트
        "sigma_annual": 0.12,   # 연간 변동성
        "jump_lambda":  2.0,    # 연간 점프 횟수 (포아송)
        "jump_mu":     -0.01,   # 점프 평균 크기 (로그)
        "jump_sigma":   0.02,   # 점프 크기 표준편차
    },
    "high_vol": {
        "mu_annual":    0.05,
        "sigma_annual": 0.25,
        "jump_lambda":  6.0,
        "jump_mu":     -0.03,
        "jump_sigma":   0.05,
    },
    "crisis": {
        "mu_annual":   -0.20,
        "sigma_annual": 0.55,
        "jump_lambda":  15.0,
        "jump_mu":     -0.06,
        "jump_sigma":   0.08,
    },
}

# 마르코프 전환 행렬 기본값 (행=현재 레짐, 열=다음 레짐)
# [low_vol, high_vol, crisis]
TRANSITION_BASE = np.array([
    [0.97, 0.025, 0.005],   # low_vol
    [0.15, 0.80,  0.050],   # high_vol
    [0.10, 0.30,  0.600],   # crisis
])


# ─────────────────────────────────────────
# 1. 시장 데이터 수집
# ─────────────────────────────────────────
def fetch_market_data() -> dict:
    """Nasdaq + VIX 실제 데이터 수집. Yahoo → FRED → fallback."""
    result = {"nasdaq": None, "vix": None, "source": "fallback"}

    if HAS_YF:
        try:
            raw = yf.download([NASDAQ_TICKER, VIX_TICKER],
                              period="1y", interval="1d",
                              progress=False, auto_adjust=True)
            for ticker, key in [(NASDAQ_TICKER, "nasdaq"), (VIX_TICKER, "vix")]:
                try:
                    col = raw[ticker]["Close"] if ticker in raw.columns.get_level_values(0) else None
                    if col is None:
                        df_s = yf.download(ticker, period="1y", interval="1d",
                                           progress=False, auto_adjust=True)
                        col = df_s["Close"]
                    arr = col.dropna().values.astype(float).flatten()
                    if len(arr) > 10:
                        result[key] = arr
                except Exception:
                    pass
            if result["nasdaq"] is not None:
                result["source"] = "Yahoo Finance"
                print(f"[Market] Yahoo Finance: Nasdaq={result['nasdaq'][-1]:.1f}, "
                      f"VIX={result['vix'][-1] if result['vix'] is not None else 'N/A'}")
                return result
        except Exception as e:
            print(f"[Market] Yahoo 실패: {e}")

    # FRED: Nasdaq composite (NASDAQCOM)
    try:
        from io import StringIO
        r = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM",
            timeout=12, headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        arr = df.dropna()["value"].values.astype(float)
        if len(arr) > 10:
            result["nasdaq"] = arr
            result["source"] = "FRED NASDAQCOM"
            print(f"[Market] FRED: Nasdaq={arr[-1]:.1f}")
            return result
    except Exception as e:
        print(f"[Market] FRED 실패: {e}")

    # Fallback: 2026-04 기준 대략값
    print("[Market] fallback 데이터 사용 (Nasdaq ~18500)")
    rng = np.random.default_rng(42)
    result["nasdaq"] = 18500 + np.cumsum(rng.normal(0, 80, 260))
    result["vix"]    = np.clip(15 + np.cumsum(rng.normal(0, 1.5, 260)), 10, 80)
    return result


# ─────────────────────────────────────────
# 2. 파라미터 캘리브레이션
# ─────────────────────────────────────────
def calibrate_params(nasdaq_series: np.ndarray,
                     vix_series: np.ndarray = None,
                     fear_index: float = 40.0) -> dict:
    """
    역사적 수익률로 일간 변동성 캘리브레이션 + 레짐 판별.
    VIX가 있으면 내재변동성 직접 사용.
    공포지수는 레짐 전환 확률에만 반영.
    """
    log_rets = np.diff(np.log(np.clip(nasdaq_series, 1, None)))
    hist_vol_annual = float(np.std(log_rets) * np.sqrt(252))
    hist_mu_annual  = float(np.mean(log_rets) * 252)

    # 현재 레짐 판별
    recent_vol = float(np.std(log_rets[-20:]) * np.sqrt(252)) if len(log_rets) >= 20 else hist_vol_annual
    vix_latest = float(vix_series[-1]) / 100.0 if vix_series is not None and len(vix_series) > 0 else None

    # VIX 직접 사용 (v2 핵심 개선)
    implied_vol = vix_latest if vix_latest else recent_vol

    if implied_vol > 0.40:
        current_regime = "crisis"
    elif implied_vol > 0.22:
        current_regime = "high_vol"
    else:
        current_regime = "low_vol"

    # Merton JDM 캘리브레이션: 점프 식별 (|수익률| > 3σ)
    threshold = 3 * np.std(log_rets)
    jump_mask = np.abs(log_rets) > threshold
    n_jumps   = jump_mask.sum()
    lambda_hat = float(n_jumps / len(log_rets) * 252)  # 연간 점프 횟수
    jump_mu_hat    = float(log_rets[jump_mask].mean()) if n_jumps > 0 else -0.03
    jump_sigma_hat = float(log_rets[jump_mask].std())  if n_jumps > 1 else 0.05

    # 레짐 전환 행렬: fear_index 반영
    # 위기 레짐 진입 확률 = fear_index / 200 (최대 50% 추가)
    fear_adj = float(np.clip(fear_index / 200.0, 0.0, 0.30))
    T = TRANSITION_BASE.copy()
    # low_vol → crisis 확률 증가, low_vol → low_vol 감소
    T[0, 2] = float(np.clip(T[0, 2] + fear_adj * 0.5, 0, 0.20))
    T[1, 2] = float(np.clip(T[1, 2] + fear_adj * 0.8, 0, 0.30))
    # 행 합계 1로 정규화
    T = T / T.sum(axis=1, keepdims=True)

    return {
        "hist_vol_annual":  round(hist_vol_annual, 4),
        "hist_mu_annual":   round(hist_mu_annual, 4),
        "implied_vol":      round(implied_vol, 4),
        "current_regime":   current_regime,
        "lambda_hat":       round(lambda_hat, 2),
        "jump_mu_hat":      round(jump_mu_hat, 4),
        "jump_sigma_hat":   round(jump_sigma_hat, 4),
        "transition_matrix": T,
        "fear_adj":         round(fear_adj, 4),
    }


# ─────────────────────────────────────────
# 3. 마르코프 레짐 + Merton JDM Monte Carlo
# ─────────────────────────────────────────
def run_monte_carlo(S0: float, params: dict,
                    days: int = 90,
                    n_sims: int = 5000,
                    seed: int = 42) -> dict:
    """
    레짐 스위칭 + Merton JDM Monte Carlo.

    각 시뮬레이션 경로:
      - 마르코프 체인으로 매일 레짐 결정
      - 해당 레짐의 (μ, σ) 적용
      - Merton JDM: 포아송 점프 추가
        price[t] = price[t-1] * exp(
            (μ_d - 0.5σ²_d - λ*k̄) * dt
            + σ_d * dW
            + sum_j(log(1 + J_j))    -- 포아송 점프
        )
        k̄ = E[J] = exp(jump_mu + 0.5*jump_sigma²) - 1
    """
    rng = np.random.default_rng(seed)
    T   = TRANSITION_BASE  # 기본값
    T   = params["transition_matrix"]

    regime_names = ["low_vol", "high_vol", "crisis"]
    regime_idx   = {"low_vol": 0, "high_vol": 1, "crisis": 2}
    init_regime  = regime_idx[params["current_regime"]]

    dt = 1.0 / 252.0  # 일간

    paths = np.zeros((n_sims, days + 1))
    paths[:, 0] = S0
    regimes_out  = np.zeros((n_sims, days), dtype=int)

    for sim in range(n_sims):
        regime = init_regime
        for t in range(1, days + 1):
            # 레짐 전환
            regime = int(rng.choice(3, p=T[regime]))
            regimes_out[sim, t - 1] = regime

            r_name = regime_names[regime]
            rp = REGIMES[r_name]

            mu_d    = rp["mu_annual"] * dt
            sigma_d = rp["sigma_annual"] * np.sqrt(dt)
            lam     = rp["jump_lambda"]
            jmu     = rp["jump_mu"]
            jsig    = rp["jump_sigma"]

            # JDM 보정항 k̄
            k_bar = np.exp(jmu + 0.5 * jsig**2) - 1

            # 연속 확산
            dW = rng.normal(0, 1)
            diffusion = (mu_d - 0.5 * sigma_d**2 - lam * k_bar * dt) + sigma_d * dW

            # 포아송 점프
            n_jumps = rng.poisson(lam * dt)
            if n_jumps > 0:
                jump_sizes = rng.normal(jmu, jsig, n_jumps)
                jump_total = float(jump_sizes.sum())
            else:
                jump_total = 0.0

            paths[sim, t] = paths[sim, t - 1] * np.exp(diffusion + jump_total)

    return {"paths": paths, "regimes": regimes_out, "S0": S0, "days": days, "n_sims": n_sims}


# ─────────────────────────────────────────
# 4. VaR / CVaR (Cornish-Fisher + Historical)
# ─────────────────────────────────────────
def compute_risk_metrics(mc: dict, confidence: float = 0.95) -> dict:
    """
    Cornish-Fisher VaR + Historical VaR + CVaR.
    Cornish-Fisher: 왜도(skewness)·첨도(kurtosis) 보정 → 정규 VaR보다 꼬리 리스크 정확.
    """
    paths = mc["paths"]
    S0    = mc["S0"]
    results = {}

    for h in [30, 60, 90]:
        if h > mc["days"]:
            continue
        final = paths[:, h]
        rets  = (final - S0) / S0  # 누적 수익률

        # Historical VaR
        var_hist = float(np.percentile(rets, (1 - confidence) * 100))

        # Cornish-Fisher VaR
        alpha = 1 - confidence
        z     = float(norm.ppf(alpha)) if HAS_SCIPY else -1.645
        if HAS_SCIPY:
            s = float(skew(rets))
            k = float(kurtosis(rets))  # excess kurtosis
        else:
            mean = rets.mean(); std = rets.std()
            s = float(np.mean(((rets - mean) / (std + 1e-9))**3))
            k = float(np.mean(((rets - mean) / (std + 1e-9))**4)) - 3

        # Cornish-Fisher 조정 z
        z_cf = (z
                + (z**2 - 1) * s / 6
                + (z**3 - 3*z) * k / 24
                - (2*z**3 - 5*z) * s**2 / 36)
        mu_r    = float(rets.mean())
        sigma_r = float(rets.std())
        var_cf  = float(mu_r + z_cf * sigma_r)

        # CVaR (Expected Shortfall)
        cvar = float(rets[rets <= var_hist].mean()) if (rets <= var_hist).sum() > 0 else var_hist

        results[h] = {
            "p10":       float(np.percentile(final, 10)),
            "p25":       float(np.percentile(final, 25)),
            "p50":       float(np.percentile(final, 50)),
            "p75":       float(np.percentile(final, 75)),
            "p90":       float(np.percentile(final, 90)),
            "mean":      float(final.mean()),
            "prob_up":   float((final > S0).mean() * 100),
            "var_hist":  round(var_hist * 100, 2),   # %
            "var_cf":    round(var_cf  * 100, 2),    # % (Cornish-Fisher)
            "cvar":      round(cvar    * 100, 2),    # %
            "skewness":  round(s, 4),
            "kurtosis":  round(k, 4),
            "crisis_prob": float(
                (mc["regimes"][:, :h] == 2).any(axis=1).mean() * 100
            ),
        }

    return results


# ─────────────────────────────────────────
# 5. Plotly 대시보드
# ─────────────────────────────────────────
def build_dashboard(mc: dict, risk: dict, params: dict,
                    nasdaq_series: np.ndarray = None,
                    fear_index: float = 40.0,
                    out_path: str = "nasdaq_mc_v2.html") -> str:
    if not HAS_PLOTLY:
        print("[SKIP] plotly 미설치")
        return ""

    S0   = mc["S0"]
    days = mc["days"]
    paths= mc["paths"]
    rng2 = np.random.default_rng(0)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"MC 경로 샘플 (Merton JDM + 마르코프 레짐, n={mc['n_sims']:,})",
            "분위수 밴드 (30/60/90일)",
            "레짐 분포 (위기/고변동/저변동)",
            "Cornish-Fisher VaR vs Historical VaR",
        ),
    )

    # P1: 경로 샘플 50개
    t_axis = list(range(days + 1))
    sample_idx = rng2.choice(mc["n_sims"], size=min(50, mc["n_sims"]), replace=False)
    for idx in sample_idx:
        fig.add_trace(go.Scatter(
            x=t_axis, y=paths[idx],
            mode="lines", line=dict(color="rgba(55,138,221,0.12)", width=0.8),
            showlegend=False,
        ), row=1, col=1)
    fig.add_hline(y=S0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_axis, y=np.percentile(paths, 50, axis=0),
        mode="lines", line=dict(color="#E24B4A", width=2), name="Median",
    ), row=1, col=1)

    # P2: 분위수 밴드
    horizons = sorted(risk.keys())
    if horizons:
        h_vals  = [0] + horizons
        p10_arr = [S0] + [risk[h]["p10"] for h in horizons]
        p50_arr = [S0] + [risk[h]["p50"] for h in horizons]
        p90_arr = [S0] + [risk[h]["p90"] for h in horizons]

        fig.add_trace(go.Scatter(
            x=h_vals, y=p90_arr,
            mode="lines+markers", name="P90",
            line=dict(color="#3FB950", width=2),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=h_vals, y=p50_arr,
            mode="lines+markers", name="Median",
            line=dict(color="#58A6FF", width=2),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=h_vals, y=p10_arr,
            mode="lines+markers", name="P10",
            line=dict(color="#E24B4A", width=2),
        ), row=1, col=2)
        fig.add_hline(y=S0, line_dash="dash", line_color="gray", row=1, col=2)

    # P3: 레짐 분포 파이
    regime_flat = mc["regimes"].flatten()
    regime_counts = [
        int((regime_flat == 0).sum()),
        int((regime_flat == 1).sum()),
        int((regime_flat == 2).sum()),
    ]
    fig.add_trace(go.Pie(
        labels=["저변동 (Low Vol)", "고변동 (High Vol)", "위기 (Crisis)"],
        values=regime_counts,
        marker_colors=["#3FB950", "#D29922", "#E24B4A"],
        hole=0.35, textinfo="label+percent",
    ), row=2, col=1)

    # P4: VaR 비교 바
    if horizons:
        var_labels = [f"{h}일" for h in horizons]
        var_hist_v = [risk[h]["var_hist"] for h in horizons]
        var_cf_v   = [risk[h]["var_cf"]   for h in horizons]
        cvar_v     = [risk[h]["cvar"]     for h in horizons]

        fig.add_trace(go.Bar(
            x=var_labels, y=var_hist_v, name="Historical VaR",
            marker_color="#58A6FF",
        ), row=2, col=2)
        fig.add_trace(go.Bar(
            x=var_labels, y=var_cf_v, name="Cornish-Fisher VaR",
            marker_color="#D29922",
        ), row=2, col=2)
        fig.add_trace(go.Bar(
            x=var_labels, y=cvar_v, name="CVaR (ES)",
            marker_color="#E24B4A",
        ), row=2, col=2)

    # 레이아웃
    regime_name_ko = {"low_vol": "저변동", "high_vol": "고변동", "crisis": "위기"}
    fig.update_layout(
        title=dict(
            text=(
                f"OilPriceValuation — Nasdaq MC v2  |  "
                f"Fear Index {fear_index:.0f}  |  "
                f"현재 레짐: {regime_name_ko[params['current_regime']]}  |  "
                f"내재변동성 {params['implied_vol']*100:.1f}%  |  "
                f"λ={params['lambda_hat']:.1f}회/년  |  "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ),
            font=dict(size=12),
        ),
        height=750,
        barmode="group",
        template="plotly_white",
    )

    fig.write_html(out_path)
    print(f"[Dashboard] 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────
# 6. 메인 실행 함수
# ─────────────────────────────────────────
def run(fear_index: float = 40.0,
        days: int = 90,
        n_sims: int = 5000,
        out_path: str = "nasdaq_mc_v2.html") -> dict:
    """
    외부 호출 진입점.
    market_sentiment_pipeline.run_pipeline()의 반환값에서
    fear_index를 꺼내 그대로 전달.
    """
    print("\n" + "=" * 60)
    print(f"  Nasdaq MC v2  |  Fear Index={fear_index:.1f}  |  Days={days}")
    print("=" * 60)

    # 데이터 수집
    mkt = fetch_market_data()
    nasdaq = mkt.get("nasdaq")
    vix    = mkt.get("vix")

    S0 = float(nasdaq[-1]) if nasdaq is not None else 18500.0
    print(f"[Nasdaq] S0=${S0:,.1f}  source={mkt['source']}")

    # 캘리브레이션
    params = calibrate_params(nasdaq, vix, fear_index)
    print(f"[Calibration] regime={params['current_regime']}, "
          f"implied_vol={params['implied_vol']*100:.1f}%, "
          f"lambda={params['lambda_hat']:.1f}, "
          f"fear_adj={params['fear_adj']:.3f}")

    # Monte Carlo
    mc = run_monte_carlo(S0, params, days=days, n_sims=n_sims)

    # 리스크 지표
    risk = compute_risk_metrics(mc)

    # 출력
    print(f"\n{'지평':>6} {'P10':>8} {'P50':>8} {'P90':>8} {'Prob↑':>7} {'CF-VaR':>8} {'CVaR':>8} {'Crisis%':>8}")
    print("-" * 65)
    for h in sorted(risk.keys()):
        r = risk[h]
        print(f"{h:>4}일  ${r['p10']:>7.0f}  ${r['p50']:>7.0f}  ${r['p90']:>7.0f}  "
              f"{r['prob_up']:>6.1f}%  {r['var_cf']:>+7.2f}%  {r['cvar']:>+7.2f}%  {r['crisis_prob']:>6.1f}%")

    # 대시보드
    dash = build_dashboard(mc, risk, params, nasdaq, fear_index, out_path)

    print("=" * 60)

    return {
        "S0":         S0,
        "params":     params,
        "risk":       risk,
        "dashboard":  dash,
        "mc_paths":   mc["paths"],
    }


# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nasdaq MC v2")
    parser.add_argument("--fear",     type=float, default=40.0,  help="공포지수 (0~100)")
    parser.add_argument("--days",     type=int,   default=90,    help="예측 기간 (일)")
    parser.add_argument("--sims",     type=int,   default=5000,  help="Monte Carlo 경로 수")
    parser.add_argument("--out",      default="nasdaq_mc_v2.html", help="출력 HTML 경로")
    parser.add_argument("--pipeline", default=None, help="감성 파이프라인 입력 파일 경로")
    args = parser.parse_args()

    fear = args.fear

    # 감성 파이프라인 연동
    if args.pipeline:
        try:
            from market_sentiment_pipeline import run_pipeline
            print(f"[Pipeline] 감성 분석 실행: {args.pipeline}")
            sentiment = run_pipeline(input_path=args.pipeline)
            fear = sentiment["fear_index"]
            print(f"[Pipeline] fear_index={fear:.1f}")
        except ImportError:
            print("[Pipeline] market_sentiment_pipeline.py 를 같은 디렉토리에 두세요.")

    run(fear_index=fear, days=args.days, n_sims=args.sims, out_path=args.out)
