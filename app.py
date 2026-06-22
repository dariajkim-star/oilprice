"""
app.py - Oil Analysis Dashboard (Light Mode)
크롤링:  data_crawler.py 모듈에서 담당 (API 과사용 방지 분리)
ML 흐름: 분류(방향 -1/0/+1) → 회귀(크기 예측) 2단계 파이프라인
모델:
  - MS-style Kalman Filter (잠재 펀더멘털)
  - GS Two-Stage Bridge (수급 → 지정학 프리미엄)
  - RF Supply Shock Index
  - Monte Carlo 30/60/90일 가격 예측
"""

import io, base64, datetime, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template_string, jsonify

# 크롤링 모듈 분리 — API 과사용 방지
import requests
from data_crawler import fetch_all_data, CSV_DIR, EIA_KEY, EIA_BASE, _cache

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

app = Flask(__name__)

# ── 팔레트 (라이트 모드) ───────────────────────────────────
L_BG     = "#ffffff"
L_PANEL  = "#f6f8fa"
L_BORDER = "#d0d7de"
L_TEXT   = "#1f2328"
L_SUB    = "#57606a"
L_BLUE   = "#0969da"
L_GREEN  = "#1a7f37"
L_RED    = "#cf222e"
L_YELLOW = "#9a6700"
L_PURPLE = "#8250df"
L_ORANGE = "#bc4c00"
L_LGRAY  = "#eaeef2"

# ── 5사 파라미터 ───────────────────────────────────────────
HOUSE_WEIGHTS = {
    "Goldman Sachs":  0.28, "Morgan Stanley": 0.27,
    "Citigroup":      0.18, "JPMorgan":       0.18,
    "Bloomberg":      0.09,
}
# 2026-06-18 업데이트: 미국-이란 MOU 60일, 호르무즈 재개방 진행
SCENARIO_PROBS = {
    "S1_Agreement": 0.60,  # 구 0.10 → 합의·재개방 현실화
    "S2_Strike":    0.15,  # 구 0.40 → Trump 경고 잔존, 재점화 리스크
    "S3_Prolonged": 0.15,  # 구 0.30 → 협상 결렬 시 재봉쇄
    "S4_Withdrawal":0.10,  # 구 0.20 → 극단 시나리오
}
# 5사 × 4 시나리오 WTI 목표가 — 2026-06-18 업데이트
# S1: 공급 정상화 + IEA 과잉 반영 → 하향
TARGETS = {
    "Goldman Sachs":  [ 72, 100, 115,  88],  # S1: $77→$72
    "Morgan Stanley": [ 68, 103, 110,  87],  # S1: $72→$68
    "Citigroup":      [ 70, 108, 128,  95],  # S1: $75→$70
    "JPMorgan":       [ 73, 105, 112,  97],  # S1: $78→$73
    "Bloomberg":      [ 74, 110, 145,  98],  # S1: $80→$74
}
SIGMA = {"S1_Agreement": 4.0, "S2_Strike": 14.0,
          "S3_Prolonged": 20.0, "S4_Withdrawal": 12.0}  # S1 변동성 축소

FORECAST_HORIZONS = [7, 14, 30, 60, 90]



# ══════════════════════════════════════════════════════════
# 2. 보조 함수  (크롤링은 data_crawler.py 참조)
# ══════════════════════════════════════════════════════════
def _align(arr, T: int, rng, mu: float, sigma: float) -> np.ndarray:
    if arr is None or len(arr) == 0:
        return mu + np.cumsum(rng.normal(0, sigma, T))
    arr = np.array(arr).flatten().astype(float)
    if len(arr) >= T:
        return arr[-T:]
    return np.concatenate([np.full(T - len(arr), arr[0]), arr])


def _kalman(obs: np.ndarray, q: float):
    n = len(obs)
    xh, P = np.zeros(n), np.zeros(n)
    xh[0], P[0] = obs[0], 1.0
    for i in range(1, n):
        Pp = P[i-1] + q
        K  = Pp / (Pp + 1.0)
        xh[i] = xh[i-1] + K * (obs[i] - xh[i-1])
        P[i]  = (1-K) * Pp
    return xh, P


def _hormuz_proxy(mkt: dict, T: int) -> tuple:
    """
    PortWatch 없을 때: WTI-Brent 스프레드와 OVX로 리스크 프록시 생성.
    역 스프레드(좁아질수록 위험) + OVX 상승 → vessel_count 감소 신호.
    """
    rng0 = np.random.default_rng(0)
    wti   = _align(mkt.get("WTI",   {}).get("series"), T, rng0, 80, 0.5)
    brent = _align(mkt.get("Brent", {}).get("series"), T, rng0, 83, 0.5)
    ovx   = _align(mkt.get("OVX",   {}).get("series"), T, rng0, 28, 0.3)

    spread = wti - brent                          # 보통 음수 (WTI < Brent)
    spread_norm = (spread - spread.mean()) / (spread.std() + 1e-9)
    ovx_norm    = (ovx    - ovx.mean())    / (ovx.std()    + 1e-9)

    # 리스크 지수↑ → 선박수↓ (호르무즈 긴장 프록시)
    risk = -0.5 * spread_norm + 0.5 * ovx_norm   # 높을수록 위험
    risk_norm = (risk - risk.min()) / (risk.max() - risk.min() + 1e-9)

    vessel_count   = np.clip(138 - risk_norm * 60, 50, 138)
    vessel_speed   = np.clip(12  - risk_norm *  2, 8,  16)
    vessel_tonnage = np.clip(50000 - risk_norm * 15000, 20000, 65000)
    tanker_ratio   = np.clip(0.4 + risk_norm * 0.2, 0.3, 0.7)
    return vessel_count, vessel_speed, vessel_tonnage, tanker_ratio


def run_tanker_forecast(arcgis: dict, n_mc: int = 1000) -> dict:
    """
    호르무즈 유조선 통과 수 예측 (7/14/30/60/90일).
    ArcGIS 365일 n_tanker 시계열 기반:
      - 7일 이동평균 스무딩
      - 최근 30일 선형 추세 추출
      - 평균회귀(Ornstein-Uhlenbeck) + 추세 + 잔차 잡음 Monte Carlo
    """
    if not arcgis or arcgis.get("error"):
        return {}
    n_tanker = arcgis.get("n_tanker")
    if n_tanker is None or len(n_tanker) < 14:
        return {}

    T = len(n_tanker)
    # 7일 이동평균 스무딩
    w      = min(7, T)
    smooth = np.convolve(n_tanker, np.ones(w) / w, mode="same")

    # 최근 30일 선형 추세
    look   = min(30, T)
    x_fit  = np.arange(look, dtype=float)
    coeffs = np.polyfit(x_fit, smooth[-look:], 1)
    trend_daily = float(coeffs[0])

    hist_mean = float(n_tanker.mean())
    resid_std = float(np.std(n_tanker - smooth))

    # Monte Carlo: 평균회귀 + 선형 추세 + 백색잡음
    rng   = np.random.default_rng(42)
    H     = 90
    theta = 0.08   # 평균회귀 속도 (일간)
    paths = np.zeros((n_mc, H + 1))
    paths[:, 0] = float(n_tanker[-1])
    noise_std = max(resid_std, 0.5)

    for t in range(1, H + 1):
        revert     = theta * (hist_mean - paths[:, t - 1])
        noise      = rng.normal(0, noise_std, n_mc)
        paths[:, t] = np.clip(
            paths[:, t - 1] + revert + trend_daily + noise, 0, 300
        )

    horizons = {}
    for h in [7, 14, 30, 60, 90]:
        p = paths[:, h]
        horizons[h] = {
            "p10":        float(np.percentile(p, 10)),
            "p50":        float(np.percentile(p, 50)),
            "p90":        float(np.percentile(p, 90)),
            "mean":       float(p.mean()),
            "crisis_prob": float((p < hist_mean * 0.6).mean() * 100),
        }

    return dict(
        smooth=smooth,
        trend_daily=trend_daily,
        hist_mean=hist_mean,
        resid_std=resid_std,
        paths=paths,
        horizons=horizons,
        n_mc=n_mc,
    )


# ══════════════════════════════════════════════════════════
# 3. oil_fair_value 분석
# ══════════════════════════════════════════════════════════
def run_fair_value(mkt: dict) -> dict:
    rng = np.random.default_rng(42)

    # WTI 시계열
    wti_arr = mkt.get("WTI", {})
    if wti_arr and wti_arr.get("series") is not None:
        wti_spot = wti_arr["series"].flatten().astype(float)
    else:
        T0 = 260
        wti_spot = np.clip(80 + np.cumsum(rng.normal(0, 0.8, T0)), 50, 160)
    T = len(wti_spot)

    # 보조 시계열
    dxy   = _align(mkt.get("DXY",   {}).get("series") if mkt.get("DXY")   else None, T, rng, 100, 0.3)
    ovx   = _align(mkt.get("OVX",   {}).get("series") if mkt.get("OVX")   else None, T, rng,  28, 0.5)
    brent = _align(mkt.get("Brent", {}).get("series") if mkt.get("Brent") else None, T, rng,  83, 0.8)

    # EIA 데이터
    eia_arr = mkt.get("eia_stock")
    eia_stock = _align(eia_arr if eia_arr is not None and len(eia_arr) > 0 else None,
                       T, rng, -0.5, 3.0)
    # EIA 재고: 절대값 → 주간 변화로 변환
    if eia_arr is not None and len(eia_arr) > 4:
        eia_aligned = _align(eia_arr, T, rng, 0, 3.0)
        eia_stock = np.diff(eia_aligned, prepend=eia_aligned[0]) / 1000.0

    spr_arr = mkt.get("spr_level")
    spr_level = _align(spr_arr if spr_arr is not None and len(spr_arr) > 0 else None,
                       T, rng, 350, 1.5)

    gap_arr = mkt.get("supply_gap")
    supply_gap = _align(gap_arr if gap_arr is not None and len(gap_arr) > 0 else None,
                        T, rng, 0, 1.2)

    # 선물 기울기
    fs_val = mkt.get("futures_slope")
    futures_slope = (np.full(T, fs_val) + rng.normal(0, 0.1, T)
                     if fs_val is not None else rng.normal(1.5, 0.8, T))

    # USO / BNO 원유 ETF
    uso_arr = mkt.get("USO", {}).get("series") if mkt.get("USO") else None
    bno_arr = mkt.get("BNO", {}).get("series") if mkt.get("BNO") else None
    uso = _align(uso_arr, T, rng, 70.0, 1.5)
    bno = _align(bno_arr, T, rng, 35.0, 0.8)

    # Brent-WTI 스프레드 (저장된 값 우선, 없으면 인라인 계산)
    sp_data = mkt.get("brent_wti_spread", {})
    if sp_data and sp_data.get("series") is not None:
        wb_spread = _align(sp_data["series"], T, rng, 3.0, 0.5)
    else:
        wb_spread = brent - wti_spot

    # 호르무즈 선박 데이터 (ArcGIS 우선, 실패시 시장 프록시)
    ag = mkt.get("arcgis_hormuz")
    if ag and ag.get("n_tanker") is not None and len(ag["n_tanker"]) > 7:
        vessel_count   = _align(ag["n_tanker"],        T, rng, 10, 2)
        vessel_speed   = _align(None,                  T, rng, 12, 1)
        vessel_tonnage = _align(ag["capacity_tanker"], T, rng, 50000, 5000)
        tanker_ratio   = _align(ag["tanker_ratio"],    T, rng, 0.45, 0.05)
    else:
        vessel_count, vessel_speed, vessel_tonnage, tanker_ratio = \
            _hormuz_proxy(mkt, T)

    # ── RF Supply Shock Index ──────────────────────────────
    X_rf = StandardScaler().fit_transform(
        np.column_stack([vessel_count, vessel_speed, vessel_tonnage,
                         tanker_ratio, supply_gap, eia_stock,
                         uso, bno, wb_spread])   # USO/BNO ETF + Brent-WTI spread 추가
    )
    rf_target = np.clip(-np.diff(wti_spot, prepend=wti_spot[0]), -10, 10)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_rf, rf_target)
    shock_index = rf.predict(X_rf)
    si_range = shock_index.max() - shock_index.min()
    shock_index = (shock_index - shock_index.min()) / (si_range + 1e-9)
    feat_names  = ["vessel_cnt","vessel_spd","tonnage","tanker_rt",
                   "supply_gap","eia_stk","uso","bno","bw_spread"]
    importances = rf.feature_importances_

    # ── Kalman Filter ──────────────────────────────────────
    obs_n = (wti_spot - wti_spot.mean()) / (wti_spot.std() + 1e-9)
    best_q, best_rmse = 0.01, 1e9
    for q in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
        xh, _ = _kalman(obs_n, q)
        rmse = np.sqrt(np.mean((obs_n - xh)**2))
        if rmse < best_rmse:
            best_rmse, best_q = rmse, q
    kn, kv = _kalman(obs_n, best_q)
    kalman_f = kn * wti_spot.std() + wti_spot.mean()
    kalman_u = np.sqrt(kv) * wti_spot.std()

    # ── GS Bridge ─────────────────────────────────────────
    d_wti = np.diff(wti_spot, prepend=wti_spot[0])
    d_spr = np.diff(spr_level, prepend=spr_level[0])
    d_dxy = np.diff(dxy, prepend=dxy[0])

    Xa = np.column_stack([supply_gap, d_spr, d_dxy, eia_stock])
    Xa_sc = np.column_stack([StandardScaler().fit_transform(Xa), np.ones(T)])
    beta_a  = np.linalg.lstsq(Xa_sc, d_wti, rcond=None)[0]
    da      = Xa_sc @ beta_a
    stage_a = wti_spot[0] + np.cumsum(da)
    ss_res  = np.sum((d_wti - da)**2)
    ss_tot  = np.sum((d_wti - d_wti.mean())**2)
    r2_a    = 1 - ss_res / (ss_tot + 1e-9)

    Xb = np.column_stack([shock_index, ovx, wb_spread, futures_slope])
    Xb_sc  = np.column_stack([StandardScaler().fit_transform(Xb), np.ones(T)])
    resid_a = d_wti - da
    beta_b  = np.linalg.lstsq(Xb_sc, resid_a, rcond=None)[0]
    geo_prem = Xb_sc @ beta_b
    gs_fair  = stage_a + np.cumsum(geo_prem)
    overshoot = wti_spot - gs_fair

    ss_rb = np.sum((resid_a - geo_prem)**2)
    ss_tb = np.sum((resid_a - resid_a.mean())**2)
    r2_b  = 1 - ss_rb / (ss_tb + 1e-9)

    return dict(
        T=T, wti_spot=wti_spot, gs_fair=gs_fair,
        kalman_f=kalman_f, kalman_u=kalman_u,
        shock_index=shock_index, stage_a=stage_a, geo_prem=geo_prem,
        overshoot=overshoot, importances=importances, feat_names=feat_names,
        r2_a=r2_a, r2_b=r2_b,
        wti_latest=float(wti_spot[-1]),
        gs_latest=float(gs_fair[-1]),
        over_latest=float(overshoot[-1]),
        kalman_latest=float(kalman_f[-1]),
        kalman_q=best_q,
        vessel_count=vessel_count,
        spr_latest=float(spr_level[-1]) if len(spr_level) > 0 else 350.0,
        eia_latest=float(eia_stock[-1]) if len(eia_stock) > 0 else 0.0,
    )


# ══════════════════════════════════════════════════════════
# 4. 피처 매트릭스 + ML 예측
# ══════════════════════════════════════════════════════════
def build_feature_matrix(mkt: dict) -> pd.DataFrame:
    """
    모든 시장/매크로/수급 데이터를 하나의 피처 DataFrame으로 조합.
    타깃: target_return_{h}d = price(t+h)/price(t) - 1
    """
    wti_info = mkt.get("WTI", {})
    if not wti_info or wti_info.get("series") is None:
        return pd.DataFrame()

    wti_arr = wti_info["series"].flatten()
    T       = len(wti_arr)
    rng     = np.random.default_rng(0)

    def _s(key, sub="series", mu=0.0, sig=1.0):
        v = mkt.get(key, {})
        arr = v.get(sub) if isinstance(v, dict) else None
        return _align(arr, T, rng, mu, sig)

    def _m(key, mu=0.0, sig=1.0):
        m = mkt.get("macro", {}).get(key, {})
        arr = m.get("series") if isinstance(m, dict) else None
        return _align(arr, T, rng, mu, sig)

    def _w(key, mu=0.0, sig=1.0):
        arr = mkt.get("wpsr", {}).get(key)
        return _align(arr, T, rng, mu, sig)

    data = {
        "oil_price":        wti_arr,
        # 시장 가격
        "brent":            _s("Brent",           mu=83,   sig=1.0),
        "dxy":              _s("DXY",             mu=103,  sig=0.3),
        "ovx":              _s("OVX",             mu=28,   sig=0.5),
        "uso":              _s("USO",             mu=70,   sig=1.0),
        "bno":              _s("BNO",             mu=35,   sig=0.5),
        "bw_spread":        _s("brent_wti_spread",mu=3,    sig=0.3),
        "futures_slope":    np.full(T, float(mkt.get("futures_slope") or 1.5)),
        # 매크로
        "tnx_yield":        _m("TNX",             mu=4.3,  sig=0.05),
        "sp500":            _m("SP500",           mu=5500, sig=20.0),
        "msci_world":       _m("MSCI_World",      mu=120,  sig=0.5),
        "gold":             _m("Gold",            mu=2300, sig=10.0),
        "copper":           _m("Copper",          mu=4.0,  sig=0.05),
        "vix":              _m("VIX",             mu=18,   sig=0.5),
        "tip_etf":          _m("TIP",             mu=105,  sig=0.3),
        "xli_etf":          _m("XLI",             mu=120,  sig=0.5),
        "fxi_etf":          _m("FXI",             mu=28,   sig=0.3),
        "mchi_etf":         _m("MCHI",            mu=50,   sig=0.3),
        # EIA 수급
        "eia_crude_stk":    _s("eia_stock",       mu=0,    sig=3.0),
        "eia_gas_stk":      _w("gasoline",        mu=230,  sig=2.0),
        "eia_distillate":   _w("distillate",      mu=120,  sig=1.0),
        "eia_imports":      _w("crude_imports",   mu=6,    sig=0.5),
        "spr":              _s("spr_level",       mu=350,  sig=2.0),
    }

    df = pd.DataFrame(data)

    # 로그 수익률
    for c in ["oil_price","brent","sp500","msci_world","gold","copper",
              "uso","bno","tip_etf","xli_etf","fxi_etf","mchi_etf"]:
        df[f"{c}_ret"] = df[c].pct_change().fillna(0)

    # 이동평균 / 변동성
    for lag in [7, 21]:
        df[f"wti_ma{lag}"]  = df["oil_price"].rolling(lag, min_periods=1).mean()
        df[f"wti_vol{lag}"] = df["oil_price"].rolling(lag, min_periods=1).std().fillna(0)

    # ── 타깃: h일 선행 수익률 ──────────────────────────────
    for h in FORECAST_HORIZONS:
        df[f"target_return_{h}d"] = df["oil_price"].shift(-h) / df["oil_price"] - 1

    return df


def run_ml_forecast(df: pd.DataFrame) -> dict:
    """
    2단계 분류→회귀 파이프라인.
    Stage 1 (RF Classifier): 방향 분류 (-1=하락, 0=중립, +1=상승)
    Stage 2 (RF Regressor) : 수익률 크기 예측
    Final                  : 방향 신뢰도 × 크기 결합 → 최종 예측 수익률
    """
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.metrics import mean_absolute_error, r2_score, balanced_accuracy_score

    DIRECTION_THRESHOLD = 0.005  # ±0.5% 이내는 중립(0)

    if df.empty:
        return {}

    feat_cols = [c for c in df.columns
                 if not c.startswith("target_") and c != "oil_price"]

    clf_param_grid = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }
    reg_param_grid = {
        "n_estimators":      [50, 100, 200, 300],
        "max_depth":         [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", 0.5, 0.7],
    }

    tscv    = TimeSeriesSplit(n_splits=5)
    results = {}

    for h in FORECAST_HORIZONS:
        tcol = f"target_return_{h}d"
        if tcol not in df.columns:
            continue

        df_c = df.dropna(subset=[tcol] + feat_cols)
        if len(df_c) < 40:
            continue

        X   = df_c[feat_cols].values
        y_ret = df_c[tcol].values
        # 방향 레이블: +1/0/-1
        y_dir = np.where(y_ret >  DIRECTION_THRESHOLD,  1,
                np.where(y_ret < -DIRECTION_THRESHOLD, -1, 0))

        split = int(len(X) * 0.8)
        X_train, X_test     = X[:split],     X[split:]
        y_ret_tr, y_ret_te  = y_ret[:split], y_ret[split:]
        y_dir_tr, y_dir_te  = y_dir[:split], y_dir[split:]

        # ── Stage 1: 방향 분류 (RF Classifier) ───────────
        clf = RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        )
        clf_search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=clf_param_grid,
            n_iter=20,
            cv=tscv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            random_state=42,
            refit=True,
        )
        clf_search.fit(X_train, y_dir_tr)
        best_clf = clf_search.best_estimator_

        dir_pred_test = best_clf.predict(X_test) if len(X_test) > 0 else np.array([0])
        dir_acc = float(balanced_accuracy_score(y_dir_te, dir_pred_test)) if len(y_dir_te) > 0 else 0.0

        dir_latest      = int(best_clf.predict(X[-1:])[0])
        dir_proba_all   = best_clf.predict_proba(X[-1:])[0]
        dir_classes     = list(best_clf.classes_)
        dir_proba_dict  = {int(c): round(float(p), 4)
                           for c, p in zip(dir_classes, dir_proba_all)}

        # ── Stage 2: 수익률 회귀 (RF Regressor) ──────────
        reg = RandomForestRegressor(random_state=42, n_jobs=-1)
        reg_search = RandomizedSearchCV(
            estimator=reg,
            param_distributions=reg_param_grid,
            n_iter=20,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
            refit=True,
        )
        reg_search.fit(X_train, y_ret_tr)
        best_reg = reg_search.best_estimator_

        reg_pred_test = best_reg.predict(X_test) if len(X_test) > 0 else np.array([0.0])
        mae = float(mean_absolute_error(y_ret_te, reg_pred_test)) if len(y_ret_te) > 0 else 0.0
        r2  = float(r2_score(y_ret_te, reg_pred_test))            if len(y_ret_te) > 1 else 0.0

        reg_latest = float(best_reg.predict(X[-1:])[0])

        # ── 결합: 방향 신뢰도 × 크기 ─────────────────────
        if dir_latest == 0:
            pred_return = reg_latest * 0.2        # 중립: 크기 감쇠
        elif np.sign(reg_latest) == dir_latest:
            pred_return = reg_latest              # 방향 일치: 그대로
        else:
            # 방향 불일치: 분류기 방향으로 부호 반전, 크기 50%
            pred_return = abs(reg_latest) * 0.5 * dir_latest

        dir_label_map = {1: "상승(▲)", 0: "중립(─)", -1: "하락(▼)"}
        dir_label     = dir_label_map.get(dir_latest, "?")

        results[h] = {
            "dir_label":    dir_label,
            "dir_acc":      round(dir_acc, 4),
            "dir_proba":    dir_proba_dict,
            "mae":          round(mae, 5),
            "r2":           round(r2, 4),
            "pred_return":  round(pred_return, 5),
            "pred_price":   round(float(df["oil_price"].iloc[-1]) * (1 + pred_return), 2),
            "importances":  {f: round(float(v), 4)
                             for f, v in zip(feat_cols, best_reg.feature_importances_)},
        }

    return results


# ══════════════════════════════════════════════════════════
# 4b. 가격 예측 (Monte Carlo GBM)
# ══════════════════════════════════════════════════════════
def monte_carlo_price_paths(S0: float, mu: float, sigma: float,
                            days: int = 90, n_sims: int = 1000) -> np.ndarray:
    """
    log-normal GBM Monte Carlo.
    daily_returns ~ N(mu, sigma)  →  price = S0 * exp(cumsum(returns))
    mu    : 일간 드리프트 (로그 수익률)
    sigma : 일간 변동성 (로그 수익률 표준편차)
    반환  : shape (days, n_sims) 가격 경로 행렬
    """
    daily_returns = np.random.normal(mu, sigma, size=(days, n_sims))
    price_paths   = S0 * np.exp(np.cumsum(daily_returns, axis=0))
    return price_paths


def run_forecast(fv: dict, n_paths: int = 2000) -> dict:
    """
    log-normal GBM Monte Carlo (monte_carlo_price_paths 사용).
    mu    = GS 공정가 수렴 드리프트 + Itô 보정 (-0.5σ²)
    sigma = 최근 20일 역사적 일간 변동성
    """
    np.random.seed(99)
    wti = fv["wti_spot"]
    gf  = fv["gs_fair"]

    # 일간 변동성
    rets      = np.diff(np.log(np.clip(wti[-30:], 1, None)))
    daily_vol = float(np.std(rets)) if len(rets) > 2 else 0.015

    # 일간 드리프트: GS 공정가 수렴 + Itô 보정
    gap_daily = (float(gf[-1]) - float(wti[-1])) / 90.0 / float(wti[-1])
    mu_daily  = gap_daily - 0.5 * daily_vol ** 2

    S0   = float(wti[-1])
    H    = 90

    # shape: (H, n_paths)  — 각 열이 하나의 시뮬레이션 경로
    price_paths = monte_carlo_price_paths(S0, mu_daily, daily_vol,
                                          days=H, n_sims=n_paths)

    # paths[i, j]: i번째 시뮬레이션 j일째 가격  → 전치해서 (n_paths, H+1)로
    paths = np.vstack([np.full((1, n_paths), S0), price_paths]).T  # (n_paths, H+1)

    result = {"paths": paths, "horizons": {}}
    for h in [30, 60, 90]:
        p = paths[:, h]
        result["horizons"][h] = {
            "p10":     float(np.percentile(p, 10)),
            "median":  float(np.percentile(p, 50)),
            "p90":     float(np.percentile(p, 90)),
            "mean":    float(p.mean()),
            "prob_up": float((p > S0).mean() * 100),
        }
    return result


# ══════════════════════════════════════════════════════════
# 5. 5사 시나리오
# ══════════════════════════════════════════════════════════
def run_scenario(wti_val: float) -> dict:
    sc  = list(SCENARIO_PROBS.keys())
    hs  = list(HOUSE_WEIGHTS.keys())
    w   = np.array([HOUSE_WEIGHTS[h]  for h in hs])
    pr  = np.array([SCENARIO_PROBS[s] for s in sc])
    TM  = np.array([TARGETS[h]        for h in hs])
    hev = TM @ pr
    w_ev = float(w @ hev);  s_ev = float(hev.mean())
    sc_w = w @ TM
    rng  = np.random.default_rng(42)
    N    = 100_000
    sim  = np.zeros(N)
    for k, s in enumerate(sc):
        n  = int(N * pr[k])
        st = int(N * pr[:k].sum())
        sim[st:st+n] = rng.normal(float(sc_w[k]), SIGMA[s], n)
    mask = sim == 0
    sim[mask] = rng.normal(w_ev, 10.0, mask.sum())
    return dict(
        wti_spot=wti_val, sc=sc, hs=hs, w=w, pr=pr, TM=TM,
        hev=hev, w_ev=w_ev, s_ev=s_ev, sc_w=sc_w, sim=sim,
        p110=float((sim>110).mean()), p120=float((sim>120).mean()),
        v5=float(np.percentile(sim,5)), v95=float(np.percentile(sim,95)),
    )


# ══════════════════════════════════════════════════════════
# 6. 차트 빌더 (라이트 모드)
# ══════════════════════════════════════════════════════════
def _ax(ax, title: str):
    ax.set_facecolor(L_BG)
    ax.tick_params(colors=L_SUB, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(L_BORDER)
    ax.set_title(title, color=L_TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.xaxis.label.set_color(L_SUB); ax.yaxis.label.set_color(L_SUB)


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=L_BG)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_fair_value(fv: dict, pw_live: bool) -> str:
    t          = np.arange(fv["T"])
    wti_spot   = fv["wti_spot"]
    gs_fair    = fv["gs_fair"]
    kf         = fv["kalman_f"]
    ku         = fv["kalman_u"]
    shock      = fv["shock_index"]
    stage_a    = fv["stage_a"]
    geo_prem   = fv["geo_prem"]
    overshoot  = fv["overshoot"]
    imp        = fv["importances"]
    fn         = fv["feat_names"]
    vc         = fv["vessel_count"]

    fig = plt.figure(figsize=(20, 14), facecolor=L_BG)
    title_suffix = "호르무즈: PortWatch LIVE" if pw_live else "호르무즈: 시장 프록시 (WTI-Brent+OVX)"
    fig.suptitle(
        f"OilPriceValuation — Kalman+GS Bridge+RF Shock  |  {title_suffix}",
        color=L_TEXT, fontsize=12, fontweight="bold", y=0.99,
    )
    gs_l = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.35,
                            left=0.06, right=0.97, top=0.94, bottom=0.06)
    axs = [
        fig.add_subplot(gs_l[0, :2]),
        fig.add_subplot(gs_l[0, 2]),
        fig.add_subplot(gs_l[1, 0]),
        fig.add_subplot(gs_l[1, 1]),
        fig.add_subplot(gs_l[1, 2]),
        fig.add_subplot(gs_l[2, :]),
    ]

    # P1: 실제가 vs GS Fair vs Kalman (전체 기간)
    ax = axs[0]; _ax(ax, "WTI Spot vs GS Fair Value vs Kalman Fundamental")
    ax.plot(t, wti_spot, color=L_BLUE,   lw=1.3, label="WTI Spot")
    ax.plot(t, gs_fair,  color=L_GREEN,  lw=1.2, ls="--", label="GS Fair Value")
    ax.plot(t, kf,       color=L_PURPLE, lw=1.0, ls=":",  label="Kalman Fundamental")
    ax.fill_between(t, kf-ku, kf+ku, alpha=0.12, color=L_PURPLE)
    ax.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER)
    ax.set_ylabel("USD/bbl")

    # P2: 호르무즈 선박수
    ax = axs[1]
    src_label = "IMF PortWatch" if pw_live else "Market Proxy"
    _ax(ax, f"Hormuz Vessel Count ({src_label})")
    ax.fill_between(t, 0, vc, color=L_ORANGE, alpha=0.35)
    ax.plot(t, vc, color=L_ORANGE, lw=0.9)
    ax.axhline(vc.mean(), color=L_RED, ls="--", lw=1.0,
               label=f"Mean {vc.mean():.0f}")
    ax.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER)
    ax.set_ylabel("Vessels")

    # P3: RF Shock Index
    ax = axs[2]; _ax(ax, "RF Supply Shock Index")
    ax.fill_between(t, 0, shock, color=L_RED, alpha=0.35)
    ax.plot(t, shock, color=L_RED, lw=0.9)
    ax.axhline(shock.mean(), color=L_YELLOW, ls="--", lw=1.0,
               label=f"Mean {shock.mean():.2f}")
    ax.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER)
    ax.set_ylim(-0.05, 1.15); ax.set_ylabel("Normalized")

    # P4: 가격 분해
    ax = axs[3]; _ax(ax, f"Price Decomposition  R²A={fv['r2_a']:.3f} R²B={fv['r2_b']:.3f}")
    ax.stackplot(t, stage_a, np.cumsum(geo_prem),
                 labels=["Stage A (Supply-Demand)", "Stage B (Geo Premium)"],
                 colors=[L_BLUE, L_RED], alpha=0.55)
    ax.plot(t, wti_spot, color=L_TEXT, lw=1.0, ls="--", label="WTI Spot")
    ax.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER)
    ax.set_ylabel("USD/bbl")

    # P5: RF 피처 중요도
    ax = axs[4]; _ax(ax, "RF Feature Importance")
    si = np.argsort(imp)
    cols = [L_GREEN if i == si[-1] else L_BLUE for i in range(len(fn))]
    ax.barh([fn[i] for i in si], imp[si],
            color=[cols[i] for i in si], edgecolor=L_LGRAY, lw=0.6)
    ax.set_xlabel("Importance")

    # P6: 오버슈트
    ax = axs[5]; ov_now = fv["over_latest"]
    _ax(ax, f"Overshoot vs GS Fair  (Current: ${ov_now:+.1f}/bbl)")
    ax.fill_between(t, 0, overshoot, where=overshoot>0, color=L_RED,  alpha=0.40, label="Overvalued")
    ax.fill_between(t, 0, overshoot, where=overshoot<=0,color=L_GREEN,alpha=0.40, label="Undervalued")
    ax.plot(t, overshoot, color=L_SUB, lw=0.6)
    ax.axhline(0,    color=L_BORDER, lw=1.0)
    ax.axhline(+10,  color=L_RED,    ls="--", lw=0.8, label="+$10 Alert")
    ax.axhline(-20,  color=L_GREEN,  ls="--", lw=0.8, label="-$20 Alert")
    ax.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER)
    ax.set_ylabel("USD/bbl")

    return _b64(fig)


def chart_forecast(fv: dict, fc: dict) -> str:
    """30/60/90일 가격 예측 차트 (MC 경로 + 분위수)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=L_BG)
    fig.suptitle("WTI 가격 예측 — Monte Carlo (GS 공정가 수렴 모델)",
                 color=L_TEXT, fontsize=12, fontweight="bold", y=1.01)

    wti_latest = fv["wti_latest"]
    paths      = fc["paths"]
    horizons   = fc["horizons"]
    H          = paths.shape[1] - 1
    t_hist     = np.arange(fv["T"])
    t_fwd      = np.arange(H + 1) + fv["T"] - 1

    # 왼쪽: 역사 + 미래 경로
    ax = axes[0]; _ax(ax, "MC 경로 + 분위수 밴드  (N=2,000)")
    ax.plot(t_hist[-60:], fv["wti_spot"][-60:], color=L_BLUE, lw=1.5, label="Historical WTI")
    ax.plot(t_hist[-60:], fv["gs_fair"][-60:],  color=L_GREEN, lw=1.0, ls="--", label="GS Fair Value")

    # 경로 샘플 50개
    for i in range(min(50, paths.shape[0])):
        ax.plot(t_fwd, np.concatenate([[wti_latest], paths[i, 1:]]),
                color=L_BLUE, lw=0.3, alpha=0.15)

    # 분위수 밴드
    p10  = np.percentile(paths, 10, axis=0)
    p50  = np.percentile(paths, 50, axis=0)
    p90  = np.percentile(paths, 90, axis=0)
    ax.fill_between(t_fwd, p10, p90, alpha=0.18, color=L_BLUE, label="10-90th pct")
    ax.plot(t_fwd, p50, color=L_BLUE, lw=2.0, ls="-", label="Median")

    # 수직선 (30/60/90일)
    for h, col in zip([30, 60, 90], [L_YELLOW, L_ORANGE, L_RED]):
        ax.axvline(fv["T"]-1+h, color=col, ls=":", lw=1.2, label=f"+{h}d")

    ax.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER, ncol=2)
    ax.set_xlabel("Trading Days"); ax.set_ylabel("WTI USD/bbl")

    # 오른쪽: 지평별 요약 바
    ax = axes[1]; _ax(ax, "지평별 가격 예측 (중앙값 + P10/P90)")
    hs_labels = ["현재", "+30일", "+60일", "+90일"]
    medians = [wti_latest] + [horizons[h]["median"] for h in [30, 60, 90]]
    p10s    = [wti_latest] + [horizons[h]["p10"]    for h in [30, 60, 90]]
    p90s    = [wti_latest] + [horizons[h]["p90"]    for h in [30, 60, 90]]
    x = np.arange(4)
    bars = ax.bar(x, medians, color=[L_SUB, L_YELLOW, L_ORANGE, L_RED],
                  edgecolor=L_LGRAY, lw=0.6, width=0.5, zorder=3)
    ax.errorbar(x, medians, yerr=[np.array(medians)-np.array(p10s),
                                   np.array(p90s)-np.array(medians)],
                fmt="none", color=L_TEXT, capsize=5, lw=1.5, zorder=4)
    for b, m, p1, p9 in zip(bars, medians, p10s, p90s):
        ax.text(b.get_x() + b.get_width()/2, p9 + 0.8,
                f"${m:.1f}", ha="center", color=L_TEXT, fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width()/2, p1 - 2.5,
                f"↕${p9-p1:.1f}", ha="center", color=L_SUB, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(hs_labels)
    ax.axhline(wti_latest, color=L_BLUE, ls="--", lw=1.0, label=f"현재가 ${wti_latest:.2f}")
    ax.axhline(fv["gs_latest"], color=L_GREEN, ls="--", lw=1.0,
               label=f"GS FV ${fv['gs_latest']:.2f}")
    ax.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER)
    ax.set_ylabel("WTI USD/bbl")
    ax.set_ylim(min(p10s) - 8, max(p90s) + 8)

    plt.tight_layout()
    return _b64(fig)


def chart_scenario(sc: dict) -> str:
    hs  = sc["hs"]; pr = sc["pr"]; TM = sc["TM"]
    hev = sc["hev"]; w_ev = sc["w_ev"]; s_ev = sc["s_ev"]
    wti = sc["wti_spot"]; sim = sc["sim"]
    v5  = sc["v5"]; v95 = sc["v95"]
    p110= sc["p110"]; p120= sc["p120"]

    PIE = [L_BLUE, L_GREEN, L_YELLOW, L_ORANGE, L_RED]
    BAR = [L_GREEN, L_RED, L_ORANGE, L_PURPLE]

    fig = plt.figure(figsize=(22, 14), facecolor=L_BG)
    fig.suptitle("5-House Weighted Scenario Analysis  |  KFBI Internal",
                 color=L_TEXT, fontsize=14, fontweight="bold", y=0.98)
    gl = fig.add_gridspec(3,3, hspace=0.48, wspace=0.38,
                          left=0.05, right=0.97, top=0.93, bottom=0.06)
    axs = [fig.add_subplot(gl[0,0]), fig.add_subplot(gl[0,1]),
           fig.add_subplot(gl[0,2]), fig.add_subplot(gl[1,:2]),
           fig.add_subplot(gl[1,2]), fig.add_subplot(gl[2,:])]

    ax = axs[0]; _ax(ax, "House Weights")
    _, _, ats = ax.pie([HOUSE_WEIGHTS[h] for h in hs],
                       labels=[h.replace(" ","\n") for h in hs],
                       autopct="%1.0f%%", colors=PIE,
                       textprops={"color":L_TEXT,"fontsize":7},
                       wedgeprops={"edgecolor":L_BG,"linewidth":1.2}, startangle=90)
    for at in ats: at.set_fontsize(7); at.set_color("white")

    ax = axs[1]; _ax(ax, "Scenario Probabilities")
    xl = ["S1\nAgreement","S2\nStrike","S3\nProlonged","S4\nWithdrawal"]
    bars = ax.bar(xl, pr*100, color=BAR, edgecolor=L_LGRAY, lw=0.6)
    for b, p in zip(bars, pr):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.8,
                f"{p*100:.0f}%", ha="center", color=L_TEXT, fontsize=8)
    ax.set_ylabel("Probability (%)"); ax.set_ylim(0,55)

    ax = axs[2]; _ax(ax, "House Expected Value (WTI)")
    bars2 = ax.barh(hs, hev, color=PIE, edgecolor=L_LGRAY, lw=0.6)
    ax.axvline(wti,  color=L_RED,  ls="--", lw=1.2, label=f"Spot ${wti:.2f}")
    ax.axvline(w_ev, color=L_GREEN,ls="--", lw=1.2, label=f"Wtd EV ${w_ev:.1f}")
    for b, v in zip(bars2, hev):
        ax.text(v+0.5, b.get_y()+b.get_height()/2,
                f"${v:.1f}", va="center", color=L_TEXT, fontsize=8)
    ax.set_xlabel("WTI (USD)"); ax.set_xlim(60,145)
    ax.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER)

    ax = axs[3]; _ax(ax, "Scenario Target Matrix (USD)")
    im = ax.imshow(TM, cmap="RdYlGn", aspect="auto", vmin=65, vmax=165)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["S1 Agree","S2 Strike","S3 Prolong","S4 Withdraw"],
                       color=L_TEXT, fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(hs, color=L_TEXT, fontsize=8)
    for i in range(5):
        for j in range(4):
            ax.text(j,i,f"${TM[i,j]}", ha="center", va="center",
                    color="black", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axs[4]; uv = w_ev - wti
    ax.set_facecolor(L_BG); ax.tick_params(colors=L_SUB, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(L_BORDER)
    ax.set_title(f"Weighted EV vs Spot  |  Delta: ${uv:+.1f}",
                 color=L_TEXT, fontsize=9, fontweight="bold", pad=6)
    cats=["Spot","Simple EV","Weighted EV"]; vals=[wti,s_ev,w_ev]
    cols=[L_SUB,L_YELLOW,L_GREEN]
    bars3 = ax.bar(cats, vals, color=cols, edgecolor=L_LGRAY, lw=0.6, width=0.5)
    for b, v in zip(bars3, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f"${v:.1f}", ha="center", color=L_TEXT, fontsize=9, fontweight="bold")
    ax.set_ylim(60,135); ax.set_ylabel("WTI USD")

    ax = axs[5]; _ax(ax, f"Monte Carlo  (N=100,000)")
    ax.hist(sim, bins=120, color=L_BLUE, edgecolor=L_LGRAY, alpha=0.55, lw=0.3)
    ax.axvspan(v5, v95, alpha=0.10, color=L_BLUE)
    ax.axvline(wti,  color=L_RED,   ls="--", lw=1.5, label=f"Spot ${wti:.2f}")
    ax.axvline(w_ev, color=L_GREEN, ls="--", lw=1.5, label=f"Wtd EV ${w_ev:.1f}")
    ax.axvline(v5,   color=L_YELLOW,ls=":",  lw=1.2, label=f"5th ${v5:.1f}")
    ax.axvline(v95,  color=L_ORANGE,ls=":",  lw=1.2, label=f"95th ${v95:.1f}")
    ax.set_xlabel("WTI USD"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=8, facecolor=L_BG, edgecolor=L_BORDER, ncol=4)

    fig.text(0.05, 0.01,
             f"Spot: ${wti:.2f} | Wtd EV: ${w_ev:.1f} (+${w_ev-wti:.1f}) | "
             f"P(>$110)={p110*100:.1f}% P(>$120)={p120*100:.1f}% | "
             "GS28% MS27% Citi18% JPM18% BB9%",
             color=L_SUB, fontsize=8)
    return _b64(fig)


# ══════════════════════════════════════════════════════════
# 6b. ML 예측 결과 차트
# ══════════════════════════════════════════════════════════
def chart_ml(ml: dict, df_feat: pd.DataFrame) -> str:
    """
    3-패널 ML 결과 차트.
      [0,0] 지평별 예측 수익률 + 현재가 대비 예측 가격 (바 차트)
      [0,1] 모델 성능 (MAE / R²) 지평별 비교
      [1, :] 상위 10 피처 중요도 (horizon 평균)
    """
    if not ml:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=L_BG)
        ax.set_facecolor(L_PANEL); ax.axis("off")
        ax.text(0.5, 0.5, "ML 모델 결과 없음\n(데이터 부족 또는 학습 실패)",
                ha="center", va="center", fontsize=11, color=L_SUB,
                transform=ax.transAxes)
        return _b64(fig)

    fig = plt.figure(figsize=(18, 9), facecolor=L_BG)
    fig.patch.set_facecolor(L_BG)
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(L_PANEL)
        ax.tick_params(colors=L_TEXT, labelsize=8)
        ax.spines[:].set_color(L_BORDER)

    horizons = sorted(ml.keys())
    labels   = [f"+{h}d" for h in horizons]

    # ── [0,0] 예측 수익률 + 방향 레이블 ─────────────────────
    rets   = [ml[h]["pred_return"] * 100 for h in horizons]
    dir_lbls = [ml[h].get("dir_label", "?") for h in horizons]
    colors = [L_GREEN if r >= 0 else L_RED for r in rets]
    bars   = ax1.bar(labels, rets, color=colors, alpha=0.8, width=0.5, edgecolor=L_BORDER)
    ax1.axhline(0, color=L_TEXT, lw=0.8, ls="--")
    for bar, r, dl in zip(bars, rets, dir_lbls):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.05 if r >= 0 else -0.15),
                 f"{r:+.2f}%", ha="center", va="bottom", fontsize=8,
                 fontweight="bold", color=L_GREEN if r >= 0 else L_RED)
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 0.02 if r >= 0 else -0.02,
                 dl, ha="center", va="bottom" if r >= 0 else "top",
                 fontsize=7, color=L_TEXT, alpha=0.75)
    for i, h in enumerate(horizons):
        ax1.text(i, min(rets) - 0.3,
                 f"${ml[h]['pred_price']:.1f}",
                 ha="center", va="top", fontsize=7, color=L_SUB)
    ax1.set_title("분류→회귀 예측 수익률 (지평별)", color=L_TEXT,
                  fontsize=10, fontweight="bold")
    ax1.set_ylabel("Predicted Return (%)", color=L_TEXT, fontsize=8)

    # ── [0,1] 방향 분류 정확도 + MAE/R² ────────────────────
    dir_accs = [ml[h].get("dir_acc", 0) * 100 for h in horizons]
    maes     = [ml[h]["mae"] * 100 for h in horizons]
    r2s      = [ml[h]["r2"]        for h in horizons]
    x        = np.arange(len(horizons))
    w        = 0.3
    ax2.bar(x - w, dir_accs, w, label="방향 정확도 (%)",
            color=L_PURPLE, alpha=0.8, edgecolor=L_BORDER)
    ax2.bar(x,     maes,     w, label="회귀 MAE (%)",
            color=L_ORANGE, alpha=0.8, edgecolor=L_BORDER)
    ax2r = ax2.twinx()
    ax2r.plot(x, r2s, "o--", color=L_BLUE, lw=2, ms=6, label="R²")
    ax2r.set_ylabel("R²", color=L_BLUE, fontsize=8)
    ax2r.tick_params(colors=L_BLUE, labelsize=8)
    ax2r.set_ylim(-0.2, 1.0)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_title("모델 성능 (방향정확도 / MAE / R²)", color=L_TEXT,
                  fontsize=10, fontweight="bold")
    ax2.set_ylabel("%", color=L_TEXT, fontsize=8)
    lines1, lbs1 = ax2.get_legend_handles_labels()
    lines2, lbs2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbs1 + lbs2, fontsize=7,
               facecolor=L_BG, edgecolor=L_BORDER)

    # ── [1,:] 피처 중요도 (horizon 평균, 상위 15) ─────────
    imp_sum: dict = {}
    for h in horizons:
        for feat, val in ml[h]["importances"].items():
            imp_sum[feat] = imp_sum.get(feat, 0) + val
    n_h = len(horizons)
    imp_avg = {k: v / n_h for k, v in imp_sum.items()}
    top15   = sorted(imp_avg.items(), key=lambda x: x[1], reverse=True)[:15]
    feats_t, vals_t = zip(*top15) if top15 else ([], [])
    bar_colors = [L_BLUE if "macro" not in f.lower() else L_PURPLE
                  for f in feats_t]
    # 색상 규칙: _ret 계열=초록, eia_=오렌지, 나머지=파랑
    bar_colors = []
    for f in feats_t:
        if "_ret" in f:     bar_colors.append(L_GREEN)
        elif "eia_" in f:   bar_colors.append(L_ORANGE)
        elif f in ("gold", "copper", "vix", "tnx_yield", "sp500", "msci_world",
                   "fxi_etf", "mchi_etf", "xli_etf", "tip_etf"):
            bar_colors.append(L_PURPLE)
        else:               bar_colors.append(L_BLUE)
    ax3.barh(list(feats_t)[::-1], list(vals_t)[::-1],
             color=bar_colors[::-1], alpha=0.85, edgecolor=L_BORDER)
    ax3.set_title(
        "피처 중요도 (RF, FORECAST_HORIZONS 평균)  "
        "│  파랑=시장  초록=수익률  오렌지=EIA수급  보라=매크로",
        color=L_TEXT, fontsize=9, fontweight="bold",
    )
    ax3.set_xlabel("Mean Feature Importance", color=L_TEXT, fontsize=8)

    fig.suptitle(
        f"ML 유가 예측 (RF 분류→회귀 2단계 + TimeSeriesSplit CV)  "
        f"│  Horizons: {FORECAST_HORIZONS}",
        fontsize=11, fontweight="bold", color=L_TEXT, y=0.999,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    return _b64(fig)


# ══════════════════════════════════════════════════════════
# 6c. 호르무즈 유조선 예측 차트
# ══════════════════════════════════════════════════════════
def chart_tanker(arcgis: dict, forecast: dict, straits: dict) -> str:  # noqa: E302
    """
    4-패널 차트:
      [0,0] 탱커 일별 통과 + 90일 MC 예측 (P10/P50/P90)
      [0,1] 선박 타입 스택 분해
      [1,0] 탱커 DWT 용량 추세
      [1,1] straits.live 현황 스냅샷 텍스트
    """
    if not arcgis or arcgis.get("error"):
        # 데이터 없음 → 빈 placeholder 반환
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=L_BG)
        ax.set_facecolor(L_PANEL)
        ax.text(0.5, 0.5, "ArcGIS 데이터 없음\n(fetch_all_data 오류)",
                ha="center", va="center", fontsize=12, color=L_SUB,
                transform=ax.transAxes)
        ax.axis("off")
        return _b64(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor=L_BG)
    fig.patch.set_facecolor(L_BG)
    for ax in axes.flat:
        ax.set_facecolor(L_PANEL)
        ax.tick_params(colors=L_TEXT, labelsize=8)
        ax.spines[:].set_color(L_BORDER)

    dates    = arcgis.get("dates", [])
    n_tanker = arcgis.get("n_tanker", np.array([]))
    n_total  = arcgis.get("n_total",  np.array([]))
    cap_tank = arcgis.get("capacity_tanker", np.array([]))
    T        = len(n_tanker)
    x_hist   = np.arange(T)

    def _date_ticks(ax, T, dates, step=None):
        step = step or max(1, T // 6)
        ticks = list(range(0, T, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [dates[i][:10] for i in ticks if i < len(dates)],
            rotation=25, ha="right", fontsize=7,
        )

    # ── [0,0] 탱커 수 + 예측 ───────────────────────────────
    ax1 = axes[0, 0]
    smooth = forecast.get("smooth", n_tanker) if forecast else n_tanker
    ax1.fill_between(x_hist, n_tanker, alpha=0.15, color=L_BLUE)
    ax1.plot(x_hist, n_tanker, color=L_BLUE, lw=0.8, alpha=0.5, label="Daily n_tanker")
    ax1.plot(x_hist, smooth,   color=L_BLUE, lw=1.8, label="7d MA")

    if forecast and "paths" in forecast:
        paths  = forecast["paths"]
        H_plot = min(90, paths.shape[1] - 1)
        x_fore = np.arange(T, T + H_plot + 1)
        p10 = np.percentile(paths[:, :H_plot + 1], 10, axis=0)
        p50 = np.percentile(paths[:, :H_plot + 1], 50, axis=0)
        p90 = np.percentile(paths[:, :H_plot + 1], 90, axis=0)
        ax1.fill_between(x_fore, p10, p90, alpha=0.20, color=L_ORANGE,
                         label="P10–P90 band")
        ax1.plot(x_fore, p50, color=L_ORANGE, lw=2.2, ls="--", label="Forecast P50")
        ax1.plot(x_fore, p10, color=L_RED,    lw=1.0, ls=":")
        ax1.plot(x_fore, p90, color=L_GREEN,  lw=1.0, ls=":")

    hist_mean = forecast.get("hist_mean", float(n_tanker.mean())) if forecast else float(n_tanker.mean())
    ax1.axhline(hist_mean, color=L_PURPLE, lw=1.0, ls=":", alpha=0.7,
                label=f"Hist mean {hist_mean:.1f}")
    ax1.axvline(T - 1, color=L_SUB, lw=1.0, ls="--", alpha=0.5)
    ax1.text(T - 0.5, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 5,
             "forecast →", fontsize=7, color=L_SUB, va="top")

    if dates:
        _date_ticks(ax1, T, dates)
    ax1.set_title("Hormuz Tanker Traffic  +  90-Day MC Forecast",
                  color=L_TEXT, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Tankers / day", color=L_TEXT, fontsize=8)
    ax1.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER, ncol=2, loc="upper left")

    # ── [0,1] 선박 타입 스택 분해 ──────────────────────────
    ax2 = axes[0, 1]
    n_dry  = arcgis.get("n_dry_bulk",  np.zeros(T))
    n_cont = arcgis.get("n_container", np.zeros(T))
    n_roro = arcgis.get("n_roro",      np.zeros(T))
    n_other = np.maximum(n_total - n_tanker - n_dry - n_cont - n_roro, 0)
    ax2.stackplot(
        x_hist,
        n_tanker, n_dry, n_cont, n_roro, n_other,
        labels=["Tanker", "Dry Bulk", "Container", "RoRo", "Other"],
        colors=[L_ORANGE, L_BLUE, L_GREEN, L_PURPLE, L_LGRAY],
        alpha=0.85,
    )
    if dates:
        _date_ticks(ax2, T, dates)
    ax2.set_title("Vessel Type Breakdown — Stacked Daily",
                  color=L_TEXT, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Vessel count", color=L_TEXT, fontsize=8)
    ax2.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER, ncol=3, loc="upper left")

    # ── [1,0] 탱커 DWT 용량 추세 ───────────────────────────
    ax3 = axes[1, 0]
    if len(cap_tank) > 0:
        w7     = min(7, len(cap_tank))
        c_smt  = np.convolve(cap_tank, np.ones(w7) / w7, mode="same")
        ax3.fill_between(x_hist, cap_tank, alpha=0.12, color=L_GREEN)
        ax3.plot(x_hist, cap_tank, color=L_GREEN, lw=0.8, alpha=0.5, label="Daily DWT")
        ax3.plot(x_hist, c_smt,   color=L_GREEN, lw=1.8, label="7d MA")
        if dates:
            _date_ticks(ax3, T, dates)
    ax3.set_title("Tanker Capacity Through Hormuz (DWT)",
                  color=L_TEXT, fontsize=10, fontweight="bold")
    ax3.set_ylabel("Tanker DWT", color=L_TEXT, fontsize=8)
    ax3.legend(fontsize=7, facecolor=L_BG, edgecolor=L_BORDER)

    # ── [1,1] straits.live 현황 스냅샷 ─────────────────────
    ax4 = axes[1, 1]
    ax4.axis("off")
    sl = straits or {}
    sl_ok = bool(sl and not sl.get("error"))

    ax4.text(0.5, 0.97, "straits.live  —  Real-Time Snapshot",
             ha="center", va="top", fontsize=10, fontweight="bold",
             color=L_TEXT, transform=ax4.transAxes)
    ax4.text(0.5, 0.91,
             sl.get("scraped_at", "데이터 없음") if sl_ok else "스크래핑 실패 (JS 렌더링 페이지)",
             ha="center", va="top", fontsize=7.5, color=L_SUB,
             transform=ax4.transAxes)

    if sl_ok:
        rows = [
            ("Commercial Transits Today",    sl.get("transits_today"),       ""),
            ("Vessels in Transit",            sl.get("vessels_in_transit"),   ""),
            ("Tankers Dark (AIS off)",        sl.get("tankers_dark"),
             f"7d baseline: {sl.get('dark_baseline_7d', '?')}"),
            ("War-Risk Insurance (VLCC)",
             f"${sl.get('war_risk_usd_m','?')}M",
             f"{sl.get('war_risk_multiplier','?')}x pre-crisis"),
            ("Hormuz Index",                  sl.get("hormuz_index"),         "Crisis Pressure"),
            ("Escalation Forecast",           sl.get("escalation_forecast"),  ""),
            ("High-Risk Vessels",             sl.get("high_risk_vessels"),    ""),
            ("Moderate-Risk Vessels",         sl.get("moderate_risk"),        ""),
        ]
        y = 0.84
        for label, val, note in rows:
            val_str = str(val) if val is not None else "N/A"
            is_risk = any(w in label for w in ("Dark", "Risk", "Escalation", "Hormuz Index"))
            col_val = L_RED if is_risk else L_TEXT
            ax4.text(0.05, y, label + ":",
                     ha="left", va="top", fontsize=8, color=L_SUB,
                     transform=ax4.transAxes)
            ax4.text(0.62, y, val_str,
                     ha="left", va="top", fontsize=9.5, fontweight="bold",
                     color=col_val, transform=ax4.transAxes)
            y -= 0.07
            if note:
                ax4.text(0.62, y, note,
                         ha="left", va="top", fontsize=7, color=L_SUB,
                         transform=ax4.transAxes)
                y -= 0.04
    else:
        ax4.text(0.5, 0.5,
                 f"오류: {sl.get('error','N/A')}\n\n"
                 "straits.live는 JavaScript 렌더링이 필요할 수 있음.\n"
                 "Playwright/Selenium 설치 시 완전 지원 가능.",
                 ha="center", va="center", fontsize=9, color=L_SUB,
                 transform=ax4.transAxes)

    n_mc = forecast.get("n_mc", 0) if forecast else 0
    fig.suptitle(
        f"호르무즈 유조선 예측  |  ArcGIS Daily + straits.live  |  MC N={n_mc:,}",
        fontsize=12, fontweight="bold", color=L_TEXT, y=0.999,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    return _b64(fig)


# ══════════════════════════════════════════════════════════
# 7. HTML 템플릿
# ══════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Oil Analysis Dashboard</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#f6f8fa;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;line-height:1.5}

header{background:#fff;border-bottom:1px solid #d0d7de;padding:14px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;position:sticky;top:0;z-index:10}
.h-left h1{font-size:1.1rem;font-weight:700}
.h-left p{font-size:.78rem;color:#57606a}
.btn{background:#0969da;color:#fff;border:none;border-radius:6px;padding:7px 16px;font-size:.88rem;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:6px}
.btn:hover{background:#0860ca}

.ticker-bar{background:#fff;border-bottom:1px solid #d0d7de;padding:8px 24px;display:flex;gap:28px;overflow-x:auto;flex-wrap:wrap}
.tick{display:flex;flex-direction:column;min-width:80px}
.t-name{font-size:.72rem;color:#57606a;font-weight:600;text-transform:uppercase}
.t-price{font-size:1.05rem;font-weight:700}
.up{color:#1a7f37}.down{color:#cf222e}

main{padding:20px 24px;max-width:1600px;margin:0 auto}

.section-hdr{display:flex;align-items:center;gap:10px;margin:24px 0 12px}
.section-hdr h2{font-size:1rem;font-weight:700}
.badge{background:#ddf4ff;color:#0969da;border:1px solid #54aeff;border-radius:20px;font-size:.72rem;font-weight:600;padding:2px 8px}
.badge-g{background:#dafbe1;color:#1a7f37;border-color:#4ac26b}
.badge-y{background:#fff8c5;color:#9a6700;border-color:#d4a72c}
.badge-r{background:#ffebe9;color:#cf222e;border-color:#ff8182}

.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-bottom:20px}
.sc{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:14px 16px}
.sl{font-size:.75rem;color:#57606a;margin-bottom:4px;font-weight:600}
.sv{font-size:1.3rem;font-weight:700}
.ss{font-size:.72rem;color:#57606a;margin-top:2px}
.cg{color:#1a7f37}.cr{color:#cf222e}.cb{color:#0969da}.cy{color:#9a6700}.cp{color:#8250df}.co{color:#bc4c00}

.chart-card{background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:16px;margin-bottom:20px}
.chart-card img{max-width:100%;height:auto;border-radius:6px;display:block}

/* 데이터 소스 테이블 */
.src-table{width:100%;border-collapse:collapse;font-size:.8rem}
.src-table th{background:#f6f8fa;color:#57606a;font-weight:600;text-align:left;padding:6px 10px;border:1px solid #d0d7de}
.src-table td{padding:6px 10px;border:1px solid #eaeef2}
.src-live{color:#1a7f37;font-weight:600}
.src-proxy{color:#9a6700;font-weight:600}
.src-sim{color:#cf222e}

/* 예측 테이블 */
.fc-table{width:100%;border-collapse:collapse;font-size:.88rem;margin-bottom:16px}
.fc-table th{background:#f6f8fa;text-align:center;padding:8px 12px;border:1px solid #d0d7de;font-weight:600;color:#57606a}
.fc-table td{text-align:center;padding:8px 12px;border:1px solid #eaeef2;font-weight:600}

#overlay{display:none;position:fixed;inset:0;background:rgba(255,255,255,.8);z-index:100;align-items:center;justify-content:center;flex-direction:column;gap:14px}
#overlay.show{display:flex}
.spinner{width:40px;height:40px;border:4px solid #d0d7de;border-top-color:#0969da;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.sp-txt{font-size:.9rem;font-weight:600}

footer{text-align:center;padding:20px;font-size:.78rem;color:#57606a;border-top:1px solid #d0d7de;margin-top:24px}
</style>
</head>
<body>
<div id="overlay"><div class="spinner"></div><div class="sp-txt">크롤링 중...</div></div>

<header>
  <div class="h-left">
    <h1>OilPriceValuation Dashboard</h1>
    <p>Kalman Filter + GS Bridge + RF Shock  &nbsp;|&nbsp;  5-House Scenario  &nbsp;|&nbsp;  30/60/90d Forecast  &nbsp;|&nbsp;  KFBI Internal</p>
  </div>
  <button class="btn" onclick="recrawl()">
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2z"/><path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/></svg>
    전체 재크롤링
  </button>
</header>

<div class="ticker-bar">
  {% for key,label in [("WTI","WTI Crude"),("Brent","Brent Crude"),("DXY","DXY Index"),("OVX","OVX Vol"),("USO","USO ETF"),("BNO","BNO ETF")] %}
  {% set it = mkt.get(key) %}
  <div class="tick">
    <span class="t-name">{{ label }}</span>
    {% if it %}
    <span class="t-price">{{ "%.2f"|format(it.latest) }}</span>
    <span class="{{ 'up' if it.chg>=0 else 'down' }}">
      {{ '+' if it.chg>=0 else '' }}{{ "%.2f"|format(it.chg) }} ({{ '+' if it.chg_pct>=0 else '' }}{{ "%.2f"|format(it.chg_pct) }}%)
    </span>
    {% else %}<span class="t-price" style="color:#57606a">N/A</span>{% endif %}
  </div>
  {% endfor %}
  <div class="tick" style="margin-left:auto">
    <span class="t-name">Updated</span>
    <span style="font-size:.82rem;color:#57606a">{{ mkt.crawled_at }}</span>
  </div>
</div>

<main>

<!-- 데이터 소스 현황 -->
<div class="section-hdr">
  <h2>크롤링 데이터 소스 현황</h2>
  <span class="badge-g badge">LIVE</span>
  <span class="badge-y badge">PROXY</span>
  <span class="badge-r badge">SIM</span>
</div>
<div class="chart-card">
<table class="src-table">
<tr><th>변수</th><th>원본 모델 주석</th><th>상태</th></tr>
{% for var, note, src in sources %}
<tr>
  <td><strong>{{ var }}</strong></td>
  <td style="color:#57606a">{{ note }}</td>
  <td class="{% if 'LIVE' in src %}src-live{% elif 'PROXY' in src %}src-proxy{% else %}src-sim{% endif %}">{{ src }}</td>
</tr>
{% endfor %}
</table>
</div>

<!-- 공정가치 모델 -->
<div class="section-hdr">
  <h2>Fair Value Model</h2>
  <span class="badge">MS Kalman Filter</span>
  <span class="badge">GS Two-Stage Bridge</span>
  <span class="badge">RF Shock Index</span>
  <span class="badge {{ 'badge-g' if pw_live else 'badge-y' }}">
    호르무즈: {{ 'PortWatch LIVE' if pw_live else 'Market Proxy' }}
  </span>
</div>

<div class="stat-grid">
  <div class="sc"><div class="sl">WTI Spot (Live)</div><div class="sv">${{ "%.2f"|format(fv.wti_latest) }}</div></div>
  <div class="sc"><div class="sl">GS Fair Value</div><div class="sv cg">${{ "%.2f"|format(fv.gs_latest) }}</div></div>
  <div class="sc">
    <div class="sl">Overshoot</div>
    <div class="sv {{ 'cr' if fv.over_latest>0 else 'cg' }}">${{ '%+.2f'|format(fv.over_latest) }}</div>
    <div class="ss">{{ '고평가' if fv.over_latest>0 else '저평가' }}</div>
  </div>
  <div class="sc"><div class="sl">Kalman Fundamental</div><div class="sv cp">${{ "%.2f"|format(fv.kalman_latest) }}</div></div>
  <div class="sc"><div class="sl">GS Bridge R² (Stage A)</div><div class="sv cb">{{ "%.3f"|format(fv.r2_a) }}</div></div>
  <div class="sc"><div class="sl">GS Bridge R² (Stage B)</div><div class="sv cb">{{ "%.3f"|format(fv.r2_b) }}</div></div>
  <div class="sc">
    <div class="sl">SPR 잔여량</div>
    <div class="sv cy">{{ "%.0f"|format(fv.spr_latest) }} Mbbl</div>
    <div class="ss">EIA {{ 'LIVE' if mkt.spr_level is not none else 'SIM' }}</div>
  </div>
  {% if mkt.futures_slope is not none %}
  <div class="sc">
    <div class="sl">WTI 선물 기울기</div>
    <div class="sv {{ 'cr' if mkt.futures_slope>0 else 'cg' }}">{{ '%+.2f'|format(mkt.futures_slope) }}%/yr</div>
    <div class="ss">{{ 'Contango' if mkt.futures_slope>0 else 'Backwardation' }}</div>
  </div>
  {% endif %}
</div>

<div class="chart-card"><img src="data:image/png;base64,{{ chart_fv }}" alt="Fair Value Chart"></div>

<!-- 가격 예측 -->
<div class="section-hdr">
  <h2>WTI 가격 예측 (30 / 60 / 90일)</h2>
  <span class="badge">Monte Carlo N=2,000</span>
  <span class="badge">GS 공정가 수렴 모델</span>
</div>

<table class="fc-table">
<tr>
  <th>지평</th><th>P10 (하방 10%)</th><th>중앙값</th><th>평균</th>
  <th>P90 (상방 10%)</th><th>상승 확률</th>
</tr>
<tr style="background:#f6f8fa">
  <td>현재</td>
  <td colspan="4" style="text-align:center">${{ "%.2f"|format(fv.wti_latest) }}</td>
  <td>—</td>
</tr>
{% for h in [30,60,90] %}
{% set info = fc_horizons[h] %}
<tr>
  <td><strong>+{{ h }}일</strong></td>
  <td class="cr">${{ "%.1f"|format(info.p10) }}</td>
  <td class="cb">${{ "%.1f"|format(info.median) }}</td>
  <td>${{ "%.1f"|format(info.mean) }}</td>
  <td class="cg">${{ "%.1f"|format(info.p90) }}</td>
  <td class="{{ 'cg' if info.prob_up>=50 else 'cr' }}">{{ "%.1f"|format(info.prob_up) }}%</td>
</tr>
{% endfor %}
</table>

<div class="chart-card"><img src="data:image/png;base64,{{ chart_fc }}" alt="Forecast Chart"></div>

<!-- 5사 시나리오 -->
<div class="section-hdr">
  <h2>5-House Scenario Analysis</h2>
  <span class="badge-g badge">GS 28% / MS 27% / Citi 18% / JPM 18% / BB 9%</span>
  <span class="badge">MC N=100,000</span>
</div>

<div class="stat-grid">
  <div class="sc"><div class="sl">WTI Spot</div><div class="sv">${{ "%.2f"|format(sc.wti_spot) }}</div></div>
  <div class="sc"><div class="sl">Weighted EV</div><div class="sv cg">${{ "%.1f"|format(sc.w_ev) }}</div></div>
  <div class="sc"><div class="sl">Simple EV</div><div class="sv cy">${{ "%.1f"|format(sc.s_ev) }}</div></div>
  <div class="sc">
    <div class="sl">Undervaluation</div>
    <div class="sv {{ 'cg' if sc.w_ev>sc.wti_spot else 'cr' }}">${{ '%+.1f'|format(sc.w_ev-sc.wti_spot) }}</div>
  </div>
  <div class="sc"><div class="sl">P(WTI &gt; $110)</div><div class="sv cb">{{ "%.1f"|format(sc.p110*100) }}%</div></div>
  <div class="sc"><div class="sl">P(WTI &gt; $120)</div><div class="sv cp">{{ "%.1f"|format(sc.p120*100) }}%</div></div>
  <div class="sc"><div class="sl">5th Percentile</div><div class="sv cr">${{ "%.1f"|format(sc.v5) }}</div></div>
  <div class="sc"><div class="sl">95th Percentile</div><div class="sv cg">${{ "%.1f"|format(sc.v95) }}</div></div>
</div>

<div class="chart-card"><img src="data:image/png;base64,{{ chart_sc }}" alt="Scenario Chart"></div>

<!-- ML 유가 예측 -->
<div class="section-hdr">
  <h2>ML &#xC720;&#xAC00; &#xC608;&#xCE21; (RandomForest + TimeSeriesSplit)</h2>
  <span class="badge">&#xD53C;&#xCCD0; {{ ml|length * 0 + 22 }}&#xAC1C;</span>
  <span class="badge">RandomizedSearchCV n_iter=50</span>
  <span class="badge">TimeSeriesSplit n_splits=5</span>
</div>

{% if ml %}
<table class="fc-table">
<tr>
  <th>&#xC9C0;&#xD3C9;</th>
  <th>&#xBC29;&#xD5A5; (&#xBD84;&#xB958;)</th>
  <th>&#xBC29;&#xD5A5; &#xC815;&#xD655;&#xB3C4;</th>
  <th>&#xC608;&#xCE21; &#xC218;&#xC775;&#xB960;</th>
  <th>&#xC608;&#xCE21; &#xAC00;&#xACA9;</th>
  <th>&#xD68C;&#xADC0; MAE</th>
  <th>R&#xB178;</th>
</tr>
{% for h in [7,14,30,60,90] %}{% if h in ml %}
{% set info = ml[h] %}
<tr>
  <td><strong>+{{ h }}&#xC77C;</strong></td>
  <td class="{{ 'cg' if '&#xC0C1;&#xC2B9;' in info.dir_label else ('cr' if '&#xD558;&#xB77D;' in info.dir_label else 'cy') }}">
    {{ info.dir_label }}
  </td>
  <td class="{{ 'cg' if info.dir_acc>=0.6 else ('cy' if info.dir_acc>=0.45 else 'cr') }}">
    {{ "%.1f"|format(info.dir_acc*100) }}%
  </td>
  <td class="{{ 'cg' if info.pred_return>=0 else 'cr' }}">
    {{ '%+.2f'|format(info.pred_return*100) }}%
  </td>
  <td class="{{ 'cg' if info.pred_return>=0 else 'cr' }}">${{ "%.1f"|format(info.pred_price) }}</td>
  <td class="cy">{{ "%.4f"|format(info.mae*100) }}%</td>
  <td class="{{ 'cg' if info.r2>0.1 else 'cr' }}">{{ "%.3f"|format(info.r2) }}</td>
</tr>
{% endif %}{% endfor %}
</table>
{% endif %}

<div class="chart-card"><img src="data:image/png;base64,{{ chart_ml }}" alt="ML Chart"></div>

<!-- 호르무즈 유조선 예측 -->
<div class="section-hdr">
  <h2>&#xD638;&#xB974;&#xBB34;&#xC988; &#xC720;&#xC870;&#xC120; &#xC608;&#xCE21;</h2>
  <span class="badge">ArcGIS chokepoint6 365&#xC77C;</span>
  <span class="badge">Monte Carlo N=1,000</span>
  <span class="badge {{ 'badge-g' if arcgis_live else 'badge-r' }}">
    ArcGIS: {{ 'LIVE' if arcgis_live else 'FAIL' }}
  </span>
  <span class="badge {{ 'badge-g' if straits_ok else 'badge-y' }}">
    straits.live: {{ 'LIVE' if straits_ok else 'N/A' }}
  </span>
</div>

{% if tanker_horizons %}
<table class="fc-table">
<tr>
  <th>&#xC9C0;&#xD3C9;</th>
  <th>P10 (&#xD558;&#xBC29;)</th>
  <th>&#xC911;&#xC559;&#xAC12; (P50)</th>
  <th>&#xD3C9;&#xADE0;</th>
  <th>P90 (&#xC0C1;&#xBC29;)</th>
  <th>&#xC704;&#xAE30; &#xD655;&#xB960; (&lt;60% &#xD3C9;&#xADE0;)</th>
</tr>
{% for h in [7,14,30,60,90] %}
{% set info = tanker_horizons[h] %}
<tr>
  <td><strong>+{{ h }}&#xC77C;</strong></td>
  <td class="cr">{{ "%.1f"|format(info.p10) }}</td>
  <td class="cb">{{ "%.1f"|format(info.p50) }}</td>
  <td>{{ "%.1f"|format(info.mean) }}</td>
  <td class="cg">{{ "%.1f"|format(info.p90) }}</td>
  <td class="{{ 'cr' if info.crisis_prob > 20 else ('cy' if info.crisis_prob > 5 else 'cg') }}">
    {{ "%.1f"|format(info.crisis_prob) }}%
  </td>
</tr>
{% endfor %}
</table>
{% endif %}

<div class="chart-card"><img src="data:image/png;base64,{{ chart_tanker }}" alt="Hormuz Tanker Chart"></div>

</main>
<!-- CSV 다운로드 -->
<div class="section-hdr">
  <h2>크롤링 CSV 다운로드</h2>
  <span class="badge">data/ 폴더 자동 저장</span>
</div>
<div class="chart-card">
  <table class="src-table" id="csv-table">
    <tr><th>파일명</th><th>도메인</th><th>크기</th><th>저장 시각</th><th>다운로드</th></tr>
  </table>
  <p id="csv-loading" style="color:#57606a;font-size:.82rem;padding:8px">파일 목록 로딩 중...</p>
</div>

<footer>OilPriceValuation &nbsp;|&nbsp; KFBI Internal &nbsp;|&nbsp; 크롤링: Yahoo Finance (yfinance) + EIA Open Data &nbsp;|&nbsp; {{ mkt.crawled_at }}</footer>

<script>
function recrawl(){
  document.getElementById('overlay').classList.add('show');
  fetch('/recrawl').then(()=>location.reload()).catch(()=>location.reload());
}

const DOMAIN_MAP = {
  "yahoo_finance_wti":           "Yahoo Finance (CL=F)",
  "yahoo_finance_brent":         "Yahoo Finance (BZ=F)",
  "yahoo_finance_dxy":           "Yahoo Finance (DX-Y.NYB)",
  "yahoo_finance_ovx":           "Yahoo Finance (^OVX)",
  "yahoo_finance_futures_slope": "Yahoo Finance (WTI Futures Curve)",
  "eia_commercial_inventory":    "EIA Open Data (WCESTUS1)",
  "eia_spr":                     "EIA Open Data (SPR 유도)",
  "eia_us_production":           "EIA Open Data (U.S. Production)",
  "imf_portwatch_hormuz":        "IMF PortWatch (Strait of Hormuz)",
};

fetch('/csv-list').then(r=>r.json()).then(data=>{
  const tbody = document.getElementById('csv-table');
  const loading = document.getElementById('csv-loading');
  if(!data.files || data.files.length===0){
    loading.textContent='저장된 CSV 없음 (페이지 첫 로드 후 생성됩니다)';
    return;
  }
  loading.style.display='none';
  data.files.forEach(f=>{
    const key = f.name.replace('.csv','');
    const domain = DOMAIN_MAP[key] || '—';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${f.name}</strong></td>
      <td style="color:#57606a">${domain}</td>
      <td>${f.size_kb} KB</td>
      <td style="color:#57606a">${f.updated}</td>
      <td><a href="/csv/${f.name}" style="color:#0969da;font-weight:600">다운로드</a></td>
    `;
    tbody.appendChild(tr);
  });
}).catch(()=>{
  document.getElementById('csv-loading').textContent='CSV 목록 불러오기 실패';
});
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════
# 8. Flask 라우트
# ══════════════════════════════════════════════════════════
@app.route("/")
def index():
    mkt       = fetch_all_data()
    fv        = run_fair_value(mkt)
    fc        = run_forecast(fv)
    sc        = run_scenario(fv["wti_latest"])

    # 호르무즈 탱커 예측
    arcgis    = mkt.get("arcgis_hormuz") or {}
    straits   = mkt.get("straits_live")  or {}
    tc        = run_tanker_forecast(arcgis)

    # ML 예측 (피처 매트릭스 구성 → RF 튜닝)
    df_feat   = build_feature_matrix(mkt)
    ml        = run_ml_forecast(df_feat)

    arcgis_live = bool(arcgis and not arcgis.get("error"))
    straits_ok  = bool(straits and not straits.get("error"))
    pw_live     = arcgis_live

    chart_fv     = chart_fair_value(fv, pw_live)
    chart_fc     = chart_forecast(fv, fc)
    chart_sc     = chart_scenario(sc)
    chart_tnk    = chart_tanker(arcgis, tc, straits)
    chart_ml_png = chart_ml(ml, df_feat)

    # 소스 현황 테이블 데이터
    src = mkt.get("sources", {})
    sources = [
        ("WTI Spot",             "Yahoo Finance CL=F",               src.get("WTI","??")),
        ("Brent Spot",           "Yahoo Finance BZ=F",               src.get("Brent","??")),
        ("DXY",                  "Yahoo Finance DX-Y.NYB",           src.get("DXY","??")),
        ("OVX (변동성)",          "Yahoo Finance ^OVX",               src.get("OVX","??")),
        ("USO (WTI ETF)",        "Yahoo Finance USO",                src.get("USO","??")),
        ("BNO (Brent ETF)",      "Yahoo Finance BNO",                src.get("BNO","??")),
        ("WTI 선물 커브 M1~M6",  "Yahoo Finance CL futures",         src.get("futures_slope","??")),
        ("FRED WTI",             "FRED DCOILWTICO (무료, 키 불필요)", src.get("FRED_WTI","??")),
        ("FRED Brent",           "FRED DCOILBRENTEU (무료)",         src.get("FRED_Brent","??")),
        ("Brent-WTI 스프레드",   "직접 계산 (Brent - WTI)",          src.get("brent_wti_spread","??")),
        # 매크로
        ("10y Treasury (^TNX)",  "yfinance ^TNX",                   src.get("macro_TNX","??")),
        ("S&P 500",              "yfinance ^GSPC",                  src.get("macro_SP500","??")),
        ("MSCI World (URTH)",    "yfinance URTH ETF",               src.get("macro_MSCI_World","??")),
        ("Gold (GC=F)",          "yfinance 금 선물",                 src.get("macro_Gold","??")),
        ("Copper (HG=F)",        "yfinance 구리 선물",               src.get("macro_Copper","??")),
        ("VIX (공포지수)",        "yfinance ^VIX",                   src.get("macro_VIX","??")),
        ("TIP ETF (CPI 프록시)", "yfinance TIP",                    src.get("macro_TIP","??")),
        ("XLI ETF (ISM 프록시)", "yfinance XLI",                    src.get("macro_XLI","??")),
        ("FXI ETF (중국 PMI)",   "yfinance FXI",                    src.get("macro_FXI","??")),
        # EIA 수급
        ("EIA 상업재고",          "EIA WPSR v2 (WCESTUS1)",          src.get("eia_stock","??")),
        ("EIA 가솔린 재고",       "EIA WPSR v2 (EPM0F)",             src.get("wpsr_gasoline","??")),
        ("EIA 중간유 재고",       "EIA WPSR v2 (EPD0)",              src.get("wpsr_distillate","??")),
        ("EIA 원유 수입량",       "EIA WPSR v2 (SAI)",               src.get("wpsr_crude_imports","??")),
        ("SPR 잔여량",            "EIA API v2 (유도)",                src.get("spr_level","??")),
        ("호르무즈 선박 (ArcGIS)","ArcGIS chokepoint6 Daily",        src.get("vessel_data","??")),
        ("straits.live 현황",    "실시간 스냅샷 스크래핑",              src.get("straits_live","??")),
    ]

    return render_template_string(
        HTML,
        mkt=mkt, fv=fv, fc_horizons=fc["horizons"], sc=sc,
        pw_live=pw_live, sources=sources,
        arcgis_live=arcgis_live, straits_ok=straits_ok,
        tanker_horizons=tc.get("horizons") if tc else None,
        ml=ml,
        chart_fv=chart_fv, chart_fc=chart_fc,
        chart_sc=chart_sc, chart_tanker=chart_tnk,
        chart_ml=chart_ml_png,
    )


@app.route("/recrawl")
def recrawl():
    d = fetch_all_data(force=True)
    return jsonify({"status": "ok", "crawled_at": d["crawled_at"]})


@app.route("/debug-crawl")
def debug_crawl():
    """크롤링 소스 상태 + EIA 직접 테스트."""
    d = _cache.get("data") or {}
    sources = d.get("sources", {})
    # EIA 직접 ping
    eia_status = {}
    try:
        r = requests.get(
            f"{EIA_BASE}/petroleum/stoc/wstk/data/",
            params={"api_key": EIA_KEY, "frequency": "weekly",
                    "data[0]": "value", "facets[product][]": "EPC0",
                    "facets[duoarea][]": "NUS", "facets[process][]": "SAX",
                    "sort[0][column]": "period", "sort[0][direction]": "desc",
                    "offset": 0, "length": 2},
            timeout=10,
        )
        eia_status = {"http_status": r.status_code,
                      "records": len(r.json().get("response", {}).get("data", []))}
    except Exception as e:
        eia_status = {"error": str(e)}
    return jsonify({"sources": sources, "eia_ping": eia_status,
                    "csv_dir": CSV_DIR})


@app.route("/csv-list")
def csv_list():
    """저장된 CSV 파일 목록 + 크기 반환."""
    files = []
    if os.path.isdir(CSV_DIR):
        for fname in sorted(os.listdir(CSV_DIR)):
            if fname.endswith(".csv"):
                fpath = os.path.join(CSV_DIR, fname)
                size  = os.path.getsize(fpath)
                mtime = datetime.datetime.fromtimestamp(
                    os.path.getmtime(fpath)
                ).strftime("%Y-%m-%d %H:%M:%S")
                files.append({"name": fname, "size_kb": round(size/1024, 1),
                               "updated": mtime})
    return jsonify({"csv_dir": CSV_DIR, "files": files})


@app.route("/csv/<filename>")
def csv_download(filename: str):
    """CSV 파일 다운로드."""
    from flask import send_from_directory, abort
    safe = os.path.basename(filename)
    if not safe.endswith(".csv"):
        abort(400)
    return send_from_directory(CSV_DIR, safe, as_attachment=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  Oil Analysis Dashboard")
    print("  Crawling: yfinance + EIA + ArcGIS Hormuz + straits.live")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000)
