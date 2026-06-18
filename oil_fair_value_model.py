"""
oil_fair_value_model.py  (Streamlit 버전)
=======================
MS-style Kalman Filter + GS-style Two-Stage Bridge Equation
+ RandomForest Supply Shock Index 통합 모델

실행: streamlit run oil_fair_value_model.py
pip install streamlit numpy scikit-learn matplotlib
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="OilPriceValuation",
    page_icon="🛢️",
    layout="wide",
)
st.title("OilPriceValuation — MS Kalman + GS Bridge + RF Shock")
st.caption("KFBI Internal | 시뮬레이션 데이터 (실제 운용 시 API로 교체)")

# ─────────────────────────────────────────
# 사이드바: 파라미터 조정
# ─────────────────────────────────────────
with st.sidebar:
    st.header("파라미터")
    seed = st.number_input("랜덤 시드", value=42, step=1)
    T = st.slider("거래일 수", min_value=60, max_value=520, value=260, step=20)
    n_estimators = st.slider("RF 트리 수", min_value=50, max_value=300, value=100, step=50)
    st.divider()
    st.caption("파라미터를 바꾸면 자동으로 재계산됩니다.")

# ─────────────────────────────────────────
# 1. 시뮬레이션 데이터 생성
#    (실제 운용 시: OPEC, EIA, FRED, PortWatch 등 API로 교체)
# ─────────────────────────────────────────
rng = np.random.default_rng(seed)
t = np.arange(T)

# WTI Spot (실제 데이터 교체 포인트 ①)
# 2026-06-18 기준: $75.54 (미국-이란 MOU, 호르무즈 재개방 진행)
wti_spot = 75.54 + np.cumsum(rng.normal(0, 0.8, T))
wti_spot = np.clip(wti_spot, 40, 160)

# 공급 갭 (mb/d, + = 과잉공급)  (실제 교체 포인트 ②: OPEC API)
# IEA 2027년 과잉 +6mb/d 반영 → 양수(과잉) 편향
supply_gap = rng.normal(0.8, 1.2, T)  # 구: 0 → 0.8 (IEA 과잉 반영)

# EIA 상업 재고 변화 (Mbbl)  (실제 교체 포인트 ③: EIA API)
# 최근 주간 -8.3 Mbbl 감소 → 평균 소폭 음수
eia_stock = rng.normal(-1.5, 3.0, T)  # 구: -0.5 → -1.5

# SPR 잔여 (Mbbl)  (실제 교체 포인트 ④: EIA API)
# 봉쇄 기간 중 SPR 방출로 잔여량 감소 반영
spr_level = 320 + np.cumsum(rng.normal(-0.1, 1.5, T))  # 구: 350 → 320

# DXY  (실제 교체 포인트 ⑤: FRED DXY)
dxy = 103 + np.cumsum(rng.normal(0, 0.3, T))  # 구: 100 → 103 (달러 강세)

# OVX (원유 변동성 지수)  (실제 교체 포인트 ⑥: Yahoo Finance ^OVX)
# 재개방 기대로 변동성 다소 안정 → 평균 하향
ovx = 30 + 8 * np.abs(rng.normal(0, 1, T))  # 구: 25+10 → 30+8

# 호르무즈 선박수 (7dMA)  (실제 교체 포인트 ⑦: IMF PortWatch)
# 2026-04 기준 6척 → 재개방 진행 중 → 점진 회복 시뮬레이션
vessel_recovery = np.clip(np.linspace(6, 80, T), 0, 138)  # 6→80 점진 회복
vessel_count = np.clip(vessel_recovery + rng.normal(0, 8, T), 0, 138)

# 선물 커브 기울기 (3M - spot)  (실제 교체 포인트 ⑧: CME)
# 콘탱고(backwardation 해소): 재개방 기대 → 근월물 하락
futures_slope = rng.normal(2.5, 0.8, T)  # 구: 1.5 → 2.5 (콘탱고 심화)


# ─────────────────────────────────────────
# 2. RandomForest Supply Shock Index (GS 보완)
# ─────────────────────────────────────────
with st.spinner("RF Supply Shock Index 학습 중..."):
    rf_features = np.column_stack([
        vessel_count,
        supply_gap,
        eia_stock,
    ])

    rf_target = -np.diff(wti_spot, prepend=wti_spot[0])
    rf_target = np.clip(rf_target, -10, 10)

    scaler_rf = StandardScaler()
    X_rf = scaler_rf.fit_transform(rf_features)

    tscv = TimeSeriesSplit(n_splits=5)
    rf_scores = []
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=5, random_state=42, n_jobs=-1
    )

    for train_idx, val_idx in tscv.split(X_rf):
        rf_model.fit(X_rf[train_idx], rf_target[train_idx])
        score = rf_model.score(X_rf[val_idx], rf_target[val_idx])
        rf_scores.append(score)

    rf_model.fit(X_rf, rf_target)
    shock_index = rf_model.predict(X_rf)
    shock_min = shock_index.min()
    shock_max = shock_index.max()
    shock_index = (shock_index - shock_min) / (shock_max - shock_min + 1e-9)

feat_names = ["vessel_cnt", "supply_gap", "eia_stock"]
importances = rf_model.feature_importances_
best_feat_idx = np.argmax(importances)


# ─────────────────────────────────────────
# 3. MS-style Kalman Filter (잠재 펀더멘털 추출)
# ─────────────────────────────────────────
def run_kalman(obs, q_ratio):
    """
    Local Level Model (Random Walk + Noise)
    q_ratio = Q/R (signal-to-noise ratio)
    """
    n = len(obs)
    Q = q_ratio
    R = 1.0
    x_hat = np.zeros(n)
    P = np.zeros(n)
    x_hat[0] = obs[0]
    P[0] = 1.0

    for i in range(1, n):
        # Predict
        x_pred = x_hat[i - 1]
        P_pred = P[i - 1] + Q
        # Update
        K = P_pred / (P_pred + R)
        x_hat[i] = x_pred + K * (obs[i] - x_pred)
        P[i] = (1 - K) * P_pred

    return x_hat, P

obs_norm = (wti_spot - wti_spot.mean()) / (wti_spot.std() + 1e-9)
best_q, best_rmse = 0.01, 1e9
for q in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
    est, _ = run_kalman(obs_norm, q)
    rmse = np.sqrt(np.mean((obs_norm - est) ** 2))
    if rmse < best_rmse:
        best_rmse = rmse
        best_q = q

kalman_norm, kalman_var = run_kalman(obs_norm, best_q)
kalman_fundamental = kalman_norm * wti_spot.std() + wti_spot.mean()
kalman_uncertainty = np.sqrt(kalman_var) * wti_spot.std()


# ─────────────────────────────────────────
# 4. GS-style Two-Stage Bridge Equation
#    Stage A: 물리적 수급 → 적정가
#    Stage B: 지정학 리스크 프리미엄 추가
# ─────────────────────────────────────────
d_spr = np.diff(spr_level, prepend=spr_level[0])
d_dxy = np.diff(dxy, prepend=dxy[0])
d_wti = np.diff(wti_spot, prepend=wti_spot[0])

X_a = np.column_stack([supply_gap, d_spr, d_dxy, eia_stock, np.ones(T)])
scaler_a = StandardScaler()
X_a_sc = scaler_a.fit_transform(X_a[:, :-1])
X_a_sc = np.column_stack([X_a_sc, np.ones(T)])

beta_a = np.linalg.lstsq(X_a_sc, d_wti, rcond=None)[0]
stage_a_delta = X_a_sc @ beta_a
stage_a_price = wti_spot[0] + np.cumsum(stage_a_delta)

ss_res = np.sum((d_wti - stage_a_delta) ** 2)
ss_tot = np.sum((d_wti - d_wti.mean()) ** 2)
r2_a = 1 - ss_res / (ss_tot + 1e-9)

X_b = np.column_stack([shock_index, ovx, futures_slope, np.ones(T)])
scaler_b = StandardScaler()
X_b_sc = scaler_b.fit_transform(X_b[:, :-1])
X_b_sc = np.column_stack([X_b_sc, np.ones(T)])

beta_b = np.linalg.lstsq(X_b_sc, d_wti - stage_a_delta, rcond=None)[0]
geo_premium = X_b_sc @ beta_b

ss_res_b = np.sum(((d_wti - stage_a_delta) - geo_premium) ** 2)
ss_tot_b = np.sum(((d_wti - stage_a_delta) - (d_wti - stage_a_delta).mean()) ** 2)
r2_b = 1 - ss_res_b / (ss_tot_b + 1e-9)

gs_fair_value = stage_a_price + np.cumsum(geo_premium)
overshoot = wti_spot - gs_fair_value


# ─────────────────────────────────────────
# 5. 핵심 지표 요약 (상단 메트릭 카드)
# ─────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("WTI 현재가", f"${wti_spot[-1]:.2f}")
col2.metric("GS Fair Value", f"${gs_fair_value[-1]:.2f}")
col3.metric("오버슈트", f"${overshoot[-1]:+.2f}",
            delta_color="inverse")
col4.metric("Stage A R²", f"{r2_a:.3f}")
col5.metric("Stage B R²", f"{r2_b:.3f}")


# ─────────────────────────────────────────
# 6. 6패널 다크 테마 시각화
# ─────────────────────────────────────────
DARK   = "#0d1117"
CARD   = "#161b22"
BORDER = "#30363d"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
BLUE   = "#58a6ff"
PURPLE = "#bc8cff"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"

fig, axes = plt.subplots(3, 2, figsize=(18, 13), facecolor=DARK)
fig.suptitle(
    "OilPriceValuation — MS Kalman + GS Bridge + RF Shock  |  KFBI Internal",
    color=WHITE, fontsize=13, fontweight="bold", y=0.99,
)

def style(ax, title):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.set_title(title, color=WHITE, fontsize=9, fontweight="bold", pad=6)

# 패널 1: 실제가 vs GS Fair Value vs Kalman
ax = axes[0, 0]
style(ax, "WTI vs GS Fair Value vs Kalman Fundamental")
ax.plot(t, wti_spot, color=BLUE, linewidth=1.2, label="WTI Spot")
ax.plot(t, gs_fair_value, color=GREEN, linewidth=1.2, linestyle="--", label="GS Fair Value")
ax.plot(t, kalman_fundamental, color=PURPLE, linewidth=1.0, linestyle=":", label="Kalman Fundamental")
ax.fill_between(t, kalman_fundamental - kalman_uncertainty,
                kalman_fundamental + kalman_uncertainty, alpha=0.15, color=PURPLE)
ax.legend(fontsize=7, labelcolor=WHITE, facecolor=CARD, edgecolor=BORDER)
ax.set_ylabel("USD", color=GRAY, fontsize=8)

# 패널 2: RF Shock Index
ax = axes[0, 1]
style(ax, "RF Supply Shock Index")
ax.fill_between(t, 0, shock_index, color=RED, alpha=0.6)
ax.plot(t, shock_index, color=RED, linewidth=0.8)
ax.axhline(shock_index.mean(), color=YELLOW, linestyle="--", linewidth=1.0,
           label=f"Mean {shock_index.mean():.2f}")
ax.legend(fontsize=7, labelcolor=WHITE, facecolor=CARD, edgecolor=BORDER)
ax.set_ylabel("Normalized Index", color=GRAY, fontsize=8)
ax.set_ylim(-0.05, 1.1)

# 패널 3: Kalman Fundamental + 불확실성 밴드
ax = axes[1, 0]
style(ax, "Kalman Latent Fundamental (MS-style)")
ax.plot(t, wti_spot, color=BLUE, linewidth=0.8, alpha=0.5, label="WTI Spot")
ax.plot(t, kalman_fundamental, color=PURPLE, linewidth=1.5, label="Kalman Fundamental")
ax.fill_between(t,
                kalman_fundamental - 2 * kalman_uncertainty,
                kalman_fundamental + 2 * kalman_uncertainty,
                alpha=0.2, color=PURPLE, label="95% Band")
ax.legend(fontsize=7, labelcolor=WHITE, facecolor=CARD, edgecolor=BORDER)
ax.set_ylabel("USD", color=GRAY, fontsize=8)

# 패널 4: 가격 분해 (누적 스택)
ax = axes[1, 1]
style(ax, "Price Decomposition (Stacked)")
ax.stackplot(t,
             [stage_a_price, np.cumsum(geo_premium)],
             labels=["Stage A (Supply-Demand)", "Stage B (Geo Premium)"],
             colors=[BLUE, RED], alpha=0.7)
ax.plot(t, wti_spot, color=WHITE, linewidth=1.0, linestyle="--", label="WTI Spot")
ax.legend(fontsize=7, labelcolor=WHITE, facecolor=CARD, edgecolor=BORDER)
ax.set_ylabel("USD", color=GRAY, fontsize=8)

# 패널 5: RF 피처 중요도
ax = axes[2, 0]
style(ax, "RF Feature Importance")
sorted_idx = np.argsort(importances)

sorted_names = []
sorted_importances = []
sorted_colors = []
for i in sorted_idx:
    sorted_names.append(feat_names[i])
    sorted_importances.append(importances[i])
    if i == sorted_idx[-1]:
        sorted_colors.append(GREEN)
    else:
        sorted_colors.append(BLUE)

ax.barh(sorted_names, sorted_importances, color=sorted_colors, edgecolor=DARK, linewidth=0.8)
ax.set_xlabel("Importance", color=GRAY, fontsize=8)

# 패널 6: 오버슈트 타임시리즈
ax = axes[2, 1]
style(ax, f"Overshoot vs GS Fair  (Final: ${overshoot[-1]:+.2f})")
ax.fill_between(t, 0, overshoot,
                where=(overshoot > 0), color=RED, alpha=0.5, label="Over")
ax.fill_between(t, 0, overshoot,
                where=(overshoot <= 0), color=GREEN, alpha=0.5, label="Under")
ax.plot(t, overshoot, color=WHITE, linewidth=0.6)
ax.axhline(0, color=BORDER, linewidth=1.0)
ax.axhline(+8,  color=RED,   linestyle="--", linewidth=0.8, label="+8 Alert (구 +10)")
ax.axhline(-12, color=GREEN, linestyle="--", linewidth=0.8, label="-12 Alert (구 -20)")
ax.legend(fontsize=7, labelcolor=WHITE, facecolor=CARD, edgecolor=BORDER)
ax.set_ylabel("USD", color=GRAY, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.98])
st.pyplot(fig)

# ─────────────────────────────────────────
# 7. RF 상세 정보 (접을 수 있는 섹션)
# ─────────────────────────────────────────
with st.expander("RF 모델 상세"):
    st.write(f"**주요 변수:** {feat_names[best_feat_idx]}  ({importances.max()*100:.1f}%)")
    st.write(f"**TimeSeriesSplit CV R² 평균:** {np.mean(rf_scores):.3f}  (±{np.std(rf_scores):.3f})")
    st.write(f"**Kalman 최적 Q/R:** {best_q:.3f}")
