"""
five_house_weighted.py
======================
5사 가중치 시나리오 분석 + Monte Carlo + 6패널 차트

하우스 가중치:
  GS 28% / MS 27% / Citi 18% / JPM 18% / Bloomberg 9%

시나리오 (2026-06-18 업데이트):
  S1 합의·호르무즈 재개방 (60%)  ← 미국-이란 60일 MOU 체결 현실화
  S2 에너지 인프라 타격 (15%)    ← Trump 경고 발언, 불확실성 잔존
  S3 장기화 (15%)               ← 협상 결렬 시 재봉쇄 가능성
  S4 봉쇄+미국 철군 (10%)        ← 극단 시나리오

pip install numpy scipy matplotlib
python3 five_house_weighted.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ─────────────────────────────────────────
# 파라미터
# ─────────────────────────────────────────
WTI_SPOT = 75.54  # 2026-06-18 실제가 (NYMEX CL=F)

HOUSE_WEIGHTS = {
    "Goldman Sachs":   0.28,
    "Morgan Stanley":  0.27,
    "Citigroup":       0.18,
    "JPMorgan":        0.18,
    "Bloomberg":       0.09,
}

# 시나리오 확률 — 2026-06-18 업데이트
# 미국-이란 60일 MOU 체결, 호르무즈 재개방 진행 중
# IEA: 2027년 공급 +8mb/d vs 수요 +2mb/d 과잉 경고
SCENARIO_PROBS = {
    "S1_Agreement":     0.60,   # 60%: 합의·재개방 현실화 (구 10%)
    "S2_Strike":        0.15,   # 15%: 재점화 리스크 (구 40%, Trump 경고 잔존)
    "S3_Prolonged":     0.15,   # 15%: 협상 결렬·장기화 (구 30%)
    "S4_Withdrawal":    0.10,   # 10%: 극단 봉쇄 (구 20%)
}

# 5사 × 4 시나리오 WTI 목표가 — 2026-06-18 업데이트
# S1: 공급 정상화 + IEA 과잉 반영 → 하향
# S2~S4: 재점화 시나리오 유지 (지정학 프리미엄)
TARGETS = {
    #                     S1    S2    S3    S4
    "Goldman Sachs":   [  72,  100,  115,   88],  # S1 $77→$72 (IEA 과잉)
    "Morgan Stanley":  [  68,  103,  110,   87],  # S1 $72→$68
    "Citigroup":       [  70,  108,  128,   95],  # S1 $75→$70
    "JPMorgan":        [  73,  105,  112,   97],  # S1 $78→$73
    "Bloomberg":       [  74,  110,  145,   98],  # S1 $80→$74 (과잉 민감)
}

SCENARIOS = list(SCENARIO_PROBS.keys())
HOUSES    = list(HOUSE_WEIGHTS.keys())
WEIGHTS   = np.array([HOUSE_WEIGHTS[h] for h in HOUSES])
PROBS     = np.array([SCENARIO_PROBS[s] for s in SCENARIOS])
TARGET_MATRIX = np.array([TARGETS[h] for h in HOUSES])  # (5, 4)

# 변동성 (시나리오별 표준편차) — S1 변동성 축소 (합의 불확실성 감소)
SIGMA = {"S1_Agreement": 4.0, "S2_Strike": 14.0, "S3_Prolonged": 20.0, "S4_Withdrawal": 12.0}

# ─────────────────────────────────────────
# 연산
# ─────────────────────────────────────────

# 각 하우스별 확률가중평균
house_ev = TARGET_MATRIX @ PROBS  # (5,)

# 최종 가중평균 (하우스 가중치 적용)
weighted_ev = float(WEIGHTS @ house_ev)
simple_ev   = float(house_ev.mean())

# 단순 가중 시나리오 컬럼 평균 (시나리오별 전체 가중 가격)
scenario_weighted = WEIGHTS @ TARGET_MATRIX  # (4,)

print("=" * 55)
print("  5사 가중치 시나리오 분석  (WTI 기준)")
print("=" * 55)
print(f"  현재가         : ${WTI_SPOT:.2f}")
print(f"  가중평균 EV    : ${weighted_ev:.1f}")
print(f"  단순평균 EV    : ${simple_ev:.1f}")
print(f"  저평가 폭      : ${weighted_ev - WTI_SPOT:.1f}")
print()
for i, h in enumerate(HOUSES):
    print(f"  {h:<18} EV=${house_ev[i]:.1f}  (w={HOUSE_WEIGHTS[h]*100:.0f}%)")
print("=" * 55)

# Monte Carlo (100,000 경로)
N_SIM = 100_000
rng = np.random.default_rng(42)
sim_prices = np.zeros(N_SIM)

for k, s in enumerate(SCENARIOS):
    n_draw = int(N_SIM * PROBS[k])
    scenario_price = float(scenario_weighted[k])
    draws = rng.normal(scenario_price, SIGMA[s], n_draw)
    start = int(N_SIM * PROBS[:k].sum())
    sim_prices[start: start + n_draw] = draws

# 남은 슬롯 채우기
sim_prices[sim_prices == 0] = rng.normal(weighted_ev, 10.0, (sim_prices == 0).sum())

p_above_110 = float((sim_prices > 110).mean())
p_above_120 = float((sim_prices > 120).mean())
var_5 = float(np.percentile(sim_prices, 5))
var_95 = float(np.percentile(sim_prices, 95))

print(f"\n  MC 100k 결과:")
print(f"  P(WTI > $110) = {p_above_110*100:.1f}%")
print(f"  P(WTI > $120) = {p_above_120*100:.1f}%")
print(f"  5th pct       = ${var_5:.1f}")
print(f"  95th pct      = ${var_95:.1f}\n")

# ─────────────────────────────────────────
# 6패널 차트
# ─────────────────────────────────────────
DARK = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
BLUE   = "#58a6ff"
PURPLE = "#bc8cff"
ORANGE = "#ffa657"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"

fig = plt.figure(figsize=(22, 14), facecolor=DARK)
fig.suptitle(
    "OilPriceValuation — 5-House Weighted Scenario Analysis  |  KFBI Internal",
    color=WHITE, fontsize=14, fontweight="bold", y=0.98,
)

gs_layout = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38,
                               left=0.05, right=0.97, top=0.93, bottom=0.06)
axes = [
    fig.add_subplot(gs_layout[0, 0]),
    fig.add_subplot(gs_layout[0, 1]),
    fig.add_subplot(gs_layout[0, 2]),
    fig.add_subplot(gs_layout[1, :2]),
    fig.add_subplot(gs_layout[1, 2]),
    fig.add_subplot(gs_layout[2, :]),
]

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.set_title(title, color=WHITE, fontsize=9, fontweight="bold", pad=6)

# ── Panel 1: 하우스 가중치 파이 ──
ax = axes[0]
style_ax(ax, "House Weights")
colors_pie = [BLUE, GREEN, YELLOW, ORANGE, RED]
wedges, texts, autotexts = ax.pie(
    [HOUSE_WEIGHTS[h] for h in HOUSES],
    labels=[h.replace(" ", "\n") for h in HOUSES],
    autopct="%1.0f%%",
    colors=colors_pie,
    textprops={"color": WHITE, "fontsize": 7},
    wedgeprops={"edgecolor": DARK, "linewidth": 1.2},
    startangle=90,
)
for at in autotexts:
    at.set_fontsize(7)
    at.set_color(DARK)

# ── Panel 2: 시나리오 확률 바 ──
ax = axes[1]
style_ax(ax, "Scenario Probabilities")
short = ["S1\nAgreement", "S2\nStrike", "S3\nProlonged", "S4\nWithdrawal"]
bar_colors = [GREEN, RED, ORANGE, PURPLE]
bars = ax.bar(short, PROBS * 100, color=bar_colors, edgecolor=DARK, linewidth=0.8)
for b, p in zip(bars, PROBS):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
            f"{p*100:.0f}%", ha="center", va="bottom", color=WHITE, fontsize=8)
ax.set_ylabel("Probability (%)", color=GRAY, fontsize=8)
ax.set_ylim(0, 55)
ax.yaxis.label.set_color(GRAY)

# ── Panel 3: 하우스별 EV ──
ax = axes[2]
style_ax(ax, "House Expected Value (WTI)")
h_colors = colors_pie
bars2 = ax.barh(HOUSES, house_ev, color=h_colors, edgecolor=DARK, linewidth=0.8)
ax.axvline(WTI_SPOT, color=RED, linestyle="--", linewidth=1.2, label=f"Spot ${WTI_SPOT:.2f}")
ax.axvline(weighted_ev, color=GREEN, linestyle="--", linewidth=1.2, label=f"Wtd EV ${weighted_ev:.1f}")
for b, v in zip(bars2, house_ev):
    ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
            f"${v:.1f}", va="center", color=WHITE, fontsize=8)
ax.set_xlabel("WTI (USD)", color=GRAY, fontsize=8)
ax.legend(fontsize=7, labelcolor=WHITE, facecolor=CARD, edgecolor=BORDER)
ax.set_xlim(60, 145)

# ── Panel 4: 시나리오 × 하우스 히트맵 ──
ax = axes[3]
style_ax(ax, "Scenario Target Matrix (USD)")
im = ax.imshow(TARGET_MATRIX, cmap="RdYlGn", aspect="auto",
               vmin=65, vmax=165)
ax.set_xticks(range(len(SCENARIOS)))
ax.set_xticklabels(["S1 Agree", "S2 Strike", "S3 Prolong", "S4 Withdraw"],
                   color=WHITE, fontsize=8)
ax.set_yticks(range(len(HOUSES)))
ax.set_yticklabels(HOUSES, color=WHITE, fontsize=8)
for i in range(len(HOUSES)):
    for j in range(len(SCENARIOS)):
        ax.text(j, i, f"${TARGET_MATRIX[i,j]}", ha="center", va="center",
                color="black", fontsize=8, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

# ── Panel 5: 가중평균 게이지 ──
ax = axes[4]
style_ax(ax, "Weighted EV vs Spot")
cats = ["Spot", "Simple EV", "Weighted EV"]
vals = [WTI_SPOT, simple_ev, weighted_ev]
cols = [GRAY, YELLOW, GREEN]
bars3 = ax.bar(cats, vals, color=cols, edgecolor=DARK, linewidth=0.8, width=0.5)
for b, v in zip(bars3, vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
            f"${v:.1f}", ha="center", color=WHITE, fontsize=9, fontweight="bold")
ax.set_ylim(60, 135)
ax.set_ylabel("WTI USD", color=GRAY, fontsize=8)
underval = weighted_ev - WTI_SPOT
ax.set_title(f"Weighted EV vs Spot  |  Undervaluation: ${underval:.1f}",
             color=WHITE, fontsize=9, fontweight="bold", pad=6)

# ── Panel 6: MC 히스토그램 ──
ax = axes[5]
style_ax(ax, f"Monte Carlo Distribution  (N={N_SIM:,})")
ax.hist(sim_prices, bins=120, color=BLUE, edgecolor=DARK, alpha=0.75, linewidth=0.4)
ax.axvline(WTI_SPOT,   color=RED,    linestyle="--", linewidth=1.5, label=f"Spot ${WTI_SPOT:.2f}")
ax.axvline(weighted_ev, color=GREEN, linestyle="--", linewidth=1.5, label=f"Wtd EV ${weighted_ev:.1f}")
ax.axvline(var_5,  color=YELLOW, linestyle=":",  linewidth=1.2, label=f"5th ${var_5:.1f}")
ax.axvline(var_95, color=ORANGE, linestyle=":",  linewidth=1.2, label=f"95th ${var_95:.1f}")
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5000],
                 var_5, var_95, alpha=0.08, color=BLUE)
ax.set_xlabel("WTI USD", color=GRAY, fontsize=9)
ax.set_ylabel("Frequency", color=GRAY, fontsize=9)
ax.legend(fontsize=8, labelcolor=WHITE, facecolor=CARD, edgecolor=BORDER, ncol=4)

# 하단 텍스트 박스
fig.text(0.05, 0.01,
         f"WTI Spot: ${WTI_SPOT:.2f}  |  Weighted EV: ${weighted_ev:.1f}  "
         f"(+${weighted_ev-WTI_SPOT:.1f} undervalued)  |  "
         f"P(>$110)={p_above_110*100:.1f}%  P(>$120)={p_above_120*100:.1f}%  "
         f"| Weights: GS28% MS27% Citi18% JPM18% BB9%",
         color=GRAY, fontsize=8)

outpath = "five_house_weighted.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"차트 저장: {outpath}")
plt.show()
