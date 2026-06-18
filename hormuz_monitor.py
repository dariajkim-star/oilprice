"""
hormuz_monitor.py
=================
호르무즈 봉쇄 실시간 모니터링 데몬
- IMF PortWatch → 선박수/봉쇄율 자동 수집
- Yahoo Finance / FRED → WTI 현재가
- GS Mar23 $105 기준 저평가/과도평가 알림
- KST 08:00, 10:00 강제 + 2시간 간격 스케줄

실행:
  python hormuz_monitor.py          # 1회 실행
  python hormuz_monitor.py --daemon # 데몬 모드 (스케줄 반복)

pip install requests pandas schedule
"""

import sys
import json
import time
import logging
import argparse
import datetime
import requests
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# ─────────────────────────────────────────
# 상수 / 파라미터
# ─────────────────────────────────────────
GS_MAR23_WTI      = 105.0   # GS Mar23 house view (WTI)
GS_BASE_WTI       = 84.0    # GS original base (reference only)
HORMUZ_NORMAL_VES = 138     # 정상 7일 MA 선박수
PIPELINE_OFFSET   = 3.5     # 파이프라인 우회 mb/d
SPR_RELEASE       = 2.0     # SPR 방출 mb/d
GS_LOSS_PER_MBD   = 10.0 / 14.0  # $0.714 / mb/d

# 알림 임계치 (GS Mar23 기준)
OVER_THRESHOLD    = +10.0   # 과도평가 경보 (현재가 - GS Mar23 > +10)
UNDER_THRESHOLD   = -20.0   # 저평가 경보   (현재가 - GS Mar23 < -20)

KST = ZoneInfo("Asia/Seoul")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 1. 호르무즈 선박수 (IMF PortWatch)
# ─────────────────────────────────────────
ARCGIS_URL = (
    "https://services3.arcgis.com/ZvidGQkLzkAMQbZQ/arcgis/rest/services/"
    "chokepoints_layer/FeatureServer/0/query"
)
ARCGIS_PARAMS = {
    "where": "port_id='chokepoint6'",
    "outFields": "date,transit_calls,vessel_7dma",
    "orderByFields": "date DESC",
    "resultRecordCount": 14,
    "f": "json",
}
GEOJSON_URL = (
    "https://opendata.arcgis.com/datasets/"
    "42132aa4e2fc4d41bdaf9a445f688931_0.geojson"
)
CSV_URL = (
    "https://data-download.imf.org/climatedata/"
    "portwatch-chokepoints-indicators.csv"
)
CACHE_CSV = "portwatch_cache.csv"


def fetch_portwatch() -> dict:
    """
    IMF PortWatch에서 호르무즈 선박 데이터 가져오기
    우선순위: ArcGIS REST → GeoJSON → CSV download → 로컬 캐시
    반환: {"vessels_7dma": float, "date": str, "source": str}
    """
    # 방법 1: ArcGIS REST API
    try:
        r = requests.get(ARCGIS_URL, params=ARCGIS_PARAMS, timeout=15)
        r.raise_for_status()
        features = r.json().get("features", [])
        if features:
            attrs = features[0]["attributes"]
            val = attrs.get("vessel_7dma") or attrs.get("transit_calls")
            date = attrs.get("date", "")
            if isinstance(date, int):
                date = datetime.datetime.utcfromtimestamp(date / 1000).strftime("%Y-%m-%d")
            log.info(f"[PortWatch ArcGIS] vessels_7dma={val}, date={date}")
            return {"vessels_7dma": float(val), "date": date, "source": "arcgis"}
    except Exception as e:
        log.warning(f"ArcGIS 실패: {e}")

    # 방법 2: GeoJSON Hub
    try:
        r = requests.get(GEOJSON_URL, timeout=20)
        r.raise_for_status()
        feats = r.json().get("features", [])
        ch6 = [f for f in feats if f["properties"].get("port_id") == "chokepoint6"]
        if ch6:
            ch6.sort(key=lambda x: x["properties"].get("date", ""), reverse=True)
            props = ch6[0]["properties"]
            val = props.get("vessel_7dma") or props.get("transit_calls")
            log.info(f"[PortWatch GeoJSON] vessels_7dma={val}")
            return {"vessels_7dma": float(val), "date": props.get("date", ""), "source": "geojson"}
    except Exception as e:
        log.warning(f"GeoJSON 실패: {e}")

    # 방법 3: CSV 직접 다운로드
    try:
        r = requests.get(CSV_URL, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.lower().strip() for c in df.columns]
        df = df[df.get("port_id", pd.Series()).str.lower() == "chokepoint6"].copy() if "port_id" in df.columns else df
        df = df.sort_values("date", ascending=False)
        val = float(df.iloc[0]["vessel_7dma"] if "vessel_7dma" in df.columns else df.iloc[0]["transit_calls"])
        date = str(df.iloc[0]["date"])
        # 캐시 저장
        df.to_csv(CACHE_CSV, index=False)
        log.info(f"[PortWatch CSV] vessels_7dma={val}, cached")
        return {"vessels_7dma": val, "date": date, "source": "csv"}
    except Exception as e:
        log.warning(f"CSV 실패: {e}")

    # 방법 4: 로컬 캐시
    try:
        df = pd.read_csv(CACHE_CSV)
        df.columns = [c.lower().strip() for c in df.columns]
        df = df.sort_values("date", ascending=False)
        val = float(df.iloc[0].get("vessel_7dma", df.iloc[0].get("transit_calls", 6)))
        log.warning(f"[PortWatch CACHE] vessels_7dma={val} (캐시 사용 — 실시간 아님)")
        return {"vessels_7dma": val, "date": str(df.iloc[0]["date"]), "source": "cache"}
    except Exception as e:
        log.error(f"캐시도 실패: {e}")

    # 최후 fallback: 하드코딩 최신값
    log.error("PortWatch 완전 실패. 하드코딩 fallback 사용 (6척)")
    return {"vessels_7dma": 6.0, "date": "fallback", "source": "fallback"}


# ─────────────────────────────────────────
# 2. WTI 현재가
# ─────────────────────────────────────────
def fetch_wti() -> float:
    """Yahoo Finance → FRED DCOILWTICO 순서로 WTI 가져오기"""
    # 방법 1: Yahoo Finance
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/CL=F?interval=1m&range=1d"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        price = r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
        log.info(f"[Yahoo] WTI = {price}")
        return float(price)
    except Exception as e:
        log.warning(f"Yahoo 실패: {e}")

    # 방법 2: FRED
    try:
        fred_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "DCOILWTICO",
            "api_key": "FRED_PUBLIC",  # public key로 접근 가능한 endpoint 사용
            "sort_order": "desc",
            "limit": 1,
            "file_type": "json",
        }
        r = requests.get(fred_url, params=params, timeout=10)
        r.raise_for_status()
        val = float(r.json()["observations"][0]["value"])
        log.info(f"[FRED] WTI = {val}")
        return val
    except Exception as e:
        log.warning(f"FRED 실패: {e}")

    log.error("WTI 가격 수집 실패. fallback = 98.34")
    return 98.34  # 최후 fallback


# ─────────────────────────────────────────
# 3. 핵심 연산
# ─────────────────────────────────────────
def compute_derived(vessels_7dma: float, wti_spot: float) -> dict:
    """
    선박수 + WTI → 봉쇄율, Goldman 프리미엄, 과/저평가 계산
    """
    blockade_rate = (HORMUZ_NORMAL_VES - vessels_7dma) / HORMUZ_NORMAL_VES

    net_loss = max(0.0, 20.0 * blockade_rate - PIPELINE_OFFSET - SPR_RELEASE)

    gs_premium = net_loss * GS_LOSS_PER_MBD

    # GS Mar23 fair value (결정 기준)
    gs_fair_mar23 = GS_MAR23_WTI  # $105 고정 house view

    # GS original formula fair value (참고용)
    gs_fair_formula = GS_BASE_WTI + gs_premium

    # 과/저평가 = 현재가 - GS Mar23 house view
    overshoot_mar23 = wti_spot - gs_fair_mar23

    # 방향 불일치 감지
    mar23_signal = "OVERVALUED" if overshoot_mar23 > 0 else "UNDERVALUED"
    formula_signal = "OVERVALUED" if (wti_spot - gs_fair_formula) > 0 else "UNDERVALUED"
    direction_mismatch = mar23_signal != formula_signal

    return {
        "blockade_rate":       blockade_rate,
        "net_loss_mbd":        net_loss,
        "gs_premium":          gs_premium,
        "gs_fair_mar23":       gs_fair_mar23,
        "gs_fair_formula":     gs_fair_formula,
        "overshoot_mar23":     overshoot_mar23,
        "overshoot_formula":   wti_spot - gs_fair_formula,
        "direction_mismatch":  direction_mismatch,
        "mar23_signal":        mar23_signal,
        "formula_signal":      formula_signal,
    }


# ─────────────────────────────────────────
# 4. 알림 판정
# ─────────────────────────────────────────
def evaluate_alerts(derived: dict, wti_spot: float) -> list:
    alerts = []
    ovs = derived["overshoot_mar23"]

    if derived["direction_mismatch"]:
        alerts.append({
            "level": "WARN",
            "msg": (
                f"[방향 불일치] GS Mar23({derived['mar23_signal']}) vs "
                f"GS Formula({derived['formula_signal']}) → 판단 유보"
            ),
        })

    if ovs > OVER_THRESHOLD:
        alerts.append({
            "level": "CRITICAL",
            "msg": (
                f"[과도평가 경보] WTI ${wti_spot:.2f} > GS Mar23 ${derived['gs_fair_mar23']:.2f} "
                f"(+${ovs:.2f}). 매도 압력 점검 필요."
            ),
        })
    elif ovs < UNDER_THRESHOLD:
        alerts.append({
            "level": "CRITICAL",
            "msg": (
                f"[극단 저평가 경보] WTI ${wti_spot:.2f} << GS Mar23 ${derived['gs_fair_mar23']:.2f} "
                f"({ovs:.2f}). 수급 충격 가능성 점검."
            ),
        })
    else:
        alerts.append({
            "level": "INFO",
            "msg": (
                f"[정상 범위] 현재가 ${wti_spot:.2f}, "
                f"GS Mar23 기준 편차 {ovs:+.2f} (임계치 미달)"
            ),
        })

    return alerts


# ─────────────────────────────────────────
# 5. 메인 실행 루틴
# ─────────────────────────────────────────
def run_once():
    now_kst = datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    log.info(f"===== 호르무즈 모니터 실행 [{now_kst}] =====")

    # 데이터 수집
    portwatch = fetch_portwatch()
    vessels   = portwatch["vessels_7dma"]
    wti       = fetch_wti()

    # 연산
    d = compute_derived(vessels, wti)

    # 출력
    print("\n" + "=" * 60)
    print(f"  호르무즈 모니터  |  {now_kst}")
    print("=" * 60)
    print(f"  선박수 (7dMA)   : {vessels:.1f}척  [{portwatch['source']}]  {portwatch['date']}")
    print(f"  봉쇄율          : {d['blockade_rate']*100:.1f}%")
    print(f"  순손실          : {d['net_loss_mbd']:.2f} mb/d")
    print(f"  GS 프리미엄     : +${d['gs_premium']:.2f}")
    print(f"  WTI 현재가      : ${wti:.2f}")
    print(f"  GS Mar23 view   : ${d['gs_fair_mar23']:.2f}  ← 결정 기준")
    print(f"  GS 공식 fair    : ${d['gs_fair_formula']:.2f}  ← 참고용")
    print(f"  편차 (Mar23기준): {d['overshoot_mar23']:+.2f}")
    print("-" * 60)

    alerts = evaluate_alerts(d, wti)
    for a in alerts:
        tag = "🔴" if a["level"] == "CRITICAL" else ("⚠️" if a["level"] == "WARN" else "✅")
        print(f"  {tag}  {a['msg']}")

    print("=" * 60 + "\n")
    return d, alerts


# ─────────────────────────────────────────
# 6. 스케줄러 (데몬 모드)
# ─────────────────────────────────────────
def build_schedule():
    """KST 기준 스케줄 계산: 08:00, 10:00, 이후 2시간 간격"""
    kst_times = ["08:00", "10:00"]
    # 10:00 이후 2시간 간격 추가
    h = 12
    while h < 24:
        kst_times.append(f"{h:02d}:00")
        h += 2
    return kst_times


def daemon_loop():
    try:
        import schedule as sched
    except ImportError:
        log.error("pip install schedule 필요")
        sys.exit(1)

    times = build_schedule()
    log.info(f"스케줄 등록: {times}")

    for t in times:
        sched.every().day.at(t).do(run_once)

    log.info("데몬 시작. Ctrl+C로 종료.")
    # 즉시 1회 실행
    run_once()

    while True:
        sched.run_pending()
        time.sleep(30)


# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hormuz Monitor")
    parser.add_argument("--daemon", action="store_true", help="스케줄 데몬 모드")
    args = parser.parse_args()

    if args.daemon:
        daemon_loop()
    else:
        run_once()
