"""
app.py - Oil Analysis Dashboard (Light Mode, Full Crawling)
크롤링 소스:
  [1] yfinance      : WTI(CL=F), Brent(BZ=F), DXY(DX-Y.NYB), OVX(^OVX), WTI선물커브
  [2] EIA DEMO_KEY  : 상업재고(WCESTUS1), 전체재고(SAE→SPR 유도), 미국 생산량
  [3] IMF PortWatch : 호르무즈 선박수/속도/톤수 (응답 실패 시 시장 프록시로 대체)
모델:
  - MS-style Kalman Filter (잠재 펀더멘털)
  - GS Two-Stage Bridge (수급 → 지정학 프리미엄)
  - RF Supply Shock Index
  - Monte Carlo 30/60/90일 가격 예측
"""

import io, base64, datetime, requests, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template_string, jsonify
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# ── CSV 저장 경로 ──────────────────────────────────────────
CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(CSV_DIR, exist_ok=True)


def _save_csv(filename: str, df: pd.DataFrame) -> str:
    """data/ 폴더에 CSV 저장, 저장 경로 반환."""
    path = os.path.join(CSV_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

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
SCENARIO_PROBS = {
    "S1_Agreement": 0.10, "S2_Strike":     0.40,
    "S3_Prolonged": 0.30, "S4_Withdrawal": 0.20,
}
TARGETS = {
    "Goldman Sachs":  [ 77, 110, 125,  96],
    "Morgan Stanley": [ 72, 113, 120,  95],
    "Citigroup":      [ 75, 120, 140, 105],
    "JPMorgan":       [ 78, 115, 122, 105],
    "Bloomberg":      [ 80, 122, 160, 107],
}
SIGMA = {"S1_Agreement": 5.0, "S2_Strike": 12.0,
          "S3_Prolonged": 18.0, "S4_Withdrawal": 10.0}

FORECAST_HORIZONS = [7, 14, 30, 60, 90]

# EIA API 키: .env 파일 또는 환경변수 EIA_API_KEY 우선, 없으면 DEMO_KEY
# 무료 등록: https://www.eia.gov/opendata/register.php  |  DEMO_KEY: 일 500회/분 5회 한도
EIA_KEY  = os.getenv("EIA_API_KEY", "DEMO_KEY")
EIA_BASE = "https://api.eia.gov/v2"

# ══════════════════════════════════════════════════════════
# 1. 크롤링 계층
# ══════════════════════════════════════════════════════════
_cache: dict = {"data": None, "ts": None}
CACHE_TTL = 300


def _eia_fetch(path: str, facets: dict, length: int = 60,
               retries: int = 3, backoff: float = 4.0) -> list:
    """EIA API v2 호출 → data 배열 반환. 429 시 최대 retries회 재시도."""
    import time
    params: dict = {
        "api_key": EIA_KEY,
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": length,
    }
    for k, v in facets.items():
        params[k] = v
    for attempt in range(retries):
        try:
            r = requests.get(f"{EIA_BASE}/{path}", params=params, timeout=12)
            if r.status_code == 429:
                wait = backoff * (attempt + 1)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("response", {}).get("data", [])
        except Exception:
            if attempt < retries - 1:
                import time as _t; _t.sleep(backoff)
    return []


def _eia_series(rows: list) -> np.ndarray:
    """EIA 레코드 → 시간순 float 배열 (오래된 것부터)."""
    vals = []
    for row in reversed(rows):
        try:
            vals.append(float(row["value"]))
        except Exception:
            pass
    return np.array(vals) if vals else np.array([])


def _fred_series(series_id: str, limit: int = 365) -> tuple:
    """
    FRED 무료 공개 CSV 엔드포인트 → (값 배열, 날짜 리스트).
    API 키 불필요.  주요 시리즈:
      DCOILWTICO  = WTI spot ($/bbl, 일별)
      DCOILBRENTEU = Brent spot ($/bbl, 일별)
    """
    from io import StringIO
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().tail(limit)
        return df["value"].values.astype(float), df["date"].tolist()
    except Exception:
        return None, []


def _yf_macro() -> dict:
    """
    매크로 지표 yfinance 배치 수집 (1회 호출로 10개 티커 동시 다운로드).
      ^TNX  : 미국 10년물 국채 수익률
      ^GSPC : S&P 500
      URTH  : MSCI World ETF
      GC=F  : 금 선물 (안전자산/지정학)
      HG=F  : 구리 선물 (글로벌 제조업)
      ^VIX  : VIX 공포지수
      TIP   : TIPS ETF → CPI/인플레 프록시
      XLI   : 산업재 ETF → ISM PMI 프록시
      FXI   : iShares China ETF → 중국 PMI 프록시
      MCHI  : MSCI China ETF (중국 경기 보완)
    """
    name_map = {
        "^TNX":  "TNX",   "^GSPC": "SP500",
        "URTH":  "MSCI_World",
        "GC=F":  "Gold",  "HG=F":  "Copper",
        "^VIX":  "VIX",   "TIP":   "TIP",
        "XLI":   "XLI",   "FXI":   "FXI",
        "MCHI":  "MCHI",
    }
    result: dict = {}
    try:
        # 배치 다운로드 - 1회 HTTP 요청으로 처리
        raw = yf.download(
            list(name_map.keys()),
            period="1y", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
        for tick, name in name_map.items():
            try:
                col = raw[tick]["Close"] if tick in raw.columns.get_level_values(0) else None
                if col is None:
                    continue
                series = col.dropna().values.astype(float).flatten()
                if len(series) < 5:
                    continue
                result[name] = {
                    "series":  series,
                    "latest":  float(series[-1]),
                    "chg_pct": float((series[-1] / series[-2] - 1) * 100),
                    "ticker":  tick,
                }
            except Exception:
                pass
    except Exception:
        pass
    # 개별 fallback
    for tick, name in name_map.items():
        if name not in result:
            try:
                df = yf.download(tick, period="1y", interval="1d",
                                 progress=False, auto_adjust=True)
                arr = df["Close"].dropna().values.astype(float).flatten()
                if len(arr) >= 5:
                    result[name] = {
                        "series":  arr,
                        "latest":  float(arr[-1]),
                        "chg_pct": float((arr[-1] / arr[-2] - 1) * 100),
                        "ticker":  tick,
                    }
            except Exception:
                pass
    return result


def _eia_wpsr() -> dict:
    """
    EIA Weekly Petroleum Status Report 추가 시리즈.
      - 가솔린 재고  (EPM0F + SAE + NUS)
      - 중간유 재고  (EPD0  + SAE + NUS)
      - 원유 수입량  (EPC0  + SAI + NUS)
    반환 dict: {"gasoline": arr, "distillate": arr, "crude_imports": arr}
    """
    import time as _t
    series_config = [
        ("gasoline",      "petroleum/stoc/wstk/data/",
         {"facets[product][]": "EPM0F", "facets[duoarea][]": "NUS",
          "facets[process][]": "SAE"}),
        ("distillate",    "petroleum/stoc/wstk/data/",
         {"facets[product][]": "EPD0",  "facets[duoarea][]": "NUS",
          "facets[process][]": "SAE"}),
        ("crude_imports", "petroleum/move/wkly/data/",
         {"facets[product][]": "EPC0",  "facets[duoarea][]": "NUS",
          "facets[process][]": "SAI"}),
    ]
    result: dict = {}
    for name, path, facets in series_config:
        _t.sleep(3)
        rows = _eia_fetch(path, facets, length=60)
        arr  = _eia_series(rows)
        result[name]         = arr if len(arr) > 4 else None
        result[f"{name}_rows"] = rows
    return result


_ARCGIS_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services"
    "/Daily_Chokepoints_Data/FeatureServer/0/query"
)


def _arcgis_hormuz(n_records: int = 365) -> dict:
    """
    ArcGIS FeatureServer에서 호르무즈(chokepoint6) 일별 선박 데이터 수집.
    반환 필드: date, n_tanker, n_total, capacity_tanker, capacity,
               n_dry_bulk, n_container
    """
    try:
        r = requests.get(
            _ARCGIS_URL,
            params={
                "where":             "portid='chokepoint6'",
                "outFields":         "date,n_tanker,n_total,capacity_tanker,"
                                     "capacity,n_dry_bulk,n_container,n_roro",
                "orderByFields":     "date DESC",
                "resultRecordCount": n_records,
                "f":                 "json",
            },
            timeout=15,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            return {}

        rows = [f["attributes"] for f in features]
        # 시간순 정렬 (오래된 것부터)
        rows = sorted(rows, key=lambda x: x.get("date", ""))

        def col(key, default=0.0):
            return np.array([float(row.get(key) or default) for row in rows])

        return {
            "dates":            [row.get("date", "") for row in rows],
            "n_tanker":         col("n_tanker"),
            "n_total":          col("n_total"),
            "capacity_tanker":  col("capacity_tanker"),
            "capacity":         col("capacity"),
            "n_dry_bulk":       col("n_dry_bulk"),
            "n_container":      col("n_container"),
            "n_roro":           col("n_roro"),
            "tanker_ratio":     col("n_tanker") / np.maximum(col("n_total"), 1),
            "source":           _ARCGIS_URL,
        }
    except Exception as e:
        return {"error": str(e)}


def _straits_live_scrape() -> dict:
    """
    straits.live 페이지 스크래핑 → 호르무즈 현황 스냅샷.
    반환: transits_today, vessels_in_transit, tankers_dark,
          war_risk_usd_m, war_risk_multiplier, hormuz_index,
          escalation_forecast, high_risk, moderate_risk, low_risk
    """
    import re
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {"error": "beautifulsoup4 미설치"}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        r = requests.get("https://straits.live/", timeout=12, headers=headers)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}", "source": "straits.live"}

        soup = BeautifulSoup(r.text, "html.parser")
        txt  = soup.get_text(separator=" ", strip=True)

        def _num(pattern, cast=float, txt=txt):
            m = re.search(pattern, txt, re.I | re.S)
            try:
                raw = m.group(1).replace(",", "") if m else None
                return cast(raw) if raw is not None else None
            except Exception:
                return None

        return {
            # 일별 상업적 통과 횟수 (오늘)
            "transits_today":      _num(r"(?:Commercial\s+transit[s]?|transits?)\D{0,15}(\d+)\s*/?\s*day|(\d+)\s*transit", int),
            # 현재 통과 중인 선박 수 (여러 표현 처리)
            "vessels_in_transit":  _num(r"(\d+)\s+vessels?\s*(?:currently\s+)?in\s+transit|in\s+transit[:\s]+(\d+)", int),
            # 유형별 탱커 수
            "tankers_in_transit":  _num(r"(\d+)\s+tanker|tanker[s]?\s*[:\-]\s*(\d+)", int),
            # AIS 소등 탱커 (숫자 먼저, 라벨 뒤)
            "tankers_dark":        _num(r"(\d+)\s*(?:AIS.{0,15}dark|dark.{0,15}AIS|tanker.{0,15}dark)", int),
            # AIS 소등 7일 기준선
            "dark_baseline_7d":    _num(r"baseline\D{0,10}(\d+\.?\d*)|(\d+\.?\d*)\s*7.?d\s*baseline", float),
            # VLCC 전쟁보험 (백만 달러)
            "war_risk_usd_m":      _num(r"\$\s*(\d+\.?\d*)\s*[Mm]", float),
            # 전쟁보험 배수
            "war_risk_multiplier": _num(r"(\d+\.?\d*)\s*[×xX]\s*(?:est|pre|crisis)", float),
            # 호르무즈 지수: "Crisis Pressure: 94" 또는 "Pressure:\s+94" 형식
            "hormuz_index":        _num(r"Crisis\s+Pressure\D{0,5}(\d{2,3})|Hormuz\s+Index\D{0,30}Pressure\D{0,5}(\d{2,3})", float),
            # 에스컬레이션 예측
            "escalation_forecast": _num(r"Escalation\s+Forecast\D{0,5}(\d{2,3})", float),
            # 위험 등급별 선박 (숫자 우선)
            "high_risk_vessels":   _num(r"(\d+)\s*high.{0,10}risk|high.{0,10}risk\D{0,5}(\d+)", int),
            "moderate_risk":       _num(r"(\d+)\s*moderate|moderate\D{0,5}(\d+)", int),
            "low_risk":            _num(r"(\d+)\s*low.{0,10}risk|low.{0,10}risk\D{0,5}(\d+)", int),
            "scraped_at":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source":              "https://straits.live/",
        }
    except Exception as e:
        return {"error": str(e), "source": "straits.live"}


def _futures_curve_yf() -> dict:
    """
    WTI 선물 커브: 근월물(M1) + M2/M3/M6 개별 계약 가격.
    반환: {"M1": float, "M2": float, "M3": float, "M6": float,
            "slope": 연율화%, "contango": bool}
    slope = (M6 - M1) / M1 * 2 * 100  (연율화)
    fallback: M1 역사적 3개월 변화율로 slope 산출
    """
    now = datetime.datetime.now()
    mc  = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",
           7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}
    curve: dict = {}

    # 근월물
    try:
        p = yf.Ticker("CL=F").fast_info.get("last_price")
        if p: curve["M1"] = round(float(p), 2)
    except Exception:
        pass

    # M2 / M3 / M6
    for n in [2, 3, 6]:
        dt   = now + datetime.timedelta(days=30 * (n - 1))
        tick = f"CL{mc[dt.month]}{str(dt.year)[2:]}.NYM"
        try:
            p = yf.Ticker(tick).fast_info.get("last_price")
            if p: curve[f"M{n}"] = round(float(p), 2)
        except Exception:
            pass

    # slope 계산
    m1 = curve.get("M1")
    if m1 and m1 > 0:
        if curve.get("M6"):
            curve["slope"] = round((curve["M6"] - m1) / m1 * 2 * 100, 3)
        elif curve.get("M3"):
            curve["slope"] = round((curve["M3"] - m1) / m1 * 4 * 100, 3)

    # fallback: 역사적 가격 변화
    if "slope" not in curve:
        try:
            df = yf.download("CL=F", period="6mo", interval="1d",
                             progress=False, auto_adjust=True)
            cl = df["Close"].dropna().values.flatten()
            if len(cl) > 60:
                curve["slope"] = round(float(cl[-1] - cl[-60]) / cl[-60] * (12/3) * 100, 3)
                if "M1" not in curve:
                    curve["M1"] = round(float(cl[-1]), 2)
        except Exception:
            pass

    curve["contango"] = curve.get("slope", 0) > 0
    return curve


def fetch_all_data(force: bool = False) -> dict:
    """통합 크롤링: yfinance + EIA + PortWatch."""
    now = datetime.datetime.now()
    if (not force and _cache["data"] is not None
            and (now - _cache["ts"]).seconds < CACHE_TTL):
        return _cache["data"]

    status: dict = {"crawled_at": now.strftime("%Y-%m-%d %H:%M:%S"), "sources": {}}

    # ── [1] yfinance 배치 다운로드 ───────────────────────────
    yf_name_map = {
        "CL=F":     "WTI",   "BZ=F":      "Brent",
        "DX-Y.NYB": "DXY",   "^OVX":      "OVX",
        "USO":      "USO",   "BNO":       "BNO",
    }
    mkt: dict = {}
    try:
        raw_batch = yf.download(
            list(yf_name_map.keys()),
            period="1y", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
    except Exception:
        raw_batch = pd.DataFrame()

    for ticker, name in yf_name_map.items():
        try:
            try:
                col = raw_batch[ticker]["Close"] if (
                    not raw_batch.empty and
                    ticker in raw_batch.columns.get_level_values(0)
                ) else None
            except Exception:
                col = None

            if col is None or col.dropna().empty:
                df_s = yf.download(ticker, period="1y", interval="1d",
                                   progress=False, auto_adjust=True)
                col = df_s["Close"].dropna() if not df_s.empty else pd.Series(dtype=float)

            closes = col.dropna() if hasattr(col, "dropna") else pd.Series(dtype=float)
            if closes.empty:
                raise ValueError("empty")

            arr   = closes.values.astype(float).flatten()
            idx   = closes.index
            dates = (idx.strftime("%Y-%m-%d")
                     if hasattr(idx, "strftime") else [str(d)[:10] for d in idx])

            mkt[name] = {
                "series":  arr,
                "latest":  float(arr[-1]),
                "chg":     float(arr[-1] - arr[-2]) if len(arr) > 1 else 0.0,
                "chg_pct": float((arr[-1] / arr[-2] - 1) * 100) if len(arr) > 1 else 0.0,
            }
            status["sources"][name] = "LIVE (Yahoo Finance)"
            _save_csv(
                f"yahoo_finance_{name.lower()}.csv",
                pd.DataFrame({"date": dates, "close": arr,
                              "ticker": ticker, "source": "Yahoo Finance"}),
            )
        except Exception as e:
            mkt[name] = None
            status["sources"][name] = f"FAIL ({e})"

    # ── WTI 선물 커브 (M1~M6) ─────────────────────────────
    curve = _futures_curve_yf()
    slope = curve.get("slope")
    if slope is not None:
        mkt["futures_slope"] = slope
        mkt["futures_curve"] = curve
        status["sources"]["futures_slope"] = (
            f"LIVE (Yahoo Finance WTI curve: "
            f"M1={curve.get('M1','?')} M3={curve.get('M3','?')} M6={curve.get('M6','?')})"
        )
        _save_csv(
            "yahoo_finance_futures_curve.csv",
            pd.DataFrame([{
                "crawled_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "M1": curve.get("M1"), "M2": curve.get("M2"),
                "M3": curve.get("M3"), "M6": curve.get("M6"),
                "slope_annualized_pct": slope,
                "contango": curve.get("contango"),
                "source": "Yahoo Finance (WTI CL= futures curve)",
            }]),
        )
    else:
        mkt["futures_slope"] = None
        mkt["futures_curve"] = {}
        status["sources"]["futures_slope"] = "SIM (yfinance 선물 실패)"

    # ── [2b] FRED WTI & Brent (보조 검증, 무료, API 키 불필요) ─
    for name, fid in [("FRED_WTI", "DCOILWTICO"), ("FRED_Brent", "DCOILBRENTEU")]:
        arr_f, dates_f = _fred_series(fid)
        if arr_f is not None and len(arr_f) > 10:
            mkt[name] = {"series": arr_f, "latest": float(arr_f[-1]),
                         "chg": float(arr_f[-1] - arr_f[-2]) if len(arr_f) > 1 else 0.0}
            status["sources"][name] = f"LIVE (FRED {fid}, n={len(arr_f)}일)"
            short = name.replace("FRED_", "").lower()
            _save_csv(
                f"fred_{short}.csv",
                pd.DataFrame({"date": dates_f, "price_usd": arr_f,
                              "series_id": fid,
                              "source": "FRED St. Louis Fed (public CSV, no key)"}),
            )
        else:
            mkt[name] = None
            status["sources"][name] = f"FAIL (FRED {fid})"

    # ── Brent-WTI 스프레드 (직접 계산) ────────────────────
    wti_s   = mkt.get("WTI",   {}).get("series") if mkt.get("WTI")   else None
    brent_s = mkt.get("Brent", {}).get("series") if mkt.get("Brent") else None
    if wti_s is not None and brent_s is not None:
        n_sp = min(len(wti_s), len(brent_s))
        sp   = brent_s[-n_sp:] - wti_s[-n_sp:]
        mkt["brent_wti_spread"] = {"series": sp, "latest": float(sp[-1])}
        status["sources"]["brent_wti_spread"] = (
            f"LIVE (Brent-WTI, 현재 ${sp[-1]:+.2f}, n={n_sp}일)"
        )
        _save_csv(
            "brent_wti_spread.csv",
            pd.DataFrame({"index": range(n_sp), "spread_usd": sp,
                          "source": "Calculated (Brent - WTI from yfinance)"}),
        )
    else:
        mkt["brent_wti_spread"] = None
        status["sources"]["brent_wti_spread"] = "SIM (Brent/WTI 없음)"

    # ── [2] EIA API (DEMO_KEY) ─────────────────────────────
    import time as _time
    _time.sleep(2)   # DEMO_KEY rate limit 완화
    # 상업재고 (excl SPR): process=SAX → WCESTUS1
    rows_excl = _eia_fetch(
        "petroleum/stoc/wstk/data/",
        {"facets[product][]": "EPC0", "facets[duoarea][]": "NUS",
         "facets[process][]": "SAX"},
        length=60,
    )
    arr_excl = _eia_series(rows_excl)
    if len(arr_excl) > 1:
        mkt["eia_stock"] = arr_excl
        status["sources"]["eia_stock"] = f"LIVE (EIA DEMO_KEY, n={len(arr_excl)}주)"
        # ── CSV 저장: eia_commercial_inventory ────────────
        if rows_excl:
            _save_csv(
                "eia_commercial_inventory.csv",
                pd.DataFrame([{
                    "period":             r.get("period"),
                    "value_mbbl":         r.get("value"),
                    "series":             r.get("series"),
                    "series_description": r.get("series-description"),
                    "units":              r.get("units"),
                    "duoarea":            r.get("duoarea"),
                    "source":             "EIA Open Data API v2 (DEMO_KEY)",
                } for r in rows_excl]),
            )
    else:
        mkt["eia_stock"] = None
        status["sources"]["eia_stock"] = "SIM (EIA 응답 없음)"

    # 전체재고 (incl SPR): process=SAE → WCRSTUS1
    _time.sleep(3)
    rows_total = _eia_fetch(
        "petroleum/stoc/wstk/data/",
        {"facets[product][]": "EPC0", "facets[duoarea][]": "NUS",
         "facets[process][]": "SAE"},
        length=60,
    )
    arr_total = _eia_series(rows_total)
    # SPR = 전체 - 상업
    if len(arr_total) > 1 and len(arr_excl) > 1:
        n = min(len(arr_total), len(arr_excl))
        spr = arr_total[-n:] - arr_excl[-n:]
        mkt["spr_level"] = spr
        status["sources"]["spr_level"] = f"LIVE (EIA 유도: total-excl, n={n}주)"
        # ── CSV 저장: eia_spr ──────────────────────────────
        if rows_total:
            rows_total_rev = list(reversed(rows_total))
            rows_excl_rev  = list(reversed(rows_excl))
            spr_rows = []
            for i in range(n):
                spr_rows.append({
                    "period":           rows_total_rev[i].get("period"),
                    "total_stocks_mbbl": float(rows_total_rev[i].get("value", 0) or 0),
                    "excl_spr_mbbl":    float(rows_excl_rev[i].get("value", 0) or 0),
                    "spr_derived_mbbl": float(rows_total_rev[i].get("value", 0) or 0)
                                       - float(rows_excl_rev[i].get("value", 0) or 0),
                    "source": "EIA Open Data API v2 (derived: total - excl_SPR)",
                })
            _save_csv("eia_spr.csv", pd.DataFrame(spr_rows))
    else:
        mkt["spr_level"] = None
        status["sources"]["spr_level"] = "SIM (EIA 유도 실패)"

    # 미국 생산량 (주간) → 공급 갭 프록시
    _time.sleep(3)
    rows_prod = _eia_fetch(
        "petroleum/sum/snd/data/",
        {"facets[duoarea][]": "NUS", "facets[product][]": "EPC0",
         "facets[process][]": "FPD"},
        length=60,
    )
    arr_prod = _eia_series(rows_prod)
    if len(arr_prod) > 4:
        d_prod = np.diff(arr_prod, prepend=arr_prod[0])
        mkt["supply_gap"] = d_prod / 100.0
        status["sources"]["supply_gap"] = f"LIVE (EIA 생산량 차분, n={len(arr_prod)}주)"
        # ── CSV 저장: eia_us_production ───────────────────
        _save_csv(
            "eia_us_production.csv",
            pd.DataFrame([{
                "period":                  r.get("period"),
                "value":                   r.get("value"),
                "series":                  r.get("series"),
                "series_description":      r.get("series-description"),
                "units":                   r.get("units"),
                "source":                  "EIA Open Data API v2 (DEMO_KEY)",
            } for r in rows_prod]),
        )
    else:
        mkt["supply_gap"] = None
        status["sources"]["supply_gap"] = "SIM (EIA 생산 응답 없음)"

    # ── [2c] 매크로 지표 (yfinance) ────────────────────────
    macro = _yf_macro()
    mkt["macro"] = macro
    macro_csv_rows = []
    for name, info in macro.items():
        status["sources"][f"macro_{name}"] = (
            f"LIVE (yfinance {info['ticker']}, n={len(info['series'])}일)"
        )
        arr = info["series"]
        macro_csv_rows.append({
            "name":       name,
            "ticker":     info["ticker"],
            "latest":     info["latest"],
            "chg_pct":    info["chg_pct"],
            "n_days":     len(arr),
            "crawled_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        })
    if macro_csv_rows:
        _save_csv("macro_snapshot.csv", pd.DataFrame(macro_csv_rows))

    # ── [2d] EIA WPSR 추가 수급 시리즈 ─────────────────────
    wpsr = _eia_wpsr()
    mkt["wpsr"] = wpsr
    for name in ("gasoline", "distillate", "crude_imports"):
        arr = wpsr.get(name)
        if arr is not None:
            status["sources"][f"wpsr_{name}"] = (
                f"LIVE (EIA WPSR {name}, n={len(arr)}주)"
            )
            rows_k = wpsr.get(f"{name}_rows", [])
            if rows_k:
                _save_csv(
                    f"eia_wpsr_{name}.csv",
                    pd.DataFrame([{
                        "period": r.get("period"), "value": r.get("value"),
                        "units": r.get("units"), "series": r.get("series"),
                        "source": "EIA WPSR v2",
                    } for r in rows_k]),
                )
        else:
            status["sources"][f"wpsr_{name}"] = f"SIM (EIA WPSR {name} 없음)"

    # ── [3] ArcGIS 호르무즈 일별 선박 데이터 ──────────────
    arcgis = _arcgis_hormuz(n_records=365)
    if arcgis and not arcgis.get("error"):
        mkt["arcgis_hormuz"] = arcgis
        n_ag = len(arcgis["n_tanker"])
        status["sources"]["vessel_data"] = (
            f"LIVE (ArcGIS chokepoint6 Daily, n={n_ag})"
        )
        _save_csv(
            "arcgis_hormuz.csv",
            pd.DataFrame({
                "date":            arcgis["dates"],
                "n_tanker":        arcgis["n_tanker"],
                "n_total":         arcgis["n_total"],
                "n_dry_bulk":      arcgis["n_dry_bulk"],
                "n_container":     arcgis["n_container"],
                "n_roro":          arcgis["n_roro"],
                "capacity_tanker": arcgis["capacity_tanker"],
                "capacity":        arcgis["capacity"],
                "tanker_ratio":    arcgis["tanker_ratio"],
                "source":          "ArcGIS FeatureServer (IMF PortWatch chokepoint6)",
            }),
        )
    else:
        mkt["arcgis_hormuz"] = None
        status["sources"]["vessel_data"] = (
            f"PROXY (ArcGIS 실패: {arcgis.get('error','no data')} → 시장 프록시)"
        )

    # ── [4] straits.live 현황 스크래핑 ─────────────────────
    straits = _straits_live_scrape()
    if straits and not straits.get("error"):
        mkt["straits_live"] = straits
        status["sources"]["straits_live"] = (
            f"LIVE (straits.live: {straits.get('scraped_at','')})"
        )
        _save_csv("straits_live_snapshot.csv", pd.DataFrame([straits]))
    else:
        mkt["straits_live"] = None
        status["sources"]["straits_live"] = (
            f"FAIL (straits.live: {straits.get('error','no data')})"
        )

    _cache["data"] = {**mkt, **status}
    _cache["ts"]   = now
    return _cache["data"]


# ══════════════════════════════════════════════════════════
# 2. 보조 함수
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
    TimeSeriesSplit + RandomizedSearchCV로 RF 하이퍼파라미터 튜닝.
    각 FORECAST_HORIZONS(7/14/30/60/90d)에 대해 최적 모델 학습 후 예측.
    """
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.metrics import mean_absolute_error, r2_score

    if df.empty:
        return {}

    feat_cols = [c for c in df.columns
                 if not c.startswith("target_") and c != "oil_price"]

    param_grid = {
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

        X = df_c[feat_cols].values
        y = df_c[tcol].values

        split   = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=20,          # 웹 앱 응답 속도: 20회 (full study: 50)
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        pred = best_model.predict(X_test) if len(X_test) > 0 else np.array([0.0])
        mae  = float(mean_absolute_error(y_test, pred)) if len(y_test) > 0 else 0.0
        r2   = float(r2_score(y_test, pred))            if len(y_test) > 1 else 0.0

        pred_latest = float(best_model.predict(X[-1:])[0])

        results[h] = {
            "best_params":  search.best_params_,
            "mae":          round(mae, 5),
            "r2":           round(r2, 4),
            "pred_return":  round(pred_latest, 5),
            "pred_price":   round(float(df["oil_price"].iloc[-1]) * (1 + pred_latest), 2),
            "importances":  {f: round(float(v), 4)
                             for f, v in zip(feat_cols, best_model.feature_importances_)},
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

    # ── [0,0] 예측 수익률 바 차트 ────────────────────────
    rets   = [ml[h]["pred_return"] * 100 for h in horizons]
    colors = [L_GREEN if r >= 0 else L_RED for r in rets]
    bars   = ax1.bar(labels, rets, color=colors, alpha=0.8, width=0.5, edgecolor=L_BORDER)
    ax1.axhline(0, color=L_TEXT, lw=0.8, ls="--")
    for bar, r in zip(bars, rets):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.05 if r >= 0 else -0.15),
                 f"{r:+.2f}%", ha="center", va="bottom", fontsize=8,
                 fontweight="bold", color=L_GREEN if r >= 0 else L_RED)
    # 예측 가격 보조 텍스트
    for i, h in enumerate(horizons):
        ax1.text(i, min(rets) - 0.3,
                 f"${ml[h]['pred_price']:.1f}",
                 ha="center", va="top", fontsize=7, color=L_SUB)
    ax1.set_title("RF 예측 수익률 (지평별)", color=L_TEXT,
                  fontsize=10, fontweight="bold")
    ax1.set_ylabel("Predicted Return (%)", color=L_TEXT, fontsize=8)

    # ── [0,1] MAE / R² 성능 ──────────────────────────────
    maes = [ml[h]["mae"] * 100 for h in horizons]
    r2s  = [ml[h]["r2"]       for h in horizons]
    x    = np.arange(len(horizons))
    w    = 0.35
    b1   = ax2.bar(x - w/2, maes, w, label="MAE (%)",
                   color=L_ORANGE, alpha=0.8, edgecolor=L_BORDER)
    ax2r = ax2.twinx()
    ax2r.plot(x, r2s, "o--", color=L_BLUE, lw=2, ms=6, label="R²")
    ax2r.set_ylabel("R²", color=L_BLUE, fontsize=8)
    ax2r.tick_params(colors=L_BLUE, labelsize=8)
    ax2r.set_ylim(-0.2, 1.0)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_title("모델 성능 (MAE / R²)", color=L_TEXT,
                  fontsize=10, fontweight="bold")
    ax2.set_ylabel("MAE (%)", color=L_ORANGE, fontsize=8)
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
        f"ML 유가 예측 (RandomForest + TimeSeriesSplit CV + RandomizedSearchCV)  "
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
  <th>&#xC608;&#xCE21; &#xC218;&#xC775;&#xB960;</th>
  <th>&#xC608;&#xCE21; &#xAC00;&#xACA9;</th>
  <th>MAE</th>
  <th>R&#xB178;</th>
  <th>&#xBCA0;&#xC2A4;&#xD2B8; n_estimators</th>
  <th>&#xBCA0;&#xC2A4;&#xD2B8; max_depth</th>
</tr>
{% for h in [7,14,30,60,90] %}{% if h in ml %}
{% set info = ml[h] %}
<tr>
  <td><strong>+{{ h }}&#xC77C;</strong></td>
  <td class="{{ 'cg' if info.pred_return>=0 else 'cr' }}">
    {{ '%+.2f'|format(info.pred_return*100) }}%
  </td>
  <td class="{{ 'cg' if info.pred_return>=0 else 'cr' }}">${{ "%.1f"|format(info.pred_price) }}</td>
  <td class="cy">{{ "%.4f"|format(info.mae*100) }}%</td>
  <td class="{{ 'cg' if info.r2>0.1 else 'cr' }}">{{ "%.3f"|format(info.r2) }}</td>
  <td>{{ info.best_params.get('n_estimators','?') }}</td>
  <td>{{ info.best_params.get('max_depth','?') }}</td>
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
