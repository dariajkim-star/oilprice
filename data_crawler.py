"""
data_crawler.py
===============
API 과사용 방지를 위해 app.py에서 분리한 독립 크롤링 모듈.

단독 실행:
  python data_crawler.py           # 전체 크롤링 후 data/ 폴더에 CSV 저장
  python data_crawler.py --force   # 캐시 무시 강제 재크롤링

app.py 연동:
  from data_crawler import fetch_all_data
"""

import io, datetime, requests, warnings, os, argparse
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

warnings.filterwarnings("ignore")

# ── 경로 / API 설정 ──────────────────────────────────────────
CSV_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(CSV_DIR, exist_ok=True)

EIA_KEY  = os.getenv("EIA_API_KEY", "DEMO_KEY")
EIA_BASE = "https://api.eia.gov/v2"

# 캐시 TTL: 3600초(1시간) — 5분보다 길게 설정해 API 과호출 방지
_cache: dict = {"data": None, "ts": None}
CACHE_TTL    = 3600


# ── 내부 유틸 ────────────────────────────────────────────────
def _save_csv(filename: str, df: pd.DataFrame) -> str:
    path = os.path.join(CSV_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _eia_fetch(path: str, facets: dict, length: int = 60,
               retries: int = 3, backoff: float = 4.0) -> list:
    import time
    params: dict = {
        "api_key":              EIA_KEY,
        "frequency":            "weekly",
        "data[0]":              "value",
        "sort[0][column]":      "period",
        "sort[0][direction]":   "desc",
        "offset": 0,
        "length": length,
    }
    for k, v in facets.items():
        params[k] = v
    for attempt in range(retries):
        try:
            r = requests.get(f"{EIA_BASE}/{path}", params=params, timeout=12)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json().get("response", {}).get("data", [])
        except Exception:
            if attempt < retries - 1:
                import time as _t; _t.sleep(backoff)
    return []


def _eia_series(rows: list) -> np.ndarray:
    vals = []
    for row in reversed(rows):
        try:
            vals.append(float(row["value"]))
        except Exception:
            pass
    return np.array(vals) if vals else np.array([])


def _fred_series(series_id: str, limit: int = 365) -> tuple:
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
        result[name]           = arr if len(arr) > 4 else None
        result[f"{name}_rows"] = rows
    return result


_ARCGIS_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services"
    "/Daily_Chokepoints_Data/FeatureServer/0/query"
)


def _arcgis_hormuz(n_records: int = 365) -> dict:
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
        rows = sorted([f["attributes"] for f in features],
                      key=lambda x: x.get("date", ""))

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

        from bs4 import BeautifulSoup
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
            "transits_today":      _num(r"(?:Commercial\s+transit[s]?|transits?)\D{0,15}(\d+)\s*/?\s*day|(\d+)\s*transit", int),
            "vessels_in_transit":  _num(r"(\d+)\s+vessels?\s*(?:currently\s+)?in\s+transit|in\s+transit[:\s]+(\d+)", int),
            "tankers_in_transit":  _num(r"(\d+)\s+tanker|tanker[s]?\s*[:\-]\s*(\d+)", int),
            "tankers_dark":        _num(r"(\d+)\s*(?:AIS.{0,15}dark|dark.{0,15}AIS|tanker.{0,15}dark)", int),
            "dark_baseline_7d":    _num(r"baseline\D{0,10}(\d+\.?\d*)|(\d+\.?\d*)\s*7.?d\s*baseline", float),
            "war_risk_usd_m":      _num(r"\$\s*(\d+\.?\d*)\s*[Mm]", float),
            "war_risk_multiplier": _num(r"(\d+\.?\d*)\s*[×xX]\s*(?:est|pre|crisis)", float),
            "hormuz_index":        _num(r"Crisis\s+Pressure\D{0,5}(\d{2,3})|Hormuz\s+Index\D{0,30}Pressure\D{0,5}(\d{2,3})", float),
            "escalation_forecast": _num(r"Escalation\s+Forecast\D{0,5}(\d{2,3})", float),
            "high_risk_vessels":   _num(r"(\d+)\s*high.{0,10}risk|high.{0,10}risk\D{0,5}(\d+)", int),
            "moderate_risk":       _num(r"(\d+)\s*moderate|moderate\D{0,5}(\d+)", int),
            "low_risk":            _num(r"(\d+)\s*low.{0,10}risk|low.{0,10}risk\D{0,5}(\d+)", int),
            "scraped_at":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source":              "https://straits.live/",
        }
    except Exception as e:
        return {"error": str(e), "source": "straits.live"}


def _futures_curve_yf() -> dict:
    now = datetime.datetime.now()
    mc  = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",
           7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}
    curve: dict = {}

    try:
        p = yf.Ticker("CL=F").fast_info.get("last_price")
        if p: curve["M1"] = round(float(p), 2)
    except Exception:
        pass

    for n in [2, 3, 6]:
        dt   = now + datetime.timedelta(days=30 * (n - 1))
        tick = f"CL{mc[dt.month]}{str(dt.year)[2:]}.NYM"
        try:
            p = yf.Ticker(tick).fast_info.get("last_price")
            if p: curve[f"M{n}"] = round(float(p), 2)
        except Exception:
            pass

    m1 = curve.get("M1")
    if m1 and m1 > 0:
        if curve.get("M6"):
            curve["slope"] = round((curve["M6"] - m1) / m1 * 2 * 100, 3)
        elif curve.get("M3"):
            curve["slope"] = round((curve["M3"] - m1) / m1 * 4 * 100, 3)

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


# ── 메인 크롤링 함수 ──────────────────────────────────────────
def fetch_all_data(force: bool = False) -> dict:
    """
    통합 크롤링: yfinance + EIA + ArcGIS + straits.live.
    CACHE_TTL(1시간) 내 재호출은 캐시 반환 → API 과사용 방지.
    """
    now = datetime.datetime.now()
    if (not force and _cache["data"] is not None
            and (now - _cache["ts"]).seconds < CACHE_TTL):
        return _cache["data"]

    status: dict = {"crawled_at": now.strftime("%Y-%m-%d %H:%M:%S"), "sources": {}}

    # [1] yfinance 배치 다운로드
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

    # WTI 선물 커브
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

    # FRED WTI & Brent
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

    # Brent-WTI 스프레드
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

    # [2] EIA API
    import time as _time
    _time.sleep(2)
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

    _time.sleep(3)
    rows_total = _eia_fetch(
        "petroleum/stoc/wstk/data/",
        {"facets[product][]": "EPC0", "facets[duoarea][]": "NUS",
         "facets[process][]": "SAE"},
        length=60,
    )
    arr_total = _eia_series(rows_total)
    if len(arr_total) > 1 and len(arr_excl) > 1:
        n = min(len(arr_total), len(arr_excl))
        spr = arr_total[-n:] - arr_excl[-n:]
        mkt["spr_level"] = spr
        status["sources"]["spr_level"] = f"LIVE (EIA 유도: total-excl, n={n}주)"
        if rows_total:
            rows_total_rev = list(reversed(rows_total))
            rows_excl_rev  = list(reversed(rows_excl))
            spr_rows = []
            for i in range(n):
                spr_rows.append({
                    "period":            rows_total_rev[i].get("period"),
                    "total_stocks_mbbl": float(rows_total_rev[i].get("value", 0) or 0),
                    "excl_spr_mbbl":     float(rows_excl_rev[i].get("value", 0) or 0),
                    "spr_derived_mbbl":  float(rows_total_rev[i].get("value", 0) or 0)
                                        - float(rows_excl_rev[i].get("value", 0) or 0),
                    "source": "EIA Open Data API v2 (derived: total - excl_SPR)",
                })
            _save_csv("eia_spr.csv", pd.DataFrame(spr_rows))
    else:
        mkt["spr_level"] = None
        status["sources"]["spr_level"] = "SIM (EIA 유도 실패)"

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
        _save_csv(
            "eia_us_production.csv",
            pd.DataFrame([{
                "period":             r.get("period"),
                "value":              r.get("value"),
                "series":             r.get("series"),
                "series_description": r.get("series-description"),
                "units":              r.get("units"),
                "source":             "EIA Open Data API v2 (DEMO_KEY)",
            } for r in rows_prod]),
        )
    else:
        mkt["supply_gap"] = None
        status["sources"]["supply_gap"] = "SIM (EIA 생산 응답 없음)"

    # 매크로 지표 (yfinance)
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

    # EIA WPSR 추가 시계열
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

    # ArcGIS 호르무즈 일별 선박 데이터
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

    # straits.live 스크랩
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


# ── 공통 출력 헬퍼 ────────────────────────────────────────────
def _print_result(data: dict) -> None:
    sources = data.get("sources", {})
    live  = sum(1 for v in sources.values() if "LIVE"  in str(v))
    fail  = sum(1 for v in sources.values() if "FAIL"  in str(v) or "SIM" in str(v))
    proxy = sum(1 for v in sources.values() if "PROXY" in str(v))
    print(f"  완료: {data.get('crawled_at')}  |  LIVE {live}  PROXY {proxy}  FAIL/SIM {fail}")
    for k, v in sources.items():
        tag = "[OK]" if "LIVE" in str(v) else ("[--]" if "PROXY" in str(v) else "[XX]")
        print(f"    {tag}  {k:<28} {str(v)[:55]}")
    print(f"  CSV → {CSV_DIR}/")


# ── 단독 실행 ─────────────────────────────────────────────────
if __name__ == "__main__":
    import time as _time_mod

    parser = argparse.ArgumentParser(description="Oil Data Crawler")
    parser.add_argument("--force",    action="store_true", help="캐시 무시 강제 재크롤링")
    parser.add_argument("--daemon",   action="store_true", help="3시간 간격 반복 실행")
    parser.add_argument("--interval", type=int, default=180, metavar="MIN",
                        help="데몬 모드 실행 간격 (분, 기본 180)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Oil Data Crawler  --  API 과사용 방지용 독립 실행 모듈")
    print(f"  저장 경로: {CSV_DIR}")
    if args.daemon:
        print(f"  모드: 데몬 (매 {args.interval}분 반복, Ctrl+C 로 종료)")
    print("=" * 60)

    # ── 최초 1회 즉시 실행 ────────────────────────────────────
    print("\n[1/1] 초기 크롤링 시작...")
    _print_result(fetch_all_data(force=True))

    if not args.daemon:
        raise SystemExit(0)

    # ── 데몬 루프 ─────────────────────────────────────────────
    interval_sec = args.interval * 60
    run_count    = 1
    while True:
        next_run = datetime.datetime.now() + datetime.timedelta(seconds=interval_sec)
        print(f"\n  다음 크롤링: {next_run.strftime('%Y-%m-%d %H:%M:%S')} "
              f"({args.interval}분 후)  -- Ctrl+C 로 종료")
        try:
            _time_mod.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n  데몬 종료.")
            raise SystemExit(0)

        run_count += 1
        print(f"\n[{run_count}회] 크롤링 시작 -- "
              f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            _print_result(fetch_all_data(force=True))
        except Exception as e:
            print(f"  [ERROR] 크롤링 실패: {e}  -- 다음 주기에 재시도")
