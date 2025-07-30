#!/usr/bin/env python3
"""
fetch_wakamatsu_env.py
若松水面の ML 特徴量用 ─ 風・潮位・潮流・波浪・気温を 10 min～1 h 解像度で収集
-------------------------------------------------------------------------
*   NOWPHAS   : 観測波浪・観測潮位
*   tide736   : 天文潮位（満潮／干潮・潮名含む）
*   MSIL 海しる: 潮流 u,v / 表層水温
*   AMeDAS    : 10 min 平均風速・瞬間風速・風向・気温・気圧
-------------------------------------------------------------------------"""
from __future__ import annotations
import os, sys, json, datetime as dt, pathlib, typing as t
import requests, pandas as pd
import io
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional dependency: pyarrow for Parquet. Fallback to CSV if unavailable.
try:
    import pyarrow  # noqa: F401
    _PARQUET_SUPPORTED = True
except ImportError:
    _PARQUET_SUPPORTED = False

# ──────────────────────────────────────────────────────────────
# 共通ヘルパ
# ──────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path("download/raw_wakamatsu_env").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

TODAY = dt.date.today()
ISO_TS = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1.5))
def fetch_json(url: str, headers: dict | None = None, params: dict | None = None):
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1.5))
def fetch_csv(url: str, headers: dict | None = None, params: dict | None = None):
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# ------------------------------------------------------------------
# Helper: write DataFrame as Parquet when possible, otherwise CSV
# ------------------------------------------------------------------
def _save_table(df: pd.DataFrame, path: pathlib.Path):
    """
    Save *df* to *path*.
    If pyarrow/fastparquet is available -> Parquet.
    Otherwise -> same stem with .csv extension.
    """
    if _PARQUET_SUPPORTED:
        df.to_parquet(path, index=False)
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
    else:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"⚠️  pyarrow not installed — saved as CSV: {csv_path.name}")

# ──────────────────────────────────────────────────────────────
# 1) tide736 ― 天文潮位（10 min）
# ──────────────────────────────────────────────────────────────
def get_tide736(pred_date: dt.date, pref_code: int = 40, port_code: int = 10) -> pd.DataFrame:
    """若松 (福岡=40, 港=10) 1 日分の天文潮位 (10 min 解像度)"""
    url = "https://api.tide736.net/get_tide.php"
    params = dict(
        pc=pref_code, hc=port_code,
        yr=pred_date.year, mn=f"{pred_date.month:02}",
        dy=f"{pred_date.day:02}", rg="day", fmt="json",
    )
    js = fetch_json(url, params=params)

    # ---- early sanity check -------------------------------------------------
    if "tide" not in js or not js["tide"]:
        print(
            "\n⚠️  tide736 returned NO tide data "
            f"for pc={pref_code}, hc={port_code}, date={pred_date}.\n"
            "Raw response ↓\n"
            f"{json.dumps(js, ensure_ascii=False, indent=2)[:800]}...\n"
            "👉  Hint: the harbor‑code *hc* is often NOT intuitive.  \n"
            "    Try listing available harbors for Fukuoka with:\n"
            "       curl -s 'https://api.tide736.net/get_harbor.php?pc=40' | jq .\n"
            "    and look for 若松 / 若松港 (the correct hc may differ from 30)."
        )
        return pd.DataFrame(columns=["timestamp_jst", "astro_cm"])

    tide_obj = js["tide"]
    # ---------------------------------------------
    # tide736 sometimes nests the actual series under
    # tide_obj["chart"][YYYY-MM-DD]["tide"].
    # If that structure exists, grab that list instead.
    # ---------------------------------------------
    date_key = pred_date.strftime("%Y-%m-%d")
    if (
        isinstance(tide_obj, dict)
        and "chart" in tide_obj
        and date_key in tide_obj["chart"]
        and "tide" in tide_obj["chart"][date_key]
    ):
        tide_obj = tide_obj["chart"][date_key]["tide"]

    print(f"🌊  tide736: {len(tide_obj)} records for {pred_date} (pc={pref_code}, hc={port_code})")
    print(f"tide_obj {tide_obj}")
    records: list[dict] = []

    # Case 1: dict keyed by "HH:MM"
    if isinstance(tide_obj, dict):
        iterator = tide_obj.items()
    # Case 2: list of dicts [{"time": "HH:MM", "cm": "123"}, ...]
    elif isinstance(tide_obj, list):
        iterator = ((d.get("time"), d) for d in tide_obj)
    else:
        print(f"⚠️  Unrecognized 'tide' structure ({type(tide_obj)}) — skipping")
        return pd.DataFrame(columns=["timestamp_jst", "astro_cm"])

    for t_str, val in iterator:
        if not t_str:
            continue
        cm_val = (
            val.get("cm") if isinstance(val, dict) else val
        )
        try:
            cm_float = float(cm_val)
            t_obj = dt.datetime.strptime(t_str, "%H:%M").time()
            records.append({
                "timestamp_jst": dt.datetime.combine(pred_date, t_obj),
                "astro_cm": cm_float,
            })
        except Exception:
            continue

    if not records:
        print(f"⚠️  No valid tide records parsed for {pred_date}")
        return pd.DataFrame(columns=["timestamp_jst", "astro_cm"])

    records.sort(key=lambda x: x["timestamp_jst"])
    return pd.DataFrame.from_records(records, columns=["timestamp_jst", "astro_cm"])

# ──────────────────────────────────────────────────────────────
# 2) NOWPHAS ― 波浪 & 観測潮位（10 min）
# 公式サイトで公開されている CSV を直リンク取得
# base: https://nowphas.mlit.go.jp/data/realtime_csv/{STATION_ID}.csv   [oai_citation:1‡国土交通省](https://www.mlit.go.jp/kowan/nowphas/?utm_source=chatgpt.com)
# 若松沖ブイ ID は 'WAKAMATSU'（サイト上で確認）
# ──────────────────────────────────────────────────────────────
def get_nowphas(station_id="WAKAMATSU") -> pd.DataFrame:
    url = f"https://nowphas.mlit.go.jp/data/realtime_csv/{station_id}.csv"
    raw = fetch_csv(url)
    # 先頭行にヘッダが 2 段あるので整形
    raw.columns = raw.columns.str.strip()
    raw = (raw.rename(columns={"時刻": "time_jst"})
              .assign(station=station_id))
    return raw


# ──────────────────────────────────────────────────────────────
# 3) 海しる API ― 潮流 u,v（cm/s）・表層水温
# ──────────────────────────────────────────────────────────────
MSIL_KEY = "0e83ad5d93214e04abf37c970c32b641"
MSIL_ENDPOINT = "https://api.msil.go.jp/v1/ocn/currents/point"

def get_msils_current(date_iso: str = ISO_TS,
                      lon: float = 130.790, lat: float = 33.903) -> pd.DataFrame:
    import numpy as np

    API_KEY = "0e83ad5d93214e04abf37c970c32b641"
    LAT, LON = 33.903, 130.790            # 若松スタートライン中央付近
    ts = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0).isoformat() + "Z"

    url = "https://api.msil.go.jp/apis/v2/ocn/currents/point"
    hdr = {"Ocp-Apim-Subscription-Key": API_KEY}
    params = {"lat": LAT, "lon": LON, "time": ts, "format": "json"}

    js = requests.get(url, headers=hdr, params=params, timeout=20).json()
    print(f"🌊  MSIL currents: {len(js)} records for {ts} (lat={LAT}, lon={LON})")
    print(f"js {js}")
    cur  = js["current"]                 # u, v, speed, direction
    temp = js.get("waterTemperature")    # °C
    sal  = js.get("salinity")            # psu

    u, v = cur["u"], cur["v"]
    speed = (u**2 + v**2) ** .5
    theta = (180/3.14159) * np.arctan2(v, u)

    rho = 1000 + 0.8*sal - 0.2*temp      # 近似密度
    print(dict(ts=ts, u=u, v=v, speed=speed,
            temp=temp, sal=sal, rho=rho))

# ──────────────────────────────────────────────────────────────
# 4) 気象庁 AMeDAS 10 分値 ― 風・気温
# ──────────────────────────────────────────────────────────────
AMEDAS_ID = os.getenv("JMA_AMEDAS_ID", "82056")  # 八幡

def _nearest_10min(dt_utc: dt.datetime) -> str:
    dt_jst = dt_utc + dt.timedelta(hours=9)
    minute = (dt_jst.minute // 10) * 10
    return dt_jst.replace(minute=minute, second=0, microsecond=0)

def get_amedas(date: dt.date = TODAY,
               amedas_id: str = AMEDAS_ID) -> pd.DataFrame:
    dt_utc = dt.datetime.utcnow()
    stamp = _nearest_10min(dt_utc).strftime("%H%M")
    yyyymmdd = date.strftime("%Y%m%d")
    url = (f"https://www.jma.go.jp/bosai/amedas/data/point/"
           f"{amedas_id}/{yyyymmdd}_{stamp}.json")
    js = fetch_json(url)
    records = []
    for ts, arr in js.items():
        try:
            jst = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
            records.append({
                "time_jst": jst,
                "temp_C": arr[1],
                "wind_ms": arr[2],
                "wind_dir_deg": arr[3],
            })
        except Exception:
            continue
    return pd.DataFrame(records)

# ──────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────
def main():
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Load environment variables from .env file
    today = TODAY
    yesterday = today - dt.timedelta(days=1)

    # 天文潮位
    # tide_df = pd.concat([get_tide736(yesterday), get_tide736(today)])
    # _save_table(tide_df, DATA_DIR / "tide736.parquet")

    # NOWPHAS 観測
    # now_df = get_nowphas()
    # _save_table(now_df, DATA_DIR / "nowphas.parquet")

    # # # 海しる（最新時刻のみ）
    # msil_df = get_msils_current()
    # _save_table(msil_df, DATA_DIR / "msil_current.parquet")

    # AMeDAS
    amedas_df = get_amedas(today)
    _save_table(amedas_df, DATA_DIR / "amedas.parquet")

    print("✅ all data fetched →", DATA_DIR)

if __name__ == "__main__":
    main()