
# pred_features_from_dfs.py
# Build pred.* dataframes from racers_df & weather_df (pandas DataFrames)

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional

__all__ = [
    "build_pred_from_dfs",
    "to_float_clean",
]

def to_float_clean(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return np.nan
    s = (s.replace("kg","")
           .replace("℃","")
           .replace("cm","")
           .replace("m","")
           .replace("−","-"))
    if s.startswith("."):
        s = "0"+s
    try:
        return float(s)
    except:
        try:
            return float(s.replace(",", ""))
        except:
            return np.nan

def _normalize_racers_df(racers_df: pd.DataFrame, race_key: str) -> pd.DataFrame:
    df = racers_df.copy()
    df["race_key"] = race_key
    if "lane" not in df.columns:
        raise ValueError("racers_df に lane 列が必要です（1..6 の1始まり）")
    # rename to pred schema
    ren = {}
    if "exhibition_time" in df.columns and "exh_time" not in df.columns:
        ren["exhibition_time"] = "exh_time"
    if "tilt" in df.columns and "tilt_deg" not in df.columns:
        ren["tilt"] = "tilt_deg"
    if "ST" in df.columns and "st_time" not in df.columns:
        ren["ST"] = "st_time"
    df = df.rename(columns=ren)
    # keep & coerce
    keep = ["race_key","lane","racer_id","weight","exh_time","tilt_deg","st_time","course","bf_st_time","bf_course","fs_flag"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df["lane"] = pd.to_numeric(df["lane"], errors="coerce").astype("Int64")
    # strip units and cast
    if "weight" in df.columns:
        df["weight"] = df["weight"].astype(str).str.replace("kg","", regex=False)
    for c in ["weight","exh_time","tilt_deg","st_time","course","bf_st_time","bf_course"]:
        df[c] = df[c].map(to_float_clean)
    # fs_flag
    df["fs_flag"] = df["fs_flag"].map(lambda v: str(v).strip().lower() in ("1","true","t","y","yes")) if "fs_flag" in df.columns else False
    out = df[keep].sort_values(["race_key","lane"]).drop_duplicates(["race_key","lane"], keep="first").reset_index(drop=True)
    return out

def _normalize_weather_df(weather_df: pd.DataFrame, race_key: str) -> pd.DataFrame:
    df = weather_df.copy()
    # prefer a row that has "weather" if present
    if "weather" in df.columns and df["weather"].notna().any():
        wsrc = df[df["weather"].notna()].head(1)
    else:
        wsrc = df.head(1)
    def _get(col, default=np.nan):
        return wsrc[col].iloc[0] if col in wsrc.columns and len(wsrc[col])>0 else default
    out = pd.DataFrame({
        "race_key": [race_key],
        "air_temp": [to_float_clean(_get("air_temp_C"))],
        "wind_speed": [to_float_clean(_get("wind_speed_m"))],
        "wave_height": [to_float_clean(_get("wave_height_cm"))],
        "water_temp": [to_float_clean(_get("water_temp_C"))],
        "weather_txt": [_get("weather", "")],
        "wind_dir_deg": [to_float_clean(_get("wind_dir_deg"))],
    })
    return out

def _pred_boat_flat(pred_boat_info: pd.DataFrame, pred_weather: pd.DataFrame) -> pd.DataFrame:
    flat = pred_boat_info.merge(pred_weather, on="race_key", how="left")
    flat = flat.sort_values(["race_key","lane"]).drop_duplicates(["race_key","lane"], keep="first").reset_index(drop=True)
    return flat

def _pivot_lane(df_long: pd.DataFrame, values) -> pd.DataFrame:
    wide = df_long.pivot_table(index="race_key", columns="lane", values=values, aggfunc="first")
    wide.columns = [f"lane{int(l)}_{c}" for c,l in wide.columns]
    return wide.reset_index()

def _pred_features(boat_flat: pd.DataFrame, race_date: Optional[str]) -> pd.DataFrame:
    lane_values = ["racer_id","weight","exh_time","st_time","course","bf_st_time","bf_course","fs_flag","tilt_deg"]
    for c in lane_values:
        if c not in boat_flat.columns:
            boat_flat[c] = np.nan
    pivoted = _pivot_lane(boat_flat[["race_key","lane"] + lane_values], lane_values)
    race_info = boat_flat[["race_key","air_temp","wind_speed","wave_height","water_temp","weather_txt","wind_dir_deg"]].drop_duplicates("race_key")
    race_info = race_info.assign(race_date=race_date if race_date else np.nan)
    features = race_info.merge(pivoted, on="race_key", how="left")
    return features

def _build_filtered_course_from_history(hist_boat_info_df: pd.DataFrame, hist_results_df: pd.DataFrame) -> pd.DataFrame:
    b = hist_boat_info_df[["race_key","lane","racer_id","course"]].copy()
    b["lane"] = pd.to_numeric(b["lane"], errors="coerce").astype("Int64")
    b["course"] = pd.to_numeric(b["course"], errors="coerce")
    r = hist_results_df[["race_key","lane","rank"]].copy()
    r["lane"] = pd.to_numeric(r["lane"], errors="coerce").astype("Int64")
    r["rank"] = pd.to_numeric(r["rank"], errors="coerce")
    df = b.merge(r, on=["race_key","lane"], how="inner").rename(columns={"racer_id":"reg_no"})
    grp = df.groupby(["reg_no","course"], dropna=False, as_index=False)
    agg = grp.agg(
        starts=("race_key","count"),
        firsts=("rank", lambda s: int((s==1).sum())),
        two_cnt=("rank", lambda s: int((s<=2).sum())),
        three_cnt=("rank", lambda s: int((s<=3).sum())),
    )
    agg["first_rate"]  = agg["firsts"]   / agg["starts"].replace(0, np.nan)
    agg["two_rate"]    = agg["two_cnt"]  / agg["starts"].replace(0, np.nan)
    agg["three_rate"]  = agg["three_cnt"]/ agg["starts"].replace(0, np.nan)
    return agg.drop(columns=["two_cnt","three_cnt"])

def _pred_tf2_lane_stats(pred_features: pd.DataFrame, filtered_course_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in range(1,7):
        rows.append(pd.DataFrame({
            "race_key": pred_features["race_key"],
            "lane_no": n,
            "reg_no": pred_features.get(f"lane{n}_racer_id"),
            "course": pd.to_numeric(pred_features.get(f"lane{n}_course"), errors="coerce"),
        }))
    long = pd.concat(rows, ignore_index=True)
    fc = filtered_course_df.copy()
    if "racer_id" in fc.columns and "reg_no" not in fc.columns:
        fc = fc.rename(columns={"racer_id": "reg_no"})

    long["reg_no"] = pd.to_numeric(long["reg_no"], errors="coerce").astype("Int64")
    if "reg_no" in fc.columns:
        fc["reg_no"] = pd.to_numeric(fc["reg_no"], errors="coerce").astype("Int64")

    long["course"] = pd.to_numeric(long["course"], errors="coerce")
    if "course" in fc.columns:
        fc["course"] = pd.to_numeric(fc["course"], errors="coerce")

    # Merge and pivot back to wide format
    stats_cols = ["starts","firsts","first_rate","two_rate","three_rate"]
    merged = long.merge(fc, on=["reg_no","course"], how="left")
    wide = merged.pivot_table(index="race_key", columns="lane_no", values=stats_cols, aggfunc="first")
    wide.columns = [f"lane{int(l)}_{c}" for c, l in wide.columns]
    return wide.reset_index()

def build_pred_from_dfs(
    racers_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    race_key: str,
    race_date: Optional[str] = None,
    filtered_course_df: Optional[pd.DataFrame] = None,
    hist_boat_info_df: Optional[pd.DataFrame] = None,
    hist_results_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict containing:
      - pred_boat_flat
      - pred_features
      - pred_tf2_lane_stats (if stats available)
      - pred_features_with_record (if stats available)
    """
    # strip whitespace
    for df in (racers_df, weather_df):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()

    pred_boat_info = _normalize_racers_df(racers_df, race_key)
    pred_weather   = _normalize_weather_df(weather_df, race_key)
    boat_flat      = _pred_boat_flat(pred_boat_info, pred_weather)
    features       = _pred_features(boat_flat, race_date)

    out = {
        "pred_boat_flat": boat_flat,
        "pred_features": features,
    }

    # lane stats
    fc = None
    if filtered_course_df is not None:
        fc = filtered_course_df
    elif (hist_boat_info_df is not None) and (hist_results_df is not None):
        fc = _build_filtered_course_from_history(hist_boat_info_df, hist_results_df)

    if fc is not None:
        lane_stats = _pred_tf2_lane_stats(features, fc)
        out["pred_tf2_lane_stats"] = lane_stats
        out["pred_features_with_record"] = features.merge(lane_stats, on="race_key", how="left")

    return out
