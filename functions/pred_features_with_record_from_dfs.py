
# pred_features_with_record_from_dfs.py
# Always return pred.features_with_record from racers_df & weather_df,
# optionally merging lane stats from either pred_tf2_lane_stats or filtered_course/history.
import pandas as pd
import numpy as np
from typing import Optional, Dict
from pred_features_from_dfs import build_pred_from_dfs

LANE_STAT_COLS = [
    "lane1_starts","lane1_firsts","lane1_first_rate","lane1_two_rate","lane1_three_rate",
    "lane2_starts","lane2_firsts","lane2_first_rate","lane2_two_rate","lane2_three_rate",
    "lane3_starts","lane3_firsts","lane3_first_rate","lane3_two_rate","lane3_three_rate",
    "lane4_starts","lane4_firsts","lane4_first_rate","lane4_two_rate","lane4_three_rate",
    "lane5_starts","lane5_firsts","lane5_first_rate","lane5_two_rate","lane5_three_rate",
    "lane6_starts","lane6_firsts","lane6_first_rate","lane6_two_rate","lane6_three_rate",
]

def build_pred_features_with_record_from_dfs(
    racers_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    race_key: str,
    race_date: Optional[str] = None,
    # Option 1: pass precomputed pred.tf2_lane_stats DataFrame (must have race_key + LANE_STAT_COLS)
    tf2_lane_stats_df: Optional[pd.DataFrame] = None,
    # Option 2: pass filtered_course (reg_no, course, starts/first_rate/...) OR historical raw to aggregate
    filtered_course_df: Optional[pd.DataFrame] = None,
    hist_boat_info_df: Optional[pd.DataFrame] = None,
    hist_results_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame equivalent to pred.features_with_record for the given race.
    If no lane stats are available, the lane-stat columns are included and filled with NaN.
    """
    out = build_pred_from_dfs(
        racers_df=racers_df,
        weather_df=weather_df,
        race_key=race_key,
        race_date=race_date,
        filtered_course_df=filtered_course_df,
        hist_boat_info_df=hist_boat_info_df,
        hist_results_df=hist_results_df,
    )
    features = out["pred_features"].copy()
    # Merge lane stats in one of the accepted ways
    if tf2_lane_stats_df is not None:
        ls = tf2_lane_stats_df.copy()
        # sanity: ensure required columns exist
        missing = [c for c in ["race_key"] + LANE_STAT_COLS if c not in ls.columns]
        if missing:
            raise ValueError(f"tf2_lane_stats_df is missing columns: {missing}")
        merged = features.merge(ls[["race_key"] + LANE_STAT_COLS], on="race_key", how="left")
        # ensure float types where appropriate
        for c in LANE_STAT_COLS:
            if c.endswith("_starts") or c.endswith("_firsts"):
                merged[c] = pd.to_numeric(merged[c], errors="coerce")
            else:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")
        return merged

    if "pred_features_with_record" in out:
        # build_pred_from_dfs already computed lane stats via filtered_course/history
        merged = out["pred_features_with_record"]
        # ensure all expected columns exist
        for c in LANE_STAT_COLS:
            if c not in merged.columns:
                merged[c] = np.nan
        return merged

    # No stats available: append empty stat columns to match 16の形
    for c in LANE_STAT_COLS:
        features[c] = np.nan
    return features
