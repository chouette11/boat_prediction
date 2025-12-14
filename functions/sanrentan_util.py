from __future__ import annotations
from typing import Iterable, Union, Optional, Dict
import pandas as pd
import numpy as np

RANK_COLS = [f"lane{i}_rank" for i in range(1, 7)]

def _ensure_dataframe(source: Union[str, pd.DataFrame], encoding: Optional[str] = None) -> pd.DataFrame:
    """
    Accept a CSV path or a preloaded DataFrame and return a DataFrame.
    Tries multiple encodings if one is not provided.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    encodings_to_try = [encoding] if encoding else ["utf-8", "utf-8-sig", "cp932", "shift_jis"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(source, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else ValueError("Failed to read CSV")

def _trifecta_from_row(row: pd.Series) -> Optional[str]:
    """
    Compute sanrentan string like '1-2-3' from laneX_rank columns.
    Ignores NaN/0/negative ranks; sorts ascending by rank and returns the top 3 lanes.
    """
    pairs = []
    for lane in range(1, 7):
        col = f"lane{lane}_rank"
        r = row.get(col, np.nan)
        try:
            r = int(r)
        except Exception:
            r = np.nan
        if isinstance(r, (int, np.integer)) and r > 0:
            pairs.append((lane, r))
    if len(pairs) < 3:
        return None
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    lanes_top3 = [lane for lane, _ in pairs_sorted[:3]]
    return f"{lanes_top3[0]}-{lanes_top3[1]}-{lanes_top3[2]}"

def compute_sanrentan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with a new 'sanrentan' column derived from laneX_rank columns.
    Requires a 'race_key' column.
    """
    missing = [c for c in ["race_key"] + RANK_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing: {missing}")
    out = df.copy()
    out["sanrentan"] = out.apply(_trifecta_from_row, axis=1)
    return out

def get_sanrentan_by_race_key(source: Union[str, pd.DataFrame],
                              race_keys: Union[str, Iterable[str]],
                              encoding: Optional[str] = None) -> pd.DataFrame:
    """
    Load CSV or use given DataFrame and return a DataFrame with ['race_key', 'sanrentan'] for the given race_key(s).
    - race_keys may be a single string or an iterable of strings.
    - Rows where the trifecta cannot be computed (e.g., insufficient rank data) will have sanrentan=None.
    """
    df = _ensure_dataframe(source, encoding=encoding)
    df2 = compute_sanrentan(df)[["race_key", "sanrentan"]].drop_duplicates("race_key")
    if isinstance(race_keys, str):
        keys = [race_keys]
    else:
        keys = list(race_keys)
    mask = df2["race_key"].isin(keys)
    return df2.loc[mask].reset_index(drop=True)
