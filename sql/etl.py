#!/usr/bin/env python3
"""load_wakamatsu_csv_staging.py

CSV → raw.*_staging へロードするスクリプト。

- *_results.csv*      → raw.results_staging  (12 列)
- *_beforeinfo.csv*   → raw.beforeinfo_staging
- *_weather.csv*      → raw.weather_staging

本番テーブル (raw.results / raw.racers / raw.weather) へは流しません。
その後の変換は 03_merge_staging.sql などで実施してください。

環境変数 (任意)
----------------
PGHOST / PGPORT / PGDATABASE / PGUSER / PGPASSWORD
    libpq 標準。未指定は localhost:5432/ver2_2/keiichiro

CSV_DIR_RESULTS / CSV_DIR_BEFOREINFO
    CSV が置いてあるディレクトリ。
    既定:
      - download/wakamatsu_off_raceresult_csv
      - download/wakamatsu_off_beforeinfo_csv

"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

# ---------------------------------------------------------------------------
# 1. *_results.csv → raw.results_staging
# ---------------------------------------------------------------------------

import re

def _extract_date_no_from_path(path: Path) -> tuple[str | None, int | None]:
    """wakamatsu_raceresult_20_20240101_10_results.csv → ('20240101', 10)"""
    m = re.search(r"_(\d{8})_(\d+)", path.name)
    if m:
        yyyymmdd, race_no = m.groups()
        return yyyymmdd, int(race_no)
    return None, None

def _load_results(engine, path: str) -> None:
    p = Path(path)
    df = pd.read_csv(p)

    # CSV 列名 → staging 列名へ補正
    df = df.rename(
        columns={
            "time": "arrival_time",
            "st_time": "st_time_raw",
        }
    )

    required_cols = [
        "position_txt",
        "lane",
        "racer_no",
        "racer_name",
        "arrival_time",
        "course",
        "st_time_raw",
        "tactic",
        "stadium",
        "race_title",
        "date_label",
        "race_no",
        "source_file",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # ファイル名から race_date / race_no を補完
    yyyymmdd, race_no_from_fn = _extract_date_no_from_path(p)
    if yyyymmdd:
        # date_label が空なら補完 (分析用に保持するだけ。staging では NOT NULL 制約なし)
        df.loc[df["date_label"].isna(), "date_label"] = yyyymmdd
    if race_no_from_fn is not None:
        df.loc[df["race_no"].isna(), "race_no"] = race_no_from_fn

    # 数値列を明示的に変換
    num_cols = ["lane", "racer_no", "course", "st_time_raw", "race_no"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # source_file が入っていなければファイル名をそのままセット
    if df["source_file"].isna().all():
        df["source_file"] = str(p)

    df.to_sql(
        "results_staging",
        con=engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
    )
# ---------------------------------------------------------------------------
# 2. *_beforeinfo.csv → raw.beforeinfo_staging
# ---------------------------------------------------------------------------

def _load_beforeinfo(engine, path: str) -> None:
    df = pd.read_csv(path)

    df = df.rename(columns={"ST": "st_time_raw"})

    required_cols = [
        "lane",
        "racer_id",
        "name",
        "weight",
        "adjust_weight",
        "exhibition_time",
        "tilt",
        "photo",
        "source_file",
        "course",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    numeric_cols = ["adjust_weight", "exhibition_time", "tilt", "course"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df.to_sql(
        "beforeinfo_staging",
        con=engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
    )

# ---------------------------------------------------------------------------
# 3. *_weather.csv → raw.weather_staging
# ---------------------------------------------------------------------------

def _load_weather(engine, path: str) -> None:
    df = pd.read_csv(path)

    # カラム名を小文字に統一 (CSV 側の大文字小文字ゆらぎ対策)
    df.columns = [c.lower() for c in df.columns]

    df.to_sql(
        "weather_staging",
        con=engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
    )

# ---------------------------------------------------------------------------
# 3.5 *_person.csv → raw.person_staging
# ---------------------------------------------------------------------------

def _load_person(engine, path: str) -> None:
    df = pd.read_csv(path)

    float_cols = [
        "winrate_natl", "2in_natl", "3in_natl",
        "motor_2in", "motor_3in",
        "boat_2in", "boat_3in"
    ]
    int_cols = [
        "boat_no", "reg_no", "age", "ability_now", "ability_prev",
        "F_now", "L_now", "nat_1st", "nat_2nd", "nat_3rd", "nat_starts",
        "loc_1st", "loc_2nd", "loc_3rd", "loc_starts",
        "motor_no", "mot_1st", "mot_2nd", "mot_3rd", "mot_starts",
        "boat_no_hw", "boa_1st", "boa_2nd", "boa_3rd", "boa_starts"
    ]

    df[float_cols] = df[float_cols].replace("", pd.NA).astype("float64")
    df[int_cols] = df[int_cols].replace("", pd.NA).astype("Int64")

    p = Path(path)
    df["source_file"] = str(p)

    yyyymmdd, race_no = _extract_date_no_from_path(p)
    df["race_no"] = race_no
    print(race_no)

    df.to_sql(
        "person_staging",
        con=engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
    )

# ---------------------------------------------------------------------------
# 3.6 *_odds3t_*.csv → raw.odds3t_staging
# ---------------------------------------------------------------------------

def load_odds3t(engine, path: str) -> None:
    """
    Load 3‑連単オッズ CSV into raw.odds3t_staging.

    Supported file formats
    ----------------------
    1) **Flat**: columns = first_lane, second_lane, third_lane, odds
       The loader just enforces numeric types and appends `source_file`.
    2) **Matrix** (official *odds_matrix.csv*):
       - 1 header row (lane numbers / racer names) + 20 data rows
       - Each data row holds 6 triples of the form (second_lane, third_lane, odds)
       - Across the 20 rows the 120 permutations are laid out in a fixed,
         lane‑ascending order. We unpivot the matrix, then reconstruct the full
         (first_lane, second_lane, third_lane) permutation list in the same
         canonical order:
             first_lane asc, second_lane asc (≠ first), third_lane asc (≠ first, second)
    """
    import csv
    from pathlib import Path

    if not "matrix" in path:
        return

    p = Path(path)

    # ──────────────────────────────────────────────────────────
    lanes = [1, 2, 3, 4, 5, 6]  # for validation / fallbacks
    expected_triples = 120      # 6P3 permutations

    # ──────────────────────────────────────────────────────────
    # 1) Try reading as a “flat” file first
    try:
        df = pd.read_csv(p)
        if {"first_lane", "second_lane", "third_lane", "odds"}.issubset(df.columns):
            num_cols = ["first_lane", "second_lane", "third_lane", "odds"]
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
        else:
            raise ValueError  # fall through to matrix parser
    except Exception:
        # ──────────────────────────────────────────────────────
        # 2) odds_matrix fallback parser
        # ──────────────────────────────────────────────────────
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        if len(rows) <= 1:
            print(f"⚠ odds file {p} has insufficient rows; skipped.", file=sys.stderr)
            return

        # ------------------------------------------------------------------
        # Parse official odds_matrix layout: 6 lanes × 20 rows → 120 triples
        # ------------------------------------------------------------------
        header = rows[0]
        data_rows = rows[1:]
        n_cols = len(header)
        group_indices = [i for i in range(0, n_cols, 3)]  # 3 cols per lane

        records = []
        for col_start in group_indices:
            # Derive first_lane purely from the column position:
            # col_start = 0 → lane 1, 3 → lane 2, 6 → lane 3, …
            fst_lane = col_start // 3 + 1

            for row in data_rows:
                # Ensure we have a complete (sec, thr, odds) triple
                if col_start + 2 >= len(row):
                    continue
                try:
                    sec_lane = int(float(row[col_start]))
                    thr_lane = int(float(row[col_start + 1]))
                    odds_val = float(row[col_start + 2])
                except ValueError:
                    # Skip rows containing non‑numeric cells
                    continue

                records.append(
                    {
                        "first_lane":  fst_lane,
                        "second_lane": sec_lane,
                        "third_lane":  thr_lane,
                        "odds":        odds_val,
                        "source_file": str(p),
                    }
                )

        if len(records) != expected_triples:
            print(
                f"⚠ {p.name}: expected {expected_triples} triples, "
                f"got {len(records)}; check parsing logic.",
                file=sys.stderr,
            )

        df = pd.DataFrame.from_records(records)
        num_cols = ["first_lane", "second_lane", "third_lane", "odds"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    if "source_file" not in df.columns:
        df["source_file"] = str(p)

    print(df.head(20))

    print(f"df.shape = {df.shape}")
    df.to_sql(
        "odds3t_staging",
        con=engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
    )

# ---------------------------------------------------------------------------
# 4. ローダ設定
# ---------------------------------------------------------------------------

_LOADERS = {
    "odds3t": ("*_odds3t_*.csv", load_odds3t),
    "person": ("*_person_*.csv", _load_person),
    "results": ("*_results.csv", _load_results),
    "beforeinfo": ("*_beforeinfo.csv", _load_beforeinfo),
    "weather": ("*_weather.csv", _load_weather),
}

# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    csv_root = {
        "results": Path(os.getenv("CSV_DIR_RESULTS", "download/wakamatsu_off_raceresult_csv")),
        "beforeinfo": Path(os.getenv("CSV_DIR_BEFOREINFO", "download/wakamatsu_off_beforeinfo_csv")),
        "weather": Path(os.getenv("CSV_DIR_BEFOREINFO", "download/wakamatsu_off_beforeinfo_csv")),
        "person": Path(os.getenv("CSV_DIR_PERSON", "download/wakamatsu_person_csv")),
        "odds3t": Path(os.getenv("CSV_DIR_ODDS", "download/wakamatsu_off_odds3t_csv")),
    }

    user = os.getenv("PGUSER", "keiichiro")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE", "ver1_0")

    dsn = f"postgresql://{user}@{host}:{port}/{database}"
    engine = create_engine(dsn)

    for kind, (pattern, loader) in _LOADERS.items():
        path = csv_root[kind]
        if not path.is_dir():
            print(f"❌ {path} not found", file=sys.stderr)
            continue

        files = sorted(path.glob(pattern))
        print(f"{kind}: {len(files)} files in {path}")
        for f in files:
            print(f"  loading {f.name}")
            loader(engine, str(f))

    print("✔ Staging import complete.")

def loader_test() -> None:
    """Test loader functions with a specific file."""
    from dotenv import load_dotenv
    load_dotenv(override=True)

    engine = create_engine(
        f"postgresql://{os.getenv('PGUSER', 'keiichiro')}@"
        f"{os.getenv('PGHOST', 'localhost')}:"
        f"{os.getenv('PGPORT', '5432')}/"
        f"{os.getenv('PGDATABASE', 'ver1_0')}"
    )

    # Example: Load a specific odds3t file
    load_odds3t(engine, "download/wakamatsu_off_odds3t_csv/wakamatsu_odds3t_20_20250717_10_odds_matrix.csv")

if __name__ == "__main__":
    main()
    # loader_test()  # Uncomment to test a specific loader function