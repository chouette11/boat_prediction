#!/usr/bin/env python3
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
        "beforeinfo_staging_off",
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
        "weather_staging_off",
        con=engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
    )



# 4. ローダ設定
# ---------------------------------------------------------------------------

_LOADERS = {
    "beforeinfo": ("*_beforeinfo.csv", _load_beforeinfo),
    "weather": ("*_weather.csv", _load_weather),
}

# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------
def predict_main() -> None:
    print("=== predict_main が呼ばれました ===")
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    csv_root = {
        "beforeinfo": Path(os.getenv("CSV_DIR_BEFOREINFO", "download/wakamatsu_off_beforeinfo_pred_csv")),
        "weather": Path(os.getenv("CSV_DIR_BEFOREINFO", "download/wakamatsu_off_beforeinfo_pred_csv")),
    }

    user = os.getenv("PGUSER", "keiichiro")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE", "ver1_0")

    dsn = f"postgresql://{user}@{host}:{port}/{database}"
    engine = create_engine(dsn)

    for kind, path in csv_root.items():
        base_kind = kind.removesuffix("_2")  # 結果: "results2"  (末尾が "_2" ではないので変化なし)
        print(base_kind)
        if base_kind not in _LOADERS:
            print(f"⚠ no loader for {base_kind}")
            continue

        pattern, loader = _LOADERS[base_kind]

        if not path.is_dir():
            print(f"❌ {path} not found", file=sys.stderr)
            continue

        files = sorted(path.glob(pattern))
        print(f"{kind}: {len(files)} files in {path}")
        for f in files:
            print(f"  loading {f.name}")
            loader(engine, str(f))

    print("✔ Staging import complete.")

if __name__ == "__main__":
    # main()
    predict_main()  # Uncomment to run the prediction loader
    # loader_test()  # Uncomment to test a specific loader function