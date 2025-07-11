#!/usr/bin/env python
"""
ETL driver for BoatRace data.
 1. Execute ddl.sql (idempotent)
 2. COPY new CSV files in ./data into raw.* tables
 3. REFRESH materialized views
"""

import os, glob, pathlib, psycopg2, dotenv

# ---------- load DB config ----------
dotenv.load_dotenv("sql/.env")

# 環境変数をしゅとく
# (デフォルト値を設定)

password = os.getenv("PGPASSWORD", "")
post = os.getenv("PGPORT")
print(f"PGPASSWORD: {password}, PGPORT: {post}")

DB_CONF = {
    "host":     os.getenv("PGHOST", "localhost"),
    "port":     int(os.getenv("PGPORT", 5432)),
    "dbname":   os.getenv("PGDATABASE", "boatrace"),
    "user":     os.getenv("PGUSER", "br_user"),
    "password": os.getenv("PGPASSWORD", "secret"),
}

# ---------- connect ----------
conn = psycopg2.connect(**DB_CONF)
conn.autocommit = True
cur = conn.cursor()

# ---------- 1. DDL ----------
ddl_sql = pathlib.Path("sql/ddl.sql").read_text(encoding="utf8")
cur.execute(ddl_sql)
print("✔ DDL executed")

# ---------- 2. CSV COPY ----------
def copy_csv(table, pattern):
    files = glob.glob(pattern)
    if not files:
        print(f"  (no files for {pattern})")
        return
    for fp in files:
        print(f"processing {fp}...")
        try:
            with open(fp, "r", encoding="utf8") as f:
                cur.copy_expert(
                    f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER 1, ENCODING 'UTF8')",
                    f,
                )
            print(f"  → {table}: {fp}")
        except Exception as e:
            print(f"❌ COPY FAILED for {fp}: {e}")

print("🔄 Copying CSV files...")
copy_csv(
    "raw.results_staging",
    "download/wakamatsu_off_result_csv/wakamatsu_result_*.csv",
)
cur.execute("""
INSERT INTO raw.results (stadium, race_date, race_no, lane,
                         position_txt, racer_no, st_time_raw, source_file)
SELECT
    stadium,
    /* 例: '..._20241106_6.html' → 2024-11-06 */
    TO_DATE( REGEXP_REPLACE(source_file, '.*_(\\d{8})_.*', '\\1'), 'YYYYMMDD') AS race_date,
    race_no,
    lane,
    position,         -- '１','２',…をそのまま
    racer_no,
    st_time_raw,
    source_file
FROM raw.results_staging
ON CONFLICT DO NOTHING;          -- ← 重複取込を無視したければ
TRUNCATE raw.results_staging;     -- 次回の ETL に備えて空に
""")
copy_csv(
    "raw.racers_staging",
    "download/wakamatsu_off_beforeinfo_csv/*_racers.csv"
)
cur.execute("""
INSERT INTO raw.racers (
    race_date, race_no, lane, racer_id,
    weight_raw, adjust_weight, exh_time, tilt_deg
)
SELECT
    TO_DATE((REGEXP_MATCHES(source_file, '.*_(\\d{8})_(\\d+)\\.html$'))[1], 'YYYYMMDD') AS race_date,
    ((REGEXP_MATCHES(source_file, '.*_(\\d{8})_(\\d+)\\.html$'))[2])::INT AS race_no,
    lane,
    racer_id,
    weight,
    adjust_weight,
    exhibition_time,
    tilt
FROM raw.racers_staging
WHERE source_file ~ '.*_(\\d{8})_(\\d+)\\.html$'
ON CONFLICT DO NOTHING;

TRUNCATE raw.racers_staging;
""")
copy_csv("raw.start_exhibition_staging",
         "download/wakamatsu_off_beforeinfo_csv/wakamatsu_beforeinfo_*_start_exhibition.csv")
cur.execute("""
            INSERT INTO raw.start_exhibition (race_date, race_no, lane, st_raw)
            SELECT
                TO_DATE(
                    REGEXP_REPLACE(source_file,
                       '.*_20_(\\d{8})_\\d+\\.html$', '\\1'),
                    'YYYYMMDD'
                ) AS race_date,
                CAST(
                    REGEXP_REPLACE(source_file,
                       '.*_20_\\d{8}_(\\d+)\\.html$', '\\1') AS INT
                ) AS race_no,
                lane,
                st_raw
            FROM   raw.start_exhibition_staging
            ON CONFLICT DO NOTHING;
            TRUNCATE raw.start_exhibition_staging;
        """)
copy_csv("raw.weather_staging",          "download/wakamatsu_off_beforeinfo_csv/wakamatsu_beforeinfo_*_weather.csv")
cur.execute("""
            INSERT INTO raw.weather (
                race_date, race_no,
                obs_time_label, weather_txt,
                air_temp_raw, wind_speed_raw,
                water_temp_raw, wave_height_raw
            )
            SELECT
                /* source_file から 8 桁日付とレース番号を抽出 */
                TO_DATE(regexp_replace(source_file,
                       '.*_20_(\\d{8})_\\d+\\.html$','\\1'),'YYYYMMDD'),
                regexp_replace(source_file,
                       '.*_20_\\d{8}_(\\d+)\\.html$','\\1')::int,
                obs_datetime_label,
                weather,
                air_temp_C,
                wind_speed_m,
                water_temp_C,
                wave_height_cm
            FROM raw.weather_staging
            ON CONFLICT DO NOTHING;
            TRUNCATE raw.weather_staging;
        """)

print("✔ CSV files copied into raw tables")
# ---------- 3. REFRESH materialized views ----------
views = [
    "core.races", "core.results", "core.boat_info",
    "core.start_exh", "core.weather",
    "feat.boat_flat", "feat.train_features"
]
for v in views:
    cur.execute(f"REFRESH MATERIALIZED VIEW {v};")
    print(f"✔ refreshed {v}")

cur.close(); conn.close()
print("🎉 ETL done")
