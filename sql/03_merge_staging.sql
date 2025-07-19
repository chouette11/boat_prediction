/*------------------------------------------------------------
  03_merge_staging.sql  (NOT NULL 対応 + CTE 分割 + right()対応)
------------------------------------------------------------*/

------------------------------------------------------------
-- 1) results_staging → raw.results
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.results
    ADD CONSTRAINT results_unique UNIQUE (stadium, race_date, race_no, lane);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint results_unique already exists.';
END
$$;

WITH rs AS (
    SELECT *,
           (regexp_match(right(source_file, 100),
                         'wakamatsu_raceresult_[0-9]+_([0-9]{8})_([0-9]+)\.html$'))[1] AS yyyymmdd,
           (regexp_match(right(source_file, 100),
                         'wakamatsu_raceresult_[0-9]+_([0-9]{8})_([0-9]+)\.html$'))[2] AS race_no_ex
    FROM raw.results_staging
    WHERE right(source_file, 100) ~ 'wakamatsu_raceresult_[0-9]+_[0-9]{8}_[0-9]+\.html$'
)
INSERT INTO raw.results (
    stadium, race_date, race_no, lane,
    position_txt, racer_no, st_time_raw, source_file
)
SELECT
    COALESCE(stadium, '若松') AS stadium,
    to_date(yyyymmdd, 'YYYYMMDD') AS race_date,
    COALESCE(race_no::int, race_no_ex::int) AS race_no,
    lane,
    position_txt,
    racer_no,
    st_time_raw,
    source_file
FROM rs
ON CONFLICT DO NOTHING;

------------------------------------------------------------
-- 2) beforeinfo_staging → raw.racers
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.racers
    ADD CONSTRAINT racers_unique UNIQUE (stadium, race_date, race_no, lane);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint racers_unique already exists.';
END
$$;

WITH bfs AS (
    SELECT *,
           (regexp_match(right(source_file, 100),
                         'beforeinfo_[0-9]+_([0-9]{8})_([0-9]+)\.html$'))[1] AS yyyymmdd,
           (regexp_match(right(source_file, 100),
                         'beforeinfo_[0-9]+_([0-9]{8})_([0-9]+)\.html$'))[2] AS race_no_ex
    FROM raw.beforeinfo_staging
    WHERE right(source_file, 100) ~ 'beforeinfo_[0-9]+_[0-9]{8}_[0-9]+\.html$'
)
INSERT INTO raw.racers (
    stadium, race_date, race_no, lane,
    racer_id, weight_raw, adjust_weight,
    exh_time, tilt_deg, st_raw, st_entry
)
SELECT
    '若松',
    to_date(yyyymmdd, 'YYYYMMDD'),
    race_no_ex::int,
    lane,
    racer_id, weight, adjust_weight,
    exhibition_time, tilt,
    st_raw, st_entry
FROM bfs
ON CONFLICT DO NOTHING;

------------------------------------------------------------
-- 3) weather_staging → raw.weather
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.weather
    ADD CONSTRAINT weather_unique UNIQUE (stadium, race_date, race_no);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint weather_unique already exists.';
END
$$;

------------------------------------------------------------
-- 3) weather_staging → raw.weather
------------------------------------------------------------
WITH ws AS (
    SELECT *,
           (regexp_match(right(source_file, 100),
                         'wakamatsu_beforeinfo_[0-9]+_([0-9]{8})_([0-9]+)\.html$'))[1] AS yyyymmdd,
           (regexp_match(right(source_file, 100),
                         'wakamatsu_beforeinfo_[0-9]+_([0-9]{8})_([0-9]+)\.html$'))[2] AS race_no_ex
    FROM raw.weather_staging
    WHERE right(source_file, 100) ~ 'wakamatsu_beforeinfo_[0-9]+_[0-9]{8}_[0-9]+\.html$'
)
INSERT INTO raw.weather (
    stadium, race_date, race_no,
    obs_time_label, weather_txt,
    air_temp_raw, wind_speed_raw,
    wind_dir_raw,
    water_temp_raw, wave_height_raw
)
SELECT
    '若松',
    to_date(yyyymmdd, 'YYYYMMDD'),
    race_no_ex::int,
    obs_datetime_label, weather,
    air_temp_c, wind_speed_m,
    wind_dir_icon,
    water_temp_c, wave_height_cm
FROM ws
ON CONFLICT DO NOTHING;
