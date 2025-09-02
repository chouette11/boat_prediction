/*------------------------------------------------------------
  02_tables_raw.sql
  RAW レイヤのテーブル + STAGING テーブル
------------------------------------------------------------*/

CREATE TABLE IF NOT EXISTS raw.beforeinfo_staging_off (
    lane              INT,
    racer_id          INT,
    name              TEXT,
    weight            TEXT,
    adjust_weight     NUMERIC,
    exhibition_time   NUMERIC,
    tilt              NUMERIC,
    photo             TEXT,
    source_file       TEXT,
    st_time_raw       TEXT,
    course            INT
);

CREATE TABLE IF NOT EXISTS raw.weather_staging_off (
    obs_datetime_label TEXT,
    weather            TEXT,
    air_temp_C         TEXT,
    wind_speed_m       TEXT,
    water_temp_C       TEXT,
    wave_height_cm     TEXT,
    wind_dir_icon      TEXT,
    source_file        TEXT
);

CREATE TABLE IF NOT EXISTS raw.beforeinfo_off (
    stadium        TEXT NOT NULL DEFAULT '若松',
    race_date      DATE NOT NULL,
    race_no        INT  NOT NULL,
    lane           INT  NOT NULL,
    racer_id       INT,
    weight_raw     TEXT,
    adjust_weight  NUMERIC,
    exh_time       NUMERIC,
    tilt_deg       NUMERIC,
    st_time_raw         TEXT,
    course         INT
);

CREATE TABLE IF NOT EXISTS raw.weather_off (
    stadium            TEXT NOT NULL DEFAULT '若松',
    race_date          DATE NOT NULL,
    race_no            INT  NOT NULL,
    obs_time_label     TEXT,
    weather_txt        TEXT,
    air_temp_raw       TEXT,
    wind_speed_raw     TEXT,
    wind_dir_raw       TEXT,
    water_temp_raw     TEXT,
    wave_height_raw    TEXT
);