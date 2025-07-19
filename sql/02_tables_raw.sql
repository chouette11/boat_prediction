/*------------------------------------------------------------
  02_tables_raw.sql
  RAW レイヤのテーブル + STAGING テーブル
------------------------------------------------------------*/

/*--- STAGING (取り込み中間テーブル) --------------------------*/
CREATE TABLE IF NOT EXISTS raw.results_staging (
    position_txt   TEXT,
    lane           INT,
    racer_no       INT,
    racer_name     TEXT,
    arrival_time   TEXT,
    st_entry       INT,
    st_time_raw    TEXT,
    tactic         TEXT,
    stadium        TEXT,
    race_title     TEXT,
    date_label     TEXT,
    race_no        INT,
    source_file    TEXT
);

CREATE TABLE IF NOT EXISTS raw.beforeinfo_staging (
    lane              INT,
    racer_id          INT,
    name              TEXT,
    weight            TEXT,
    adjust_weight     NUMERIC,
    exhibition_time   NUMERIC,
    tilt              NUMERIC,
    photo             TEXT,
    source_file       TEXT,
    st_raw            TEXT,
    st_entry          INT
);

CREATE TABLE IF NOT EXISTS raw.weather_staging (
    obs_datetime_label TEXT,
    weather            TEXT,
    air_temp_C         TEXT,
    wind_speed_m       TEXT,
    water_temp_C       TEXT,
    wave_height_cm     TEXT,
    wind_dir_icon      TEXT,
    source_file        TEXT
);

/*--- 本テーブル --------------------------------------------*/
CREATE TABLE IF NOT EXISTS raw.results (
    stadium        TEXT NOT NULL DEFAULT '若松',
    race_date      DATE NOT NULL,
    race_no        INT  NOT NULL,
    lane           INT  NOT NULL,
    position_txt   TEXT,
    racer_no       INT,
    st_time_raw    TEXT,
    source_file    TEXT
);

CREATE TABLE IF NOT EXISTS raw.racers (
    stadium        TEXT NOT NULL DEFAULT '若松',
    race_date      DATE NOT NULL,
    race_no        INT  NOT NULL,
    lane           INT  NOT NULL,
    racer_id       INT,
    weight_raw     TEXT,
    adjust_weight  NUMERIC,
    exh_time       NUMERIC,
    tilt_deg       NUMERIC,
    st_raw         TEXT,
    st_entry       INT
);

CREATE TABLE IF NOT EXISTS raw.weather (
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
