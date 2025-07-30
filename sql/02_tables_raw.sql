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
    course         INT,
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
    st_time_raw       TEXT,
    course            INT
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

CREATE TABLE IF NOT EXISTS raw.person_staging (
    boat_no         INT,
    reg_no          INT,
    name            TEXT,
    age             INT,
    class_now       TEXT,
    class_hist1     TEXT,
    class_hist2     TEXT,
    class_hist3     TEXT,
    ability_now     INT,
    ability_prev    INT,
    "F_now"         INT,
    "L_now"         INT,
    winrate_natl    NUMERIC,
    "2in_natl"      NUMERIC,
    "3in_natl"      NUMERIC,
    nat_1st         INT,
    nat_2nd         INT,
    nat_3rd         INT,
    nat_starts      INT,
    loc_1st         INT,
    loc_2nd         INT,
    loc_3rd         INT,
    loc_starts      INT,
    motor_no        INT,
    motor_2in       NUMERIC,
    motor_3in       NUMERIC,
    mot_1st         INT,
    mot_2nd         INT,
    mot_3rd         INT,
    mot_starts      INT,
    boat_no_hw      INT,
    boat_2in        NUMERIC,
    boat_3in        NUMERIC,
    boa_1st         INT,
    boa_2nd         INT,
    boa_3rd         INT,
    boa_starts      INT,
    source_file     TEXT,
    race_no         INT
);

CREATE TABLE IF NOT EXISTS raw.odds3t_staging (
    first_lane      INT,
    second_lane     INT,
    third_lane      INT,
    odds            NUMERIC,
    source_file     TEXT
);

/*--- 本テーブル --------------------------------------------*/
CREATE TABLE IF NOT EXISTS raw.results (
    stadium        TEXT NOT NULL DEFAULT '若松',
    race_date      DATE NOT NULL,
    race_no        INT  NOT NULL,
    lane           INT  NOT NULL,
    course         INT,
    position_txt   TEXT,
    racer_no       INT,
    st_time_raw    TEXT,
    source_file    TEXT
);

CREATE TABLE IF NOT EXISTS raw.beforeinfo (
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

CREATE TABLE IF NOT EXISTS raw.person (
    stadium         TEXT NOT NULL DEFAULT '若松',
    race_date       DATE NOT NULL,
    race_no         INT,
    boat_no         INT,
    reg_no          INT,
    name            TEXT,
    age             INT,
    class_now       TEXT,
    class_hist1     TEXT,
    class_hist2     TEXT,
    class_hist3     TEXT,
    ability_now     INT,
    -- ability_prev    INT,
    "F_now"         INT,
    "L_now"         INT,
    winrate_natl    NUMERIC,
    "2in_natl"      NUMERIC,
    "3in_natl"      NUMERIC,
    nat_1st         INT,
    nat_2nd         INT,
    nat_3rd         INT,
    nat_starts      INT,
    loc_1st         INT,
    loc_2nd         INT,
    loc_3rd         INT,
    loc_starts      INT,
    motor_no        INT,
    motor_2in       NUMERIC,
    motor_3in       NUMERIC,
    mot_1st         INT,
    mot_2nd         INT,
    mot_3rd         INT,
    mot_starts      INT,
    boat_no_hw      INT,
    boat_2in        NUMERIC,
    boat_3in        NUMERIC,
    boa_1st         INT,
    boa_2nd         INT,
    boa_3rd         INT,
    boa_starts      INT,
    source_file     TEXT
);

CREATE TABLE IF NOT EXISTS raw.odds3t (
    stadium        TEXT NOT NULL DEFAULT '若松',
    race_date      DATE NOT NULL,
    race_no        INT  NOT NULL,
    first_lane     INT  NOT NULL,
    second_lane    INT  NOT NULL,
    third_lane     INT  NOT NULL,
    odds           NUMERIC,
    source_file    TEXT
);
