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

/*--- STAGING (レーサー統計テーブル) --------------------------*/
CREATE TABLE IF NOT EXISTS raw.racertech_staging (
    reg_no          INT,
    name            TEXT,
    course_label    TEXT,
    starts          INT,
    firsts          INT,
    nige            INT,
    sashi           INT,
    makuri          INT,
    makurisashi     INT,
    nuki            INT,
    megumare        INT,
    source_file     TEXT
);

CREATE TABLE IF NOT EXISTS raw.racerstadium_staging (
    reg_no          INT,
    name            TEXT,
    stadium_label   TEXT,
    meeting_entries INT,
    starts          INT,
    firsts          INT,
    winrate         NUMERIC,
    first_rate      NUMERIC,
    two_rate        NUMERIC,
    three_rate      NUMERIC,
    finalist_cnt    INT,
    champion_cnt    INT,
    avg_st          NUMERIC,
    source_file     TEXT
);

CREATE TABLE IF NOT EXISTS raw.racerresult1_staging (
    reg_no          INT,
    name            TEXT,
    grade           TEXT,
    meeting_entries INT,
    starts          INT,
    firsts          INT,
    winrate         NUMERIC,
    first_rate      NUMERIC,
    two_rate        NUMERIC,
    three_rate      NUMERIC,
    finalist_cnt    INT,
    champion_cnt    INT,
    avg_st          NUMERIC,
    avg_st_rank     NUMERIC,
    source_file     TEXT
);

CREATE TABLE IF NOT EXISTS raw.racerresult2_staging (
    reg_no          INT,
    name            TEXT,
    grade           TEXT,
    starts          INT,
    firsts          INT,
    seconds         INT,
    thirds          INT,
    fourths         INT,
    fifths          INT,
    sixths          INT,
    s0              INT,
    s1              INT,
    s2              INT,
    f_cnt           INT,
    l0              INT,
    l1              INT,
    k0              INT,
    k1              INT,
    source_file     TEXT
);

CREATE TABLE IF NOT EXISTS raw.racercourse_staging (
    reg_no          INT,
    name            TEXT,
    course_label    TEXT,
    starts          INT,
    firsts          INT,
    first_rate      NUMERIC,
    two_rate        NUMERIC,
    three_rate      NUMERIC,
    avg_st          NUMERIC,
    avg_st_rank     NUMERIC,
    source_file     TEXT
);

CREATE TABLE IF NOT EXISTS raw.racerboatcourse_staging (
    reg_no          INT,
    name            TEXT,
    boat_no_label   TEXT,
    starts          INT,
    lane1_cnt       INT,
    lane2_cnt       INT,
    lane3_cnt       INT,
    lane4_cnt       INT,
    lane5_cnt       INT,
    lane6_cnt       INT,
    other_cnt       INT,
    source_file     TEXT
);

CREATE TABLE IF NOT EXISTS raw.racerboat_staging (
    reg_no          INT,
    name            TEXT,
    boat_no_label   TEXT,
    starts          INT,
    firsts          INT,
    first_rate      NUMERIC,
    two_rate        NUMERIC,
    three_rate      NUMERIC,
    finalist_cnt    INT,
    champion_cnt    INT,
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

/*--- 本テーブル（レーサー統計） ----------------------------*/
CREATE TABLE IF NOT EXISTS raw.racertech (
    reg_no          INT,
    course          INT,
    starts          INT,
    firsts          INT,
    nige            INT,
    sashi           INT,
    makuri          INT,
    makurisashi     INT,
    nuki            INT,
    megumare        INT
);

CREATE TABLE IF NOT EXISTS raw.racerstadium (
    reg_no          INT,
    stadium_code    TEXT,
    meeting_entries INT,
    starts          INT,
    firsts          INT,
    winrate         NUMERIC,
    first_rate      NUMERIC,
    two_rate        NUMERIC,
    three_rate      NUMERIC,
    finalist_cnt    INT,
    champion_cnt    INT,
    avg_st          NUMERIC
);

CREATE TABLE IF NOT EXISTS raw.racerresult1 (
    reg_no          INT,
    grade           TEXT,
    meeting_entries INT,
    starts          INT,
    firsts          INT,
    winrate         NUMERIC,
    first_rate      NUMERIC,
    two_rate        NUMERIC,
    three_rate      NUMERIC,
    finalist_cnt    INT,
    champion_cnt    INT,
    avg_st          NUMERIC,
    avg_st_rank     NUMERIC
);

CREATE TABLE IF NOT EXISTS raw.racerresult2 (
    reg_no          INT,
    grade           TEXT,
    starts          INT,
    firsts          INT,
    seconds         INT,
    thirds          INT,
    fourths         INT,
    fifths          INT,
    sixths          INT,
    s0              INT,
    s1              INT,
    s2              INT,
    f_cnt           INT,
    l0              INT,
    l1              INT,
    k0              INT,
    k1              INT
);

CREATE TABLE IF NOT EXISTS raw.racercourse (
    reg_no          INT,
    course          INT,
    starts          INT,
    firsts          INT,
    first_rate      NUMERIC,
    two_rate        NUMERIC,
    three_rate      NUMERIC,
    avg_st          NUMERIC,
    avg_st_rank     NUMERIC
);

CREATE TABLE IF NOT EXISTS raw.racerboatcourse (
    reg_no          INT,
    lane            INT,
    starts          INT,
    lane1_cnt       INT,
    lane2_cnt       INT,
    lane3_cnt       INT,
    lane4_cnt       INT,
    lane5_cnt       INT,
    lane6_cnt       INT,
    other_cnt       INT
);

CREATE TABLE IF NOT EXISTS raw.racerboat (
    reg_no          INT,
    lane            INT,
    starts          INT,
    firsts          INT,
    first_rate      NUMERIC,
    two_rate        NUMERIC,
    three_rate      NUMERIC,
    finalist_cnt    INT,
    champion_cnt    INT
);
