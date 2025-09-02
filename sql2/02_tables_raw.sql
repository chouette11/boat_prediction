CREATE SCHEMA IF NOT EXISTS raw;
/*------------------------------------------------------------
  02_tables_raw.sql
  RAW レイヤのテーブル + STAGING テーブル
------------------------------------------------------------*/

/*--- STAGING (取り込み中間テーブル) --------------------------*/

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

/*------------------------------------------------------------
  02_tables_raw.sql 追加変更（結果TXT取り込み強化）
  - results / results_staging の拡張
  - レース情報テーブル（race_info / staging）
  - 払戻・オッズの統合テーブル（payouts / staging）
  - 既存odds3tは互換のため残置
------------------------------------------------------------*/

/*--- 既存 STAGING テーブルの拡張 ---------------------------*/
ALTER TABLE IF EXISTS raw.results_staging
    ADD COLUMN IF NOT EXISTS motor_no      INT,
    ADD COLUMN IF NOT EXISTS boat_no       INT,
    ADD COLUMN IF NOT EXISTS exh_time      NUMERIC;

/*--- レース情報 STAGING ------------------------------------*/
CREATE TABLE IF NOT EXISTS raw.race_info_staging (
    stadium        TEXT,
    race_date      DATE,
    race_no        INT,
    meeting_title  TEXT,
    day_label      TEXT,
    category_label TEXT,         -- 例: 一般, 準優勝戦, 進入固定 等
    distance_m     INT,
    weather        TEXT,         -- 晴, 雨 など
    wind_dir       TEXT,         -- 北西 など（方位をそのまま格納）
    wind_speed_m   INT,          -- 風速(m)
    wave_cm        INT,          -- 波高(cm)
    source_file    TEXT
);

/*--- 払戻 STAGING（ベット種別を1テーブルに統合） ----------*/
CREATE TABLE IF NOT EXISTS raw.payouts_staging (
    stadium        TEXT NOT NULL,
    race_date      DATE NOT NULL,
    race_no        INT  NOT NULL,
    bet_type       TEXT NOT NULL,      -- WIN, PLACE, EXACTA, QUINELLA, WIDE, TRIFECTA, TRIO
    first_lane     INT,
    second_lane    INT,
    third_lane     INT,
    odds           NUMERIC,
    popularity     INT,                -- 「人気」欄
    source_file    TEXT
);

/*--- 本テーブルの拡張：results に項目追加 ------------------*/
ALTER TABLE IF EXISTS raw.results
    ADD COLUMN IF NOT EXISTS racer_name   TEXT,
    ADD COLUMN IF NOT EXISTS motor_no     INT,
    ADD COLUMN IF NOT EXISTS boat_no      INT,
    ADD COLUMN IF NOT EXISTS exh_time     NUMERIC,
    ADD COLUMN IF NOT EXISTS race_time    TEXT,   -- 例: 1.49.2 / 欠場は'. .'
    ADD COLUMN IF NOT EXISTS tactic       TEXT;   -- 例: 逃げ, まくり差し 等（基本1着のみ）

/*--- レース情報（本テーブル） ------------------------------*/
CREATE TABLE IF NOT EXISTS raw.race_info (
    stadium        TEXT NOT NULL,
    race_date      DATE NOT NULL,
    race_no        INT  NOT NULL,
    meeting_title  TEXT,
    day_label      TEXT,
    category_label TEXT,
    distance_m     INT,
    weather        TEXT,
    wind_dir       TEXT,
    wind_speed_m   INT,
    wave_cm        INT,
    source_file    TEXT,
    PRIMARY KEY (stadium, race_date, race_no)
);

/*--- 払戻（本テーブル） ------------------------------------*/
CREATE TABLE IF NOT EXISTS raw.payouts (
    stadium        TEXT NOT NULL,
    race_date      DATE NOT NULL,
    race_no        INT  NOT NULL,
    bet_type       TEXT NOT NULL,      -- WIN, PLACE, EXACTA, QUINELLA, WIDE, TRIFECTA, TRIO
    first_lane     INT,
    second_lane    INT,
    third_lane     INT,
    odds           NUMERIC,
    popularity     INT,
    source_file    TEXT,
    PRIMARY KEY (stadium, race_date, race_no, bet_type, first_lane, second_lane, third_lane)
);

/*--- 参考：既存3連単テーブル用のインデックス ---------------*/
CREATE INDEX IF NOT EXISTS idx_raw_odds3t_key
    ON raw.odds3t (stadium, race_date, race_no, first_lane, second_lane, third_lane);

/*--- 推奨インデックス --------------------------------------*/
CREATE INDEX IF NOT EXISTS idx_raw_results_key
    ON raw.results (stadium, race_date, race_no, lane);

CREATE INDEX IF NOT EXISTS idx_raw_payouts_type
    ON raw.payouts (stadium, race_date, race_no, bet_type);
