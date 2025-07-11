/* ---------- ① スキーマ ---------- */
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS feat;

/* ---------- ② RAW テーブル ---------- */
CREATE TABLE IF NOT EXISTS raw.results_staging (
    position      TEXT,
    lane          INT,
    racer_no      INT,
    racer_name    TEXT,
    arrival_time  TEXT,
    st_entry      INT,
    st_time_raw   TEXT,
    tactic        TEXT,
    stadium       TEXT,
    race_title    TEXT,
    date_label    TEXT,
    race_no       INT,
    source_file   TEXT
);

CREATE TABLE IF NOT EXISTS raw.racers_staging (
    lane              INT,
    racer_id          INT,
    name              TEXT,
    weight            TEXT,
    adjust_weight     NUMERIC,
    exhibition_time   NUMERIC,
    tilt              NUMERIC,
    photo             TEXT,
    source_file       TEXT
);

CREATE TABLE IF NOT EXISTS raw.start_exhibition_staging (
    lane       INT,
    st_raw     TEXT,          -- .03 / F.09 など
    source_file TEXT          -- ファイル名を COPY 時に埋め込む
);

CREATE TABLE IF NOT EXISTS raw.weather_staging (
    obs_datetime_label TEXT,
    weather            TEXT,
    air_temp_C         TEXT,
    wind_speed_m       TEXT,
    water_temp_C       TEXT,
    wave_height_cm     TEXT,
    wind_dir_icon      TEXT,
    source_file        TEXT          -- COPY 後に埋めても良い
);

CREATE TABLE IF NOT EXISTS raw.results (
    stadium        TEXT,
    race_date      DATE,
    race_no        INT,
    lane           INT,
    position_txt   TEXT,
    racer_no       INT,
    st_time_raw    TEXT,
    source_file    TEXT
);

CREATE TABLE IF NOT EXISTS raw.racers (
    race_date  DATE,
    race_no    INT,
    lane       INT,
    racer_id   INT,
    weight_raw TEXT,
    exh_time   NUMERIC,
    tilt_deg   NUMERIC
);

ALTER TABLE raw.racers
ADD COLUMN IF NOT EXISTS adjust_weight NUMERIC;

CREATE TABLE IF NOT EXISTS raw.start_exhibition (
    race_date  DATE,
    race_no    INT,
    lane       INT,
    st_raw     TEXT
);

CREATE TABLE IF NOT EXISTS raw.weather (
    race_date         DATE,
    race_no           INT,
    obs_time_label    TEXT,
    weather_txt       TEXT,
    air_temp_raw      TEXT,
    wind_speed_raw    TEXT,
    water_temp_raw    TEXT,
    wave_height_raw   TEXT
);

/* ---------- ③ 便利関数 ---------- */
CREATE OR REPLACE FUNCTION core.f_race_key(d DATE, n INT, s TEXT)
RETURNS TEXT LANGUAGE sql IMMUTABLE AS
$$ SELECT CONCAT(s, '_', d, '_', n) $$;

/* ---------- ④ CORE マテビュー ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS core.races AS
SELECT DISTINCT
       stadium,
       race_date,
       race_no,
       core.f_race_key(race_date, race_no, stadium) AS race_key
FROM raw.results;

/* 各艇の結果（rank 変換付き） */
CREATE MATERIALIZED VIEW IF NOT EXISTS core.results AS
SELECT DISTINCT ON (race_key, lane)
       core.f_race_key(race_date, race_no, stadium) AS race_key,
       lane,
       CASE position_txt
            WHEN '１' THEN 1 WHEN '２' THEN 2 WHEN '３' THEN 3
            WHEN '４' THEN 4 WHEN '５' THEN 5 WHEN '６' THEN 6
            ELSE 7          -- 失格／転覆など
       END AS rank
FROM raw.results
ORDER BY race_key, lane, position_txt;  -- “公式順位行” が先に来るよう並べる


/* 前検情報 */
CREATE MATERIALIZED VIEW IF NOT EXISTS core.boat_info AS
SELECT  core.f_race_key(race_date, race_no, '若松') AS race_key,
        lane,
        racer_id,
        CAST(regexp_replace(weight_raw, '[^0-9.]', '', 'g') AS NUMERIC) AS weight,
        exh_time,
        tilt_deg
FROM raw.racers;

/* スタート展示 */
CREATE MATERIALIZED VIEW IF NOT EXISTS core.start_exh AS
SELECT  core.f_race_key(race_date, race_no, '若松') AS race_key,
        lane,
        (substr(st_raw,1,1) = 'F')                          AS fs_flag,
        (
          NULLIF(regexp_replace(st_raw, '[^0-9.]', '', 'g'), '')::NUMERIC
          / 100          --  ← ここまで計算してから
        )::NUMERIC(4,2)  --  ← キャスト
        AS st_time
FROM raw.start_exhibition;

/* 天候 */
CREATE MATERIALIZED VIEW IF NOT EXISTS core.weather AS
SELECT  core.f_race_key(race_date, race_no, '若松') AS race_key,
        NULLIF(regexp_replace(air_temp_raw   ,'[^0-9.]','','g'), '')::NUMERIC AS air_temp,
        NULLIF(regexp_replace(wind_speed_raw ,'[^0-9.]','','g'), '')::NUMERIC AS wind_speed,
        NULLIF(regexp_replace(wave_height_raw,'[^0-9.]','','g'), '')::NUMERIC AS wave_height,
        NULLIF(regexp_replace(water_temp_raw ,'[^0-9.]','','g'), '')::NUMERIC AS water_temp,
        weather_txt
FROM raw.weather;

/* ---------- ⑤ FEAT マテビュー ---------- */
/* 縦持ちフラット (デバッグ用) */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.boat_flat AS
SELECT b.race_key,
       b.lane,
       b.racer_id,
       w.air_temp, w.wind_speed, w.wave_height, w.water_temp, w.weather_txt,
       s.st_time, s.fs_flag,
       b.weight, b.exh_time, b.tilt_deg,
       r.rank,
       core.races.race_date                 -- 解析用に日付も保持
FROM   core.boat_info  b
JOIN   core.weather    w USING (race_key)
JOIN   core.start_exh  s USING (race_key, lane)
JOIN   core.results    r USING (race_key, lane)
JOIN   core.races      USING (race_key);

/* 横持ち最終ビュー (CPL-Net 直食い) */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.train_features AS
WITH flat AS (SELECT * FROM feat.boat_flat)
SELECT
    MAX(race_date)      AS race_date,
    race_key,

    /* --- レース共通コンテキスト (単一値なので MAX) --- */
    MAX(air_temp)         AS air_temp,
    MAX(wind_speed)       AS wind_speed,
    MAX(wave_height)      AS wave_height,
    MAX(water_temp)       AS water_temp,
    MAX(weather_txt)      AS weather_txt,

    /* --- 6艇×主要特徴 --- */
    /* lane1_*** */
    MAX(CASE WHEN lane=1 THEN racer_id   END) AS lane1_racer_id,
    MAX(CASE WHEN lane=1 THEN weight     END) AS lane1_weight,
    MAX(CASE WHEN lane=1 THEN exh_time   END) AS lane1_exh_time,
    MAX(CASE WHEN lane=1 THEN st_time    END) AS lane1_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=1) AS lane1_fs_flag,
    MAX(CASE WHEN lane=1 THEN rank       END) AS lane1_rank,

    /* lane2_*** */
    MAX(CASE WHEN lane=2 THEN racer_id END)  AS lane2_racer_id,
    MAX(CASE WHEN lane=2 THEN weight   END)  AS lane2_weight,
    MAX(CASE WHEN lane=2 THEN exh_time END)  AS lane2_exh_time,
    MAX(CASE WHEN lane=2 THEN st_time  END)  AS lane2_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=2) AS lane2_fs_flag,
    MAX(CASE WHEN lane=2 THEN rank     END)  AS lane2_rank,

    /* … lane3～6 同様に展開 … */
    MAX(CASE WHEN lane=3 THEN racer_id END)  AS lane3_racer_id,
    MAX(CASE WHEN lane=3 THEN weight   END)  AS lane3_weight,
    MAX(CASE WHEN lane=3 THEN exh_time END)  AS lane3_exh_time,
    MAX(CASE WHEN lane=3 THEN st_time  END)  AS lane3_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=3) AS lane3_fs_flag,
    MAX(CASE WHEN lane=3 THEN rank     END)  AS lane3_rank,

    MAX(CASE WHEN lane=4 THEN racer_id END)  AS lane4_racer_id,
    MAX(CASE WHEN lane=4 THEN weight   END)  AS lane4_weight,
    MAX(CASE WHEN lane=4 THEN exh_time END)  AS lane4_exh_time,
    MAX(CASE WHEN lane=4 THEN st_time  END)  AS lane4_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=4) AS lane4_fs_flag,
    MAX(CASE WHEN lane=4 THEN rank     END)  AS lane4_rank,

    MAX(CASE WHEN lane=5 THEN racer_id END)  AS lane5_racer_id,
    MAX(CASE WHEN lane=5 THEN weight   END)  AS lane5_weight,
    MAX(CASE WHEN lane=5 THEN exh_time END)  AS lane5_exh_time,
    MAX(CASE WHEN lane=5 THEN st_time  END)  AS lane5_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=5) AS lane5_fs_flag,
    MAX(CASE WHEN lane=5 THEN rank     END)  AS lane5_rank,

    MAX(CASE WHEN lane=6 THEN racer_id END)  AS lane6_racer_id,
    MAX(CASE WHEN lane=6 THEN weight   END)  AS lane6_weight,
    MAX(CASE WHEN lane=6 THEN exh_time END)  AS lane6_exh_time,
    MAX(CASE WHEN lane=6 THEN st_time  END)  AS lane6_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=6) AS lane6_fs_flag,
    MAX(CASE WHEN lane=6 THEN rank     END)  AS lane6_rank

FROM flat
GROUP BY race_key;
/* ---------- ⑥ 一意インデックス (CONCURRENTLY 用) ---------- */
CREATE UNIQUE INDEX IF NOT EXISTS idx_core_races_pk  ON core.races        (race_key);
CREATE UNIQUE INDEX IF NOT EXISTS idx_core_results_pk ON core.results     (race_key, lane);
CREATE UNIQUE INDEX IF NOT EXISTS idx_core_boatinfo_pk ON core.boat_info  (race_key, lane);
CREATE UNIQUE INDEX IF NOT EXISTS idx_start_exh_pk    ON core.start_exh   (race_key, lane);
CREATE UNIQUE INDEX IF NOT EXISTS idx_weather_pk      ON core.weather     (race_key);
CREATE UNIQUE INDEX IF NOT EXISTS idx_boat_flat_pk    ON feat.boat_flat   (race_key, lane);
CREATE UNIQUE INDEX IF NOT EXISTS idx_train_feat_pk   ON feat.train_features (race_key);

