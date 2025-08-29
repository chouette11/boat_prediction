CREATE SCHEMA IF NOT EXISTS feat;

-- Session tuning for faster REFRESH and JOINs
SET work_mem = '256MB';
SET maintenance_work_mem = '512MB';
SET max_parallel_workers_per_gather = 4;
SET parallel_leader_participation = on;


-- Bottleneck profiling: use EXPLAIN ANALYZE to identify slow steps
-- Example: run this in psql to analyze the train_features3 view
-- EXPLAIN (ANALYZE, BUFFERS)
-- SELECT *
--   FROM feat.train_features3
--  WHERE race_date <= CURRENT_DATE;

CREATE MATERIALIZED VIEW IF NOT EXISTS feat.boat_flat AS
SELECT DISTINCT ON (b.race_key, b.lane)
       b.race_key,
       b.lane,
       b.racer_id,
       w.air_temp,
       w.wind_speed,
       w.wave_height,
       w.water_temp,
       w.weather_txt,
       w.wind_dir_deg,
       b.bf_st_time,
       b.bf_course,
       b.st_time,
       b.course,
       b.fs_flag,
       b.weight,
       b.exh_time,
       b.tilt_deg,
       r.rank,
       r.win_pattern,
       cr.race_date
FROM core.boat_info b
JOIN core.weather  w  USING (race_key)
JOIN core.results  r  USING (race_key, lane)
JOIN core.races    cr USING (race_key)
ORDER BY b.race_key, b.lane;

/* ---------- filtered_course マテリアライズドビュー定義を追加 ---------- */
DROP MATERIALIZED VIEW IF EXISTS feat.filtered_course CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.filtered_course AS
SELECT
    b.racer_id AS reg_no,
    b.course,
    COUNT(*) AS starts,
    SUM(CASE WHEN r.rank = 1 THEN 1 ELSE 0 END) AS firsts,
    SUM(CASE WHEN r.rank = 1 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS first_rate,
    SUM(CASE WHEN r.rank <= 2 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS two_rate,
    SUM(CASE WHEN r.rank <= 3 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS three_rate
FROM core.boat_info b
JOIN core.results r ON b.race_key = r.race_key AND b.lane = r.lane
JOIN core.races cr ON b.race_key = cr.race_key
WHERE cr.stadium = '若 松'
GROUP BY b.racer_id, b.course
WITH NO DATA;

-- Index to speed joins & filters on filtered_course (after MV exists)
CREATE INDEX IF NOT EXISTS idx_feat_filtered_course_reg_course
  ON feat.filtered_course (reg_no, course);

/* ---------- 決まり手×コース別 成績（feat.filtered_course_pat） ---------- */
DROP MATERIALIZED VIEW IF EXISTS feat.filtered_course_pat CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.filtered_course_pat AS
SELECT
    b.racer_id AS reg_no,
    b.course,
    COUNT(*) AS starts,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'NIGE'          THEN 1 ELSE 0 END) AS firsts_nige,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'SASHI'         THEN 1 ELSE 0 END) AS firsts_sashi,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'MAKURI'        THEN 1 ELSE 0 END) AS firsts_makuri,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'MAKURI_SASHI'  THEN 1 ELSE 0 END) AS firsts_makurizashi,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'NUKI'          THEN 1 ELSE 0 END) AS firsts_nuki,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'MEGUMARE'      THEN 1 ELSE 0 END) AS firsts_megumare,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'OTHER'         THEN 1 ELSE 0 END) AS firsts_other,
    -- starts を分母にした「勝利展開率」
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'NIGE'          THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS pat_nige_rate,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'SASHI'         THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS pat_sashi_rate,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'MAKURI'        THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS pat_makuri_rate,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'MAKURI_SASHI'  THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS pat_makurizashi_rate,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'NUKI'          THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS pat_nuki_rate,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'MEGUMARE'      THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS pat_megumare_rate,
    SUM(CASE WHEN r.rank = 1 AND r.win_pattern = 'OTHER'         THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*),0) AS pat_other_rate
FROM core.boat_info b
JOIN core.results r ON b.race_key = r.race_key AND b.lane = r.lane
JOIN core.races   cr ON b.race_key = cr.race_key
WHERE cr.stadium = '若 松'
GROUP BY b.racer_id, b.course
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_feat_filtered_course_pat_reg_course
  ON feat.filtered_course_pat (reg_no, course);

/* ---------- 1号艇の敗北展開率（feat.filtered_lane1_lose_pat） ---------- */
DROP MATERIALIZED VIEW IF EXISTS feat.filtered_lane1_lose_pat CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.filtered_lane1_lose_pat AS
SELECT
    b.racer_id AS reg_no,
    COUNT(*) FILTER (WHERE b.lane = 1 AND r.rank > 1) AS defeats,
    SUM(CASE WHEN b.lane = 1 AND r.rank > 1 AND r.win_pattern = 'SASHI'        THEN 1 ELSE 0 END) AS lose_sashi,
    SUM(CASE WHEN b.lane = 1 AND r.rank > 1 AND r.win_pattern = 'MAKURI'       THEN 1 ELSE 0 END) AS lose_makuri,
    SUM(CASE WHEN b.lane = 1 AND r.rank > 1 AND r.win_pattern = 'MAKURI_SASHI' THEN 1 ELSE 0 END) AS lose_makurizashi,
    SUM(CASE WHEN b.lane = 1 AND r.rank > 1 AND r.win_pattern = 'NUKI'         THEN 1 ELSE 0 END) AS lose_nuki,
    -- 事故・F等（rank=7 も含める）
    SUM(CASE WHEN b.lane = 1 AND (r.rank = 7 OR b.fs_flag)                      THEN 1 ELSE 0 END) AS lose_penalty,
    -- 敗北内訳に対する比率（分母は defeats）
    (SUM(CASE WHEN b.lane = 1 AND r.rank > 1 AND r.win_pattern = 'SASHI'        THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*) FILTER (WHERE b.lane = 1 AND r.rank > 1),0)) AS lose_sashi_rate,
    (SUM(CASE WHEN b.lane = 1 AND r.rank > 1 AND r.win_pattern = 'MAKURI'       THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*) FILTER (WHERE b.lane = 1 AND r.rank > 1),0)) AS lose_makuri_rate,
    (SUM(CASE WHEN b.lane = 1 AND r.rank > 1 AND r.win_pattern = 'MAKURI_SASHI' THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*) FILTER (WHERE b.lane = 1 AND r.rank > 1),0)) AS lose_makurizashi_rate,
    (SUM(CASE WHEN b.lane = 1 AND r.rank > 1 AND r.win_pattern = 'NUKI'         THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*) FILTER (WHERE b.lane = 1 AND r.rank > 1),0)) AS lose_nuki_rate,
    (SUM(CASE WHEN b.lane = 1 AND (r.rank = 7 OR b.fs_flag)                      THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*) FILTER (WHERE b.lane = 1 AND r.rank > 1),0)) AS lose_penalty_rate
FROM core.boat_info b
JOIN core.results r ON b.race_key = r.race_key AND b.lane = r.lane
JOIN core.races   cr ON b.race_key = cr.race_key
WHERE cr.stadium = '若 松'
GROUP BY b.racer_id
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_feat_filtered_lane1_lose_pat_reg
  ON feat.filtered_lane1_lose_pat (reg_no);

/* ---------- 学習用特徴量（feat.train_features） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.train_features2 AS
WITH flat AS (
    SELECT bf.*,
           cr.stadium AS venue
    FROM feat.boat_flat bf
    JOIN core.races cr USING (race_key)
)
SELECT
    race_key,
    MAX(race_date)   AS race_date,
    MAX(venue)       AS venue,
    MAX(win_pattern) AS win_pattern,
    MAX(air_temp)    AS air_temp,
    MAX(wind_speed)  AS wind_speed,
    MAX(wave_height) AS wave_height,
    MAX(water_temp)  AS water_temp,
    MAX(weather_txt) AS weather_txt,
    MAX(wind_dir_deg) AS wind_dir_deg,
    MAX(CASE WHEN lane=1 THEN racer_id END) AS lane1_racer_id,
    MAX(CASE WHEN lane=1 THEN weight END) AS lane1_weight,
    MAX(CASE WHEN lane=1 THEN exh_time END) AS lane1_exh_time,
    MAX(CASE WHEN lane=1 THEN bf_st_time END) AS lane1_bf_st_time,
    MAX(CASE WHEN lane=1 THEN bf_course END) AS lane1_bf_course,
    MAX(CASE WHEN lane=1 THEN st_time END) AS lane1_st,
    MAX(CASE WHEN lane=1 THEN course END) AS lane1_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=1)  AS lane1_fs_flag,
    MAX(CASE WHEN lane=1 THEN rank END) AS lane1_rank,
    MAX(CASE WHEN lane=2 THEN racer_id END) AS lane2_racer_id,
    MAX(CASE WHEN lane=2 THEN weight END) AS lane2_weight,
    MAX(CASE WHEN lane=2 THEN exh_time END) AS lane2_exh_time,
    MAX(CASE WHEN lane=2 THEN bf_st_time END) AS lane2_bf_st_time,
    MAX(CASE WHEN lane=2 THEN bf_course END) AS lane2_bf_course,
    MAX(CASE WHEN lane=2 THEN st_time END) AS lane2_st,
    MAX(CASE WHEN lane=2 THEN course END) AS lane2_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=2)  AS lane2_fs_flag,
    MAX(CASE WHEN lane=2 THEN rank END) AS lane2_rank,
    MAX(CASE WHEN lane=3 THEN racer_id END) AS lane3_racer_id,
    MAX(CASE WHEN lane=3 THEN weight END) AS lane3_weight,
    MAX(CASE WHEN lane=3 THEN exh_time END) AS lane3_exh_time,
    MAX(CASE WHEN lane=3 THEN bf_st_time END) AS lane3_bf_st_time,
    MAX(CASE WHEN lane=3 THEN bf_course END) AS lane3_bf_course,
    MAX(CASE WHEN lane=3 THEN st_time END) AS lane3_st,
    MAX(CASE WHEN lane=3 THEN course END) AS lane3_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=3)  AS lane3_fs_flag,
    MAX(CASE WHEN lane=3 THEN rank END) AS lane3_rank,
    MAX(CASE WHEN lane=4 THEN racer_id END) AS lane4_racer_id,
    MAX(CASE WHEN lane=4 THEN weight END) AS lane4_weight,
    MAX(CASE WHEN lane=4 THEN exh_time END) AS lane4_exh_time,
    MAX(CASE WHEN lane=4 THEN bf_st_time END) AS lane4_bf_st_time,
    MAX(CASE WHEN lane=4 THEN bf_course END) AS lane4_bf_course,
    MAX(CASE WHEN lane=4 THEN st_time END) AS lane4_st,
    MAX(CASE WHEN lane=4 THEN course END) AS lane4_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=4)  AS lane4_fs_flag,
    MAX(CASE WHEN lane=4 THEN rank END) AS lane4_rank,
    MAX(CASE WHEN lane=5 THEN racer_id END) AS lane5_racer_id,
    MAX(CASE WHEN lane=5 THEN weight END) AS lane5_weight,
    MAX(CASE WHEN lane=5 THEN exh_time END) AS lane5_exh_time,
    MAX(CASE WHEN lane=5 THEN bf_st_time END) AS lane5_bf_st_time,
    MAX(CASE WHEN lane=5 THEN bf_course END) AS lane5_bf_course,
    MAX(CASE WHEN lane=5 THEN st_time END) AS lane5_st,
    MAX(CASE WHEN lane=5 THEN course END) AS lane5_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=5)  AS lane5_fs_flag,
    MAX(CASE WHEN lane=5 THEN rank END) AS lane5_rank,
    MAX(CASE WHEN lane=6 THEN racer_id END) AS lane6_racer_id,
    MAX(CASE WHEN lane=6 THEN weight END) AS lane6_weight,
    MAX(CASE WHEN lane=6 THEN exh_time END) AS lane6_exh_time,
    MAX(CASE WHEN lane=6 THEN bf_st_time END) AS lane6_bf_st_time,
    MAX(CASE WHEN lane=6 THEN bf_course END) AS lane6_bf_course,
    MAX(CASE WHEN lane=6 THEN st_time END) AS lane6_st,
    MAX(CASE WHEN lane=6 THEN course END) AS lane6_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=6)  AS lane6_fs_flag,
    MAX(CASE WHEN lane=6 THEN rank END) AS lane6_rank

FROM flat
GROUP BY race_key
HAVING COUNT(DISTINCT race_date)=1
   AND COUNT(DISTINCT venue)=1;

CREATE MATERIALIZED VIEW IF NOT EXISTS feat.tf2_lane_stats AS
WITH tf2_long AS (
  SELECT race_key, 1 AS lane_no, lane1_racer_id AS reg_no, lane1_course AS course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 2, lane2_racer_id, lane2_course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 3, lane3_racer_id, lane3_course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 4, lane4_racer_id, lane4_course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 5, lane5_racer_id, lane5_course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 6, lane6_racer_id, lane6_course FROM feat.train_features2
)
SELECT
    l.race_key,
    MAX(fc.starts)      FILTER (WHERE l.lane_no = 1) AS lane1_starts,
    MAX(fc.firsts)      FILTER (WHERE l.lane_no = 1) AS lane1_firsts,
    MAX(fc.first_rate)  FILTER (WHERE l.lane_no = 1) AS lane1_first_rate,
    MAX(fc.two_rate)    FILTER (WHERE l.lane_no = 1) AS lane1_two_rate,
    MAX(fc.three_rate)  FILTER (WHERE l.lane_no = 1) AS lane1_three_rate,
    MAX(fc.starts)      FILTER (WHERE l.lane_no = 2) AS lane2_starts,
    MAX(fc.firsts)      FILTER (WHERE l.lane_no = 2) AS lane2_firsts,
    MAX(fc.first_rate)  FILTER (WHERE l.lane_no = 2) AS lane2_first_rate,
    MAX(fc.two_rate)    FILTER (WHERE l.lane_no = 2) AS lane2_two_rate,
    MAX(fc.three_rate)  FILTER (WHERE l.lane_no = 2) AS lane2_three_rate,
    MAX(fc.starts)      FILTER (WHERE l.lane_no = 3) AS lane3_starts,
    MAX(fc.firsts)      FILTER (WHERE l.lane_no = 3) AS lane3_firsts,
    MAX(fc.first_rate)  FILTER (WHERE l.lane_no = 3) AS lane3_first_rate,
    MAX(fc.two_rate)    FILTER (WHERE l.lane_no = 3) AS lane3_two_rate,
    MAX(fc.three_rate)  FILTER (WHERE l.lane_no = 3) AS lane3_three_rate,
    MAX(fc.starts)      FILTER (WHERE l.lane_no = 4) AS lane4_starts,
    MAX(fc.firsts)      FILTER (WHERE l.lane_no = 4) AS lane4_firsts,
    MAX(fc.first_rate)  FILTER (WHERE l.lane_no = 4) AS lane4_first_rate,
    MAX(fc.two_rate)    FILTER (WHERE l.lane_no = 4) AS lane4_two_rate,
    MAX(fc.three_rate)  FILTER (WHERE l.lane_no = 4) AS lane4_three_rate,
    MAX(fc.starts)      FILTER (WHERE l.lane_no = 5) AS lane5_starts,
    MAX(fc.firsts)      FILTER (WHERE l.lane_no = 5) AS lane5_firsts,
    MAX(fc.first_rate)  FILTER (WHERE l.lane_no = 5) AS lane5_first_rate,
    MAX(fc.two_rate)    FILTER (WHERE l.lane_no = 5) AS lane5_two_rate,
    MAX(fc.three_rate)  FILTER (WHERE l.lane_no = 5) AS lane5_three_rate,
    MAX(fc.starts)      FILTER (WHERE l.lane_no = 6) AS lane6_starts,
    MAX(fc.firsts)      FILTER (WHERE l.lane_no = 6) AS lane6_firsts,
    MAX(fc.first_rate)  FILTER (WHERE l.lane_no = 6) AS lane6_first_rate,
    MAX(fc.two_rate)    FILTER (WHERE l.lane_no = 6) AS lane6_two_rate,
    MAX(fc.three_rate)  FILTER (WHERE l.lane_no = 6) AS lane6_three_rate
FROM tf2_long l
LEFT JOIN feat.filtered_course fc
  ON fc.reg_no = l.reg_no
 AND fc.course = l.course
GROUP BY l.race_key
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS feat.tf2_lane_pat_stats AS
WITH tf2_long AS (
  SELECT race_key, 1 AS lane_no, lane1_racer_id AS reg_no, lane1_course AS course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 2, lane2_racer_id, lane2_course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 3, lane3_racer_id, lane3_course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 4, lane4_racer_id, lane4_course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 5, lane5_racer_id, lane5_course FROM feat.train_features2
  UNION ALL
  SELECT race_key, 6, lane6_racer_id, lane6_course FROM feat.train_features2
)
SELECT
    l.race_key,
    MAX(fcpat.pat_nige_rate)         FILTER (WHERE l.lane_no = 1) AS lane1_pat_nige_rate,
    MAX(fcpat.pat_sashi_rate)        FILTER (WHERE l.lane_no = 1) AS lane1_pat_sashi_rate,
    MAX(fcpat.pat_makuri_rate)       FILTER (WHERE l.lane_no = 1) AS lane1_pat_makuri_rate,
    MAX(fcpat.pat_makurizashi_rate)  FILTER (WHERE l.lane_no = 1) AS lane1_pat_makurizashi_rate,
    MAX(fcpat.pat_nuki_rate)         FILTER (WHERE l.lane_no = 1) AS lane1_pat_nuki_rate,
    MAX(fcpat.pat_megumare_rate)     FILTER (WHERE l.lane_no = 1) AS lane1_pat_megumare_rate,
    MAX(fcpat.pat_other_rate)        FILTER (WHERE l.lane_no = 1) AS lane1_pat_other_rate,

    MAX(fcpat.pat_nige_rate)         FILTER (WHERE l.lane_no = 2) AS lane2_pat_nige_rate,
    MAX(fcpat.pat_sashi_rate)        FILTER (WHERE l.lane_no = 2) AS lane2_pat_sashi_rate,
    MAX(fcpat.pat_makuri_rate)       FILTER (WHERE l.lane_no = 2) AS lane2_pat_makuri_rate,
    MAX(fcpat.pat_makurizashi_rate)  FILTER (WHERE l.lane_no = 2) AS lane2_pat_makurizashi_rate,
    MAX(fcpat.pat_nuki_rate)         FILTER (WHERE l.lane_no = 2) AS lane2_pat_nuki_rate,
    MAX(fcpat.pat_megumare_rate)     FILTER (WHERE l.lane_no = 2) AS lane2_pat_megumare_rate,
    MAX(fcpat.pat_other_rate)        FILTER (WHERE l.lane_no = 2) AS lane2_pat_other_rate,

    MAX(fcpat.pat_nige_rate)         FILTER (WHERE l.lane_no = 3) AS lane3_pat_nige_rate,
    MAX(fcpat.pat_sashi_rate)        FILTER (WHERE l.lane_no = 3) AS lane3_pat_sashi_rate,
    MAX(fcpat.pat_makuri_rate)       FILTER (WHERE l.lane_no = 3) AS lane3_pat_makuri_rate,
    MAX(fcpat.pat_makurizashi_rate)  FILTER (WHERE l.lane_no = 3) AS lane3_pat_makurizashi_rate,
    MAX(fcpat.pat_nuki_rate)         FILTER (WHERE l.lane_no = 3) AS lane3_pat_nuki_rate,
    MAX(fcpat.pat_megumare_rate)     FILTER (WHERE l.lane_no = 3) AS lane3_pat_megumare_rate,
    MAX(fcpat.pat_other_rate)        FILTER (WHERE l.lane_no = 3) AS lane3_pat_other_rate,

    MAX(fcpat.pat_nige_rate)         FILTER (WHERE l.lane_no = 4) AS lane4_pat_nige_rate,
    MAX(fcpat.pat_sashi_rate)        FILTER (WHERE l.lane_no = 4) AS lane4_pat_sashi_rate,
    MAX(fcpat.pat_makuri_rate)       FILTER (WHERE l.lane_no = 4) AS lane4_pat_makuri_rate,
    MAX(fcpat.pat_makurizashi_rate)  FILTER (WHERE l.lane_no = 4) AS lane4_pat_makurizashi_rate,
    MAX(fcpat.pat_nuki_rate)         FILTER (WHERE l.lane_no = 4) AS lane4_pat_nuki_rate,
    MAX(fcpat.pat_megumare_rate)     FILTER (WHERE l.lane_no = 4) AS lane4_pat_megumare_rate,
    MAX(fcpat.pat_other_rate)        FILTER (WHERE l.lane_no = 4) AS lane4_pat_other_rate,

    MAX(fcpat.pat_nige_rate)         FILTER (WHERE l.lane_no = 5) AS lane5_pat_nige_rate,
    MAX(fcpat.pat_sashi_rate)        FILTER (WHERE l.lane_no = 5) AS lane5_pat_sashi_rate,
    MAX(fcpat.pat_makuri_rate)       FILTER (WHERE l.lane_no = 5) AS lane5_pat_makuri_rate,
    MAX(fcpat.pat_makurizashi_rate)  FILTER (WHERE l.lane_no = 5) AS lane5_pat_makurizashi_rate,
    MAX(fcpat.pat_nuki_rate)         FILTER (WHERE l.lane_no = 5) AS lane5_pat_nuki_rate,
    MAX(fcpat.pat_megumare_rate)     FILTER (WHERE l.lane_no = 5) AS lane5_pat_megumare_rate,
    MAX(fcpat.pat_other_rate)        FILTER (WHERE l.lane_no = 5) AS lane5_pat_other_rate,

    MAX(fcpat.pat_nige_rate)         FILTER (WHERE l.lane_no = 6) AS lane6_pat_nige_rate,
    MAX(fcpat.pat_sashi_rate)        FILTER (WHERE l.lane_no = 6) AS lane6_pat_sashi_rate,
    MAX(fcpat.pat_makuri_rate)       FILTER (WHERE l.lane_no = 6) AS lane6_pat_makuri_rate,
    MAX(fcpat.pat_makurizashi_rate)  FILTER (WHERE l.lane_no = 6) AS lane6_pat_makurizashi_rate,
    MAX(fcpat.pat_nuki_rate)         FILTER (WHERE l.lane_no = 6) AS lane6_pat_nuki_rate,
    MAX(fcpat.pat_megumare_rate)     FILTER (WHERE l.lane_no = 6) AS lane6_pat_megumare_rate,
    MAX(fcpat.pat_other_rate)        FILTER (WHERE l.lane_no = 6) AS lane6_pat_other_rate
FROM tf2_long l
LEFT JOIN feat.filtered_course_pat fcpat
  ON fcpat.reg_no = l.reg_no
 AND fcpat.course = l.course
GROUP BY l.race_key
WITH NO DATA;

/* ---------- レース単位へ貼り付け（feat.tf2_lane1_lose_pat_stats） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.tf2_lane1_lose_pat_stats AS
SELECT
    tf.race_key,
    flp.lose_sashi_rate       AS lane1_lose_sashi_rate,
    flp.lose_makuri_rate      AS lane1_lose_makuri_rate,
    flp.lose_makurizashi_rate AS lane1_lose_makurizashi_rate,
    flp.lose_nuki_rate        AS lane1_lose_nuki_rate,
    flp.lose_penalty_rate     AS lane1_lose_penalty_rate
FROM feat.train_features2 tf
LEFT JOIN feat.filtered_lane1_lose_pat flp
  ON flp.reg_no = tf.lane1_racer_id
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uq_feat_tf2_lane1_lose_pat_stats_race_key
  ON feat.tf2_lane1_lose_pat_stats (race_key);

DROP MATERIALIZED VIEW IF EXISTS feat.train_features3 CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.train_features3 AS
SELECT
    tf.*,
    ls.lane1_starts, ls.lane1_firsts, ls.lane1_first_rate, ls.lane1_two_rate, ls.lane1_three_rate,
    ls.lane2_starts, ls.lane2_firsts, ls.lane2_first_rate, ls.lane2_two_rate, ls.lane2_three_rate,
    ls.lane3_starts, ls.lane3_firsts, ls.lane3_first_rate, ls.lane3_two_rate, ls.lane3_three_rate,
    ls.lane4_starts, ls.lane4_firsts, ls.lane4_first_rate, ls.lane4_two_rate, ls.lane4_three_rate,
    ls.lane5_starts, ls.lane5_firsts, ls.lane5_first_rate, ls.lane5_two_rate, ls.lane5_three_rate,
    ls.lane6_starts, ls.lane6_firsts, ls.lane6_first_rate, ls.lane6_two_rate, ls.lane6_three_rate
    , lps.lane1_pat_nige_rate, lps.lane1_pat_sashi_rate, lps.lane1_pat_makuri_rate, lps.lane1_pat_makurizashi_rate, lps.lane1_pat_nuki_rate, lps.lane1_pat_megumare_rate, lps.lane1_pat_other_rate
    , lps.lane2_pat_nige_rate, lps.lane2_pat_sashi_rate, lps.lane2_pat_makuri_rate, lps.lane2_pat_makurizashi_rate, lps.lane2_pat_nuki_rate, lps.lane2_pat_megumare_rate, lps.lane2_pat_other_rate
    , lps.lane3_pat_nige_rate, lps.lane3_pat_sashi_rate, lps.lane3_pat_makuri_rate, lps.lane3_pat_makurizashi_rate, lps.lane3_pat_nuki_rate, lps.lane3_pat_megumare_rate, lps.lane3_pat_other_rate
    , lps.lane4_pat_nige_rate, lps.lane4_pat_sashi_rate, lps.lane4_pat_makuri_rate, lps.lane4_pat_makurizashi_rate, lps.lane4_pat_nuki_rate, lps.lane4_pat_megumare_rate, lps.lane4_pat_other_rate
    , lps.lane5_pat_nige_rate, lps.lane5_pat_sashi_rate, lps.lane5_pat_makuri_rate, lps.lane5_pat_makurizashi_rate, lps.lane5_pat_nuki_rate, lps.lane5_pat_megumare_rate, lps.lane5_pat_other_rate
    , lps.lane6_pat_nige_rate, lps.lane6_pat_sashi_rate, lps.lane6_pat_makuri_rate, lps.lane6_pat_makurizashi_rate, lps.lane6_pat_nuki_rate, lps.lane6_pat_megumare_rate, lps.lane6_pat_other_rate
    -- 追加: lane1の敗北展開率
    , l1los.lane1_lose_sashi_rate
    , l1los.lane1_lose_makuri_rate
    , l1los.lane1_lose_makurizashi_rate
    , l1los.lane1_lose_nuki_rate
    , l1los.lane1_lose_penalty_rate
FROM feat.train_features2 tf
LEFT JOIN feat.tf2_lane_stats             ls    USING (race_key)
LEFT JOIN feat.tf2_lane_pat_stats         lps   USING (race_key)
LEFT JOIN feat.tf2_lane1_lose_pat_stats   l1los USING (race_key)
WHERE tf.venue = '若 松'
WITH NO DATA;

-- /* ---------- 評価用特徴量（feat.eval_features） ---------- */
DROP MATERIALIZED VIEW IF EXISTS feat.eval_features3 CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.eval_features3 AS
WITH ord AS (
  SELECT
    r.race_key,
    r.lane,
    r.rank,
    ROW_NUMBER() OVER (PARTITION BY r.race_key ORDER BY r.rank, r.lane) AS ord
  FROM core.results r
  WHERE r.rank IS NOT NULL
),
pos AS (
  SELECT
    race_key,
    MIN(lane) FILTER (WHERE ord = 1) AS first_lane,
    MIN(lane) FILTER (WHERE ord = 2) AS second_lane,
    MIN(lane) FILTER (WHERE ord = 3) AS third_lane
  FROM ord
  GROUP BY race_key
)
SELECT
    tf.*,
    pos.first_lane,
    pos.second_lane,
    pos.third_lane,
    (p.payout_yen / 100.0)       AS trifecta_odds,
    p.popularity_rank  AS trifecta_popularity_rank
FROM feat.train_features3 tf
LEFT JOIN pos USING (race_key)
LEFT JOIN core.payout3t p
  ON p.race_key = tf.race_key
 AND p.first_lane = pos.first_lane
 AND p.second_lane = pos.second_lane
 AND p.third_lane  = pos.third_lane
WITH NO DATA;

/* ---------- REFRESH 文 ---------- */
\echo '--- boat_flat 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.boat_flat;
\echo '--- filtered_course 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.filtered_course;
\echo '--- filtered_course_pat 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.filtered_course_pat;
-- REFRESH/ANALYZE for new MVs
\echo '--- train_features2 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.train_features2;
\echo '--- tf2_lane_stats 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.tf2_lane_stats;
\echo '--- tf2_lane_pat_stats 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.tf2_lane_pat_stats;
\echo '--- filtered_lane1_lose_pat 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.filtered_lane1_lose_pat;
\echo '--- tf2_lane1_lose_pat_stats 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.tf2_lane1_lose_pat_stats;
\echo '--- train_features3 層のマテリアライズドビューを更新中 ---'
ANALYZE feat.train_features2;
ANALYZE feat.filtered_course;
ANALYZE feat.filtered_course_pat;
ANALYZE feat.tf2_lane_stats;
ANALYZE feat.tf2_lane_pat_stats;
ANALYZE feat.filtered_lane1_lose_pat;
ANALYZE feat.tf2_lane1_lose_pat_stats;
REFRESH MATERIALIZED VIEW feat.train_features3;
-- Prepare for concurrent refreshes in future runs
CREATE UNIQUE INDEX IF NOT EXISTS idx_tf3_race_key ON feat.train_features3 (race_key);
\echo '--- eval_features3 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.eval_features3;

/* ---------- データ存在チェック ---------- */
\echo '--- feat 層データ存在チェック ---'
SELECT format(
          $$SELECT 'feat.%I' AS view_name,
                       CASE WHEN EXISTS (SELECT 1 FROM feat.%I LIMIT 1)
                            THEN '✔ data' ELSE '✖ empty' END AS has_data;$$,
          matviewname, matviewname
       ) AS cmd
FROM   pg_matviews
WHERE  schemaname = 'feat'
\gexec
\echo '--- feat 層データ存在チェック完了 ---'

-- ===== Indexes created after materialized views exist =====
CREATE INDEX IF NOT EXISTS idx_feat_tf2_lane1
  ON feat.train_features2 (lane1_racer_id, lane1_course);
CREATE INDEX IF NOT EXISTS idx_feat_tf2_lane2
  ON feat.train_features2 (lane2_racer_id, lane2_course);
CREATE INDEX IF NOT EXISTS idx_feat_tf2_lane3
  ON feat.train_features2 (lane3_racer_id, lane3_course);
CREATE INDEX IF NOT EXISTS idx_feat_tf2_lane4
  ON feat.train_features2 (lane4_racer_id, lane4_course);
CREATE INDEX IF NOT EXISTS idx_feat_tf2_lane5
  ON feat.train_features2 (lane5_racer_id, lane5_course);
CREATE INDEX IF NOT EXISTS idx_feat_tf2_lane6
  ON feat.train_features2 (lane6_racer_id, lane6_course);

CREATE INDEX IF NOT EXISTS idx_feat_tf3_race_date
  ON feat.train_features3 (race_date);

CREATE UNIQUE INDEX IF NOT EXISTS uq_feat_tf2_lane_stats_race_key
  ON feat.tf2_lane_stats (race_key);

CREATE INDEX IF NOT EXISTS idx_feat_filtered_course_pat_reg_course
  ON feat.filtered_course_pat (reg_no, course);
CREATE UNIQUE INDEX IF NOT EXISTS uq_feat_tf2_lane_pat_stats_race_key
  ON feat.tf2_lane_pat_stats (race_key);
CREATE INDEX IF NOT EXISTS idx_feat_filtered_lane1_lose_pat_reg
  ON feat.filtered_lane1_lose_pat (reg_no);
CREATE UNIQUE INDEX IF NOT EXISTS uq_feat_tf2_lane1_lose_pat_stats_race_key
  ON feat.tf2_lane1_lose_pat_stats (race_key);
