-- Session tuning for faster REFRESH and JOINs
SET work_mem = '256MB';
SET maintenance_work_mem = '512MB';
SET max_parallel_workers_per_gather = 4;
SET parallel_leader_participation = on;
CREATE INDEX IF NOT EXISTS idx_core_rsc_reg_course
  ON core.racerstats_course (reg_no, course);

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

-- Indexes to speed joins and filters on materialized outputs
CREATE INDEX IF NOT EXISTS idx_feat_filtered_course_reg_course
  ON feat.filtered_course (reg_no, course);

CREATE INDEX IF NOT EXISTS idx_feat_tf3_race_date
  ON feat.train_features3 (race_date);

CREATE INDEX IF NOT EXISTS idx_feat_tf2_lane_stats_race_key
  ON feat.tf2_lane_stats (race_key);
CREATE UNIQUE INDEX IF NOT EXISTS uq_feat_tf2_lane_stats_race_key
  ON feat.tf2_lane_stats (race_key);


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
       b.class_now,
       b.ability_now,
       b.winrate_natl,
       b."2in_natl",
       b."3in_natl",
       b.age,
       b.class_hist1,
       b.class_hist2,
       b.class_hist3,
    --    b.ability_prev,
       b."F_now",
       b."L_now",
       b.nat_1st,
       b.nat_2nd,
       b.nat_3rd,
       b.nat_starts,
       b.loc_1st,
       b.loc_2nd,
       b.loc_3rd,
       b.loc_starts,
       b.motor_no,
       b.motor_2in,
       b.motor_3in,
       b.mot_1st,
       b.mot_2nd,
       b.mot_3rd,
       b.mot_starts,
       b.boat_no_hw,
       b.boat_2in,
       b.boat_3in,
       b.boa_1st,
       b.boa_2nd,
       b.boa_3rd,
       b.boa_starts,
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
       cr.race_date
FROM core.boat_info b
JOIN core.weather  w  USING (race_key)
JOIN core.results  r  USING (race_key, lane)
JOIN core.races    cr USING (race_key)
ORDER BY b.race_key, b.lane;

/* ---------- filtered_course マテリアライズドビュー定義を追加 ---------- */
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
GROUP BY b.racer_id, b.course
WITH NO DATA;

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

CREATE MATERIALIZED VIEW IF NOT EXISTS feat.train_features3 AS
SELECT
    tf.*,
    ls.lane1_starts, ls.lane1_firsts, ls.lane1_first_rate, ls.lane1_two_rate, ls.lane1_three_rate,
    ls.lane2_starts, ls.lane2_firsts, ls.lane2_first_rate, ls.lane2_two_rate, ls.lane2_three_rate,
    ls.lane3_starts, ls.lane3_firsts, ls.lane3_first_rate, ls.lane3_two_rate, ls.lane3_three_rate,
    ls.lane4_starts, ls.lane4_firsts, ls.lane4_first_rate, ls.lane4_two_rate, ls.lane4_three_rate,
    ls.lane5_starts, ls.lane5_firsts, ls.lane5_first_rate, ls.lane5_two_rate, ls.lane5_three_rate,
    ls.lane6_starts, ls.lane6_firsts, ls.lane6_first_rate, ls.lane6_two_rate, ls.lane6_three_rate
FROM feat.train_features2 tf
LEFT JOIN feat.tf2_lane_stats ls USING (race_key)
WITH NO DATA;

/* ---------- 評価用特徴量（feat.eval_features） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.eval_features2 AS
SELECT
    tf.*,
    o.first_lane,
    o.second_lane,
    o.third_lane,
    o.odds
FROM feat.train_features2 tf
JOIN core.odds3t o USING (race_key)
WITH NO DATA;

/* ---------- 評価用特徴量（feat.eval_features） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.eval_features3 AS
SELECT
    tf.*,
    o.first_lane,
    o.second_lane,
    o.third_lane,
    o.odds
FROM feat.train_features3 tf
JOIN core.odds3t o USING (race_key)
WITH NO DATA;

/* ---------- REFRESH 文 ---------- */
\echo '--- boat_flat 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.boat_flat;
\echo '--- filtered_course 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.filtered_course;
\echo '--- train_features2 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.train_features2;
\echo '--- tf2_lane_stats 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.tf2_lane_stats;
\echo '--- train_features3 層のマテリアライズドビューを更新中 ---'
ANALYZE feat.train_features2;
ANALYZE feat.filtered_course;
ANALYZE feat.tf2_lane_stats;
REFRESH MATERIALIZED VIEW feat.train_features3;
-- Prepare for concurrent refreshes in future runs
CREATE UNIQUE INDEX IF NOT EXISTS idx_tf3_race_key ON feat.train_features3 (race_key);
\echo '--- eval_features2 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.eval_features2;
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
