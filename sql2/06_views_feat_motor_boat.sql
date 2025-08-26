CREATE SCHEMA IF NOT EXISTS feat;

-- Session tuning for faster REFRESH and JOINs
SET work_mem = '256MB';
SET maintenance_work_mem = '512MB';
SET max_parallel_workers_per_gather = 4;
SET parallel_leader_participation = on;
CREATE INDEX IF NOT EXISTS idx_core_rsc_reg_course
  ON core.racerstats_course (reg_no, course);


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
       b.motor_no,
       b.boat_no,
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
    MAX(CASE WHEN lane=1 THEN motor_no END) AS lane1_motor_no,
    MAX(CASE WHEN lane=1 THEN boat_no  END) AS lane1_boat_no,
    MAX(CASE WHEN lane=2 THEN racer_id END) AS lane2_racer_id,
    MAX(CASE WHEN lane=2 THEN weight END) AS lane2_weight,
    MAX(CASE WHEN lane=2 THEN exh_time END) AS lane2_exh_time,
    MAX(CASE WHEN lane=2 THEN bf_st_time END) AS lane2_bf_st_time,
    MAX(CASE WHEN lane=2 THEN bf_course END) AS lane2_bf_course,
    MAX(CASE WHEN lane=2 THEN st_time END) AS lane2_st,
    MAX(CASE WHEN lane=2 THEN course END) AS lane2_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=2)  AS lane2_fs_flag,
    MAX(CASE WHEN lane=2 THEN rank END) AS lane2_rank,
    MAX(CASE WHEN lane=2 THEN motor_no END) AS lane2_motor_no,
    MAX(CASE WHEN lane=2 THEN boat_no  END) AS lane2_boat_no,
    MAX(CASE WHEN lane=3 THEN racer_id END) AS lane3_racer_id,
    MAX(CASE WHEN lane=3 THEN weight END) AS lane3_weight,
    MAX(CASE WHEN lane=3 THEN exh_time END) AS lane3_exh_time,
    MAX(CASE WHEN lane=3 THEN bf_st_time END) AS lane3_bf_st_time,
    MAX(CASE WHEN lane=3 THEN bf_course END) AS lane3_bf_course,
    MAX(CASE WHEN lane=3 THEN st_time END) AS lane3_st,
    MAX(CASE WHEN lane=3 THEN course END) AS lane3_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=3)  AS lane3_fs_flag,
    MAX(CASE WHEN lane=3 THEN rank END) AS lane3_rank,
    MAX(CASE WHEN lane=3 THEN motor_no END) AS lane3_motor_no,
    MAX(CASE WHEN lane=3 THEN boat_no  END) AS lane3_boat_no,
    MAX(CASE WHEN lane=4 THEN racer_id END) AS lane4_racer_id,
    MAX(CASE WHEN lane=4 THEN weight END) AS lane4_weight,
    MAX(CASE WHEN lane=4 THEN exh_time END) AS lane4_exh_time,
    MAX(CASE WHEN lane=4 THEN bf_st_time END) AS lane4_bf_st_time,
    MAX(CASE WHEN lane=4 THEN bf_course END) AS lane4_bf_course,
    MAX(CASE WHEN lane=4 THEN st_time END) AS lane4_st,
    MAX(CASE WHEN lane=4 THEN course END) AS lane4_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=4)  AS lane4_fs_flag,
    MAX(CASE WHEN lane=4 THEN rank END) AS lane4_rank,
    MAX(CASE WHEN lane=4 THEN motor_no END) AS lane4_motor_no,
    MAX(CASE WHEN lane=4 THEN boat_no  END) AS lane4_boat_no,
    MAX(CASE WHEN lane=5 THEN racer_id END) AS lane5_racer_id,
    MAX(CASE WHEN lane=5 THEN weight END) AS lane5_weight,
    MAX(CASE WHEN lane=5 THEN exh_time END) AS lane5_exh_time,
    MAX(CASE WHEN lane=5 THEN bf_st_time END) AS lane5_bf_st_time,
    MAX(CASE WHEN lane=5 THEN bf_course END) AS lane5_bf_course,
    MAX(CASE WHEN lane=5 THEN st_time END) AS lane5_st,
    MAX(CASE WHEN lane=5 THEN course END) AS lane5_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=5)  AS lane5_fs_flag,
    MAX(CASE WHEN lane=5 THEN rank END) AS lane5_rank,
    MAX(CASE WHEN lane=5 THEN motor_no END) AS lane5_motor_no,
    MAX(CASE WHEN lane=5 THEN boat_no  END) AS lane5_boat_no,
    MAX(CASE WHEN lane=6 THEN racer_id END) AS lane6_racer_id,
    MAX(CASE WHEN lane=6 THEN weight END) AS lane6_weight,
    MAX(CASE WHEN lane=6 THEN exh_time END) AS lane6_exh_time,
    MAX(CASE WHEN lane=6 THEN bf_st_time END) AS lane6_bf_st_time,
    MAX(CASE WHEN lane=6 THEN bf_course END) AS lane6_bf_course,
    MAX(CASE WHEN lane=6 THEN st_time END) AS lane6_st,
    MAX(CASE WHEN lane=6 THEN course END) AS lane6_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=6)  AS lane6_fs_flag,
    MAX(CASE WHEN lane=6 THEN rank END) AS lane6_rank,
    MAX(CASE WHEN lane=6 THEN motor_no END) AS lane6_motor_no,
    MAX(CASE WHEN lane=6 THEN boat_no  END) AS lane6_boat_no
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

/* ---------- 装備統計（場×装備ID、当日時点まで） ---------- */
/* ---------- 装備別 日別成績（前集計：1日×場×装備ID） ---------- */
DROP MATERIALIZED VIEW IF EXISTS feat.equip_day_stats_motor CASCADE;
DROP TABLE IF EXISTS feat.equip_day_stats_motor CASCADE;
CREATE TABLE IF NOT EXISTS feat.equip_day_stats_motor (
  stadium   text      NOT NULL,
  race_date date      NOT NULL,
  motor_no  integer   NOT NULL,
  starts    integer   NOT NULL,
  firsts    integer   NOT NULL,
  top2      integer   NOT NULL,
  top3      integer   NOT NULL,
  PRIMARY KEY (stadium, motor_no, race_date)
);
-- Optional helper index to speed date range scans
CREATE INDEX IF NOT EXISTS idx_feat_equip_day_motor_date
  ON feat.equip_day_stats_motor (race_date);

DROP MATERIALIZED VIEW IF EXISTS feat.equip_day_stats_boat CASCADE;
DROP TABLE IF EXISTS feat.equip_day_stats_boat CASCADE;
CREATE TABLE IF NOT EXISTS feat.equip_day_stats_boat (
  stadium   text      NOT NULL,
  race_date date      NOT NULL,
  boat_no   integer   NOT NULL,
  starts    integer   NOT NULL,
  firsts    integer   NOT NULL,
  top2      integer   NOT NULL,
  top3      integer   NOT NULL,
  PRIMARY KEY (stadium, boat_no, race_date)
);
-- Optional helper index to speed date range scans
CREATE INDEX IF NOT EXISTS idx_feat_equip_day_boat_date
  ON feat.equip_day_stats_boat (race_date);

DROP MATERIALIZED VIEW IF EXISTS feat.tf2_equip_stats_timeboxed CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.tf2_equip_stats_timeboxed AS
WITH tf2_long AS (
  SELECT tf.race_key, tf.race_date, tf.venue AS stadium, 1 AS lane_no,
         tf.lane1_motor_no AS motor_no, tf.lane1_boat_no AS boat_no
  FROM feat.train_features2 tf
  UNION ALL
  SELECT tf.race_key, tf.race_date, tf.venue, 2, tf.lane2_motor_no, tf.lane2_boat_no FROM feat.train_features2 tf
  UNION ALL
  SELECT tf.race_key, tf.race_date, tf.venue, 3, tf.lane3_motor_no, tf.lane3_boat_no FROM feat.train_features2 tf
  UNION ALL
  SELECT tf.race_key, tf.race_date, tf.venue, 4, tf.lane4_motor_no, tf.lane4_boat_no FROM feat.train_features2 tf
  UNION ALL
  SELECT tf.race_key, tf.race_date, tf.venue, 5, tf.lane5_motor_no, tf.lane5_boat_no FROM feat.train_features2 tf
  UNION ALL
  SELECT tf.race_key, tf.race_date, tf.venue, 6, tf.lane6_motor_no, tf.lane6_boat_no FROM feat.train_features2 tf
)
SELECT
  l.race_key,

  -- motor (lane1-6)
  MAX(m.starts)     FILTER (WHERE l.lane_no=1) AS lane1_motor_starts,
  MAX(m.firsts)     FILTER (WHERE l.lane_no=1) AS lane1_motor_firsts,
  MAX(m.first_rate) FILTER (WHERE l.lane_no=1) AS lane1_motor_first_rate,
  MAX(m.two_rate)   FILTER (WHERE l.lane_no=1) AS lane1_motor_two_rate,
  MAX(m.three_rate) FILTER (WHERE l.lane_no=1) AS lane1_motor_three_rate,

  MAX(m.starts)     FILTER (WHERE l.lane_no=2) AS lane2_motor_starts,
  MAX(m.firsts)     FILTER (WHERE l.lane_no=2) AS lane2_motor_firsts,
  MAX(m.first_rate) FILTER (WHERE l.lane_no=2) AS lane2_motor_first_rate,
  MAX(m.two_rate)   FILTER (WHERE l.lane_no=2) AS lane2_motor_two_rate,
  MAX(m.three_rate) FILTER (WHERE l.lane_no=2) AS lane2_motor_three_rate,

  MAX(m.starts)     FILTER (WHERE l.lane_no=3) AS lane3_motor_starts,
  MAX(m.firsts)     FILTER (WHERE l.lane_no=3) AS lane3_motor_firsts,
  MAX(m.first_rate) FILTER (WHERE l.lane_no=3) AS lane3_motor_first_rate,
  MAX(m.two_rate)   FILTER (WHERE l.lane_no=3) AS lane3_motor_two_rate,
  MAX(m.three_rate) FILTER (WHERE l.lane_no=3) AS lane3_motor_three_rate,

  MAX(m.starts)     FILTER (WHERE l.lane_no=4) AS lane4_motor_starts,
  MAX(m.firsts)     FILTER (WHERE l.lane_no=4) AS lane4_motor_firsts,
  MAX(m.first_rate) FILTER (WHERE l.lane_no=4) AS lane4_motor_first_rate,
  MAX(m.two_rate)   FILTER (WHERE l.lane_no=4) AS lane4_motor_two_rate,
  MAX(m.three_rate) FILTER (WHERE l.lane_no=4) AS lane4_motor_three_rate,

  MAX(m.starts)     FILTER (WHERE l.lane_no=5) AS lane5_motor_starts,
  MAX(m.firsts)     FILTER (WHERE l.lane_no=5) AS lane5_motor_firsts,
  MAX(m.first_rate) FILTER (WHERE l.lane_no=5) AS lane5_motor_first_rate,
  MAX(m.two_rate)   FILTER (WHERE l.lane_no=5) AS lane5_motor_two_rate,
  MAX(m.three_rate) FILTER (WHERE l.lane_no=5) AS lane5_motor_three_rate,

  MAX(m.starts)     FILTER (WHERE l.lane_no=6) AS lane6_motor_starts,
  MAX(m.firsts)     FILTER (WHERE l.lane_no=6) AS lane6_motor_firsts,
  MAX(m.first_rate) FILTER (WHERE l.lane_no=6) AS lane6_motor_first_rate,
  MAX(m.two_rate)   FILTER (WHERE l.lane_no=6) AS lane6_motor_two_rate,
  MAX(m.three_rate) FILTER (WHERE l.lane_no=6) AS lane6_motor_three_rate,

  -- boat (lane1-6)
  MAX(b.starts)     FILTER (WHERE l.lane_no=1) AS lane1_boat_starts,
  MAX(b.firsts)     FILTER (WHERE l.lane_no=1) AS lane1_boat_firsts,
  MAX(b.first_rate) FILTER (WHERE l.lane_no=1) AS lane1_boat_first_rate,
  MAX(b.two_rate)   FILTER (WHERE l.lane_no=1) AS lane1_boat_two_rate,
  MAX(b.three_rate) FILTER (WHERE l.lane_no=1) AS lane1_boat_three_rate,

  MAX(b.starts)     FILTER (WHERE l.lane_no=2) AS lane2_boat_starts,
  MAX(b.firsts)     FILTER (WHERE l.lane_no=2) AS lane2_boat_firsts,
  MAX(b.first_rate) FILTER (WHERE l.lane_no=2) AS lane2_boat_first_rate,
  MAX(b.two_rate)   FILTER (WHERE l.lane_no=2) AS lane2_boat_two_rate,
  MAX(b.three_rate) FILTER (WHERE l.lane_no=2) AS lane2_boat_three_rate,

  MAX(b.starts)     FILTER (WHERE l.lane_no=3) AS lane3_boat_starts,
  MAX(b.firsts)     FILTER (WHERE l.lane_no=3) AS lane3_boat_firsts,
  MAX(b.first_rate) FILTER (WHERE l.lane_no=3) AS lane3_boat_first_rate,
  MAX(b.two_rate)   FILTER (WHERE l.lane_no=3) AS lane3_boat_two_rate,
  MAX(b.three_rate) FILTER (WHERE l.lane_no=3) AS lane3_boat_three_rate,

  MAX(b.starts)     FILTER (WHERE l.lane_no=4) AS lane4_boat_starts,
  MAX(b.firsts)     FILTER (WHERE l.lane_no=4) AS lane4_boat_firsts,
  MAX(b.first_rate) FILTER (WHERE l.lane_no=4) AS lane4_boat_first_rate,
  MAX(b.two_rate)   FILTER (WHERE l.lane_no=4) AS lane4_boat_two_rate,
  MAX(b.three_rate) FILTER (WHERE l.lane_no=4) AS lane4_boat_three_rate,

  MAX(b.starts)     FILTER (WHERE l.lane_no=5) AS lane5_boat_starts,
  MAX(b.firsts)     FILTER (WHERE l.lane_no=5) AS lane5_boat_firsts,
  MAX(b.first_rate) FILTER (WHERE l.lane_no=5) AS lane5_boat_first_rate,
  MAX(b.two_rate)   FILTER (WHERE l.lane_no=5) AS lane5_boat_two_rate,
  MAX(b.three_rate) FILTER (WHERE l.lane_no=5) AS lane5_boat_three_rate,

  MAX(b.starts)     FILTER (WHERE l.lane_no=6) AS lane6_boat_starts,
  MAX(b.firsts)     FILTER (WHERE l.lane_no=6) AS lane6_boat_firsts,
  MAX(b.first_rate) FILTER (WHERE l.lane_no=6) AS lane6_boat_first_rate,
  MAX(b.two_rate)   FILTER (WHERE l.lane_no=6) AS lane6_boat_two_rate,
  MAX(b.three_rate) FILTER (WHERE l.lane_no=6) AS lane6_boat_three_rate

 FROM tf2_long l
LEFT JOIN core.v_equip_cycles ec_m
  ON ec_m.stadium = l.stadium
 AND ec_m.equip_type = 'motor'
 AND l.race_date >= ec_m.start_date
 AND l.race_date <  ec_m.next_start_date
LEFT JOIN core.v_equip_cycles ec_b
  ON ec_b.stadium = l.stadium
 AND ec_b.equip_type = 'boat'
 AND l.race_date >= ec_b.start_date
 AND l.race_date <  ec_b.next_start_date
/* 装備成績を当日より前に限定（載せ替え考慮、場で絞る） */
LEFT JOIN LATERAL (
  SELECT
    SUM(d.starts) AS starts,
    SUM(d.firsts) AS firsts,
    CASE WHEN SUM(d.starts) > 0 THEN SUM(d.firsts)::float / SUM(d.starts) ELSE NULL END AS first_rate,
    CASE WHEN SUM(d.starts) > 0 THEN SUM(d.top2)::float   / SUM(d.starts) ELSE NULL END AS two_rate,
    CASE WHEN SUM(d.starts) > 0 THEN SUM(d.top3)::float   / SUM(d.starts) ELSE NULL END AS three_rate
  FROM feat.equip_day_stats_motor d
  WHERE d.stadium  = l.stadium
    AND d.motor_no = l.motor_no
    AND d.race_date >= COALESCE(ec_m.start_date, (l.race_date - INTERVAL '365 days'))
    AND d.race_date <  l.race_date
) m ON TRUE
LEFT JOIN LATERAL (
  SELECT
    SUM(d.starts) AS starts,
    SUM(d.firsts) AS firsts,
    CASE WHEN SUM(d.starts) > 0 THEN SUM(d.firsts)::float / SUM(d.starts) ELSE NULL END AS first_rate,
    CASE WHEN SUM(d.starts) > 0 THEN SUM(d.top2)::float   / SUM(d.starts) ELSE NULL END AS two_rate,
    CASE WHEN SUM(d.starts) > 0 THEN SUM(d.top3)::float   / SUM(d.starts) ELSE NULL END AS three_rate
  FROM feat.equip_day_stats_boat d
  WHERE d.stadium = l.stadium
    AND d.boat_no  = l.boat_no
    AND d.race_date >= COALESCE(ec_b.start_date, (l.race_date - INTERVAL '365 days'))
    AND d.race_date <  l.race_date
) b ON TRUE
GROUP BY l.race_key
WITH NO DATA;

DROP MATERIALIZED VIEW IF EXISTS feat.train_features3 CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.train_features3 AS
SELECT
    tf.*,
    ls.lane1_starts, ls.lane1_firsts, ls.lane1_first_rate, ls.lane1_two_rate, ls.lane1_three_rate,
    ls.lane2_starts, ls.lane2_firsts, ls.lane2_first_rate, ls.lane2_two_rate, ls.lane2_three_rate,
    ls.lane3_starts, ls.lane3_firsts, ls.lane3_first_rate, ls.lane3_two_rate, ls.lane3_three_rate,
    ls.lane4_starts, ls.lane4_firsts, ls.lane4_first_rate, ls.lane4_two_rate, ls.lane4_three_rate,
    ls.lane5_starts, ls.lane5_firsts, ls.lane5_first_rate, ls.lane5_two_rate, ls.lane5_three_rate,
    ls.lane6_starts, ls.lane6_firsts, ls.lane6_first_rate, ls.lane6_two_rate, ls.lane6_three_rate,

    es.lane1_motor_starts, es.lane1_motor_firsts, es.lane1_motor_first_rate, es.lane1_motor_two_rate, es.lane1_motor_three_rate,
    es.lane2_motor_starts, es.lane2_motor_firsts, es.lane2_motor_first_rate, es.lane2_motor_two_rate, es.lane2_motor_three_rate,
    es.lane3_motor_starts, es.lane3_motor_firsts, es.lane3_motor_first_rate, es.lane3_motor_two_rate, es.lane3_motor_three_rate,
    es.lane4_motor_starts, es.lane4_motor_firsts, es.lane4_motor_first_rate, es.lane4_motor_two_rate, es.lane4_motor_three_rate,
    es.lane5_motor_starts, es.lane5_motor_firsts, es.lane5_motor_first_rate, es.lane5_motor_two_rate, es.lane5_motor_three_rate,
    es.lane6_motor_starts, es.lane6_motor_firsts, es.lane6_motor_first_rate, es.lane6_motor_two_rate, es.lane6_motor_three_rate,

    es.lane1_boat_starts, es.lane1_boat_firsts, es.lane1_boat_first_rate, es.lane1_boat_two_rate, es.lane1_boat_three_rate,
    es.lane2_boat_starts, es.lane2_boat_firsts, es.lane2_boat_first_rate, es.lane2_boat_two_rate, es.lane2_boat_three_rate,
    es.lane3_boat_starts, es.lane3_boat_firsts, es.lane3_boat_first_rate, es.lane3_boat_two_rate, es.lane3_boat_three_rate,
    es.lane4_boat_starts, es.lane4_boat_firsts, es.lane4_boat_first_rate, es.lane4_boat_two_rate, es.lane4_boat_three_rate,
    es.lane5_boat_starts, es.lane5_boat_firsts, es.lane5_boat_first_rate, es.lane5_boat_two_rate, es.lane5_boat_three_rate,
    es.lane6_boat_starts, es.lane6_boat_firsts, es.lane6_boat_first_rate, es.lane6_boat_two_rate, es.lane6_boat_three_rate
FROM feat.train_features2 tf
LEFT JOIN feat.tf2_lane_stats            ls USING (race_key)
LEFT JOIN feat.tf2_equip_stats_timeboxed es USING (race_key)
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
\echo '--- train_features2 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.train_features2;
 \echo '--- tf2_lane_stats 層のマテリアライズドビューを更新中 ---'
 REFRESH MATERIALIZED VIEW feat.tf2_lane_stats;
\echo '--- equip_day_stats_* インクリメンタル更新（boat_flat 由来） ---'
DO $$
BEGIN
  -- ========== motor ==========
  WITH last AS (
    SELECT COALESCE(MAX(race_date), DATE '1900-01-01') AS maxd
    FROM feat.equip_day_stats_motor
  ), start AS (
    SELECT GREATEST((SELECT maxd FROM last) - INTERVAL '7 days', DATE '1900-01-01')::date AS d
  )
  DELETE FROM feat.equip_day_stats_motor m
  USING start s
  WHERE m.race_date >= s.d;

  INSERT INTO feat.equip_day_stats_motor (stadium, race_date, motor_no, starts, firsts, top2, top3)
  SELECT cr.stadium,
         bf.race_date::date AS race_date,
         bf.motor_no,
         COUNT(*) AS starts,
         SUM((bf.rank = 1)::int) AS firsts,
         SUM((bf.rank <= 2)::int) AS top2,
         SUM((bf.rank <= 3)::int) AS top3
  FROM feat.boat_flat bf
  JOIN core.races cr USING (race_key)
  WHERE bf.race_date::date >= (SELECT d FROM start)
  GROUP BY cr.stadium, bf.race_date::date, bf.motor_no
  ON CONFLICT (stadium, motor_no, race_date) DO UPDATE
    SET starts = EXCLUDED.starts,
        firsts = EXCLUDED.firsts,
        top2   = EXCLUDED.top2,
        top3   = EXCLUDED.top3;

  -- ========== boat ==========
  WITH last AS (
    SELECT COALESCE(MAX(race_date), DATE '1900-01-01') AS maxd
    FROM feat.equip_day_stats_boat
  ), start AS (
    SELECT GREATEST((SELECT maxd FROM last) - INTERVAL '7 days', DATE '1900-01-01')::date AS d
  )
  DELETE FROM feat.equip_day_stats_boat b
  USING start s
  WHERE b.race_date >= s.d;

  INSERT INTO feat.equip_day_stats_boat (stadium, race_date, boat_no, starts, firsts, top2, top3)
  SELECT cr.stadium,
         bf.race_date::date AS race_date,
         bf.boat_no,
         COUNT(*) AS starts,
         SUM((bf.rank = 1)::int) AS firsts,
         SUM((bf.rank <= 2)::int) AS top2,
         SUM((bf.rank <= 3)::int) AS top3
  FROM feat.boat_flat bf
  JOIN core.races cr USING (race_key)
  WHERE bf.race_date::date >= (SELECT d FROM start)
  GROUP BY cr.stadium, bf.race_date::date, bf.boat_no
  ON CONFLICT (stadium, boat_no, race_date) DO UPDATE
    SET starts = EXCLUDED.starts,
        firsts = EXCLUDED.firsts,
        top2   = EXCLUDED.top2,
        top3   = EXCLUDED.top3;
END $$;
\echo '--- tf2_equip_stats_timeboxed 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.tf2_equip_stats_timeboxed;
\echo '--- train_features3 層のマテリアライズドビューを更新中 ---'
ANALYZE feat.train_features2;
ANALYZE feat.filtered_course;
ANALYZE feat.tf2_lane_stats;
ANALYZE feat.equip_day_stats_motor;
ANALYZE feat.equip_day_stats_boat;
ANALYZE feat.tf2_equip_stats_timeboxed;
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
CREATE UNIQUE INDEX IF NOT EXISTS uq_feat_tf2_equip_stats_timeboxed_race_key
  ON feat.tf2_equip_stats_timeboxed (race_key);
