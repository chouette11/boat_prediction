/*------------------------------------------------------------
  05_views_core.sql
  CORE レイヤ（集約ビュー）
  マテビューを DROP/CREATE ではなく REFRESH で更新する版
------------------------------------------------------------*/

-- =========================================================
-- マテビュー定義 & 初回のみ作成（IF NOT EXISTS）
-- その後に REFRESH でデータを最新化
-- =========================================================

-- レース一覧 ------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.races AS
SELECT DISTINCT
       stadium,
       race_date,
       race_no,
       core.f_race_key(race_date, race_no, stadium) AS race_key
FROM raw.results;

-- 着順 ------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.results AS
SELECT DISTINCT ON (race_key, lane)
       core.f_race_key(race_date, race_no, stadium) AS race_key,
       lane,
       CASE position_txt
            WHEN '１' THEN 1 WHEN '２' THEN 2 WHEN '３' THEN 3
            WHEN '４' THEN 4 WHEN '５' THEN 5 WHEN '６' THEN 6
            ELSE 7  -- 棄権・失格など
       END AS rank
FROM raw.results
ORDER BY race_key, lane, source_file DESC;

-- ボート・選手情報（通常 VIEW） ------------------------------
CREATE OR REPLACE VIEW core.boat_info AS
SELECT DISTINCT ON (b.race_key, b.lane)
       b.race_key,
       b.lane,
       b.racer_id,
       CAST(regexp_replace(b.weight_raw, '[^0-9.]', '', 'g') AS NUMERIC(5,1)) AS weight,
       b.exh_time,
       b.tilt_deg,
       (substr(b.st_time_raw, 1, 1) = 'F') AS fs_flag,
       CASE
           WHEN b.st_time_raw LIKE 'F%' THEN
               CASE
                   WHEN regexp_replace(b.st_time_raw, '^F', '') ~ '^\.\d{2}$' THEN
                       -CAST('0' || regexp_replace(b.st_time_raw, '^F', '') AS NUMERIC(4,2))
                   WHEN regexp_replace(b.st_time_raw, '^F', '') ~ '^\d{1}\.\d{1,2}$' THEN
                       -CAST(regexp_replace(b.st_time_raw, '^F', '') AS NUMERIC(4,2))
                   ELSE
                       NULL
               END
           WHEN regexp_replace(b.st_time_raw, '^F', '') ~ '^\.\d{2}$' THEN
               CAST('0' || regexp_replace(b.st_time_raw, '^F', '') AS NUMERIC(4,2))
           WHEN regexp_replace(b.st_time_raw, '^F', '') ~ '^\d{1}\.\d{1,2}$' THEN
               CAST(regexp_replace(b.st_time_raw, '^F', '') AS NUMERIC(4,2))
           ELSE
               NULL
       END AS bf_st_time,
       b.course AS bf_course,
       r.course AS course,
       CASE
           WHEN regexp_replace(r.st_time_raw, '^F', '') ~ '^\.\d{2}$' THEN
               CAST('0' || regexp_replace(r.st_time_raw, '^F', '') AS NUMERIC(4,2))
           WHEN regexp_replace(r.st_time_raw, '^F', '') ~ '^\d{1}\.\d{1,2}$' THEN
               CAST(regexp_replace(r.st_time_raw, '^F', '') AS NUMERIC(4,2))
           ELSE
               NULL
       END AS st_time,
       p.class_now,
       p.ability_now,
       p.winrate_natl,
       p."2in_natl",
       p."3in_natl",
       p.age,
       p.class_hist1,
       p.class_hist2,
       p.class_hist3,
       -- p.ability_prev,
       p."F_now",
       p."L_now",
       p.nat_1st,
       p.nat_2nd,
       p.nat_3rd,
       p.nat_starts,
       p.loc_1st,
       p.loc_2nd,
       p.loc_3rd,
       p.loc_starts,
       p.motor_no,
       p.motor_2in,
       p.motor_3in,
       p.mot_1st,
       p.mot_2nd,
       p.mot_3rd,
       p.mot_starts,
       p.boat_no_hw,
       p.boat_2in,
       p.boat_3in,
       p.boa_1st,
       p.boa_2nd,
       p.boa_3rd,
       p.boa_starts
FROM (
    SELECT *,
           core.f_race_key(race_date, race_no, stadium) AS race_key
    FROM raw.beforeinfo
) b
LEFT JOIN raw.person p
  ON p.reg_no = b.racer_id
 AND core.f_race_key(p.race_date, p.race_no, p.stadium) = b.race_key
 AND p.boat_no = b.lane
LEFT JOIN raw.results r
  ON core.f_race_key(r.race_date, r.race_no, r.stadium) = b.race_key
 AND r.lane = b.lane
ORDER BY b.race_key, b.lane, b.course;

-- 天候 ------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.weather AS
SELECT DISTINCT ON (race_key)
       core.f_race_key(race_date, race_no, stadium) AS race_key,
       NULLIF(regexp_replace(air_temp_raw   ,'[^0-9.]','','g'), '')::NUMERIC AS air_temp,
       NULLIF(regexp_replace(wind_speed_raw ,'[^0-9.]','','g'), '')::NUMERIC AS wind_speed,
       CASE wind_dir_raw
           WHEN 'is-wind1'  THEN 0
           WHEN 'is-wind2'  THEN 22.5
           WHEN 'is-wind3'  THEN 45
           WHEN 'is-wind4'  THEN 67.5
           WHEN 'is-wind5'  THEN 90
           WHEN 'is-wind6'  THEN 112.5
           WHEN 'is-wind7'  THEN 135
           WHEN 'is-wind8'  THEN 157.5
           WHEN 'is-wind9'  THEN 180
           WHEN 'is-wind10' THEN 202.5
           WHEN 'is-wind11' THEN 225
           WHEN 'is-wind12' THEN 247.5
           WHEN 'is-wind13' THEN 270
           WHEN 'is-wind14' THEN 292.5
           WHEN 'is-wind15' THEN 315
           WHEN 'is-wind16' THEN 337.5
           ELSE NULL
       END AS wind_dir_deg,
       NULLIF(regexp_replace(wave_height_raw,'[^0-9.]','','g'), '')::NUMERIC AS wave_height,
       NULLIF(regexp_replace(water_temp_raw ,'[^0-9.]','','g'), '')::NUMERIC AS water_temp,
       weather_txt
FROM raw.weather
ORDER BY race_key, obs_time_label DESC;

-- オッズ（3連単） -------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.odds3t AS
SELECT DISTINCT ON (race_key, first_lane, second_lane, third_lane)
       core.f_race_key(race_date, race_no, stadium) AS race_key,
       first_lane,
       second_lane,
       third_lane,
       odds
FROM raw.odds3t
ORDER BY race_key,
         first_lane,
         second_lane,
         third_lane,
         source_file DESC;

-- エントリー（racelist_entries） ---------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.entries AS
SELECT DISTINCT ON (race_key, lane)
       core.f_race_key(race_date, race_no, stadium) AS race_key,
       lane,
       reg_no,
       grade,
       name,
       branch,
       birthplace,
       age,
       f_count,
       l_count,
       avg_st,
       national_win,
       national_2ren,
       national_3ren,
       local_win,
       local_2ren,
       local_3ren,
       jcd,
       place,
       title,
       day_label,
       distance_m
FROM raw.racelist_entries
ORDER BY race_key, lane, source_file DESC;

-- レーサー統計（コース別：走法＋率 を統合） -------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.racerstats_course AS
SELECT
       COALESCE(rt.reg_no, rc.reg_no)       AS reg_no,
       COALESCE(rt.course, rc.course)       AS course,
       COALESCE(rt.starts, rc.starts)       AS starts,
       COALESCE(rt.firsts, rc.firsts)       AS firsts,
       rt.nige,
       rt.sashi,
       rt.makuri,
       rt.makurisashi,
       rt.nuki,
       rt.megumare,
       rc.first_rate,
       rc.two_rate,
       rc.three_rate,
       rc.avg_st,
       rc.avg_st_rank
FROM raw.racertech   rt
FULL JOIN raw.racercourse rc
  ON rt.reg_no = rc.reg_no
 AND rt.course = rc.course;

-- レーサー成績（グレード別：率＋着順分布 を統合） --------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.racerstats_grade AS
SELECT
       rr1.reg_no,
       rr1.grade,
       rr1.meeting_entries,
       COALESCE(rr1.starts, rr2.starts) AS starts,
       COALESCE(rr1.firsts, rr2.firsts) AS firsts,
       rr2.seconds,
       rr2.thirds,
       rr2.fourths,
       rr2.fifths,
       rr2.sixths,
       rr1.winrate,
       rr1.first_rate,
       rr1.two_rate,
       rr1.three_rate,
       rr1.finalist_cnt,
       rr1.champion_cnt,
       rr1.avg_st,
       rr1.avg_st_rank,
       rr2.s0,
       rr2.s1,
       rr2.s2,
       rr2.f_cnt,
       rr2.l0,
       rr2.l1,
       rr2.k0,
       rr2.k1
FROM raw.racerresult1 rr1
LEFT JOIN raw.racerresult2 rr2
  ON rr1.reg_no = rr2.reg_no
 AND rr1.grade  = rr2.grade;

-- レーサー統計（ボート＋レーン別：分布＋率 を統合） ------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.racerstats_boatcourse AS
SELECT
       COALESCE(rbc.reg_no, rb.reg_no)           AS reg_no,
       COALESCE(rbc.lane, rb.lane)               AS lane,
       COALESCE(rbc.starts, rb.starts)           AS starts,
       rbc.lane1_cnt,
       rbc.lane2_cnt,
       rbc.lane3_cnt,
       rbc.lane4_cnt,
       rbc.lane5_cnt,
       rbc.lane6_cnt,
       rbc.other_cnt,
       rb.firsts,
       rb.first_rate,
       rb.two_rate,
       rb.three_rate,
       rb.finalist_cnt,
       rb.champion_cnt
FROM raw.racerboatcourse rbc
FULL JOIN raw.racerboat      rb
  ON rbc.reg_no        = rb.reg_no
 AND rbc.lane = rb.lane;

-- レーサー統計（スタジアム） -------------------

-- レーサー統計（スタジアム別） ------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.racerstats_stadium AS
SELECT
       reg_no,
       CASE WHEN stadium_code ~ '^\d+$' THEN stadium_code::INT ELSE NULL END AS stadium,
       meeting_entries,
       starts,
       firsts,
       winrate,
       first_rate,
       two_rate,
       three_rate,
       finalist_cnt,
       champion_cnt,
       avg_st
FROM raw.racerstadium;

-- ==========================================================
-- REFRESH MATERIALIZED VIEWS
-- ==========================================================

REFRESH MATERIALIZED VIEW core.races;
REFRESH MATERIALIZED VIEW core.results;
REFRESH MATERIALIZED VIEW core.weather;
REFRESH MATERIALIZED VIEW core.odds3t;
REFRESH MATERIALIZED VIEW core.entries;

REFRESH MATERIALIZED VIEW core.racerstats_course;
REFRESH MATERIALIZED VIEW core.racerstats_grade;
REFRESH MATERIALIZED VIEW core.racerstats_boatcourse;
REFRESH MATERIALIZED VIEW core.racerstats_stadium;

-- ==========================================================
-- データ存在チェック（core.* のマテビューすべて）
-- ==========================================================
\echo '--- core 層データ存在チェック ---'
SELECT format(
          $$SELECT 'core.%I' AS view_name,
                       CASE WHEN EXISTS (SELECT 1 FROM core.%I LIMIT 1)
                            THEN '✔ data' ELSE '✖ empty' END AS has_data;$$,
          matviewname, matviewname
       ) AS cmd
FROM   pg_matviews
WHERE  schemaname = 'core'
\gexec
\echo '--- core 層データ存在チェック完了 ---'
