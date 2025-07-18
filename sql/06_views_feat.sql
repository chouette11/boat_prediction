/*------------------------------------------------------------
  06_views_feat.sql
  FEAT レイヤ（学習用特徴量）
------------------------------------------------------------*/

-- フラット化 ------------------------------------------------
CREATE MATERIALIZED VIEW feat.boat_flat AS
SELECT DISTINCT ON (b.race_key, b.lane)
       b.race_key,
       b.lane,
       b.racer_id,
       w.air_temp, w.wind_speed, w.wave_height, w.water_temp, w.weather_txt,
       b.st_time, b.fs_flag,
       b.weight, b.exh_time, b.tilt_deg,
       r.rank,
       cr.race_date
FROM core.boat_info b
JOIN core.weather  w  USING (race_key)
JOIN core.results  r  USING (race_key, lane)
JOIN core.races    cr USING (race_key)
ORDER BY b.race_key, b.lane;

-- 学習用特徴量 ---------------------------------------------
CREATE MATERIALIZED VIEW feat.train_features AS
WITH flat AS (
    SELECT bf.*,
           cr.stadium AS venue
    FROM feat.boat_flat bf
    JOIN core.races   cr USING (race_key)
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

    MAX(CASE WHEN lane=1 THEN racer_id END) AS lane1_racer_id,
    MAX(CASE WHEN lane=1 THEN weight   END) AS lane1_weight,
    MAX(CASE WHEN lane=1 THEN exh_time END) AS lane1_exh_time,
    MAX(CASE WHEN lane=1 THEN st_time  END) AS lane1_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=1)  AS lane1_fs_flag,
    MAX(CASE WHEN lane=1 THEN rank     END) AS lane1_rank,

    MAX(CASE WHEN lane=2 THEN racer_id END) AS lane2_racer_id,
    MAX(CASE WHEN lane=2 THEN weight   END) AS lane2_weight,
    MAX(CASE WHEN lane=2 THEN exh_time END) AS lane2_exh_time,
    MAX(CASE WHEN lane=2 THEN st_time  END) AS lane2_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=2)  AS lane2_fs_flag,
    MAX(CASE WHEN lane=2 THEN rank     END) AS lane2_rank,

    MAX(CASE WHEN lane=3 THEN racer_id END) AS lane3_racer_id,
    MAX(CASE WHEN lane=3 THEN weight   END) AS lane3_weight,
    MAX(CASE WHEN lane=3 THEN exh_time END) AS lane3_exh_time,
    MAX(CASE WHEN lane=3 THEN st_time  END) AS lane3_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=3)  AS lane3_fs_flag,
    MAX(CASE WHEN lane=3 THEN rank     END) AS lane3_rank,

    MAX(CASE WHEN lane=4 THEN racer_id END) AS lane4_racer_id,
    MAX(CASE WHEN lane=4 THEN weight   END) AS lane4_weight,
    MAX(CASE WHEN lane=4 THEN exh_time END) AS lane4_exh_time,
    MAX(CASE WHEN lane=4 THEN st_time  END) AS lane4_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=4)  AS lane4_fs_flag,
    MAX(CASE WHEN lane=4 THEN rank     END) AS lane4_rank,

    MAX(CASE WHEN lane=5 THEN racer_id END) AS lane5_racer_id,
    MAX(CASE WHEN lane=5 THEN weight   END) AS lane5_weight,
    MAX(CASE WHEN lane=5 THEN exh_time END) AS lane5_exh_time,
    MAX(CASE WHEN lane=5 THEN st_time  END) AS lane5_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=5)  AS lane5_fs_flag,
    MAX(CASE WHEN lane=5 THEN rank     END) AS lane5_rank,

    MAX(CASE WHEN lane=6 THEN racer_id END) AS lane6_racer_id,
    MAX(CASE WHEN lane=6 THEN weight   END) AS lane6_weight,
    MAX(CASE WHEN lane=6 THEN exh_time END) AS lane6_exh_time,
    MAX(CASE WHEN lane=6 THEN st_time  END) AS lane6_st,
    BOOL_OR(fs_flag) FILTER (WHERE lane=6)  AS lane6_fs_flag,
    MAX(CASE WHEN lane=6 THEN rank     END) AS lane6_rank
FROM flat
GROUP BY race_key
HAVING COUNT(DISTINCT race_date) = 1
   AND COUNT(DISTINCT venue)     = 1;

-- ==========================================================
-- データ存在チェック（feat.* のマテビューすべて）
-- ==========================================================
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
