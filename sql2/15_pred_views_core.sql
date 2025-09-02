/*------------------------------------------------------------
  05_views_core.sql
  CORE レイヤ（集約ビュー）
  マテビューを DROP/CREATE ではなく REFRESH で更新する版
------------------------------------------------------------*/

-- =========================================================
-- マテビュー定義 & 初回のみ作成（IF NOT EXISTS）
-- その後に REFRESH でデータを最新化
-- =========================================================
-- ボート・選手情報（通常 VIEW） ------------------------------
-- ボート・選手情報（予測用：beforeinfoのみ）
CREATE OR REPLACE VIEW core.pred_boat_info AS
WITH b AS (
  SELECT *,
         core.f_race_key(race_date, race_no, stadium) AS race_key
  FROM raw.beforeinfo_off
)
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
                   WHEN regexp_replace(b.st_time_raw, '^F', '') ~ '^\.\d{2}$'    THEN -CAST('0' || regexp_replace(b.st_time_raw, '^F', '') AS NUMERIC(4,2))
                   WHEN regexp_replace(b.st_time_raw, '^F', '') ~ '^\d{1}\.\d{1,2}$' THEN -CAST(regexp_replace(b.st_time_raw, '^F', '')           AS NUMERIC(4,2))
                   ELSE NULL
               END
           WHEN regexp_replace(b.st_time_raw, '^F', '') ~ '^\.\d{2}$'    THEN CAST('0' || regexp_replace(b.st_time_raw, '^F', '') AS NUMERIC(4,2))
           WHEN regexp_replace(b.st_time_raw, '^F', '') ~ '^\d{1}\.\d{1,2}$' THEN CAST(regexp_replace(b.st_time_raw, '^F', '')           AS NUMERIC(4,2))
           ELSE NULL
       END AS bf_st_time,
       b.course AS bf_course
FROM b
ORDER BY b.race_key, b.lane, b.course;

-- 天候 ------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.pred_weather AS
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
FROM raw.weather_off
ORDER BY race_key, obs_time_label DESC;

-- レースキー→日付/会場マップ（予測用）
CREATE MATERIALIZED VIEW IF NOT EXISTS core.pred_races AS
SELECT DISTINCT ON (core.f_race_key(race_date, race_no, stadium))
       core.f_race_key(race_date, race_no, stadium) AS race_key,
       race_date,
       stadium AS venue
FROM raw.beforeinfo_off
ORDER BY core.f_race_key(race_date, race_no, stadium), race_date DESC;

CREATE UNIQUE INDEX IF NOT EXISTS uq_core_pred_races_race_key
  ON core.pred_races (race_key);


-- ==========================================================
-- REFRESH MATERIALIZED VIEWS
-- ==========================================================

REFRESH MATERIALIZED VIEW core.pred_weather;
REFRESH MATERIALIZED VIEW core.pred_races;
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
