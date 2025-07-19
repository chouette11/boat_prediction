/*------------------------------------------------------------
  05_views_core.sql
  CORE レイヤ（集約ビュー）
------------------------------------------------------------*/

-- レース一覧 ------------------------------------------------
CREATE MATERIALIZED VIEW core.races AS
SELECT DISTINCT
       stadium,
       race_date,
       race_no,
       core.f_race_key(race_date, race_no, stadium) AS race_key
FROM raw.results;

-- 着順 ------------------------------------------------------
CREATE MATERIALIZED VIEW core.results AS
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

CREATE OR REPLACE VIEW core.boat_info AS
SELECT DISTINCT ON (r.race_key, r.lane)
       r.race_key,
       r.lane,
       r.racer_id,
       CAST(regexp_replace(r.weight_raw, '[^0-9.]', '', 'g') AS NUMERIC(5,1)) AS weight,
       r.exh_time,
       r.tilt_deg,
       (substr(r.st_raw, 1, 1) = 'F') AS fs_flag,
       CASE
           WHEN regexp_replace(r.st_raw, '^F', '') ~ '^\.\d{2}$' THEN
               CAST('0' || regexp_replace(r.st_raw, '^F', '') AS NUMERIC(4,2))
           WHEN regexp_replace(r.st_raw, '^F', '') ~ '^\d{1}\.\d{1,2}$' THEN
               CAST(regexp_replace(r.st_raw, '^F', '') AS NUMERIC(4,2))
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
       p.ability_prev,
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
    FROM raw.racers
) r
LEFT JOIN raw.person p
  ON p.reg_no = r.racer_id
 AND core.f_race_key(p.race_date, p.race_no, p.stadium) = r.race_key
 AND p.boat_no = r.lane
ORDER BY r.race_key, r.lane, r.st_entry;

-- 天候 ------------------------------------------------------
CREATE MATERIALIZED VIEW core.weather AS
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
