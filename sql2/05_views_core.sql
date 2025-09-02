/*------------------------------------------------------------
  05_views_core.sql
  CORE レイヤ（集約ビュー）
  マテビューを DROP/CREATE ではなく REFRESH で更新する版
------------------------------------------------------------*/

-- ===== Pre-requisites: schema & helper function ============================
CREATE SCHEMA IF NOT EXISTS core;

-- official venue number resolver (01–24)
CREATE OR REPLACE FUNCTION core.f_official_venue_no(p_stadium text)
RETURNS text
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
  WITH norm AS (
    SELECT regexp_replace(translate(coalesce(p_stadium,''), '　', ' '), '\\s+', '', 'g') AS s
  ),
  mapping AS (
    SELECT * FROM (VALUES
      ('01','桐生'), ('02','戸田'), ('03','江戸川'), ('04','平和島'), ('05','多摩川'),
      ('06','浜名湖'), ('07','蒲郡'), ('08','常滑'), ('09','津'),    ('10','三国'),
      ('11','びわこ'), ('12','住之江'), ('13','尼崎'), ('14','鳴門'), ('15','丸亀'),
      ('16','児島'),  ('17','宮島'),  ('18','徳山'),  ('19','下関'), ('20','若松'),
      ('21','芦屋'),  ('22','福岡'),  ('23','唐津'),  ('24','大村')
    ) AS t(code, key_txt)
  ),
  mapping_norm AS (
    SELECT code, regexp_replace(translate(key_txt, '　', ' '), '\\s+', '', 'g') AS key_norm
    FROM mapping
  )
  SELECT COALESCE(
    CASE WHEN (SELECT s FROM norm) ~ '^\\d{1,2}$' THEN lpad((SELECT s FROM norm), 2, '0') END,
    (SELECT code FROM mapping_norm WHERE key_norm = (SELECT s FROM norm) LIMIT 1),
    (SELECT s FROM norm)
  );
$$;

-- race_key generator using official venue no
CREATE OR REPLACE FUNCTION core.f_race_key(p_date date, p_race_no int, p_stadium text)
RETURNS text
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
  SELECT to_char(p_date,'YYYY-MM-DD') || '-' ||
         lpad(p_race_no::text, 2, '0') || '-' ||
         core.f_official_venue_no(p_stadium);
$$;

-- =========================================================
-- マテビュー定義 & 初回のみ作成（IF NOT EXISTS）
-- その後に REFRESH でデータを最新化
-- =========================================================

-- レース一覧 ------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.races AS
SELECT DISTINCT
       v.name                                       AS stadium,
       e.event_date                                 AS race_date,
       r.race_no,
       core.f_race_key(e.event_date, r.race_no, v.name) AS race_key
FROM   raw.race r
JOIN   event e  ON r.event_id  = e.event_id
JOIN   venue v  ON e.venue_id  = v.venue_id;

-- 着順 ------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS core.results CASCADE;
CREATE MATERIALIZED VIEW core.results AS
SELECT DISTINCT ON (rk, rs.lane)
       rk AS race_key,
       rs.lane,
       CASE
            WHEN rs.finish_order BETWEEN 1 AND 6 THEN rs.finish_order
            ELSE 7  -- 棄権・失格など
       END AS rank,
       rs.winning_method,
       -- NORMALIZE: raw.winning_method is stored in Japanese; map with longest-first order per WIN_METHODS.
       CASE rs.winning_method
         WHEN 'まくり差し' THEN 'MAKURI_SASHI'
         WHEN 'まくり'     THEN 'MAKURI'
         WHEN '差し'       THEN 'SASHI'
         WHEN '逃げ'       THEN 'NIGE'
         WHEN '抜き'       THEN 'NUKI'
         WHEN '恵まれ'     THEN 'MEGUMARE'
         ELSE 'OTHER'
       END AS win_pattern
FROM (
    SELECT core.f_race_key(e.event_date, r.race_no, v.name) AS rk,
           rs.result_id,
           rs.lane,
           rs.finish_order,
           r.winning_method
    FROM   raw.result rs
    JOIN   raw.race   r ON rs.race_id = r.race_id
    JOIN   event  e ON r.event_id = e.event_id
    JOIN   venue  v ON e.venue_id = v.venue_id
) rs
ORDER BY rk, lane, result_id DESC;

-- ボート・選手情報（通常 VIEW） ------------------------------
CREATE OR REPLACE VIEW core.boat_info AS
WITH base AS (
    SELECT
        core.f_race_key(e.event_date, r.race_no, v.name) AS race_key,
        rs.lane,
        rs.reg_no                                       AS racer_id,
        rs.motor_no                                     AS motor_no,
        rs.boat_no                                      AS boat_no,
        rs.tenji_time                                   AS exh_time,
        rs.course_entry                                 AS course,
        rs.start_timing,
        rs.status
    FROM   raw.result rs
    JOIN   raw.race   r ON rs.race_id = r.race_id
    JOIN   event  e ON r.event_id = e.event_id
    JOIN   venue  v ON e.venue_id = v.venue_id
), prog AS (
    SELECT
        core.f_race_key(e.event_date, pr.race_no, v.name) AS race_key,
        pe.lane,
        pe.weight_kg
    FROM   raw.program_race  pr
    JOIN   event e ON pr.event_id = e.event_id
    JOIN   venue v ON e.venue_id = v.venue_id
    JOIN   raw.program_entry pe ON pe.program_race_id = pr.program_race_id
)
SELECT DISTINCT ON (b.race_key, b.lane)
       b.race_key,
       b.lane,
       b.racer_id,
       b.motor_no,
       b.boat_no,
       p.weight_kg                                 AS weight,
       b.exh_time,
       NULL::NUMERIC                                AS tilt_deg,
       (b.status = 'F')                             AS fs_flag,
       NULL::NUMERIC                                AS bf_st_time,
       NULL::INTEGER                                AS bf_course,
       b.course                                     AS course,
       CASE WHEN b.status = 'F' THEN -b.start_timing ELSE b.start_timing END AS st_time
FROM   base b
LEFT JOIN prog p
       ON p.race_key = b.race_key AND p.lane = b.lane
ORDER BY b.race_key, b.lane;

-- 天候 ------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.weather AS
SELECT DISTINCT ON (rk)
       rk AS race_key,
       NULL::NUMERIC                               AS air_temp,
       r.wind_speed_m::NUMERIC                     AS wind_speed,
       CASE r.wind_direction
           WHEN '北'     THEN 0
           WHEN '北北東' THEN 22.5
           WHEN '北東'   THEN 45
           WHEN '東北東' THEN 67.5
           WHEN '東'     THEN 90
           WHEN '東南東' THEN 112.5
           WHEN '南東'   THEN 135
           WHEN '南南東' THEN 157.5
           WHEN '南'     THEN 180
           WHEN '南南西' THEN 202.5
           WHEN '南西'   THEN 225
           WHEN '西南西' THEN 247.5
           WHEN '西'     THEN 270
           WHEN '西北西' THEN 292.5
           WHEN '北西'   THEN 315
           WHEN '北北西' THEN 337.5
           ELSE NULL
       END AS wind_dir_deg,
       (r.wave_height_cm::NUMERIC / 100.0)         AS wave_height,  -- m に換算（cm→m）
       NULL::NUMERIC                               AS water_temp,
       r.weather                                   AS weather_txt
FROM   raw.race r
JOIN   event e ON r.event_id = e.event_id
JOIN   venue v ON e.venue_id = v.venue_id
CROSS JOIN LATERAL (
    SELECT core.f_race_key(e.event_date, r.race_no, v.name) AS rk
) k
ORDER BY rk, r.race_id DESC;

-- 払戻（汎用：全券種） -------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.payouts AS
WITH p AS (
  SELECT
    core.f_race_key(e.event_date, r.race_no, v.name) AS race_key,
    CASE
      WHEN p.bet_type IN ('三連単','３連単','3連単','3T','3t','TRIFECTA') THEN '3連単'
      WHEN p.bet_type IN ('三連複','３連複','3連複','3F','3f','TRIO') THEN '3連複'
      WHEN p.bet_type IN ('二連単','２連単','2連単','2T','2t','EXACTA') THEN '2連単'
      WHEN p.bet_type IN ('二連複','２連複','2連複','2F','2f','QUINELLA') THEN '二連複'
      WHEN p.bet_type IN ('拡連複','ワイド','WIDE') THEN '拡連複'
      WHEN p.bet_type IN ('単勝','WIN') THEN '単勝'
      WHEN p.bet_type IN ('複勝','PLACE') THEN '複勝'
      ELSE p.bet_type
    END AS bet_type,
    p.combination,
    p.payout_yen,
    p.popularity_rank,
    p.payout_id
  FROM raw.payout p
  JOIN raw.race r ON p.race_id = r.race_id
  JOIN event e ON r.event_id = e.event_id
  JOIN venue v ON e.venue_id = v.venue_id
)
SELECT DISTINCT ON (race_key, bet_type, combination)
       race_key, bet_type, combination, payout_yen, popularity_rank
FROM p
ORDER BY race_key, bet_type, combination, payout_id DESC;

-- 払戻（3連単：レーン分解） ---------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS core.payout3t AS
WITH base AS (
  SELECT
    race_key,
    -- 全角数字→半角へ正規化後、「数字以外」をすべて除去
    regexp_replace(
      translate(combination, '０１２３４５６７８９', '0123456789'),
      '[^0-9]+', '', 'g'
    ) AS comb,
    payout_yen,
    popularity_rank
  FROM core.payouts
  -- ラベルは環境により半角/全角が混在するため両方許容
  WHERE bet_type IN ('3連単','３連単')
),
clean AS (
  -- 1〜6 の数字がちょうど3桁だけ残ったものに限定（返還/特払などを除外）
  SELECT race_key, comb, payout_yen, popularity_rank
  FROM base
  WHERE comb ~ '^[1-6]{3}$'
)
SELECT DISTINCT ON (race_key, first_lane, second_lane, third_lane)
  race_key,
  substring(comb, 1, 1)::int AS first_lane,
  substring(comb, 2, 1)::int AS second_lane,
  substring(comb, 3, 1)::int AS third_lane,
  payout_yen,
  popularity_rank
FROM clean
ORDER BY race_key, first_lane, second_lane, third_lane;

-- Indexes for payout join performance
CREATE INDEX IF NOT EXISTS idx_core_payouts_key_type_comb
  ON core.payouts (race_key, bet_type, combination);
CREATE INDEX IF NOT EXISTS idx_core_payout3t_key_lanes
  ON core.payout3t (race_key, first_lane, second_lane, third_lane);

-- ==========================================================
-- REFRESH MATERIALIZED VIEWS
-- ==========================================================

REFRESH MATERIALIZED VIEW core.races;
REFRESH MATERIALIZED VIEW core.results;
REFRESH MATERIALIZED VIEW core.weather;
REFRESH MATERIALIZED VIEW core.payouts;
REFRESH MATERIALIZED VIEW core.payout3t;

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
