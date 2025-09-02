/*------------------------------------------------------------
  03_merge_staging.sql  (NOT NULL 対応 + CTE 分割 + right()対応)
------------------------------------------------------------*/

------------------------------------------------------------
-- 4) odds3t_staging → raw.odds3t
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.odds3t
    ADD CONSTRAINT odds3t_unique UNIQUE (stadium, race_date, race_no,
                                         first_lane, second_lane, third_lane);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint odds3t_unique already exists.';
END
$$;

WITH os AS (
    SELECT *,
           (regexp_match(right(source_file, 100),
                  'wakamatsu_odds3t_[0-9]+_([0-9]{8})_([0-9]+)_odds_matrix\.csv$'))[1] AS yyyymmdd,
            (regexp_match(right(source_file, 100),
                  'wakamatsu_odds3t_[0-9]+_([0-9]{8})_([0-9]+)_odds_matrix\.csv$'))[2] AS race_no_ex
    FROM raw.odds3t_staging
    WHERE right(source_file, 100) ~ 'wakamatsu_odds3t_[0-9]+_[0-9]{8}_[0-9]+_odds_matrix\.csv$'
)
INSERT INTO raw.odds3t (
    stadium, race_date, race_no,
    first_lane, second_lane, third_lane, odds, source_file
)
SELECT
    '若松',
    to_date(yyyymmdd, 'YYYYMMDD'),
    race_no_ex::int,
    first_lane, second_lane, third_lane,
    odds, source_file
FROM os
ON CONFLICT DO NOTHING;

------------------------------------------------------------
-- 5) racertech_staging → raw.racertech
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.racertech
    ADD CONSTRAINT racertech_unique UNIQUE (reg_no, course_label);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint racertech_unique already exists.';
END
$$;

INSERT INTO raw.racertech (
    reg_no, course, starts, firsts,
    nige, sashi, makuri, makurisashi, nuki, megumare
)
SELECT
    reg_no, CAST(REGEXP_REPLACE(course_label, '[^0-9]', '', 'g') AS INTEGER) AS course, starts, firsts,
    nige, sashi, makuri, makurisashi, nuki, megumare
FROM raw.racertech_staging
ON CONFLICT DO NOTHING;

------------------------------------------------------------
-- 6) racerstadium_staging → raw.racerstadium
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.racerstadium
    ADD CONSTRAINT racerstadium_unique UNIQUE (reg_no, stadium_label);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint racerstadium_unique already exists.';
END
$$;

INSERT INTO raw.racerstadium (
    reg_no, stadium_code, meeting_entries, starts, firsts,
    winrate, first_rate, two_rate, three_rate,
    finalist_cnt, champion_cnt, avg_st
)
SELECT
    reg_no,     
    CASE stadium_label
      WHEN '桐生'     THEN '01'
      WHEN '戸田'     THEN '02'
      WHEN '江戸川'   THEN '03'
      WHEN '平和島'   THEN '04'
      WHEN '多摩川'   THEN '05'
      WHEN '浜名湖'   THEN '06'
      WHEN '蒲郡'     THEN '07'
      WHEN '常滑'     THEN '08'
      WHEN '津'       THEN '09'
      WHEN '三国'     THEN '10'
      WHEN 'びわこ'   THEN '11'
      WHEN '住之江'   THEN '12'
      WHEN '尼崎'     THEN '13'
      WHEN '鳴門'     THEN '14'
      WHEN '丸亀'     THEN '15'
      WHEN '児島'     THEN '16'
      WHEN '宮島'     THEN '17'
      WHEN '徳山'     THEN '18'
      WHEN '下関'     THEN '19'
      WHEN '若松'     THEN '20'
      WHEN '芦屋'     THEN '21'
      WHEN '福岡'     THEN '22'
      WHEN '唐津'     THEN '23'
      WHEN '大村'     THEN '24'
      ELSE NULL
    END AS stadium_code, 
    meeting_entries, starts, firsts,
    winrate, first_rate, two_rate, three_rate,
    finalist_cnt, champion_cnt, avg_st
FROM raw.racerstadium_staging
ON CONFLICT DO NOTHING;

------------------------------------------------------------
-- 7) racerresult1_staging → raw.racerresult1
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.racerresult1
    ADD CONSTRAINT racerresult1_unique UNIQUE (reg_no, grade);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint racerresult1_unique already exists.';
END
$$;

INSERT INTO raw.racerresult1 (
    reg_no, grade, meeting_entries, starts, firsts,
    winrate, first_rate, two_rate, three_rate,
    finalist_cnt, champion_cnt, avg_st, avg_st_rank
)
SELECT
    reg_no, grade, meeting_entries, starts, firsts,
    winrate, first_rate, two_rate, three_rate,
    finalist_cnt, champion_cnt, avg_st, avg_st_rank
FROM raw.racerresult1_staging
ON CONFLICT DO NOTHING;

------------------------------------------------------------
-- 8) racerresult2_staging → raw.racerresult2
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.racerresult2
    ADD CONSTRAINT racerresult2_unique UNIQUE (reg_no, grade);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint racerresult2_unique already exists.';
END
$$;

INSERT INTO raw.racerresult2 (
    reg_no, grade, starts, firsts, seconds, thirds,
    fourths, fifths, sixths,
    s0, s1, s2, f_cnt, l0, l1, k0, k1
)
SELECT
    reg_no, grade, starts, firsts, seconds, thirds,
    fourths, fifths, sixths,
    s0, s1, s2, f_cnt, l0, l1, k0, k1
FROM raw.racerresult2_staging
ON CONFLICT DO NOTHING;

------------------------------------------------------------
-- 9) racercourse_staging → raw.racercourse
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.racercourse
    ADD CONSTRAINT racercourse_unique UNIQUE (reg_no, course_label);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint racercourse_unique already exists.';
END
$$;

INSERT INTO raw.racercourse (
    reg_no, course, starts, firsts,
    first_rate, two_rate, three_rate,
    avg_st, avg_st_rank
)
SELECT
    reg_no, CAST(REGEXP_REPLACE(course_label, '[^0-9]', '', 'g') AS INTEGER) AS course, starts, firsts,
    first_rate, two_rate, three_rate,
    avg_st, avg_st_rank
FROM raw.racercourse_staging
ON CONFLICT DO NOTHING;

------------------------------------------------------------
-- 10) racerboatcourse_staging → raw.racerboatcourse
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.racerboatcourse
    ADD CONSTRAINT racerboatcourse_unique UNIQUE (reg_no, boat_no_label);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint racerboatcourse_unique already exists.';
END
$$;

INSERT INTO raw.racerboatcourse (
    reg_no, lane, starts,
    lane1_cnt, lane2_cnt, lane3_cnt, lane4_cnt,
    lane5_cnt, lane6_cnt, other_cnt
)
SELECT
    reg_no, CAST(REGEXP_REPLACE(boat_no_label, '[^0-9]', '', 'g') AS INTEGER) AS lane, starts,
    lane1_cnt, lane2_cnt, lane3_cnt, lane4_cnt,
    lane5_cnt, lane6_cnt, other_cnt
FROM raw.racerboatcourse_staging
ON CONFLICT DO NOTHING;

------------------------------------------------------------
-- 11) racerboat_staging → raw.racerboat
------------------------------------------------------------
DO $$
BEGIN
    ALTER TABLE raw.racerboat
    ADD CONSTRAINT racerboat_unique UNIQUE (reg_no, boat_no_label);
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint racerboat_unique already exists.';
END
$$;

INSERT INTO raw.racerboat (
    reg_no, lane, starts, firsts,
    first_rate, two_rate, three_rate,
    finalist_cnt, champion_cnt
)
SELECT
    reg_no, CAST(REGEXP_REPLACE(boat_no_label, '[^0-9]', '', 'g') AS INTEGER) AS lane, starts, firsts,
    first_rate, two_rate, three_rate,
    finalist_cnt, champion_cnt
FROM raw.racerboat_staging
ON CONFLICT DO NOTHING;
