
/*------------------------------------------------------------
  06_views_feat.sql
  FEAT レイヤ（学習用特徴量）
  IF NOT EXISTS + REFRESH 運用
------------------------------------------------------------*/

/* ---------- flat 化（feat.boat_flat） -------------------- */
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

/* ---------- flat + レーサー統計付与（feat.boat_flat_enriched） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.boat_flat_enriched AS
SELECT DISTINCT ON (bf.race_key, bf.lane)
       bf.*,
       rsc.nige,
       rsc.sashi,
       rsc.makuri,
       rsc.makurisashi,
       rsc.nuki,
       rsc.megumare,
       rsc.first_rate   AS course_first_rate,
       rsc.two_rate     AS course_two_rate,
       rsc.three_rate   AS course_three_rate,
       rsc.avg_st       AS course_avg_st,
       rsg.winrate      AS grade_winrate,
       rsg.first_rate   AS grade_first_rate,
       rsg.two_rate     AS grade_two_rate,
       rsg.three_rate   AS grade_three_rate,
       rsg.finalist_cnt AS grade_finalist_cnt,
       rsg.champion_cnt AS grade_champion_cnt
FROM feat.boat_flat bf

/* --- コース別統計: 空白を無視してマッチ -------- */
LEFT JOIN core.racerstats_course rsc
       ON rsc.reg_no = bf.racer_id
      AND regexp_replace(rsc.course_label, '\s', '', 'g')
          = (bf.course::text || 'コース')

/* --- グレード別統計: レーサー単位に集約し 1 行だけ JOIN --- */
LEFT JOIN (
    SELECT
        reg_no,
        AVG(winrate)      AS winrate,
        AVG(first_rate)   AS first_rate,
        AVG(two_rate)     AS two_rate,
        AVG(three_rate)   AS three_rate,
        SUM(finalist_cnt) AS finalist_cnt,
        SUM(champion_cnt) AS champion_cnt
    FROM core.racerstats_grade
    GROUP BY reg_no
) rsg
       ON rsg.reg_no = bf.racer_id
ORDER BY bf.race_key, bf.lane;

/* ---------- レーサー辞書（feat.racer_flat） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.racer_flat AS
SELECT
       bf.racer_id,
       MAX(bf.age)                           AS age,
       MAX(bf.class_now)                     AS class_now,
       AVG(bf.winrate_natl)                  AS winrate_natl_avg,
       AVG(bf.course_first_rate)             AS course_first_rate_avg,
       AVG(bf.grade_winrate)                 AS grade_winrate_avg,
       SUM(bf.nige)                          AS nige_cnt,
       SUM(bf.makuri)                        AS makuri_cnt,
       SUM(bf.sashi)                         AS sashi_cnt,
       SUM(bf.makurisashi)                   AS makurisashi_cnt,
       SUM(bf.nuki)                          AS nuki_cnt,
       SUM(bf.megumare)                      AS megumare_cnt,
       COUNT(*)                              AS starts_total
FROM feat.boat_flat_enriched bf
GROUP BY bf.racer_id;


/* ---------- レース × レーン × レーサー統計（feat.race_racer_features） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.race_racer_features AS
SELECT
    bf.race_key,
    bf.lane,
    bf.racer_id,
    rf.course_first_rate_avg,
    rf.grade_winrate_avg,
    rf.starts_total,
    rf.nige_cnt,
    rf.makuri_cnt,
    rf.sashi_cnt,
    rf.makurisashi_cnt,
    rf.nuki_cnt,
    rf.megumare_cnt
FROM feat.boat_flat bf
LEFT JOIN feat.racer_flat rf
       ON rf.racer_id = bf.racer_id;

/* ---------- 学習用特徴量（feat.train_features） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.train_features AS
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
    MAX(CASE WHEN lane=1 THEN st_time END) AS lane1_st,
    MAX(CASE WHEN lane=1 THEN course END) AS lane1_course,
    MAX(CASE WHEN lane=1 THEN bf_st_time END) AS lane1_bf_st_time,
    MAX(CASE WHEN lane=1 THEN bf_course END) AS lane1_bf_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=1)  AS lane1_fs_flag,
    MAX(CASE WHEN lane=1 THEN rank END) AS lane1_rank,
    MAX(CASE WHEN lane=1 THEN class_now END) AS lane1_class_now,
    MAX(CASE WHEN lane=1 THEN ability_now END) AS lane1_ability_now,
    MAX(CASE WHEN lane=1 THEN winrate_natl END) AS lane1_winrate_natl,
    MAX(CASE WHEN lane=1 THEN "2in_natl" END) AS lane1_2in_natl,
    MAX(CASE WHEN lane=1 THEN "3in_natl" END) AS lane1_3in_natl,
    MAX(CASE WHEN lane=1 THEN age END) AS lane1_age,
    MAX(CASE WHEN lane=1 THEN class_hist1 END) AS lane1_class_hist1,
    MAX(CASE WHEN lane=1 THEN class_hist2 END) AS lane1_class_hist2,
    MAX(CASE WHEN lane=1 THEN class_hist3 END) AS lane1_class_hist3,
    -- MAX(CASE WHEN lane=1 THEN ability_prev END) AS lane1_ability_prev,
    MAX(CASE WHEN lane=1 THEN "F_now" END) AS lane1_F_now,
    MAX(CASE WHEN lane=1 THEN "L_now" END) AS lane1_L_now,
    MAX(CASE WHEN lane=1 THEN nat_1st END) AS lane1_nat_1st,
    MAX(CASE WHEN lane=1 THEN nat_2nd END) AS lane1_nat_2nd,
    MAX(CASE WHEN lane=1 THEN nat_3rd END) AS lane1_nat_3rd,
    MAX(CASE WHEN lane=1 THEN nat_starts END) AS lane1_nat_starts,
    MAX(CASE WHEN lane=1 THEN loc_1st END) AS lane1_loc_1st,
    MAX(CASE WHEN lane=1 THEN loc_2nd END) AS lane1_loc_2nd,
    MAX(CASE WHEN lane=1 THEN loc_3rd END) AS lane1_loc_3rd,
    MAX(CASE WHEN lane=1 THEN loc_starts END) AS lane1_loc_starts,
    MAX(CASE WHEN lane=1 THEN motor_no END) AS lane1_motor_no,
    MAX(CASE WHEN lane=1 THEN motor_2in END) AS lane1_motor_2in,
    MAX(CASE WHEN lane=1 THEN motor_3in END) AS lane1_motor_3in,
    MAX(CASE WHEN lane=1 THEN mot_1st END) AS lane1_mot_1st,
    MAX(CASE WHEN lane=1 THEN mot_2nd END) AS lane1_mot_2nd,
    MAX(CASE WHEN lane=1 THEN mot_3rd END) AS lane1_mot_3rd,
    MAX(CASE WHEN lane=1 THEN mot_starts END) AS lane1_mot_starts,
    MAX(CASE WHEN lane=1 THEN boat_no_hw END) AS lane1_boat_no_hw,
    MAX(CASE WHEN lane=1 THEN boat_2in END) AS lane1_boat_2in,
    MAX(CASE WHEN lane=1 THEN boat_3in END) AS lane1_boat_3in,
    MAX(CASE WHEN lane=1 THEN boa_1st END) AS lane1_boa_1st,
    MAX(CASE WHEN lane=1 THEN boa_2nd END) AS lane1_boa_2nd,
    MAX(CASE WHEN lane=1 THEN boa_3rd END) AS lane1_boa_3rd,
    MAX(CASE WHEN lane=1 THEN boa_starts END) AS lane1_boa_starts,
    MAX(CASE WHEN lane=2 THEN racer_id END) AS lane2_racer_id,
    MAX(CASE WHEN lane=2 THEN weight END) AS lane2_weight,
    MAX(CASE WHEN lane=2 THEN exh_time END) AS lane2_exh_time,
    MAX(CASE WHEN lane=2 THEN bf_st_time END) AS lane2_bf_st_time,
    MAX(CASE WHEN lane=2 THEN bf_course END) AS lane2_bf_course,
    MAX(CASE WHEN lane=2 THEN st_time END) AS lane2_st,
    MAX(CASE WHEN lane=2 THEN course END) AS lane2_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=2)  AS lane2_fs_flag,
    MAX(CASE WHEN lane=2 THEN rank END) AS lane2_rank,
    MAX(CASE WHEN lane=2 THEN class_now END) AS lane2_class_now,
    MAX(CASE WHEN lane=2 THEN ability_now END) AS lane2_ability_now,
    MAX(CASE WHEN lane=2 THEN winrate_natl END) AS lane2_winrate_natl,
    MAX(CASE WHEN lane=2 THEN "2in_natl" END) AS lane2_2in_natl,
    MAX(CASE WHEN lane=2 THEN "3in_natl" END) AS lane2_3in_natl,
    MAX(CASE WHEN lane=2 THEN age END) AS lane2_age,
    MAX(CASE WHEN lane=2 THEN class_hist1 END) AS lane2_class_hist1,
    MAX(CASE WHEN lane=2 THEN class_hist2 END) AS lane2_class_hist2,
    MAX(CASE WHEN lane=2 THEN class_hist3 END) AS lane2_class_hist3,
    -- MAX(CASE WHEN lane=2 THEN ability_prev END) AS lane2_ability_prev,
    MAX(CASE WHEN lane=2 THEN "F_now" END) AS lane2_F_now,
    MAX(CASE WHEN lane=2 THEN "L_now" END) AS lane2_L_now,
    MAX(CASE WHEN lane=2 THEN nat_1st END) AS lane2_nat_1st,
    MAX(CASE WHEN lane=2 THEN nat_2nd END) AS lane2_nat_2nd,
    MAX(CASE WHEN lane=2 THEN nat_3rd END) AS lane2_nat_3rd,
    MAX(CASE WHEN lane=2 THEN nat_starts END) AS lane2_nat_starts,
    MAX(CASE WHEN lane=2 THEN loc_1st END) AS lane2_loc_1st,
    MAX(CASE WHEN lane=2 THEN loc_2nd END) AS lane2_loc_2nd,
    MAX(CASE WHEN lane=2 THEN loc_3rd END) AS lane2_loc_3rd,
    MAX(CASE WHEN lane=2 THEN loc_starts END) AS lane2_loc_starts,
    MAX(CASE WHEN lane=2 THEN motor_no END) AS lane2_motor_no,
    MAX(CASE WHEN lane=2 THEN motor_2in END) AS lane2_motor_2in,
    MAX(CASE WHEN lane=2 THEN motor_3in END) AS lane2_motor_3in,
    MAX(CASE WHEN lane=2 THEN mot_1st END) AS lane2_mot_1st,
    MAX(CASE WHEN lane=2 THEN mot_2nd END) AS lane2_mot_2nd,
    MAX(CASE WHEN lane=2 THEN mot_3rd END) AS lane2_mot_3rd,
    MAX(CASE WHEN lane=2 THEN mot_starts END) AS lane2_mot_starts,
    MAX(CASE WHEN lane=2 THEN boat_no_hw END) AS lane2_boat_no_hw,
    MAX(CASE WHEN lane=2 THEN boat_2in END) AS lane2_boat_2in,
    MAX(CASE WHEN lane=2 THEN boat_3in END) AS lane2_boat_3in,
    MAX(CASE WHEN lane=2 THEN boa_1st END) AS lane2_boa_1st,
    MAX(CASE WHEN lane=2 THEN boa_2nd END) AS lane2_boa_2nd,
    MAX(CASE WHEN lane=2 THEN boa_3rd END) AS lane2_boa_3rd,
    MAX(CASE WHEN lane=2 THEN boa_starts END) AS lane2_boa_starts,
    MAX(CASE WHEN lane=3 THEN racer_id END) AS lane3_racer_id,
    MAX(CASE WHEN lane=3 THEN weight END) AS lane3_weight,
    MAX(CASE WHEN lane=3 THEN exh_time END) AS lane3_exh_time,
    MAX(CASE WHEN lane=3 THEN bf_st_time END) AS lane3_bf_st_time,
    MAX(CASE WHEN lane=3 THEN bf_course END) AS lane3_bf_course,
    MAX(CASE WHEN lane=3 THEN st_time END) AS lane3_st,
    MAX(CASE WHEN lane=3 THEN course END) AS lane3_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=3)  AS lane3_fs_flag,
    MAX(CASE WHEN lane=3 THEN rank END) AS lane3_rank,
    MAX(CASE WHEN lane=3 THEN class_now END) AS lane3_class_now,
    MAX(CASE WHEN lane=3 THEN ability_now END) AS lane3_ability_now,
    MAX(CASE WHEN lane=3 THEN winrate_natl END) AS lane3_winrate_natl,
    MAX(CASE WHEN lane=3 THEN "2in_natl" END) AS lane3_2in_natl,
    MAX(CASE WHEN lane=3 THEN "3in_natl" END) AS lane3_3in_natl,
    MAX(CASE WHEN lane=3 THEN age END) AS lane3_age,
    MAX(CASE WHEN lane=3 THEN class_hist1 END) AS lane3_class_hist1,
    MAX(CASE WHEN lane=3 THEN class_hist2 END) AS lane3_class_hist2,
    MAX(CASE WHEN lane=3 THEN class_hist3 END) AS lane3_class_hist3,
    -- MAX(CASE WHEN lane=3 THEN ability_prev END) AS lane3_ability_prev,
    MAX(CASE WHEN lane=3 THEN "F_now" END) AS lane3_F_now,
    MAX(CASE WHEN lane=3 THEN "L_now" END) AS lane3_L_now,
    MAX(CASE WHEN lane=3 THEN nat_1st END) AS lane3_nat_1st,
    MAX(CASE WHEN lane=3 THEN nat_2nd END) AS lane3_nat_2nd,
    MAX(CASE WHEN lane=3 THEN nat_3rd END) AS lane3_nat_3rd,
    MAX(CASE WHEN lane=3 THEN nat_starts END) AS lane3_nat_starts,
    MAX(CASE WHEN lane=3 THEN loc_1st END) AS lane3_loc_1st,
    MAX(CASE WHEN lane=3 THEN loc_2nd END) AS lane3_loc_2nd,
    MAX(CASE WHEN lane=3 THEN loc_3rd END) AS lane3_loc_3rd,
    MAX(CASE WHEN lane=3 THEN loc_starts END) AS lane3_loc_starts,
    MAX(CASE WHEN lane=3 THEN motor_no END) AS lane3_motor_no,
    MAX(CASE WHEN lane=3 THEN motor_2in END) AS lane3_motor_2in,
    MAX(CASE WHEN lane=3 THEN motor_3in END) AS lane3_motor_3in,
    MAX(CASE WHEN lane=3 THEN mot_1st END) AS lane3_mot_1st,
    MAX(CASE WHEN lane=3 THEN mot_2nd END) AS lane3_mot_2nd,
    MAX(CASE WHEN lane=3 THEN mot_3rd END) AS lane3_mot_3rd,
    MAX(CASE WHEN lane=3 THEN mot_starts END) AS lane3_mot_starts,
    MAX(CASE WHEN lane=3 THEN boat_no_hw END) AS lane3_boat_no_hw,
    MAX(CASE WHEN lane=3 THEN boat_2in END) AS lane3_boat_2in,
    MAX(CASE WHEN lane=3 THEN boat_3in END) AS lane3_boat_3in,
    MAX(CASE WHEN lane=3 THEN boa_1st END) AS lane3_boa_1st,
    MAX(CASE WHEN lane=3 THEN boa_2nd END) AS lane3_boa_2nd,
    MAX(CASE WHEN lane=3 THEN boa_3rd END) AS lane3_boa_3rd,
    MAX(CASE WHEN lane=3 THEN boa_starts END) AS lane3_boa_starts,
    MAX(CASE WHEN lane=4 THEN racer_id END) AS lane4_racer_id,
    MAX(CASE WHEN lane=4 THEN weight END) AS lane4_weight,
    MAX(CASE WHEN lane=4 THEN exh_time END) AS lane4_exh_time,
    MAX(CASE WHEN lane=4 THEN bf_st_time END) AS lane4_bf_st_time,
    MAX(CASE WHEN lane=4 THEN bf_course END) AS lane4_bf_course,
    MAX(CASE WHEN lane=4 THEN st_time END) AS lane4_st,
    MAX(CASE WHEN lane=4 THEN course END) AS lane4_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=4)  AS lane4_fs_flag,
    MAX(CASE WHEN lane=4 THEN rank END) AS lane4_rank,
    MAX(CASE WHEN lane=4 THEN class_now END) AS lane4_class_now,
    MAX(CASE WHEN lane=4 THEN ability_now END) AS lane4_ability_now,
    MAX(CASE WHEN lane=4 THEN winrate_natl END) AS lane4_winrate_natl,
    MAX(CASE WHEN lane=4 THEN "2in_natl" END) AS lane4_2in_natl,
    MAX(CASE WHEN lane=4 THEN "3in_natl" END) AS lane4_3in_natl,
    MAX(CASE WHEN lane=4 THEN age END) AS lane4_age,
    MAX(CASE WHEN lane=4 THEN class_hist1 END) AS lane4_class_hist1,
    MAX(CASE WHEN lane=4 THEN class_hist2 END) AS lane4_class_hist2,
    MAX(CASE WHEN lane=4 THEN class_hist3 END) AS lane4_class_hist3,
    -- MAX(CASE WHEN lane=4 THEN ability_prev END) AS lane4_ability_prev,
    MAX(CASE WHEN lane=4 THEN "F_now" END) AS lane4_F_now,
    MAX(CASE WHEN lane=4 THEN "L_now" END) AS lane4_L_now,
    MAX(CASE WHEN lane=4 THEN nat_1st END) AS lane4_nat_1st,
    MAX(CASE WHEN lane=4 THEN nat_2nd END) AS lane4_nat_2nd,
    MAX(CASE WHEN lane=4 THEN nat_3rd END) AS lane4_nat_3rd,
    MAX(CASE WHEN lane=4 THEN nat_starts END) AS lane4_nat_starts,
    MAX(CASE WHEN lane=4 THEN loc_1st END) AS lane4_loc_1st,
    MAX(CASE WHEN lane=4 THEN loc_2nd END) AS lane4_loc_2nd,
    MAX(CASE WHEN lane=4 THEN loc_3rd END) AS lane4_loc_3rd,
    MAX(CASE WHEN lane=4 THEN loc_starts END) AS lane4_loc_starts,
    MAX(CASE WHEN lane=4 THEN motor_no END) AS lane4_motor_no,
    MAX(CASE WHEN lane=4 THEN motor_2in END) AS lane4_motor_2in,
    MAX(CASE WHEN lane=4 THEN motor_3in END) AS lane4_motor_3in,
    MAX(CASE WHEN lane=4 THEN mot_1st END) AS lane4_mot_1st,
    MAX(CASE WHEN lane=4 THEN mot_2nd END) AS lane4_mot_2nd,
    MAX(CASE WHEN lane=4 THEN mot_3rd END) AS lane4_mot_3rd,
    MAX(CASE WHEN lane=4 THEN mot_starts END) AS lane4_mot_starts,
    MAX(CASE WHEN lane=4 THEN boat_no_hw END) AS lane4_boat_no_hw,
    MAX(CASE WHEN lane=4 THEN boat_2in END) AS lane4_boat_2in,
    MAX(CASE WHEN lane=4 THEN boat_3in END) AS lane4_boat_3in,
    MAX(CASE WHEN lane=4 THEN boa_1st END) AS lane4_boa_1st,
    MAX(CASE WHEN lane=4 THEN boa_2nd END) AS lane4_boa_2nd,
    MAX(CASE WHEN lane=4 THEN boa_3rd END) AS lane4_boa_3rd,
    MAX(CASE WHEN lane=4 THEN boa_starts END) AS lane4_boa_starts,
    MAX(CASE WHEN lane=5 THEN racer_id END) AS lane5_racer_id,
    MAX(CASE WHEN lane=5 THEN weight END) AS lane5_weight,
    MAX(CASE WHEN lane=5 THEN exh_time END) AS lane5_exh_time,
    MAX(CASE WHEN lane=5 THEN bf_st_time END) AS lane5_bf_st_time,
    MAX(CASE WHEN lane=5 THEN bf_course END) AS lane5_bf_course,
    MAX(CASE WHEN lane=5 THEN st_time END) AS lane5_st,
    MAX(CASE WHEN lane=5 THEN course END) AS lane5_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=5)  AS lane5_fs_flag,
    MAX(CASE WHEN lane=5 THEN rank END) AS lane5_rank,
    MAX(CASE WHEN lane=5 THEN class_now END) AS lane5_class_now,
    MAX(CASE WHEN lane=5 THEN ability_now END) AS lane5_ability_now,
    MAX(CASE WHEN lane=5 THEN winrate_natl END) AS lane5_winrate_natl,
    MAX(CASE WHEN lane=5 THEN "2in_natl" END) AS lane5_2in_natl,
    MAX(CASE WHEN lane=5 THEN "3in_natl" END) AS lane5_3in_natl,
    MAX(CASE WHEN lane=5 THEN age END) AS lane5_age,
    MAX(CASE WHEN lane=5 THEN class_hist1 END) AS lane5_class_hist1,
    MAX(CASE WHEN lane=5 THEN class_hist2 END) AS lane5_class_hist2,
    MAX(CASE WHEN lane=5 THEN class_hist3 END) AS lane5_class_hist3,
    -- MAX(CASE WHEN lane=5 THEN ability_prev END) AS lane5_ability_prev,
    MAX(CASE WHEN lane=5 THEN "F_now" END) AS lane5_F_now,
    MAX(CASE WHEN lane=5 THEN "L_now" END) AS lane5_L_now,
    MAX(CASE WHEN lane=5 THEN nat_1st END) AS lane5_nat_1st,
    MAX(CASE WHEN lane=5 THEN nat_2nd END) AS lane5_nat_2nd,
    MAX(CASE WHEN lane=5 THEN nat_3rd END) AS lane5_nat_3rd,
    MAX(CASE WHEN lane=5 THEN nat_starts END) AS lane5_nat_starts,
    MAX(CASE WHEN lane=5 THEN loc_1st END) AS lane5_loc_1st,
    MAX(CASE WHEN lane=5 THEN loc_2nd END) AS lane5_loc_2nd,
    MAX(CASE WHEN lane=5 THEN loc_3rd END) AS lane5_loc_3rd,
    MAX(CASE WHEN lane=5 THEN loc_starts END) AS lane5_loc_starts,
    MAX(CASE WHEN lane=5 THEN motor_no END) AS lane5_motor_no,
    MAX(CASE WHEN lane=5 THEN motor_2in END) AS lane5_motor_2in,
    MAX(CASE WHEN lane=5 THEN motor_3in END) AS lane5_motor_3in,
    MAX(CASE WHEN lane=5 THEN mot_1st END) AS lane5_mot_1st,
    MAX(CASE WHEN lane=5 THEN mot_2nd END) AS lane5_mot_2nd,
    MAX(CASE WHEN lane=5 THEN mot_3rd END) AS lane5_mot_3rd,
    MAX(CASE WHEN lane=5 THEN mot_starts END) AS lane5_mot_starts,
    MAX(CASE WHEN lane=5 THEN boat_no_hw END) AS lane5_boat_no_hw,
    MAX(CASE WHEN lane=5 THEN boat_2in END) AS lane5_boat_2in,
    MAX(CASE WHEN lane=5 THEN boat_3in END) AS lane5_boat_3in,
    MAX(CASE WHEN lane=5 THEN boa_1st END) AS lane5_boa_1st,
    MAX(CASE WHEN lane=5 THEN boa_2nd END) AS lane5_boa_2nd,
    MAX(CASE WHEN lane=5 THEN boa_3rd END) AS lane5_boa_3rd,
    MAX(CASE WHEN lane=5 THEN boa_starts END) AS lane5_boa_starts,
    MAX(CASE WHEN lane=6 THEN racer_id END) AS lane6_racer_id,
    MAX(CASE WHEN lane=6 THEN weight END) AS lane6_weight,
    MAX(CASE WHEN lane=6 THEN exh_time END) AS lane6_exh_time,
    MAX(CASE WHEN lane=6 THEN bf_st_time END) AS lane6_bf_st_time,
    MAX(CASE WHEN lane=6 THEN bf_course END) AS lane6_bf_course,
    MAX(CASE WHEN lane=6 THEN st_time END) AS lane6_st,
    MAX(CASE WHEN lane=6 THEN course END) AS lane6_course,
    BOOL_OR(fs_flag) FILTER (WHERE lane=6)  AS lane6_fs_flag,
    MAX(CASE WHEN lane=6 THEN rank END) AS lane6_rank,
    MAX(CASE WHEN lane=6 THEN class_now END) AS lane6_class_now,
    MAX(CASE WHEN lane=6 THEN ability_now END) AS lane6_ability_now,
    MAX(CASE WHEN lane=6 THEN winrate_natl END) AS lane6_winrate_natl,
    MAX(CASE WHEN lane=6 THEN "2in_natl" END) AS lane6_2in_natl,
    MAX(CASE WHEN lane=6 THEN "3in_natl" END) AS lane6_3in_natl,
    MAX(CASE WHEN lane=6 THEN age END) AS lane6_age,
    MAX(CASE WHEN lane=6 THEN class_hist1 END) AS lane6_class_hist1,
    MAX(CASE WHEN lane=6 THEN class_hist2 END) AS lane6_class_hist2,
    MAX(CASE WHEN lane=6 THEN class_hist3 END) AS lane6_class_hist3,
    -- MAX(CASE WHEN lane=6 THEN ability_prev END) AS lane6_ability_prev,
    MAX(CASE WHEN lane=6 THEN "F_now" END) AS lane6_F_now,
    MAX(CASE WHEN lane=6 THEN "L_now" END) AS lane6_L_now,
    MAX(CASE WHEN lane=6 THEN nat_1st END) AS lane6_nat_1st,
    MAX(CASE WHEN lane=6 THEN nat_2nd END) AS lane6_nat_2nd,
    MAX(CASE WHEN lane=6 THEN nat_3rd END) AS lane6_nat_3rd,
    MAX(CASE WHEN lane=6 THEN nat_starts END) AS lane6_nat_starts,
    MAX(CASE WHEN lane=6 THEN loc_1st END) AS lane6_loc_1st,
    MAX(CASE WHEN lane=6 THEN loc_2nd END) AS lane6_loc_2nd,
    MAX(CASE WHEN lane=6 THEN loc_3rd END) AS lane6_loc_3rd,
    MAX(CASE WHEN lane=6 THEN loc_starts END) AS lane6_loc_starts,
    MAX(CASE WHEN lane=6 THEN motor_no END) AS lane6_motor_no,
    MAX(CASE WHEN lane=6 THEN motor_2in END) AS lane6_motor_2in,
    MAX(CASE WHEN lane=6 THEN motor_3in END) AS lane6_motor_3in,
    MAX(CASE WHEN lane=6 THEN mot_1st END) AS lane6_mot_1st,
    MAX(CASE WHEN lane=6 THEN mot_2nd END) AS lane6_mot_2nd,
    MAX(CASE WHEN lane=6 THEN mot_3rd END) AS lane6_mot_3rd,
    MAX(CASE WHEN lane=6 THEN mot_starts END) AS lane6_mot_starts,
    MAX(CASE WHEN lane=6 THEN boat_no_hw END) AS lane6_boat_no_hw,
    MAX(CASE WHEN lane=6 THEN boat_2in END) AS lane6_boat_2in,
    MAX(CASE WHEN lane=6 THEN boat_3in END) AS lane6_boat_3in,
    MAX(CASE WHEN lane=6 THEN boa_1st END) AS lane6_boa_1st,
    MAX(CASE WHEN lane=6 THEN boa_2nd END) AS lane6_boa_2nd,
    MAX(CASE WHEN lane=6 THEN boa_3rd END) AS lane6_boa_3rd,
    MAX(CASE WHEN lane=6 THEN boa_starts END) AS lane6_boa_starts

FROM flat
GROUP BY race_key
HAVING COUNT(DISTINCT race_date)=1
   AND COUNT(DISTINCT venue)=1;

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


/* ---------- 学習用特徴量（feat.train_features3） ---------- *
 * 仕様:
 *   - ベースは train_features2 の全列 (tf.*)
 *   - レーサー統計は feat.racer_flat を lane ごとに JOIN
 *   - 追加列 (例): course_first_rate_avg, grade_winrate_avg, starts_total
 *   - 結果: train_features2 の列を “完全継承” しつつ統計列を上乗せ
 */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.train_features3 AS
SELECT
    tf.*,

    -- lane1 追加統計
    rf1.course_first_rate_avg AS lane1_course_first_rate_avg,
    rf1.grade_winrate_avg     AS lane1_grade_winrate_avg,
    rf1.starts_total          AS lane1_starts_total,

    -- lane2
    rf2.course_first_rate_avg AS lane2_course_first_rate_avg,
    rf2.grade_winrate_avg     AS lane2_grade_winrate_avg,
    rf2.starts_total          AS lane2_starts_total,

    -- lane3
    rf3.course_first_rate_avg AS lane3_course_first_rate_avg,
    rf3.grade_winrate_avg     AS lane3_grade_winrate_avg,
    rf3.starts_total          AS lane3_starts_total,

    -- lane4
    rf4.course_first_rate_avg AS lane4_course_first_rate_avg,
    rf4.grade_winrate_avg     AS lane4_grade_winrate_avg,
    rf4.starts_total          AS lane4_starts_total,

    -- lane5
    rf5.course_first_rate_avg AS lane5_course_first_rate_avg,
    rf5.grade_winrate_avg     AS lane5_grade_winrate_avg,
    rf5.starts_total          AS lane5_starts_total,

    -- lane6
    rf6.course_first_rate_avg AS lane6_course_first_rate_avg,
    rf6.grade_winrate_avg     AS lane6_grade_winrate_avg,
    rf6.starts_total          AS lane6_starts_total

FROM feat.train_features2 tf
LEFT JOIN feat.racer_flat rf1 ON rf1.racer_id = tf.lane1_racer_id
LEFT JOIN feat.racer_flat rf2 ON rf2.racer_id = tf.lane2_racer_id
LEFT JOIN feat.racer_flat rf3 ON rf3.racer_id = tf.lane3_racer_id
LEFT JOIN feat.racer_flat rf4 ON rf4.racer_id = tf.lane4_racer_id
LEFT JOIN feat.racer_flat rf5 ON rf5.racer_id = tf.lane5_racer_id
LEFT JOIN feat.racer_flat rf6 ON rf6.racer_id = tf.lane6_racer_id;

CREATE MATERIALIZED VIEW IF NOT EXISTS feat.racer_hist_30d AS
SELECT
    b.racer_id,
    COUNT(*)                                AS starts_30d,
    AVG(CASE WHEN r.rank = 1 THEN 1 ELSE 0 END)::float AS winrate_30d,
    AVG(b.exh_time)                         AS exh_mean_30d,
    AVG(b.st_time)                          AS st_mean_30d
FROM core.boat_info b
JOIN core.results r
  ON b.race_key = r.race_key AND b.lane = r.lane
JOIN core.races cr
  ON b.race_key = cr.race_key
WHERE cr.race_date >= NOW() - INTERVAL '30 days'
GROUP BY b.racer_id;


/* ---------- 評価用特徴量（feat.eval_features） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.eval_features AS
SELECT
    tf.*,
    o.first_lane,
    o.second_lane,
    o.third_lane,
    o.odds
FROM feat.train_features tf
JOIN core.odds3t o USING (race_key);

/* ---------- 評価用特徴量（feat.eval_features） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.eval_features2 AS
SELECT
    tf.*,
    o.first_lane,
    o.second_lane,
    o.third_lane,
    o.odds
FROM feat.train_features2 tf
JOIN core.odds3t o USING (race_key);

/* ---------- 評価用特徴量（feat.eval_features） ---------- */
CREATE MATERIALIZED VIEW IF NOT EXISTS feat.eval_features3 AS
SELECT
    tf.*,
    o.first_lane,
    o.second_lane,
    o.third_lane,
    o.odds
FROM feat.train_features3 tf
JOIN core.odds3t o USING (race_key);

/* ---------- REFRESH 文 ---------- */
\echo '--- boat_flat 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.boat_flat;
\echo '--- boat_flat_enriched 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.boat_flat_enriched;
\echo '--- racer_flat 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.racer_flat;
\echo '--- race_racer_features 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.race_racer_features;
\echo '--- train_features 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.train_features;
\echo '--- train_features2 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.train_features2;
\echo '--- train_features3 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.train_features3;
\echo '--- racer_hist_30d 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.racer_hist_30d;
\echo '--- eval_features 層のマテリアライズドビューを更新中 ---'
REFRESH MATERIALIZED VIEW feat.eval_features;
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
