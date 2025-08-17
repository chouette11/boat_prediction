/*------------------------------------------------------------
  07_indexes.sql
  主要インデックス
------------------------------------------------------------*/
CREATE UNIQUE INDEX idx_core_races_pk     ON core.races          (race_key);
CREATE UNIQUE INDEX idx_core_results_pk   ON core.results        (race_key, lane);
CREATE UNIQUE INDEX idx_core_boatinfo_pk  ON core.boat_info      (race_key, lane);
CREATE UNIQUE INDEX idx_core_weather_pk   ON core.weather        (race_key);
CREATE UNIQUE INDEX idx_boat_flat_pk      ON feat.boat_flat      (race_key, lane);
CREATE UNIQUE INDEX idx_train_feat_pk     ON feat.train_features (race_key);
