/*------------------------------------------------------------
  00_cleanup.sql
  既存のビュー／テーブルを一括削除してクリーンな状態にする
------------------------------------------------------------*/
DROP MATERIALIZED VIEW IF EXISTS feat.train_features;
DROP MATERIALIZED VIEW IF EXISTS feat.boat_flat;
DROP MATERIALIZED VIEW IF EXISTS core.weather;
DROP MATERIALIZED VIEW IF EXISTS core.boat_info;
DROP MATERIALIZED VIEW IF EXISTS core.results;

-- races は型違いの VIEW／TABLE もまとめて掃除
DROP MATERIALIZED VIEW IF EXISTS core.races;
DROP      VIEW IF EXISTS core.races;
DROP     TABLE IF EXISTS core.races;