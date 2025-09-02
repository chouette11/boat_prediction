\echo '[RUNNING 11_create_schema.sql]'
\i sql/11_pred_schema.sql

\echo '[RUNNING 03_merge_staging.sql]'
\i sql/03_merge_staging.sql

\echo '[RUNNING 04_functions.sql]'
\i sql/04_functions.sql

\echo '[RUNNING 15_pred_views_core.sql]'
\i sql/15_pred_views_core.sql

\echo '[RUNNING 16_pred_views_feat.sql]'
\i sql/16_pred_views_feat.sql

\echo '[RUNNING 07_indexes.sql]'
\i sql/07_indexes.sql
