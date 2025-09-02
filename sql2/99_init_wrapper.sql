\echo '[RUNNING 00_cleanup.sql]'
\i sql/00_cleanup.sql

\echo '[RUNNING 01_schema.sql]'
\i sql/01_schema.sql

\echo '[RUNNING 02_tables_raw.sql]'
\i sql/02_tables_raw.sql

\echo '[RUNNING 03_merge_staging.sql]'
\i sql/03_merge_staging.sql

\echo '[RUNNING 04_functions.sql]'
\i sql/04_functions.sql

\echo '[RUNNING 05_views_core.sql]'
\i sql/05_views_core.sql

\echo '[RUNNING 06_views_feat.sql]'
\i sql/06_views_feat.sql

\echo '[RUNNING 07_indexes.sql]'
\i sql/07_indexes.sql

\echo '[REFRESHING MATERIALIZED VIEWS]'
REFRESH MATERIALIZED VIEW CONCURRENTLY core.races_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY feat.boat_flat_mv;