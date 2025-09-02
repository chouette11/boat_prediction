/*------------------------------------------------------------
  04_functions.sql
  汎用関数（IMMUTABLE）
------------------------------------------------------------*/
CREATE OR REPLACE FUNCTION core.f_race_key(
    d DATE,
    n INT,
    s TEXT
) RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    /* 例: 若松_20250115_12 */
    SELECT CONCAT(s, '_', to_char(d, 'YYYYMMDD'), '_', n);
$$;
