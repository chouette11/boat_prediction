def check(conn):
    cur = conn.cursor()
    # 今日(Asia/Tokyo)の日付で絞り込み
    import datetime as dt
    today = dt.date.today()
    ex = f"""
-- 例）3桁以外（パース対象外）を抽出
SELECT *
FROM (
  SELECT race_key, combination,
         regexp_replace(combination, '\D', '', 'g') AS comb_digits
  FROM core.payouts
  WHERE bet_type = '３連単'
) t
WHERE length(comb_digits) <> 3
LIMIT 100;
"""
    print(ex)
    cur.execute(ex)
    print(cur.fetchall())

    cur.close(); conn.close()

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import psycopg2
    load_dotenv(override=True)
    conn = psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "ver2_2"),
        user=os.getenv("PGUSER", "keiichiro"),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432")
    )
    # confirm(conn=conn)
    check(conn=conn)