def check(conn):
    cur = conn.cursor()
    ex = """
SELECT
  COUNT(*) AS total,
  COUNT(st_time) AS with_st_time,
  COUNT(*) - COUNT(st_time) AS null_st_time,
  COUNT(wind_dir_deg) AS with_wind_dir_deg,
  COUNT(*) - COUNT(wind_dir_deg) AS null_wind_dir_deg
FROM core.boat_info
JOIN feat.train_features USING (race_key);
"""
    print(ex)
    cur.execute(ex)
    for row in cur.fetchall():
        print(row)
    print("\n--- wind_dir_deg が NULL の race_key 一覧（10件）---")
    cur.execute("""
        SELECT race_key
        FROM feat.train_features
        WHERE wind_dir_deg IS NULL
        LIMIT 10;
    """)
    for row in cur.fetchall():
        print(row)

    print("\n--- raw.weather の wind_dir_raw NULL 行（10件）---")
    cur.execute("""
        SELECT *
        FROM raw.weather
        WHERE wind_dir_raw IS NULL
        LIMIT 10;
    """)
    for row in cur.fetchall():
        print(row)

    print("\n--- raw.weather の wind_dir_raw のユニーク値一覧 ---")
    cur.execute("""
        SELECT DISTINCT wind_dir_raw
        FROM raw.weather
        ORDER BY 1;
    """)
    for row in cur.fetchall():
        print(row)

    print("\n--- wind_dir_raw = 'is-wind17' の日付・レース番号 ---")
    cur.execute("""
        SELECT DISTINCT race_date, race_no
        FROM raw.weather
        WHERE wind_dir_raw = 'is-wind17'
        ORDER BY race_date, race_no;
    """)
    for row in cur.fetchall():
        print(row)

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