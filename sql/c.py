import psycopg2
from dotenv import load_dotenv


def confirm(conn):
    cur = conn.cursor()
    for table in [
        "raw.results_staging", "raw.beforeinfo_staging","raw.weather_staging",
        "raw.results", "raw.racers", "raw.weather",
        "core.boat_info", "core.weather",
        "feat.boat_flat", "feat.train_features"
    ]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"{table}: {count} rows")
    print(cur.fetchall()[:10])  # 最初の10行を表示
    cur.close(); conn.close()

def fix(conn):
    cur = conn.cursor()
    ex ="""
SELECT
  COUNT(*) AS total,
  COUNT(st_time) AS with_st_time,
  COUNT(*) - COUNT(st_time) AS null_st_time
FROM core.boat_info;

"""
    
    cur.execute(ex)
    print(ex)

    columns = cur.fetchall()
    for col in columns:
        print(f"total: {col[0]}, with_st_time: {col[1]}, null_st_time: {col[2]}")

    query = '''
    SELECT
      b.race_key,
      b.lane,
      r.st_raw
    FROM core.boat_info b
    JOIN raw.racers r
      ON b.race_key = core.f_race_key(r.race_date, r.race_no, r.stadium)
     AND b.lane = r.lane
    WHERE b.st_time IS NULL
    LIMIT 20;
    '''
    cur.execute(query)
    rows = cur.fetchall()
    for row in rows:
        print(f"race_key: {row[0]}, lane: {row[1]}, st_raw: {row[2]}")

    query = '''
    SELECT COUNT(*)
    FROM core.boat_info b
    LEFT JOIN core.results r
      USING (race_key, lane)
    WHERE r.race_key IS NULL;
    '''
    cur.execute(query)
    result = cur.fetchone()
    print(f"Unmatched rows in core.results join: {result[0]}")

    query = '''
    SELECT b.race_key, b.lane
    FROM core.boat_info b
    LEFT JOIN core.results r
      USING (race_key, lane)
    WHERE r.race_key IS NULL
    ORDER BY b.race_key, b.lane
    LIMIT 50;
    '''
    cur.execute(query)
    rows = cur.fetchall()
    print("Missing (race_key, lane) pairs in core.results:")
    for row in rows:
        print(f"  race_key: {row[0]}, lane: {row[1]}")

    print("\nChecking presence of missing race_keys in raw.results:")
    for row in rows:
        race_key = row[0]
        lane = row[1]
        try:
            stadium, date_str, no_str = race_key.split("_")
            race_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            race_no = int(no_str)
            cur.execute("""
                SELECT COUNT(*) FROM raw.results
                WHERE race_date = %s AND race_no = %s AND stadium = %s AND lane = %s;
            """, (race_date, race_no, stadium, lane))
            count = cur.fetchone()[0]
            print(f"  {race_key}, lane {lane} → raw.results: {'FOUND' if count > 0 else 'NOT FOUND'}")
        except Exception as e:
            print(f"  Error parsing race_key {race_key}: {e}")

    cur.close(); conn.close()

if __name__ == "__main__":
    import os
    load_dotenv(override=True)
    conn = psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "ver2_2"),
        user=os.getenv("PGUSER", "keiichiro"),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432")
    )
    # confirm(conn=conn)
    fix(conn=conn)