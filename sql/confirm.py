import psycopg2
from dotenv import load_dotenv

def confirm(conn):
    cur = conn.cursor()
    for table in [
        "raw.results_staging", "raw.beforeinfo_staging","raw.weather_staging",
        "raw.results", "raw.beforeinfo", "raw.weather",
        "core.boat_info", "core.weather",
        "feat.boat_flat", "feat.train_features"
    ]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"{table}: {count} rows")
    print(cur.fetchall()[:10])  # 最初の10行を表示
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
    confirm(conn=conn)
