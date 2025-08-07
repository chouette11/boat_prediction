import psycopg2
from dotenv import load_dotenv

def confirm(conn):
    cur = conn.cursor()

    tables = [
        # ----- STAGING -----
        "raw.results_staging",
        "raw.beforeinfo_staging",
        "raw.weather_staging",
        "raw.racertech_staging",
        "raw.racercourse_staging",
        "raw.racerstadium_staging",
        "raw.racerresult1_staging",
        "raw.racerresult2_staging",
        "raw.racerboatcourse_staging",
        "raw.racerboat_staging",
        "raw.odds3t_staging",
        # ----- RAW -----
        "raw.results",
        "raw.beforeinfo",
        "raw.weather",
        "raw.racertech",
        "raw.racercourse",
        "raw.racerstadium",
        "raw.racerresult1",
        "raw.racerresult2",
        "raw.racerboatcourse",
        "raw.racerboat",
        "raw.odds3t",
        # ----- CORE -----
        "core.boat_info",
        "core.weather",
        "core.racerstats_course",
        "core.racerstats_grade",
        "core.racerstats_boatcourse",
        # ----- FEAT -----
        "feat.boat_flat",
        "feat.train_features2",
        "feat.train_features3",
        "feat.eval_features2",
        "feat.eval_features3",
    ]
    for table in tables:
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
