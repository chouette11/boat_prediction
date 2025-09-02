import psycopg2
from dotenv import load_dotenv

def session(conn):
    cur = conn.cursor()

    cur.execute(f"""
                
                """)
    count = cur.fetchone()[0]
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
    session(conn=conn)
