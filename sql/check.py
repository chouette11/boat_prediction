import psycopg2
import psycopg2.extras
import pandas as pd
from shutil import get_terminal_size

def check(conn):
    # DictCursorで列名を一緒に取得
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    ex = """
SELECT *
FROM pred.features_with_record
WHERE race_key = '2025-07-24-07-20'
"""
    print(ex)
    # 列名を取得
    cur.execute(ex)
    colnames = [desc[0] for desc in cur.description]
    print(colnames)
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