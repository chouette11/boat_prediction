import psycopg2
import psycopg2.extras
import pandas as pd
from shutil import get_terminal_size

def check(conn):
    # DictCursorで列名を一緒に取得
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    sql = """
    COPY (
    SELECT *
    FROM feat.filtered_course
    ) TO STDOUT WITH CSV HEADER
    """

    with open("filtered_course.csv", "w", encoding="utf-8", newline="") as f:
        cur.copy_expert(sql, f)

    cur.close()
    conn.close()

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