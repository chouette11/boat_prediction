def check(conn):
    cur = conn.cursor()
    # 数を確認
    ex = """
SELECT * FROM core.boat_info LIMIT 100;
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