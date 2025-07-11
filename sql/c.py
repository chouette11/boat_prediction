import psycopg2
conn = psycopg2.connect(
    host="localhost", dbname="postgres",
    user="keiichiro", password=""
)
cur = conn.cursor()
cur.execute("SELECT * FROM feat.train_features;")
print(cur.fetchall()[:10])  # 最初の10行を表示
cur.close(); conn.close()