DB_NAME      := ver2_3
DB_USER      := keiichiro
DB_HOST      := localhost

.PHONY: psql
psql:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME)

.PHONY: db_drop
db_drop:
	psql -h $(DB_HOST) -U $(DB_USER) -d postgres -c "DROP DATABASE IF EXISTS $(DB_NAME);"

.PHONY: db_drop_all
db_drop_all:
	psql -h $(DB_HOST) -U $(DB_USER) -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$(DB_NAME)';"
	psql -h $(DB_HOST) -U $(DB_USER) -d postgres -c "DROP DATABASE IF EXISTS $(DB_NAME);"

.PHONY: db_create
db_create:
	psql -h $(DB_HOST) -U $(DB_USER) -d postgres -c "CREATE DATABASE $(DB_NAME);"
	# .envのバージョンを更新
	@if grep -q '^PGDATABASE=' .env; then \
		sed -i '' 's/^PGDATABASE=.*/PGDATABASE=$(DB_NAME)/' .env; \
	else \
		echo "PGDATABASE=$(DB_NAME)" >> .env; \
	fi

.PHONY: db_init
db_init:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/99_init_wrapper.sql

.PHONY: db_merge
db_merge:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/03_merge_staging.sql

.PHONY: db_5
db_5:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/05_views_core.sql

.PHONY: db_6
db_6:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/06_views_feat.sql

.PHONY: initdb
initdb:
	# ① データベースを削除
	make db_drop
	# ② データベースを作成
	make db_create
	# ③ スキーマを初期化
	make db_init
	# ④ データを流し込む
	python sql/etl.py

	# ⑤ マイグレーションを実行
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/03_merge_staging.sql

	# ⑥ ビューを作成
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/99_init_wrapper.sql

	python sql/confirm.py

.PHONY: db_list
db_list:
	psql -h $(DB_HOST) -U $(DB_USER) -d postgres -c "\list"

.PHONY: to_ipynb
to_ipynb:
	# Jupyter Notebookに変換
	jupytext model/main3.py --to notebook

.PHONY: tensorboard
tensorboard:
	# TensorBoardを起動
	tensorboard --logdir model/artifacts/tb --port 6006

.PHONY: session
session:
	@psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -c "SELECT pid, usename AS user, datname AS database, application_name, state, query, query_start FROM pg_stat_activity WHERE state != 'idle' ORDER BY query_start DESC;"
