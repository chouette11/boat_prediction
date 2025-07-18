DB_NAME      := ver1_2
DB_USER      := keiichiro
DB_HOST      := localhost

.PHONY: psql
psql:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME)

.PHONY: db_drop
db_drop:
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

.PHONY: db_list
db_list:
	psql -h $(DB_HOST) -U $(DB_USER) -d postgres -c "\list"

