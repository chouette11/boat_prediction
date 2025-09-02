DB_NAME      := all_1
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

.PHONY: db_3
db_3:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/03_merge_staging.sql

.PHONY: db_5
db_5:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/05_views_core.sql

.PHONY: db_6
db_6:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/06_views_feat.sql

.PHONY: db_pred
db_pred:
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/100_pred.sql

.PHONY: pred
pred:
	python download/download_pred.py
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql/100_pred.sql

.PHONY: initalldb
initalldb:
	# ① データベースを削除
	make db_drop
	# ② データベースを作成
	make db_create
	# ③ スキーマを初期化
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/01_schema.sql

	# ④ データを流し込む
	python sql2/parse_programs.py
	python sql2/parse_results.py

	# ⑥ ビューを作成
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/05_views_core.sql

	# ⑦ 特徴量を作成
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/06_views_feat.sql

	python sql2/confirm.py

.PHONY: rebuild_alldb
rebuild_alldb:
	# ⑤ マイグレーションを実行
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/03_merge_staging.sql

	# ⑥ ビューを作成
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/05_views_core.sql

	# ⑦ 特徴量を作成
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/06_views_feat.sql

	python sql2/confirm.py

.PHONY: alldb_pred
alldb_pred:
# 	# schema作成
# 	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/11_pred_schema.sql
# 	# rawを作成
# 	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/12_pred_raw.sql
	python pred.py
	# 流し込む
	python sql2/etl_pred.py
	# マージ
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/13_pred_merge_staging.sql
	# 05_views_core
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/15_pred_views_core.sql
	# 06_views_feat
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -f sql2/16_pred_views_feat.sql

.PHONY: db_list
db_list:
	psql -h $(DB_HOST) -U $(DB_USER) -d postgres -c "\list"

.PHONY: to_ipynb
to_ipynb:
	# Jupyter Notebookに変換
	jupytext model/main4.py --to notebook

.PHONY: base_to_ipynb
base_to_ipynb:
	# Jupyter Notebookに変換
	jupytext model/main_base.py --to notebook


.PHONY: tensorboard
tensorboard:
	# TensorBoardを起動
	tensorboard --logdir model/artifacts/tb --port 6006

.PHONY: session
session:
	@psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -c "SELECT pid, usename AS user, datname AS database, application_name, state, query, query_start FROM pg_stat_activity WHERE state != 'idle' ORDER BY query_start DESC;"

.PHONY: delete_session
delete_session:
	python sql/session_delete.py