.PHONY: psql
psql: ## activate_fvm
	psql --host=localhost --dbname=postgres --user=keiichiro

.PHONY: psql-remake
psql-remake: ## activate_fvm
	psql --host=localhost --dbname=postgres --user=keiichiro -f sql/ddl.sql