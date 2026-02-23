.PHONY: install clean lint test

uv-sync:
	uv sync

uv-sync-dev:
	uv sync --extra dev

lint:
	uv run ruff check . --fix
	uv run ruff format .

clean-pycache:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .ruff_cache

get-data:
	uv run python scripts/get_data.py

train:
	uv run python scripts/train.py

mlflow:
	uv run mlflow ui \
		--backend-store-uri sqlite:///data/mlflow.db \
		--port 5000

dvc-init:
	dvc init
	dvc remote add -d carplus gs://person-carplusdata-weichun/linego-control-trips
	dvc remote modify --local carplus credentialpath 'sa/weichun.sa.json'
	dvc config core.autostage false
	git add .dvc/config
	git commit -m "Initialize DVC"

dvc-stage:
	dvc stage add -n get_data \
		-d src/queries/data.sql \
		-d scripts/get_data.py \
		-o data/data.parquet \
		uv run python scripts/get_data.py

	dvc stage add -n train \
		-d config.yml \
		-d scripts/train.py \
		-d data/data.parquet \
		uv run python scripts/train.py

	git commit -m "Add dvc stage"

dvc-run:
	uv run dvc pull
	dvc repro
	git add dvc.lock && git commit -m "Update dvc.lock"
	uv run dvc push

run:
	dvc repro

test:
	dvc repro --no-commit

exp:
	uv run dvc exp run --no-apply