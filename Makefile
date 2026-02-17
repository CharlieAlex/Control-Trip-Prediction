.PHONY: install clean lint test

install:
	uv sync

lint:
	uv run ruff check . --fix
	uv run ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .ruff_cache .venv

dvc-init:
	dvc init
	dvc remote add carplus gs://person-carplusdata-weichun/linego-control-trips
	dvc remote modify --local carplus credentialpath 'sa/weichun.sa.json'
	dvc config core.autostage true
	git commit -m "Initialize DVC"

get_data:
	uv run python scripts/get_data.py

train:
	uv run python scripts/train.py

mlflow:
	uv run mlflow ui \
		--backend-store-uri sqlite:///data/mlflow.db \
		--port 5000

dvc-test:
	dvc add data/data.csv
	git add data/data.csv.dvc data/.gitignore
	git commit -m "Add data/data.csv"
	uv run dvc push

dvc-test2:
	dvc stage add -n get_data \
		-d src/queries/data.sql \
		-d scripts/get_data.py \
		-o data/data.parquet \
		uv run python scripts/get_data.py

	dvc stage add -n train \
		-d src/config.yml \
		-d scripts/train.py \
		-o data/data.parquet \
		uv run python scripts/train.py

	git add dvc.yaml data/.gitignore
	git commit -m "Add train stage"