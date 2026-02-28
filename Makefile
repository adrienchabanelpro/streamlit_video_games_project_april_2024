.PHONY: run test lint format train clean install help dvc-repro

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

run: ## Run the Streamlit app
	cd source && streamlit run main.py

test: ## Run pytest test suite
	python -m pytest tests/ -v

lint: ## Run ruff linter
	ruff check source/ scripts/ tests/

format: ## Run ruff formatter
	ruff format source/ scripts/ tests/

train: ## Run model training pipeline
	python scripts/train_model.py

dvc-repro: ## Reproduce DVC pipeline (train + track outputs)
	dvc repro

clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name '*.pyc' -delete 2>/dev/null; \
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; \
	true
