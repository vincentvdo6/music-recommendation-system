.PHONY: help setup dev lint test run-api serve-ann build-index eval migrate clean
.DEFAULT_GOAL := help

# Variables
EMB_VER ?= mert_v2
INDEX_VERSION ?= ann_v1_local

help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

setup: ## Install dependencies and pre-commit hooks
	uv sync --dev || (echo "uv not found, trying pip..." && pip install -e .[dev])
	pre-commit install || echo "pre-commit installation failed, continuing..."

dev: ## Start local development stack
	docker compose up -d db redis minio
	@echo "Waiting for services to be ready..."
	@sleep 5
	uv run alembic upgrade head || echo "Migration failed, database might not be ready"

lint: ## Run code quality checks
	uv run ruff check .
	uv run mypy .

test: ## Run test suite
	uv run pytest -v

run-api: ## Start the FastAPI development server
	uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

serve-ann: ## Start ANN index server
	uv run python services/ann_index/server.py --index $(INDEX_VERSION)

build-index: ## Build FAISS index from embeddings
	uv run python scripts/build_index.py --embedding-version $(EMB_VER) --index-version $(INDEX_VERSION)

eval: ## Run evaluation harness
	uv run python eval/harness.py --index-version $(INDEX_VERSION)

migrate: ## Run database migrations
	uv run alembic upgrade head

clean: ## Clean up generated files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/ build/ dist/ || true

docker-build: ## Build docker images
	docker compose build

docker-down: ## Stop and remove all containers
	docker compose down -v

reset-db: ## Reset database (destructive!)
	docker compose down -v
	docker compose up -d db
	@sleep 5
	uv run alembic upgrade head