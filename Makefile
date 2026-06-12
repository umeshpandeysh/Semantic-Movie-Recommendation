# Semantic-Movie-Recommendation — Project Makefile
.PHONY: help install test lint embeddings serve benchmark clean

PYTHON ?= python

help:       ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:    ## Install all dependencies
	pip install -r requirements.txt

test:       ## Run the test suite
	pytest tests/ -v --tb=short

lint:       ## Lint with flake8
	flake8 backend/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503

embeddings: ## Generate movie embeddings (downloads ~80MB model on first run)
	$(PYTHON) backend/generate_embeddings.py \
		--metadata embeddings/movies_metadata.csv \
		--output_dir embeddings/

serve:      ## Start the FastAPI server
	uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

benchmark:  ## Run recommendation quality benchmark
	$(PYTHON) scripts/benchmark.py \
		--metadata embeddings/movies_metadata.csv \
		--embeddings embeddings/movie_embeddings.npy

clean:      ## Remove generated embedding files
	rm -f embeddings/movie_embeddings.npy embeddings/movie_documents.txt
	@echo "Cleaned embedding files."
