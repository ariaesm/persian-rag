PY ?= python3
RUFF ?= ruff
export PYTHONPATH := src

.PHONY: help install test lint format check eval demo clean

help:  ## Show available targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

install:  ## Editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint and check formatting
	$(RUFF) check .
	$(RUFF) format --check .

format:  ## Auto-format and fix lint issues
	$(RUFF) format .
	$(RUFF) check --fix .

check: lint test  ## Lint + tests (what CI runs)

eval:  ## Evaluate retrievers on the sample corpus
	$(PY) -m persian_rag eval

demo:  ## Index the sample corpus and run a search
	$(PY) -m persian_rag ingest examples/corpus --index ./index
	$(PY) -m persian_rag search "زعفران کجا کشت می‌شود؟" --index ./index

clean:  ## Remove caches and local indexes
	rm -rf index .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
