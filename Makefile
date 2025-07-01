#! /usr/bin/make 


.PHONY: help 
help:  ## Print the help documentation
	@grep -E '^[a-zA-Z_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: test
test:  ## Run tests
	source ~/.bashrc && uv run pytest easier/tests/

.PHONY: cover
cover:  ## Run tests with coverage reporting
	source ~/.bashrc && uv run pytest easier/tests/ --cov=easier --cov-report=term-missing --cov-report=html

.PHONY: publish
publish:  ## Publish to pypi
	python check_version.py
	rm -rf dist/ build/
	pip install build twine
	python -m build
	twine upload dist/*

.PHONY: clean
clean:  ## Remove build artifacts, caches, and coverage results
	rm -rf dist/ build/ *.egg-info
	rm -rf htmlcov/ .coverage
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
