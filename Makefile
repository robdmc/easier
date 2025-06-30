#! /usr/bin/make 


.PHONY: help 
help:  ## Print the help documentation
	@grep -E '^[a-zA-Z_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: test
test:  ## Run tests
	python -m pytest easier/tests/

.PHONY: cover
cover:  ## Run tests with coverage reporting
	python -m pytest easier/tests/ --cov=easier --cov-report=term-missing --cov-report=html

.PHONY: publish
publish:  ## Publish to pypi
	rm -rf dist/ build/
	python -m build
	twine upload dist/*

