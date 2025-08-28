#! /usr/bin/env make 

SHELL := /usr/bin/env bash


.PHONY: help 
help:  ## Print the help documentation
	@grep -E '^[a-zA-Z_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all: final_result.ddb ## The final artifact for the pipeline

.PHONY: clean
clean: ## Remove all generated files
	rm -f stuff_from_replica.ddb
	rm -f final_result.ddb

stuff_from_replica.ddb: ## Holds data pulled from replica
	python template_replica_query.nb.py

final_result.ddb: stuff_from_replica.ddb ## Records returned from llm processing
	python template_process_with_llm.nb.py

.PHONY: run_result_notebook
run_result_notebook: # Run the result notebook
	uv run marimo run template_process_with_llm.nb.py


.PHONY: uv_env
uv_env: ## Create an env for this project
	rm -rf .venv || true
	uv venv

.PHONY: uv_lock
uv_lock:  ## Create lock file
	uv lock

.PHONY: uv_sync
uv_sync: ## Installs lock file into env (creating lockfile if needed)
	uv sync

.PHONY: uv_activate
uv_activate: ## Activate the virtual environment
	@exec bash --rcfile <(echo '. ~/.bashrc; source .venv/bin/activate; echo "Virtual environment activated"')

.PHONY: uv_nuke
uv_nuke: ## Delete the virtual environment
	rm -rf .venv || true


