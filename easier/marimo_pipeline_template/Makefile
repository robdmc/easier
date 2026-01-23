#! /usr/bin/env make 

SHELL := /usr/bin/env bash


.PHONY: help 
help:  ## Print the help documentation
	@grep -E '^[a-zA-Z_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: bundle
bundle: ## Create zip bundle of pipeline template files
	rm -f manifest/marimo_pipeline_template.zip
	cd pipeline_template_files && zip -r ../manifest/marimo_pipeline_template.zip Makefile pyproject.toml template_process_with_llm.nb.py template_replica_query.nb.py .claude/prompts/SQL_COMMENTING_GUIDE.md
