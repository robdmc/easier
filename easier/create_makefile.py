from fuzzypicker import picker
import argparse
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a Makefile for running marimo notebooks"
    )
    parser.add_argument("--notebook", "-n", help="Name of the notebook file")
    return parser.parse_args()


MAKEFILE_TEMPLATE = """#! /usr/bin/make
.PHONY: help
help:  ## Print the help documentation
	@grep -E "^[a-zA-Z_.-]+:.*?## .*$$" $(MAKEFILE_LIST) | sort | awk 'BEGIN {{FS = ":.*?## "}}; {{printf "\033[36m%-30s\033[0m %s\\n", $$1, $$2}}'

.PHONY: run
run: ## Run the notebook as an app
\tuv run --with marimo marimo run --watch --sandbox {notebook_name}

.PHONY: edit
edit: ## Run the notebook in edit mode
\tuv run --with marimo marimo edit --watch --sandbox {notebook_name}
"""


def get_makefile(notebook_name):
    if notebook_name is None:
        try:
            notebook_name = picker(
                ["notebook.nb.py"] + [f.name for f in pathlib.Path(".").glob("*.py")],
                default="notebook.nb.py",
            )
        except ImportError:
            return "notebook.nb.py"
    makefile_text = MAKEFILE_TEMPLATE.format(notebook_name=notebook_name)
    return makefile_text


def main():
    args = parse_args()
    with open("Makefile", "w") as f:
        f.write(get_makefile(args.notebook))


if __name__ == "__main__":
    main()
