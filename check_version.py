import subprocess
import sys
import os
import re
import requests
from pydantic import BaseModel

import typer
from typing import Callable, Any, List, Dict


app = typer.Typer()

PYPROJECT_PATH = os.path.join(os.path.dirname(__file__), "pyproject.toml")
INIT_PATH = os.path.join(os.path.dirname(__file__), "easier", "__init__.py")


def update_pyproject_toml(version: str) -> None:
    with open(PYPROJECT_PATH, "r") as f:
        content = f.read()
    # Replace the version line in the [project] section
    new_content, n = re.subn(
        r'(?m)^version\s*=\s*["\'][^"\']*["\']',
        f'version = "{version}"',
        content,
    )
    if n == 0:
        print("Warning: version line not found in pyproject.toml!")
    with open(PYPROJECT_PATH, "w") as f:
        f.write(new_content)


def update_init(version: str) -> None:
    with open(INIT_PATH, "r") as f:
        content = f.read()
    # Replace any line like __version__ = "..."
    new_content, n = re.subn(
        r'__version__\s*=\s*["\'][^"\']*["\']',
        f'__version__ = "{version}"',
        content,
    )
    if n == 0:
        # If not found, add it at the top
        new_content = f'__version__ = "{version}"\n' + content
    with open(INIT_PATH, "w") as f:
        f.write(new_content)


def check_on_master_branch() -> None:
    result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Could not check current git branch.")
        sys.exit(1)
    current_branch = result.stdout.strip()
    if current_branch != "master":
        print(
            f"Error: You must be on the master branch to publish. Currently on '{current_branch}'.\n"
            f"\nTo fix:\n"
            f"  git checkout master && git merge {current_branch}"
        )
        sys.exit(1)


def check_clean_working_directory() -> None:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Could not check git status.")
        sys.exit(1)
    # Ignore untracked files (lines starting with '??')
    lines = [line for line in result.stdout.splitlines() if not line.startswith("??")]
    if lines:
        # Extract file names from git status output
        modified_files = []
        for line in lines:
            filename = line[3:].strip()
            modified_files.append(filename)

        files_to_add = " ".join(modified_files)
        print(
            "Error: You have uncommitted changes. Please commit or stash them before proceeding.\n"
            "\nTo fix:\n"
            f'  git add {files_to_add} && git commit -m "Committing to bump version"\n'
        )
        sys.exit(1)


def check_no_unpushed_commits() -> None:
    result = subprocess.run(["git", "status", "-sb"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Could not check git branch status.")
        sys.exit(1)
    status = result.stdout.splitlines()[0]
    if "ahead" in status:
        print(
            "Error: You have commits that have not been pushed to origin. Please push them before proceeding.\n"
            "\nTo fix:\n"
            "  git push origin"
        )
        sys.exit(1)


def get_latest_tag() -> str:
    try:
        tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], encoding="utf-8").strip()
        return tag.lstrip("v")
    except subprocess.CalledProcessError:
        print("Error: Could not get the latest git tag.")
        sys.exit(1)


def get_pyproject_version() -> str:
    with open(PYPROJECT_PATH, "r") as f:
        content = f.read()
    m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not m:
        print("Error: Could not find version in pyproject.toml.")
        sys.exit(1)
    return m.group(1)


def get_init_version() -> str:
    with open(INIT_PATH, "r") as f:
        content = f.read()
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not m:
        print("Error: Could not find __version__ in easier/__init__.py.")
        sys.exit(1)
    return m.group(1)


def check_current_commit_tagged() -> None:
    result = subprocess.run(["git", "tag", "--points-at", "HEAD"], capture_output=True, text=True)
    tags = [tag for tag in result.stdout.splitlines() if tag.strip()]
    if not tags:
        print(
            "Error: The current commit is not tagged. Please tag this commit.\n"
            "\nTo fix:\n"
            "  git tag <VERSION>    # e.g., git tag 1.9.5\n"
            "  git push origin <VERSION>"
        )
        sys.exit(1)


def check_commit_pushed_to_origin() -> None:
    # Get the hash of HEAD
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Could not get current commit hash.")
        sys.exit(1)
    head_hash = result.stdout.strip()
    # Get the hash of origin/master
    result = subprocess.run(["git", "rev-parse", "origin/master"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Could not get origin/master commit hash.")
        sys.exit(1)
    origin_hash = result.stdout.strip()
    if head_hash != origin_hash:
        print("Error: The current commit has not been pushed to origin/master.\n" "\nTo fix:\n" "  git push origin")
        sys.exit(1)


def check_tag_pushed_to_origin(tag: str) -> None:
    # Check if the tag exists on the remote
    result = subprocess.run(["git", "ls-remote", "--tags", "origin", tag], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Could not check tags on origin.")
        sys.exit(1)
    if not result.stdout.strip():
        print(f"Error: Tag '{tag}' has not been pushed to origin.\n" f"\nTo fix:\n" f"  git push origin {tag}")
        sys.exit(1)


def check_version_not_on_pypi(project_name: str, version: str) -> None:
    url = f"https://pypi.org/pypi/{project_name}/json"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 404:
            # Project not published yet
            return
        resp.raise_for_status()
        data = resp.json()
        if version in data.get("releases", {}):
            print(
                f"Error: Version {version} is already published on PyPI.\n"
                f"\nTo fix:\n"
                f"  python update_version.py <NEW_VERSION>    # Choose a different version number"
            )
            sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not check PyPI for published versions: {e}")


class Task(BaseModel):
    function: Callable[..., Any]
    args: List[Any] | None = None

    def run(self) -> Any:
        if self.args is None:
            return self.function()
        return self.function(*self.args)

    def __str__(self) -> str:
        return self.function.__name__


def run_starting_at(
    tasks: list[Task],
    task_name: str | None,
    message_template: str = 'Function "{task}" failed.',
    steps: int | None = None,
) -> None:
    """ """
    tasks_to_run = list(tasks)
    while task_name is not None and tasks_to_run and tasks_to_run[0].function.__name__ != task_name:
        tasks_to_run.pop(0)

    if steps is not None:
        tasks_to_run = tasks_to_run[:steps]

    for task in tasks_to_run:
        try:
            task.run()
        except SystemExit:
            print(message_template.format(task=str(task)))
            sys.exit(1)
        except Exception:
            print(message_template.format(task=str(task)))
            sys.exit(1)

    if task_name is not None and not tasks_to_run:
        print(f"Error: Task '{task_name}' not found in task list.")
        sys.exit(1)


@app.command()
def main(
    starting_at: str | None = typer.Option(None, help="Task name to start from"),
    utility: str | None = typer.Option(None, help="Utility function to run"),
    utility_args: List[str] | None = typer.Option(None, help="Arguments to pass to the utility"),
) -> None:
    # Validate mutual exclusion of starting_at and utility
    if starting_at is not None and utility is not None:
        print("Error: --starting-at and --utility are mutually exclusive options.")
        print("\nUsage:")
        print("  check_version.py --starting-at <task_name>")
        print("  check_version.py --utility <utility_name>")
        print("  check_version.py  # Run all tasks")
        sys.exit(1)

    utility_args = utility_args or []

    project_name = "easier"
    current_version = get_pyproject_version()
    latest_tag = get_latest_tag()

    tasks = [
        Task(function=check_on_master_branch),
        Task(function=check_version_not_on_pypi, args=[project_name, current_version]),
        Task(function=check_clean_working_directory),
        Task(function=check_no_unpushed_commits),
        Task(function=check_current_commit_tagged),
        Task(function=check_commit_pushed_to_origin),
        Task(function=check_tag_pushed_to_origin, args=[latest_tag]),
    ]

    utilities = [
        Task(function=update_pyproject_toml, args=utility_args),
        Task(function=update_init, args=utility_args),
    ]

    template = "After fixing the error, resume publishing by running `check_version.py --starting-at {task_name}"

    if utility is not None:
        run_starting_at(utilities, utility, steps=1)
    else:
        run_starting_at(tasks, starting_at, template)


if __name__ == "__main__":
    app()
