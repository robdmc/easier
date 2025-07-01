import subprocess
import sys
import os
import re

PYPROJECT_PATH = os.path.join(os.path.dirname(__file__), "pyproject.toml")
INIT_PATH = os.path.join(os.path.dirname(__file__), "easier", "__init__.py")


def check_clean_working_directory():
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error: Could not check git status.")
        sys.exit(1)
    # Ignore untracked files (lines starting with '??')
    lines = [line for line in result.stdout.splitlines() if not line.startswith("??")]
    if lines:
        print(
            "Error: You have uncommitted changes. "
            "Please commit or stash them before proceeding."
        )
        sys.exit(1)


def check_no_unpushed_commits():
    result = subprocess.run(["git", "status", "-sb"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Could not check git branch status.")
        sys.exit(1)
    status = result.stdout.splitlines()[0]
    if "ahead" in status:
        print(
            "Error: You have commits that have not been pushed to origin. "
            "Please push them before proceeding."
        )
        sys.exit(1)


def get_latest_tag():
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], encoding="utf-8"
        ).strip()
        return tag.lstrip("v")
    except subprocess.CalledProcessError:
        print("Error: Could not get the latest git tag.")
        sys.exit(1)


def get_pyproject_version():
    with open(PYPROJECT_PATH, "r") as f:
        content = f.read()
    m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not m:
        print("Error: Could not find version in pyproject.toml.")
        sys.exit(1)
    return m.group(1)


def get_init_version():
    with open(INIT_PATH, "r") as f:
        content = f.read()
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not m:
        print("Error: Could not find __version__ in easier/__init__.py.")
        sys.exit(1)
    return m.group(1)


def check_current_commit_tagged():
    result = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"], capture_output=True, text=True
    )
    tags = [tag for tag in result.stdout.splitlines() if tag.strip()]
    if not tags:
        print("Error: The current commit is not tagged. Please tag this commit.")
        sys.exit(1)


def check_commit_pushed_to_origin():
    # Get the hash of HEAD
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error: Could not get current commit hash.")
        sys.exit(1)
    head_hash = result.stdout.strip()
    # Get the hash of origin/master
    result = subprocess.run(
        ["git", "rev-parse", "origin/master"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error: Could not get origin/master commit hash.")
        sys.exit(1)
    origin_hash = result.stdout.strip()
    if head_hash != origin_hash:
        print("Error: The current commit has not been pushed to origin/master.")
        sys.exit(1)


def check_tag_pushed_to_origin(tag):
    # Check if the tag exists on the remote
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", tag], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error: Could not check tags on origin.")
        sys.exit(1)
    if not result.stdout.strip():
        print(f"Error: Tag '{tag}' has not been pushed to origin.")
        sys.exit(1)


def main():
    check_clean_working_directory()
    check_no_unpushed_commits()
    check_current_commit_tagged()
    check_commit_pushed_to_origin()
    tag_version = get_latest_tag()
    check_tag_pushed_to_origin(tag_version)
    pyproject_version = get_pyproject_version()
    init_version = get_init_version()
    if tag_version != pyproject_version:
        print(
            f"Error: Latest git tag ({tag_version}) does not match "
            f"pyproject.toml version ({pyproject_version})."
        )
        sys.exit(1)
    if tag_version != init_version:
        print(
            f"Error: Latest git tag ({tag_version}) does not match "
            f"easier/__init__.py version ({init_version})."
        )
        sys.exit(1)
    print("All version checks passed. Repository is clean and up to date.")


if __name__ == "__main__":
    main()
