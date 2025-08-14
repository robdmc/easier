import subprocess
import sys
import os
import re
import requests
import click

PYPROJECT_PATH = os.path.join(os.path.dirname(__file__), "pyproject.toml")
INIT_PATH = os.path.join(os.path.dirname(__file__), "easier", "__init__.py")


def check_on_master_branch():
    result = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error: Could not check current git branch.")
        sys.exit(1)
    current_branch = result.stdout.strip()
    if current_branch != "master":
        print(
            f"Error: You must be on the master branch to publish. Currently on '{current_branch}'.\n"
            f"\nTo fix:\n"
            f"  Option 1 (Switch only): git checkout master\n"
            f"  Option 2 (Merge and switch): git checkout master && git merge {current_branch}"
        )
        sys.exit(1)


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
            "Error: You have uncommitted changes. Please commit or stash them before proceeding.\n"
            "\nTo fix:\n"
            "  Option 1 (Commit): git add pyproject.toml easier/__init__.py && git commit -m \"[Release]: Bump version to v<VERSION> for PyPI release\"\n"
            "  Option 2 (Stash): git stash"
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
            "Error: You have commits that have not been pushed to origin. Please push them before proceeding.\n"
            "\nTo fix:\n"
            "  git push origin"
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
        print(
            "Error: The current commit is not tagged. Please tag this commit.\n"
            "\nTo fix:\n"
            "  git tag <VERSION>    # e.g., git tag 1.9.5\n"
            "  git push origin <VERSION>"
        )
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
        print(
            "Error: The current commit has not been pushed to origin/master.\n"
            "\nTo fix:\n"
            "  git push origin"
        )
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
        print(
            f"Error: Tag '{tag}' has not been pushed to origin.\n"
            f"\nTo fix:\n"
            f"  git push origin {tag}"
        )
        sys.exit(1)


def check_version_not_on_pypi(project_name, version):
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


@click.command()
@click.option('--master-branch', is_flag=True, help='Check currently on master branch')
@click.option('--clean-working-dir', is_flag=True, help='Check for uncommitted changes')
@click.option('--no-unpushed', is_flag=True, help='Check for unpushed commits')
@click.option('--commit-tagged', is_flag=True, help='Check current commit is tagged')
@click.option('--commit-pushed', is_flag=True, help='Check commit pushed to origin')
@click.option('--tag-pushed', is_flag=True, help='Check tag pushed to origin')
@click.option('--pypi-available', is_flag=True, help='Check version not on PyPI')
@click.option('--version-sync', is_flag=True, help='Check version consistency across files')
def main(master_branch, clean_working_dir, no_unpushed, commit_tagged, commit_pushed, tag_pushed, pypi_available, version_sync):
    """Check version consistency and publishing requirements.
    
    If no flags are provided, all checks will be run (default behavior).
    """
    # If no flags are provided, run all checks
    run_all = not any([master_branch, clean_working_dir, no_unpushed, commit_tagged, commit_pushed, tag_pushed, pypi_available, version_sync])
    
    # Get project name and version for checks that need them
    project_name = None
    pyproject_version = None
    tag_version = None
    init_version = None
    
    if run_all or pypi_available or version_sync:
        with open(PYPROJECT_PATH, "r") as f:
            content = f.read()
        m = re.search(r'^name\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if not m:
            print("Error: Could not find project name in pyproject.toml.")
            sys.exit(1)
        project_name = m.group(1)
        pyproject_version = get_pyproject_version()
    
    if run_all or tag_pushed or version_sync:
        tag_version = get_latest_tag()
    
    if run_all or version_sync:
        init_version = get_init_version()
    
    # Run individual checks based on flags - master branch check first
    if run_all or master_branch:
        check_on_master_branch()
    
    if run_all or pypi_available:
        check_version_not_on_pypi(project_name, pyproject_version)
    
    if run_all or clean_working_dir:
        check_clean_working_directory()
    
    if run_all or no_unpushed:
        check_no_unpushed_commits()
    
    if run_all or commit_tagged:
        check_current_commit_tagged()
    
    if run_all or commit_pushed:
        check_commit_pushed_to_origin()
    
    if run_all or tag_pushed:
        check_tag_pushed_to_origin(tag_version)
    
    if run_all or version_sync:
        if tag_version != pyproject_version:
            print(
                f"Error: Latest git tag ({tag_version}) does not match pyproject.toml version ({pyproject_version}).\n"
                f"\nTo fix:\n"
                f"  Option 1: python update_version.py {tag_version}    # Update files to match tag\n"
                f"  Option 2: git tag {pyproject_version} && git push origin {pyproject_version}    # Create new tag"
            )
            sys.exit(1)
        if tag_version != init_version:
            print(
                f"Error: Latest git tag ({tag_version}) does not match easier/__init__.py version ({init_version}).\n"
                f"\nTo fix:\n"
                f"  python update_version.py {tag_version}    # Update files to match tag"
            )
            sys.exit(1)
    
    if run_all:
        print("All version checks passed. Repository is clean and up to date.")
    else:
        print("Selected checks passed.")


if __name__ == "__main__":
    main()
