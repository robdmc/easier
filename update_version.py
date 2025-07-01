import sys
import re
import os

PYPROJECT_PATH = os.path.join(os.path.dirname(__file__), "pyproject.toml")
INIT_PATH = os.path.join(os.path.dirname(__file__), "easier", "__init__.py")


def update_pyproject_toml(version):
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


def update_init(version):
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    version = sys.argv[1].lstrip("v")  # Optionally strip leading 'v'
    print(f"Updating version to: {version}")
    update_pyproject_toml(version)
    update_init(version)
    print("Done.")


if __name__ == "__main__":
    main()
