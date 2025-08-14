# Publish Package to PyPI

This command guides you through the complete publishing workflow with version management and pre-publish validation.

## Step 1: Version Management

### Check Current PyPI Version
```bash
python -c "
import requests
try:
    resp = requests.get('https://pypi.org/pypi/easier/json', timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        current_version = data['info']['version']
        print(f'Current PyPI version: {current_version}')
        
        # Parse version and suggest next patch version
        parts = current_version.split('.')
        if len(parts) == 3:
            major, minor, patch = parts
            new_patch = str(int(patch) + 1)
            suggested = f'{major}.{minor}.{new_patch}'
            print(f'Suggested new version: {suggested}')
        else:
            print('Could not parse version for suggestion')
    else:
        print('Package not found on PyPI')
except Exception as e:
    print(f'Error checking PyPI: {e}')
"
```

### Update Version
**Prompt:** What version do you want to publish? (Use format like 1.9.5)

Once you decide on a version, update both files:

**Update both files at once:**
```bash
# Replace X.Y.Z with your chosen version
python update_version.py X.Y.Z
```

**Verify updates:**
```bash
echo "pyproject.toml version:"
grep '^version = ' pyproject.toml
echo "easier/__init__.py version:"
grep '__version__ = ' easier/__init__.py
```

## Step 2: Pre-Publish Validation

### Run All Checks
```bash
python check_version.py
```

**Expected output:** `All version checks passed. Repository is clean and up to date.`

If any check fails, the error message will include specific instructions to fix the issue. Simply follow the "To fix:" instructions provided in the error output.

## Step 3: Ready to Publish

� **WARNING: PyPI publishing is IRREVERSIBLE. Once published, a version cannot be deleted or modified.**

### Publish Command (DO NOT RUN YET - REVIEW FIRST)
```bash
make publish
```

**Before running `make publish`:**
1. Double-check the version number is correct
2. Ensure all tests pass: `make test`
3. Verify the package builds correctly: `make clean && python -m build`
4. Review the built package contents in `dist/`

**Only run `make publish` when you are absolutely certain everything is correct.**

## Troubleshooting Failed Checks

If `python check_version.py` fails, here are detailed troubleshooting steps for each type of error:

### Not on Master Branch
**Error:** `You must be on the master branch to publish. Currently on 'branch_name'.`
- **Fix Option 1 (Switch only):** `git checkout master`
- **Fix Option 2 (Merge and switch):** `git checkout master && git merge branch_name`

### PyPI Version Already Exists
**Error:** `Version X.Y.Z is already published on PyPI.`
- **Fix:** Choose a different version number and run `python update_version.py <new_version>`

### Uncommitted Changes
**Error:** `You have uncommitted changes.`
- **Fix Option 1 (Commit):** `git add pyproject.toml easier/__init__.py && git commit -m "[Release]: Bump version to vX.Y.Z for PyPI release"`
- **Fix Option 2 (Stash):** `git stash`

### Unpushed Commits
**Error:** `You have commits that have not been pushed to origin.`
- **Fix:** `git push origin`

### Current Commit Not Tagged
**Error:** `The current commit is not tagged.`
- **Fix:** `git tag <VERSION> && git push origin <VERSION>`

### Commit Not Pushed to Origin
**Error:** `The current commit has not been pushed to origin/master.`
- **Fix:** `git push origin`

### Tag Not Pushed to Origin
**Error:** `Tag 'X.Y.Z' has not been pushed to origin.`
- **Fix:** `git push origin X.Y.Z`

### Version Sync Issues
**Error:** `Latest git tag does not match pyproject.toml version.`
- **Fix Option 1:** `python update_version.py <tag_version>` (update files to match tag)
- **Fix Option 2:** `git tag <pyproject_version> && git push origin <pyproject_version>` (create new tag)

## Diagnostic Commands

If you encounter issues, these commands help diagnose the current state:

```bash
# Check git status
git status

# Check recent commits
git log --oneline -5

# Check recent tags
git tag -l | tail -5

# Check current branch and tracking
git branch -vv

# Check if current commit is on origin/master
git log origin/master..HEAD

# Check current versions
echo "Current versions:"
echo "pyproject.toml: $(grep '^version = ' pyproject.toml)"
echo "easier/__init__.py: $(grep '__version__ = ' easier/__init__.py)"
echo "Latest git tag: $(git describe --tags --abbrev=0 2>/dev/null || echo 'No tags')"
```