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

### Check 1: PyPI Availability
```bash
python check_version.py --pypi-available
```

**If this fails with:** `Error: Version X.Y.Z is already published on PyPI. Run update_version.py`
- **Fix:** Choose a different version number and run `python update_version.py <new_version>`, then repeat Step 1

### Check 2: Clean Working Directory
```bash
python check_version.py --clean-working-dir
```

**If this fails with:** `Error: You have uncommitted changes. Please commit or stash them before proceeding.`
- **Diagnose:** `git status`
- **Fix Option 1 (Commit):** 
  ```bash
  git add pyproject.toml easier/__init__.py
  git commit -m "[Release]: Bump version to vX.Y.Z for PyPI release"
  ```
- **Fix Option 2 (Stash):** `git stash`

### Check 3: No Unpushed Commits
```bash
python check_version.py --no-unpushed
```

**If this fails with:** `Error: You have commits that have not been pushed to origin. Please push them before proceeding.`
- **Diagnose:** `git status -sb`
- **Fix:** `git push origin`

### Check 4: Current Commit Tagged
```bash
python check_version.py --commit-tagged
```

**If this fails with:** `Error: The current commit is not tagged. Please tag this commit.`
- **Fix:** 
  ```bash
  git tag X.Y.Z
  echo "Created tag X.Y.Z"
  ```

### Check 5: Commit Pushed to Origin
```bash
python check_version.py --commit-pushed
```

**If this fails with:** `Error: The current commit has not been pushed to origin/master.`
- **Fix:** `git push origin`

### Check 6: Tag Pushed to Origin
```bash
python check_version.py --tag-pushed
```

**If this fails with:** `Error: Tag 'X.Y.Z' has not been pushed to origin.`
- **Diagnose:** `git tag -l | grep X.Y.Z`
- **Fix:** `git push origin X.Y.Z`

### Check 7: Version Sync
```bash
python check_version.py --version-sync
```

**If this fails with:** `Error: Latest git tag (A.B.C) does not match pyproject.toml version (X.Y.Z).`
- **Fix Option 1:** Update pyproject.toml to match tag A.B.C
- **Fix Option 2:** Create new tag matching pyproject.toml version:
  ```bash
  git tag X.Y.Z
  git push origin X.Y.Z
  ```

**If this fails with:** `Error: Latest git tag (A.B.C) does not match easier/__init__.py version (X.Y.Z).`
- **Fix:** Update easier/__init__.py: `sed -i '' 's/__version__ = .*/__version__ = "A.B.C"/' easier/__init__.py`

## Step 3: Final Validation

### Run All Checks
```bash
python check_version.py
```

**Expected output:** `All version checks passed. Repository is clean and up to date.`

If any check fails, return to the appropriate step above to fix the issue.

## Step 4: Ready to Publish

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