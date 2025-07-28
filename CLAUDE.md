# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
- Run all tests: `make test` or `uv run pytest easier/tests/`
- Run tests with coverage: `make cover`
- Run a single test file: `uv run pytest easier/tests/test_<module>.py`

### Building and Publishing
- Clean build artifacts: `make clean`
- Publish to PyPI: `make publish`

### Package Publishing
When the user requests to publish the package, execute this complete publish flow:

**Publishing Flow Process:**
1. **Show current PyPI version** - Display the current published version (currently 1.9.2)
2. **Prompt for new version** - Ask user to specify the desired version tag (e.g., "1.9.3", "2.0.0")
3. **Version validation** - Check that both `pyproject.toml` and `easier/__init__.py` versions match the requested tag
4. **Version updates** - If versions don't match the requested tag, update both files with the new version
5. **Git operations with individual file confirmation**:
   - Run `git status` to show current state
   - Individually add `pyproject.toml` with user confirmation: `git add pyproject.toml`
   - Individually add `easier/__init__.py` with user confirmation: `git add easier/__init__.py`
   - Create descriptive commit message: `[Release]: Bump version to vX.Y.Z for PyPI release`
   - Commit changes: `git commit -m "message"`
   - Push to origin: `git push origin`
6. **Git tagging**:
   - Create version tag: `git tag X.Y.Z` (e.g., `git tag 1.9.3`)
   - Push tag to origin: `git push origin X.Y.Z`
7. **PyPI publish** - Run `make publish` to release the package

**Safety Requirements:**
- NEVER use `git add .` - always add files individually with user confirmation
- Verify version consistency across both files before proceeding
- Use descriptive commit messages for version bumps following the format: `[Release]: Bump version to vX.Y.Z for PyPI release`
- Include confirmation steps throughout the process
- Show git status before and after operations for transparency

### Linting and Code Quality
This project uses:
- **ruff** for linting (line length: 120, ignores F821)
- **black** for formatting (line length: 120)
- **pyright** for type checking (basic mode)

Run linting with: `uv run ruff check easier/`
Run formatting with: `uv run black easier/`
Run type checking with: `uv run pyright easier/`

## Git Workflow

### Task Completion and Git Integration
When the user requests the git workflow, automatically commit all changes to the current branch with a descriptive commit message based on the work context.

Process:
1. **Show git status** - Display current repository state and modified files
2. **Ask which files to add** - Present list of changed files and ask user to specify which to stage
3. **Use work context** - Generate commit message from tasks completed and changes made during the session
4. **Create descriptive message** - Describe the actual functionality changes, not just file diffs
5. **Stage specified files and commit** - Execute git add on selected files and git commit with context message

### Git Commit Message Format
Use context-aware commit messages based on the work performed:

```
[Component]: Brief summary of changes

- Specific functionality added or modified
- New features implemented
- Bug fixes or improvements made
- Tests added or updated
- Key impacts on library behavior

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Context Sources for Commit Messages
Create messages using:
- **TodoWrite tasks** - What objectives were completed
- **File operations** - Which modules were modified and why
- **Code changes** - Actual functionality changes made
- **Test results** - Testing outcomes and new test coverage

## Architecture

This is the "easier" Python library - a collection of analysis tools designed to minimize boilerplate code in Jupyter Notebook workflows. The codebase is organized as follows:

### Core Structure
- **easier/**: Main package directory containing all modules
- **easier/tests/**: Test suite using pytest
- **docs/**: Sphinx documentation

### Key Components

**Data Analysis Tools:**
- `fitter.py` - Curve fitting wrapper around scipy optimization
- `param_state.py` - Parameter management for optimization problems
- `dataframe_tools.py` - Pandas DataFrame utilities
- `outlier_tools.py` - IQR-based outlier detection
- `ecdf_lib.py` - Empirical cumulative distribution functions

**System Utilities:**
- `timer.py` - Context manager for timing code sections
- `clock.py` - Stopwatch functionality for measuring code performance
- `memory.py` - Memory usage monitoring

**Data Sources:**
- `postgres.py` - PostgreSQL query wrapper with JinjaSQL templating
- `bigquery.py` - Google BigQuery integration
- `gsheet.py` - Google Sheets API wrapper
- `duckcacher.py` - DuckDB caching utilities

**Plotting (Holoviews-based):**
- `plotting.py` - Matplotlib and Holoviews plotting utilities
- `hvtools.py` - Holoviews-specific tools

**AI/ML Tools:**
- `ez_agent.py` - Simplified wrapper for pydantic-ai Agent creation with Gemini models
- `gemini.py` - Google Gemini API integration

**Utilities:**
- `utils.py` - General utility functions and classes
- `item.py` - Generic data container with dict/attribute access
- `crypt.py` - Password encryption/decryption utilities
- `iterify.py` - Iterator utilities

### Development Setup
This project uses:
- **uv** for Python package management
- **pytest** for testing with coverage support
- Standard Python packaging via setuptools
- Python 3.11+ required

### Key Design Patterns
- Uses `Item` class for flexible data containers supporting both dict and attribute access
- `ParamState` class for managing optimization parameters with scipy
- Context managers for resource management (Timer, Clock)
- JinjaSQL templating for dynamic database queries
- Holoviews for all plotting functionality