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
When completing work on this repository:

1. **Complete the assigned task** - Make all necessary modifications to code, tests, and documentation
2. **Track all changes** - Maintain awareness of what was modified, added, or removed
3. **Generate comprehensive commit message** - Create detailed commit message describing all changes
4. **Propose commit to user** - Present the proposed commit and ask for explicit approval
5. **Wait for user confirmation** - Never execute git commands without user saying "yes"
6. **Execute only on approval** - Stage files and create commit only after user approval

### Git Commit Message Format
When proposing commits, use this structure:

```
[Component]: Brief summary of changes

Detailed description:
- Specific modules/functions modified
- New functionality added or logic changed
- Tests added or modified
- Dependencies or imports changed
- Impact on library functionality

Files changed:
- module_name.py: [specific changes]
- test_module.py: [test updates]
- [other files if applicable]

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### User Control Requirements
- **No automatic git operations** - All git commands require explicit user approval
- **Present proposed commits clearly** - Show the full commit message before asking for approval
- **Allow user modifications** - User can request changes to commit messages
- **Respect user decisions** - If user rejects a commit, ask how they'd like to proceed

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