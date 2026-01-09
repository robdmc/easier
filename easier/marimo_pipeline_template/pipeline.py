#!/usr/bin/env python3
"""Creates files for marimo processing pipeline"""

import os
import urllib.request
import zipfile
from pathlib import Path
try:
    from importlib import resources
except ImportError:
    # Fallback for Python < 3.9
    import importlib_resources as resources

import click


@click.command()
@click.help_option('--help', '-h')
def main():
    """Creates files for marimo processing pipeline"""
    click.echo("Creating marimo processing pipeline files...")
    
    try:
        # Get current working directory
        current_dir = os.getcwd()
        click.echo(f"Target directory: {current_dir}")
        
        # Find the zip file in the package
        try:
            # Use importlib.resources to find the zip file
            with resources.path('easier.marimo_pipeline_template.manifest', 'marimo_pipeline_template.zip') as zip_path:
                zip_file_path = str(zip_path)
        except (ModuleNotFoundError, FileNotFoundError):
            # Fallback: construct path relative to this file
            this_dir = Path(__file__).parent
            zip_file_path = this_dir / 'manifest' / 'marimo_pipeline_template.zip'
            if not zip_file_path.exists():
                click.echo("Error: Could not find marimo_pipeline_template.zip", err=True)
                return 1
        
        click.echo(f"Using template from: {zip_file_path}")
        
        # Extract the zip file to current directory
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            files_to_extract = zip_ref.namelist()
            click.echo(f"Extracting {len(files_to_extract)} files:")
            
            # Check for existing files
            existing_files = []
            for file_name in files_to_extract:
                target_path = Path(current_dir) / file_name
                if target_path.exists():
                    existing_files.append(file_name)
            
            if existing_files:
                click.echo(f"Warning: The following files already exist and will be overwritten:")
                for file_name in existing_files:
                    click.echo(f"  - {file_name}")
                if not click.confirm("Continue?"):
                    click.echo("Aborted.")
                    return 0

            # Extract all files
            for file_name in files_to_extract:
                zip_ref.extract(file_name, current_dir)
                click.echo(f"  ✓ {file_name}")

        # Create .claude/prompts directory and download marimo CLAUDE.md
        claude_prompts_dir = Path(current_dir) / '.claude' / 'prompts'
        claude_prompts_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"  ✓ .claude/prompts/")

        marimo_md_path = claude_prompts_dir / 'marimo.md'
        try:
            click.echo("Downloading marimo CLAUDE.md...")
            urllib.request.urlretrieve(
                'https://docs.marimo.io/CLAUDE.md',
                marimo_md_path
            )
            click.echo(f"  ✓ .claude/prompts/marimo.md")
        except Exception as e:
            click.echo(f"  ⚠ Warning: Could not download marimo CLAUDE.md: {e}", err=True)

        # Ensure CLAUDE.md exists with marimo check instructions
        claude_md_path = Path(current_dir) / 'CLAUDE.md'
        marimo_check_marker = '<!-- MARIMO_CHECK_INSTRUCTIONS -->'

        if not claude_md_path.exists():
            claude_md_path.write_text('# CLAUDE.md\n\n')
            click.echo(f"  ✓ Created CLAUDE.md")

        claude_md_content = claude_md_path.read_text()
        if marimo_check_marker not in claude_md_content:
            marimo_check_block = '''
<!-- MARIMO_CHECK_INSTRUCTIONS -->
## Marimo Notebooks

After editing any marimo notebook (`.py` files with marimo cell markers), always run:

```bash
uv run marimo check <notebook_file>
```

This validates the notebook structure and catches errors before runtime.
<!-- /MARIMO_CHECK_INSTRUCTIONS -->
'''
            claude_md_path.write_text(claude_md_content + marimo_check_block)
            click.echo(f"  ✓ Added marimo check instructions to CLAUDE.md")
        else:
            click.echo(f"  ✓ CLAUDE.md already has marimo check instructions")

        click.echo("\n✅ Marimo processing pipeline files created successfully!")
        click.echo("You can now start working with your marimo notebooks.")

        return 0

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


if __name__ == '__main__':
    main()
