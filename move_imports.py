#!/usr/bin/env python3
"""
Script to move all function-level imports to the top of files.
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
import re

class ImportMover(ast.NodeTransformer):
    def __init__(self, filename: str):
        self.filename = filename
        self.function_imports = set()
        self.function_stack = []
        
    def visit_FunctionDef(self, node):
        # Track function/method context
        self.function_stack.append(node.name)
        
        # Remove imports from function body and collect them
        new_body = []
        for child in node.body:
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                self.function_imports.add(ast.unparse(child))
            else:
                new_body.append(child)
        
        node.body = new_body
        
        # Continue visiting nested functions
        self.generic_visit(node)
        self.function_stack.pop()
        return node
    
    def visit_AsyncFunctionDef(self, node):
        # Handle async functions the same way
        return self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        # Track class context and visit methods
        self.function_stack.append(f"class {node.name}")
        self.generic_visit(node)
        self.function_stack.pop()
        return node

def extract_existing_imports(content: str) -> Tuple[List[str], int]:
    """Extract existing module-level imports and find where to insert new ones."""
    lines = content.split('\n')
    imports = []
    last_import_line = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('import ') or stripped.startswith('from ')) and not line.startswith('    '):
            imports.append(stripped)
            last_import_line = i
        elif stripped.startswith('#') or stripped == '' or stripped.startswith('"""') or stripped.startswith("'''"):
            # Skip comments, empty lines, and docstrings at the top
            continue
        elif not (stripped.startswith('import ') or stripped.startswith('from ')):
            # If we hit non-import code, stop looking for imports
            break
    
    return imports, last_import_line

def move_imports_in_file(file_path: str) -> bool:
    """Move function-level imports to top of file. Returns True if changes were made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Parse and transform AST
        tree = ast.parse(original_content)
        mover = ImportMover(file_path)
        new_tree = mover.visit(tree)
        
        if not mover.function_imports:
            return False
        
        # Get existing imports
        existing_imports, last_import_line = extract_existing_imports(original_content)
        
        # Combine and deduplicate imports
        all_imports = set(existing_imports) | mover.function_imports
        sorted_imports = sorted(all_imports)
        
        # Generate new content
        new_content = ast.unparse(new_tree)
        
        # Insert imports at the top (after any initial comments/docstrings)
        lines = new_content.split('\n')
        
        # Find insertion point (after module docstring if present)
        insert_idx = 0
        in_docstring = False
        docstring_quote = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_quote = stripped[:3]
                    if stripped.count(docstring_quote) == 2 and len(stripped) > 3:
                        # Single-line docstring
                        insert_idx = i + 1
                    else:
                        # Multi-line docstring
                        in_docstring = True
                elif stripped.startswith('#') or stripped == '':
                    # Skip comments and empty lines
                    continue
                else:
                    # Found code, insert here
                    insert_idx = i
                    break
            else:
                if docstring_quote in line:
                    # End of multi-line docstring
                    in_docstring = False
                    insert_idx = i + 1
                    break
        
        # Insert imports
        import_lines = sorted_imports + ['']  # Add empty line after imports
        lines = lines[:insert_idx] + import_lines + lines[insert_idx:]
        
        final_content = '\n'.join(lines)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"✅ {file_path}: moved {len(mover.function_imports)} imports")
        return True
        
    except Exception as e:
        print(f"❌ {file_path}: {e}")
        return False

def main():
    print("Moving function-level imports to top of files...")
    print("=" * 60)
    
    files_changed = 0
    total_files = 0
    
    for py_file in Path('easier/').rglob('*.py'):
        # Skip __pycache__ and other generated files
        if '__pycache__' in str(py_file) or 'build/' in str(py_file):
            continue
            
        total_files += 1
        if move_imports_in_file(str(py_file)):
            files_changed += 1
    
    print(f"\n📊 Summary: Modified {files_changed} out of {total_files} files")

if __name__ == "__main__":
    main()