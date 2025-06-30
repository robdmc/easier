#!/usr/bin/env python3
"""
Script to find all imports inside functions/methods across the codebase.
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

class ImportFinder(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.function_imports = []
        self.function_stack = []
        
    def visit_FunctionDef(self, node):
        # Track function/method context
        if self.function_stack:
            func_name = f"{'.'.join(self.function_stack)}.{node.name}"
        else:
            func_name = node.name
        
        self.function_stack.append(node.name)
        
        # Visit the function body
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                import_info = self._get_import_info(child, func_name, child.lineno)
                self.function_imports.append(import_info)
        
        # Continue visiting nested functions
        self.generic_visit(node)
        self.function_stack.pop()
    
    def visit_AsyncFunctionDef(self, node):
        # Handle async functions the same way
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        # Track class context
        self.function_stack.append(f"class {node.name}")
        self.generic_visit(node)
        self.function_stack.pop()
    
    def _get_import_info(self, node, func_name: str, line_no: int) -> Dict[str, Any]:
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
            return {
                'file': self.filename,
                'function': func_name,
                'line': line_no,
                'type': 'import',
                'statement': f"import {', '.join(imports)}",
                'modules': imports
            }
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            imports = [alias.name for alias in node.names]
            if node.names[0].name == '*':
                statement = f"from {module} import *"
            else:
                statement = f"from {module} import {', '.join(imports)}"
            return {
                'file': self.filename,
                'function': func_name,
                'line': line_no,
                'type': 'from_import',
                'statement': statement,
                'module': module,
                'imports': imports
            }

def find_function_imports(directory: str) -> List[Dict[str, Any]]:
    """Find all imports inside functions across Python files in directory."""
    all_imports = []
    
    for py_file in Path(directory).rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            finder = ImportFinder(str(py_file))
            finder.visit(tree)
            all_imports.extend(finder.function_imports)
            
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
    
    return all_imports

def main():
    print("Finding function-level imports in the easier/ directory...")
    print("=" * 60)
    
    imports = find_function_imports('easier/')
    
    if not imports:
        print("No function-level imports found!")
        return
    
    # Group by file
    by_file = {}
    for imp in imports:
        file_path = imp['file']
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(imp)
    
    # Print results
    for file_path, file_imports in sorted(by_file.items()):
        print(f"\n📁 {file_path}")
        print("-" * 40)
        
        for imp in sorted(file_imports, key=lambda x: x['line']):
            print(f"  Line {imp['line']:3d}: {imp['function']}")
            print(f"           {imp['statement']}")
    
    print(f"\n📊 Summary: Found {len(imports)} function-level imports in {len(by_file)} files")

if __name__ == "__main__":
    main()