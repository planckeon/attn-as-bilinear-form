#!/usr/bin/env python3
"""
Fix LaTeX escaping issues for Zola/CommonMark + KaTeX rendering.

Issues fixed:
1. Matrix newlines: \\\\ gets consumed to \\, so we need \\\\\\\\ 
   (actually \\\\\\\\, which becomes \\\\ in HTML, which KaTeX interprets as newline)
2. Underscore subscripts: }_{...} gets interpreted as markdown emphasis
   Fixed by: }&#95;{

Usage:
    python scripts/fix_latex_escapes.py site/content/
    python scripts/fix_latex_escapes.py site/content/theory/bilinear.md
"""

import re
import sys
from pathlib import Path


def fix_matrix_newlines(content: str) -> str:
    """
    Fix matrix newlines in LaTeX blocks.
    
    In markdown source: \\\\ (two backslashes)
    After markdown processing: \\ (one backslash) - WRONG!
    
    We need: \\\\\\\\ (four backslashes in source)
    Which becomes: \\\\ in HTML (two backslashes)
    Which KaTeX interprets as: newline
    """
    # Match display math blocks with matrices
    def fix_block(match):
        block = match.group(0)
        # Only fix if it contains matrix environments
        if any(env in block for env in ['pmatrix', 'bmatrix', 'matrix', 'vmatrix', 'Bmatrix', 'Vmatrix']):
            # Replace \\ with \\\\ but only where it's a row separator
            # Be careful not to affect other backslash uses
            # Match \\ that's followed by newline or space (row separator)
            # But not already escaped (not \\\\)
            
            # First, normalize any existing over-escaping
            block = re.sub(r'\\\\\\\\', r'\\\\', block)
            
            # Now double the backslashes for row separators
            # Match \\ followed by whitespace/newline before next row or \end
            block = re.sub(r'\\\\(\s*\n?\s*)((?:\d|[a-zA-Z]|\\|&|\-|\s|\^|_|\{|\}|\(|\)|\[|\]|\.|,|;|:|\+|\*|/|=|\'|"|`|!|@|#|\$|%|\||~|<|>)+?)(\s*(?:\\\\|\\end))', 
                          r'\\\\\\\\\\1\\2\\3', block)
        return block
    
    # Process display math blocks ($$...$$)
    content = re.sub(r'\$\$[^$]+\$\$', fix_block, content, flags=re.DOTALL)
    
    return content


def fix_matrix_newlines_simple(content: str) -> str:
    """
    Simpler approach: in any pmatrix/bmatrix environment, replace \\\\ with \\\\\\\\
    """
    lines = content.split('\n')
    result = []
    in_matrix = False
    matrix_depth = 0
    
    for line in lines:
        # Check if we're entering a matrix
        if re.search(r'\\begin\{[pbvBV]?matrix\}', line):
            in_matrix = True
            matrix_depth += line.count('\\begin{')
        
        # Check if we're exiting a matrix  
        if re.search(r'\\end\{[pbvBV]?matrix\}', line):
            matrix_depth -= line.count('\\end{')
            if matrix_depth <= 0:
                in_matrix = False
                matrix_depth = 0
        
        # If in matrix, double the backslashes for row separators
        if in_matrix:
            # Replace \\ with \\\\ but not if already \\\\
            # Use negative lookbehind/lookahead
            line = re.sub(r'(?<!\\)\\\\(?!\\)', r'\\\\\\\\', line)
        
        result.append(line)
    
    return '\n'.join(result)


def fix_underscore_subscripts(content: str) -> str:
    """
    Fix underscore subscripts: }_{...} → }&#95;{
    Already implemented in fix_math_underscores.py
    """
    # Pattern: }_{  inside math contexts
    content = re.sub(r'\}_{', r'}&#95;{', content)
    return content


def process_file(filepath: Path, dry_run: bool = False) -> tuple[bool, list[str]]:
    """Process a single markdown file. Returns (changed, list of changes)."""
    content = filepath.read_text(encoding='utf-8')
    original = content
    changes = []
    
    # Fix matrix newlines
    new_content = fix_matrix_newlines_simple(content)
    if new_content != content:
        changes.append("Fixed matrix newlines (doubled backslashes)")
        content = new_content
    
    # Fix underscore subscripts (if not already done)
    new_content = fix_underscore_subscripts(content)
    if new_content != content:
        changes.append("Fixed underscore subscripts")
        content = new_content
    
    if content != original:
        if not dry_run:
            filepath.write_text(content, encoding='utf-8')
        return True, changes
    return False, changes


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_latex_escapes.py <path> [--dry-run]")
        print("  <path> can be a file or directory")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv
    
    if dry_run:
        print("DRY RUN - no files will be modified\n")
    
    files_to_process = []
    if target.is_file():
        files_to_process = [target]
    elif target.is_dir():
        files_to_process = list(target.rglob('*.md'))
    else:
        print(f"Error: {target} not found")
        sys.exit(1)
    
    total_changed = 0
    for filepath in files_to_process:
        changed, changes = process_file(filepath, dry_run)
        if changed:
            total_changed += 1
            print(f"{'Would fix' if dry_run else 'Fixed'}: {filepath}")
            for change in changes:
                print(f"  - {change}")
    
    print(f"\n{'Would modify' if dry_run else 'Modified'} {total_changed} file(s)")


if __name__ == '__main__':
    main()
