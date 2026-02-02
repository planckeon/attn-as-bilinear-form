#!/usr/bin/env python3
"""
Fix LaTeX math in Zola markdown files.

Zola's CommonMark parser interprets underscores as emphasis, which breaks
LaTeX math expressions. This script escapes underscores within math blocks
by using &#95; HTML entities, which CommonMark passes through untouched
and KaTeX interprets correctly as underscores.
"""

import re
import sys
from pathlib import Path


def escape_underscores_in_math(content: str) -> str:
    """
    Escape underscores in math blocks using HTML entities.
    
    This handles both inline ($...$) and display ($$...$$) math.
    """
    result = []
    i = 0
    
    while i < len(content):
        # Check for display math ($$...$$)
        if content[i:i+2] == '$$':
            # Find the closing $$
            end = content.find('$$', i + 2)
            if end != -1:
                math_content = content[i+2:end]
                # Only escape underscores that are:
                # 1. After } (like }_{...})
                # 2. Before { when preceded by letters/numbers (like x_{...})
                # This avoids escaping subscript _ that work fine
                
                # Pattern: }_{  -> }&#95;{
                math_content = re.sub(r'\}_\{', r'}&#95;{', math_content)
                
                result.append('$$')
                result.append(math_content)
                result.append('$$')
                i = end + 2
                continue
            
        # Check for inline math ($...$) - but not $$
        if content[i] == '$' and (i == 0 or content[i-1] != '$') and (i + 1 < len(content) and content[i+1] != '$'):
            # Find the closing $
            end = content.find('$', i + 1)
            while end != -1 and end + 1 < len(content) and content[end + 1] == '$':
                # Skip if this is the start of $$
                end = content.find('$', end + 2)
            
            if end != -1:
                math_content = content[i+1:end]
                # Same escaping pattern
                math_content = re.sub(r'\}_\{', r'}&#95;{', math_content)
                
                result.append('$')
                result.append(math_content)
                result.append('$')
                i = end + 1
                continue
        
        result.append(content[i])
        i += 1
    
    return ''.join(result)


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """Process a single file. Returns True if changes were made."""
    content = filepath.read_text(encoding='utf-8')
    new_content = escape_underscores_in_math(content)
    
    if content != new_content:
        if dry_run:
            print(f"Would modify: {filepath}")
        else:
            filepath.write_text(new_content, encoding='utf-8')
            print(f"Modified: {filepath}")
        return True
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix LaTeX underscores in Zola markdown files')
    parser.add_argument('files', nargs='+', type=Path, help='Markdown files to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    
    args = parser.parse_args()
    
    modified = 0
    for filepath in args.files:
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist", file=sys.stderr)
            continue
        if filepath.suffix not in ['.md', '.markdown']:
            print(f"Warning: {filepath} is not a markdown file", file=sys.stderr)
            continue
        
        if process_file(filepath, args.dry_run):
            modified += 1
    
    print(f"\n{'Would modify' if args.dry_run else 'Modified'} {modified} file(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
