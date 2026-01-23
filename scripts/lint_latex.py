#!/usr/bin/env python3
"""
LaTeX/MathJax linter for markdown and Python files.

This script checks for common LaTeX formatting errors in markdown files
and Python docstrings, particularly in inline math ($...$) and display
math ($$...$$) blocks.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class LatexLinter:
    """Linter for LaTeX math in markdown files."""

    def __init__(self):
        self.errors = []
        self._content_for_linting = ""  # Will store content with code blocks removed
        # Pattern to find multi-letter words in subscripts/superscripts
        # that should be wrapped in \text{} or similar commands
        self.patterns = [
            # Subscripts with multi-letter words (not wrapped in \text{})
            (
                r'_\{([a-zA-Z]{2,})\}',
                r'Subscript with multi-letter word "{}" should use \\text{{}} or be a single letter/symbol',
                'subscript'
            ),
            # Superscripts with multi-letter words (not wrapped in \text{})
            (
                r'\^\{([a-zA-Z]{2,})\}',
                r'Superscript with multi-letter word "{}" should use \\text{{}} or be a single letter/symbol',
                'superscript'
            ),
        ]

    def _is_valid_subscript(self, word: str) -> bool:
        """Check if a subscript word is valid without \text{}."""
        # Single letters are OK (e.g., _i, _j, _k, _a, _b)
        if len(word) == 1:
            return True
        
        # Common math abbreviations and dimension names that don't need \text{}
        valid_subscripts = {
            'min', 'max', 'avg', 'sin', 'cos', 'tan', 'log', 'ln', 'exp',
            'id', 'op', 'im', 're', 'tr',  # common math operators
            'model',  # d_model is standard notation
        }
        if word.lower() in valid_subscripts:
            return True
        
        # Multiple-letter tensor indices (all lowercase letters) are OK
        # (like 'ab', 'ij', 'ia', 'bij', 'hia', etc.)
        # This is standard tensor notation
        if all(c in 'abcdefghijklmnopqrstuvwxyz' for c in word):
            return True
        
        return False

    def _is_valid_superscript(self, word: str) -> bool:
        """Check if a superscript word is valid without \text{}."""
        # Single letters are OK
        if len(word) == 1:
            return True
        
        # Common exponents
        valid_superscripts = {'st', 'nd', 'rd', 'th'}  # ordinals
        if word in valid_superscripts:
            return True
        
        # Multiple-letter tensor indices (all lowercase letters) are OK
        # This is standard tensor notation
        if all(c in 'abcdefghijklmnopqrstuvwxyz' for c in word):
            return True
        
        return False

    def extract_math_blocks(self, content: str) -> List[Tuple[str, int, int]]:
        """Extract all math blocks from markdown content, excluding code blocks."""
        # First, remove all code blocks to avoid false positives
        # Remove fenced code blocks (```...```)
        content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        # Remove inline code (`...`)
        content_no_code = re.sub(r'`[^`]+`', '', content_no_code)
        
        math_blocks = []
        
        # Find display math blocks ($$...$$)
        for match in re.finditer(r'\$\$(.*?)\$\$', content_no_code, re.DOTALL):
            # Get the position in the original content
            start_in_cleaned = match.start()
            # We need to find this math block in the original content
            # For simplicity, we'll use the cleaned content for linting
            math_blocks.append((match.group(1), match.start(), match.end()))
        
        # Find inline math blocks ($...$) that aren't part of $$...$$
        # We need to be careful not to match the $ signs from $$
        inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'
        for match in re.finditer(inline_pattern, content_no_code):
            math_blocks.append((match.group(1), match.start(), match.end()))
        
        # Adjust positions to use cleaned content
        # Update the lint_math_block to use cleaned content
        self._content_for_linting = content_no_code
        
        return math_blocks

    def lint_math_block(self, math_content: str, start_pos: int, filepath: str):
        """Lint a single math block for errors."""
        # Get line number from cleaned content
        line_num = self._content_for_linting[:start_pos].count('\n') + 1
        
        for pattern, error_msg, pattern_type in self.patterns:
            for match in re.finditer(pattern, math_content):
                word = match.group(1)
                # Check if this word needs \text{} wrapper based on pattern type
                validator = self._is_valid_subscript if pattern_type == 'subscript' else self._is_valid_superscript
                if not validator(word):
                    self.errors.append({
                        'file': filepath,
                        'line': line_num,
                        'error': error_msg.format(word),
                        'match': match.group(0)
                    })

    def lint_file(self, filepath: Path) -> int:
        """Lint a single file and return number of errors found."""
        try:
            content = filepath.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)
            return 0
        
        math_blocks = self.extract_math_blocks(content)
        
        for math_content, start_pos, _ in math_blocks:
            self.lint_math_block(math_content, start_pos, str(filepath))
        
        return len([e for e in self.errors if e['file'] == str(filepath)])

    def print_errors(self):
        """Print all errors found."""
        if not self.errors:
            print("✓ No LaTeX formatting errors found!")
            return
        
        print(f"\n{'='*70}")
        print(f"Found {len(self.errors)} LaTeX formatting error(s):")
        print(f"{'='*70}\n")
        
        for error in self.errors:
            print(f"{error['file']}:{error['line']}")
            print(f"  Error: {error['error']}")
            print(f"  Found: {error['match']}")
            print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Lint LaTeX math in markdown files'
    )
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='Markdown or Python files to lint'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically fix errors (not implemented yet)'
    )
    
    args = parser.parse_args()
    
    linter = LatexLinter()
    total_errors = 0
    
    for filepath in args.files:
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist", file=sys.stderr)
            continue
        
        if filepath.suffix not in ['.md', '.markdown', '.py']:
            print(f"Warning: {filepath} is not a markdown or Python file", file=sys.stderr)
            continue
        
        errors = linter.lint_file(filepath)
        total_errors += errors
    
    linter.print_errors()
    
    return 1 if total_errors > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
