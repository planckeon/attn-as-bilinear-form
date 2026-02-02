#!/usr/bin/env python3
"""
LaTeX/MathJax linter for markdown files.

This script checks for common LaTeX formatting errors in markdown files,
particularly in inline math ($...$) and display math ($$...$$) blocks.
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
        self._original_content = ""  # Will store original content for line mapping
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
        
        # Patterns that might cause Markdown/LaTeX conflicts
        self.conflict_patterns = [
            # Underscore followed by word chars (potential emphasis conflict)
            (
                r'(?<!\\)_[a-zA-Z]{2,}_(?![{])',
                'Underscores around word may be interpreted as markdown emphasis: "{}"',
            ),
            # Asterisks that might be interpreted as emphasis
            (
                r'(?<!\\)\*[a-zA-Z]+\*(?![{])',
                'Asterisks around word may be interpreted as markdown emphasis: "{}"',
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
            math_blocks.append((match.group(1), match.start(), match.end()))
        
        # Find inline math blocks ($...$) that aren't part of $$...$$
        # We need to be careful not to match the $ signs from $$
        inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'
        for match in re.finditer(inline_pattern, content_no_code):
            math_blocks.append((match.group(1), match.start(), match.end()))
        
        # Store both original and cleaned content for line number mapping
        self._content_for_linting = content_no_code
        self._original_content = content
        
        return math_blocks

    def lint_math_block(self, math_content: str, start_pos: int, filepath: str):
        """Lint a single math block for errors."""
        # Calculate line number by finding the math block in the original content
        # We search for the math content to get accurate line numbers even when code blocks were removed
        
        line_num = self._content_for_linting[:start_pos].count('\n') + 1  # Default fallback
        
        # Try to find this specific math block in the original content for accurate line numbers
        # We look for the dollar-delimited math containing this content
        try:
            # Search for both inline ($...$) and display ($$...$$) math blocks
            # containing this exact math content in the original file
            escaped_content = re.escape(math_content)
            # Try display math first ($$...$$), then inline math ($...$)
            for delim in [r'\$\$' + escaped_content + r'\$\$', r'\$' + escaped_content + r'\$']:
                match = re.search(delim, self._original_content, re.DOTALL)
                if match:
                    line_num = self._original_content[:match.start()].count('\n') + 1
                    break
        except re.error:
            # If regex construction fails, use fallback
            pass
        
        # Mapping of pattern types to validator functions
        validators = {
            'subscript': self._is_valid_subscript,
            'superscript': self._is_valid_superscript
        }
        
        for pattern, error_msg, pattern_type in self.patterns:
            for match in re.finditer(pattern, math_content):
                word = match.group(1)
                # Check if this word needs \text{} wrapper based on pattern type
                validator = validators[pattern_type]
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
        
        return sum(1 for e in self.errors if e['file'] == str(filepath))

    def print_errors(self):
        """Print all errors found."""
        if not self.errors:
            print("[OK] No LaTeX formatting errors found!")
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
        help='Markdown files to lint'
    )
    
    args = parser.parse_args()
    
    linter = LatexLinter()
    total_errors = 0
    
    for filepath in args.files:
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist", file=sys.stderr)
            continue
        
        if filepath.suffix not in ['.md', '.markdown']:
            print(f"Warning: {filepath} is not a markdown file", file=sys.stderr)
            continue
        
        errors = linter.lint_file(filepath)
        total_errors += errors
    
    linter.print_errors()
    
    return 1 if total_errors > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
