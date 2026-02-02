# justfile for attn-as-bilinear-form project

# Vale version to install
vale_version := "3.0.5"

# Default recipe (shows available commands)
default:
    @just --list

# Install vale linter
install-vale:
    #!/usr/bin/env bash
    set -euo pipefail
    VALE_BIN="${HOME}/.local/bin/vale"
    echo "Installing vale v{{ vale_version }}..."
    
    # Check if vale is already available
    if command -v vale >/dev/null 2>&1; then
        echo "vale is already available in PATH"
        exit 0
    fi
    if [ -f "${VALE_BIN}" ]; then
        echo "vale binary already exists at ${VALE_BIN}"
        exit 0
    fi
    
    # Detect OS and architecture
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    case "${OS}" in
        Linux*)
            case "${ARCH}" in
                x86_64|amd64)
                    PLATFORM="Linux_64-bit"
                    ;;
                aarch64|arm64)
                    PLATFORM="Linux_arm64"
                    ;;
                *)
                    echo "Unsupported Linux architecture: ${ARCH}"
                    exit 1
                    ;;
            esac
            ;;
        Darwin*)
            case "${ARCH}" in
                x86_64)
                    PLATFORM="macOS_64-bit"
                    ;;
                arm64)
                    PLATFORM="macOS_arm64"
                    ;;
                *)
                    echo "Unsupported macOS architecture: ${ARCH}"
                    exit 1
                    ;;
            esac
            ;;
        MINGW*|MSYS*|CYGWIN*)
            case "${ARCH}" in
                x86_64|amd64)
                    PLATFORM="Windows_64-bit"
                    ;;
                *)
                    echo "Unsupported Windows architecture: ${ARCH}"
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "Unsupported operating system: ${OS}"
            exit 1
            ;;
    esac
    
    echo "Detected platform: ${PLATFORM}"
    mkdir -p "$(dirname "${VALE_BIN}")"
    
    # Download and install
    if [[ "${OS}" == MINGW* ]] || [[ "${OS}" == MSYS* ]] || [[ "${OS}" == CYGWIN* ]]; then
        # Windows uses .zip
        ARCHIVE="vale.zip"
        wget -O "${ARCHIVE}" "https://github.com/errata-ai/vale/releases/download/v{{ vale_version }}/vale_{{ vale_version }}_${PLATFORM}.zip"
        unzip -o "${ARCHIVE}" vale.exe
        mv vale.exe "${VALE_BIN}.exe"
        chmod +x "${VALE_BIN}.exe"
        rm "${ARCHIVE}"
    else
        # Linux and macOS use .tar.gz
        ARCHIVE="vale.tar.gz"
        wget -O "${ARCHIVE}" "https://github.com/errata-ai/vale/releases/download/v{{ vale_version }}/vale_{{ vale_version }}_${PLATFORM}.tar.gz"
        tar -xzf "${ARCHIVE}" vale
        mv vale "${VALE_BIN}"
        chmod +x "${VALE_BIN}"
        rm "${ARCHIVE}"
    fi
    
    echo "vale v{{ vale_version }} installed successfully to ${VALE_BIN}"

# Sync vale styles (requires vale to be installed)
sync-vale: install-vale
    vale sync

# Run vale linter on markdown files
lint-vale: sync-vale
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s globstar
    vale --output=line site/content/*.md site/content/**/*.md *.md

# Run LaTeX/MathJax linter on markdown files
lint-latex:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 scripts/lint_latex.py \
        $(find . -name "*.md" -type f -not -path "./.git/*")

# Run vale linter on markdown files (deprecated, use lint-vale)
lint: sync-vale
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s globstar
    vale --output=line site/content/*.md site/content/**/*.md *.md

# Clean up vale binary and styles
clean-vale:
    rm -f "${HOME}/.local/bin/vale"
    rm -rf .vale/styles/

# Run all linters
lint-all: lint-vale lint-latex

# Fix underscores in LaTeX math blocks (prevents Markdown emphasis interpretation)
fix-math:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 scripts/fix_math_underscores.py \
        $(find site/content -name "*.md" -type f)

# Build the Zola site
build:
    cd site && zola build

# Serve the Zola site locally
serve:
    cd site && zola serve

# Full build pipeline: fix math, lint, build
build-all: fix-math lint-all build
