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
    echo "Installing vale v{{ vale_version }}..."
    if [ -f vale ]; then
    VALE_BIN="${HOME}/.local/bin/vale"
    echo "Installing vale v{{ vale_version }}..."
    if command -v vale >/dev/null 2>&1; then
        echo "vale is already available in PATH"
        exit 0
    fi
    if [ -f "${VALE_BIN}" ]; then
        echo "vale binary already exists at ${VALE_BIN}"
        exit 0
    fi
    mkdir -p "$(dirname "${VALE_BIN}")"
    wget -O vale.tar.gz "https://github.com/errata-ai/vale/releases/download/v{{ vale_version }}/vale_{{ vale_version }}_Linux_64-bit.tar.gz"
    tar -xzf vale.tar.gz vale
    mv vale "${VALE_BIN}"
    chmod +x "${VALE_BIN}"
    rm vale.tar.gz
    echo "vale v{{ vale_version }} installed successfully to ${VALE_BIN}"

# Sync vale styles (requires vale to be installed)
sync-vale: install-vale
    vale sync

# Run vale linter on markdown files
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
lint-all: lint
