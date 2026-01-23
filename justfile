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
        echo "vale binary already exists"
        exit 0
    fi
    wget -O vale.tar.gz "https://github.com/errata-ai/vale/releases/download/v{{ vale_version }}/vale_{{ vale_version }}_Linux_64-bit.tar.gz"
    tar -xzf vale.tar.gz vale
    chmod +x vale
    rm vale.tar.gz
    echo "vale v{{ vale_version }} installed successfully"

# Sync vale styles (requires vale to be installed)
sync-vale: install-vale
    ./vale sync

# Run vale linter on markdown files
lint: sync-vale
    ./vale --output=line site/content/*.md site/content/**/*.md *.md

# Clean up vale binary and styles
clean-vale:
    rm -f vale
    rm -rf .vale/styles/

# Run all linters
lint-all: lint
