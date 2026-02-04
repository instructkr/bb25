#!/usr/bin/env bash
set -euo pipefail

if ! command -v pyodide >/dev/null 2>&1; then
  echo "pyodide CLI not found. Install pyodide-build and retry."
  echo "Example: python -m pip install pyodide-build"
  exit 1
fi

# Builds a wheel suitable for Pyodide.
# You may need to set up a Pyodide toolchain and Emscripten beforehand.
pyodide build --wheel -o dist
