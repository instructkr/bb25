# Pyodide Build Notes

This project targets Pyodide by keeping the Python API pure (no file I/O, no OS dependencies) and using PyO3 for bindings. The actual build depends on your Pyodide toolchain setup.

## Minimal Flow

1) Install a Pyodide build toolchain (pyodide-build + Emscripten).
2) From `bayesian_bm25_rs/`, run:

```
./scripts/build_pyodide.sh
```

This will place a wheel in `dist/`, which can be loaded in Pyodide.

## Notes

- If your Pyodide Python version is newer than the PyO3 version supports, set:

```
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
```

- For reproducibility, pin your Pyodide toolchain and Python version.
