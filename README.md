# Bayesian BM25 Experimental Validation (Rust)

This repository is a Rust port of the Python experimental validation for the Bayesian BM25 paper. It re-implements the same corpus, queries, scorers, and experiments, and is intended to reproduce the same numerical properties and PASS/FAIL outcomes. The implementation is self-contained and uses only the Rust standard library.

## Abstract

Hybrid search typically combines lexical signals (e.g., BM25) with vector similarity using heuristics such as weighted sums or Reciprocal Rank Fusion, which either require tuning or discard score magnitudes. The Bayesian BM25 formulation converts raw BM25 scores into calibrated probabilities via Bayes' theorem using a sigmoid likelihood and informative priors. Once both lexical and vector signals are valid probabilities, they can be combined using standard probability theory (AND = joint; OR = union), yielding bounded, interpretable outputs without ad-hoc scaling. This port validates the theory via ten numerical experiments that test formula equivalence, calibration, monotonicity, bounds, and optimization behavior.

## Quick Start

```
cargo run --manifest-path bayesian_bm25_rs/Cargo.toml
```

Expected output: 10 experiments with PASS/FAIL and supporting details. All should pass.

For Python, the extension module name is `bayesian_bm25` (distribution name `bayesian_bm25_rs`).

### Python (PyO3)

Build and install locally with maturin:

```
cd bayesian_bm25_rs
python -m pip install maturin
maturin develop
```

Smoke test:

```
python -c "import bayesian_bm25 as bb; print(len(bb.run_experiments()))"
```

If your Python version is newer than the PyO3 version supports, set:

```
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
```

### Pyodide / WASM (outline)

This project is designed to support a Pyodide build, but the exact build steps depend on your Pyodide toolchain and target environment. The binding layer avoids file I/O and uses only basic Python types, which keeps the API Pyodide-friendly.

If you are using `pyodide-build`, the typical flow is:

1) Install the Pyodide build toolchain.
2) Build the package from `pyproject.toml`.
3) Load the resulting wheel in your Pyodide runtime.

If you prefer a direct Rust build, ensure a WASM toolchain is installed and then build a wheel for the `wasm32-unknown-emscripten` target using maturin.

See `docs/pyodide.md` and `scripts/build_pyodide.sh` for a minimal scaffold.

## Scope and Validation Goals

The experiments are designed to check the core claims of the Bayesian BM25 formulation:

1) **BM25 Formula Equivalence**: The standard and rewritten BM25 expressions are algebraically identical.
2) **Score Calibration**: Bayesian outputs are bounded in [0, 1] and preserve BM25 ordering.
3) **Monotonicity Preservation**: Higher term frequency increases posterior probability (for similar lengths).
4) **Prior Bounds**: Composite priors remain within conservative bounds [0.1, 0.9].
5) **IDF Properties**: Non-negativity (for df <= N/2), monotone decrease with df, and score upper bounds.
6) **Hybrid Search Bounds**: AND <= min(inputs), OR >= max(inputs) for query-document pairs.
7) **Method Comparison**: Bayesian fusion vs naive sum vs RRF for ranking effects.
8) **Numerical Stability**: Log-space computation avoids NaN/Inf under extreme inputs.
9) **Parameter Learning**: Gradient descent reduces cross-entropy and learns alpha > 0.
10) **Conjunction/Disjunction Bounds**: AND <= OR and within min/max across many probability tuples.

## Experiments (Correspondence with the Paper)

| # | Experiment | Paper Reference | Check |
|---|-----------|-----------------|-------|
| 1 | BM25 Formula Equivalence | Def 1.1.1, 3.2.1 | Standard vs rewritten BM25 scores identical |
| 2 | Score Calibration | Sec 4 | Bayesian outputs in [0,1], ranking order preserved |
| 3 | Monotonicity Preservation | Thm 4.3.1 | Higher TF yields higher posterior (same length) |
| 4 | Prior Bounds | Thm 4.2.4 | Composite priors bounded in [0.1, 0.9] |
| 5 | IDF Properties | Thm 3.1.2, 3.1.3, 3.2.3 | IDF properties and score upper bound |
| 6 | Hybrid Search Quality | Thm 5.1.2, 5.2.2 | AND/OR bounds for lexical+vector probabilities |
| 7 | Naive vs RRF vs Bayesian | Thm 1.2.2, Def 1.3.1 | Ranking comparisons across fusion methods |
| 8 | Log-space Stability | Thm 5.3.1, Def 5.3.2 | No NaN/Inf for extreme values |
| 9 | Parameter Learning | Alg 8.3.1, Def 8.1.1 | Loss decreases; alpha > 0 |
| 10 | Conjunction/Disjunction Bounds | Thm 5.1.2, 5.2.2 | AND <= min, OR >= max, AND <= OR |

## Package Structure

```
bayesian_bm25_rs/
  Cargo.toml
  README.md
  src/
    lib.rs
    main.rs
    math_utils.rs
    tokenizer.rs
    corpus.rs
    bm25_scorer.rs
    bayesian_scorer.rs
    vector_scorer.rs
    hybrid_scorer.rs
    parameter_learner.rs
    experiments.rs
```

Module dependency flow:

```
math_utils        (no deps)
tokenizer         (no deps)
corpus            -> tokenizer
bm25_scorer       -> corpus, math_utils
bayesian_scorer   -> bm25_scorer, math_utils
vector_scorer     -> math_utils
hybrid_scorer     -> bayesian_scorer, vector_scorer, math_utils
parameter_learner -> math_utils
experiments       -> all above
main              -> experiments, corpus, tokenizer
```

## Test Corpus

- 20 documents in 4 thematic clusters (ML, IR, DB, and cross-cutting).
- Each document has an 8-dimensional, hand-crafted embedding.
- 7 queries include tokenized terms, query embeddings, and relevance judgments.

Embedding dimensions: `[ML, DL/neural, IR/search, ranking, DB, distributed, probability, vectors]`.

## Key Formulas Implemented

**BM25 (Definition 1.1.1)**

score(t, d) = IDF(t) * (k1 + 1) * tf / (k1 * norm + tf)

**Robertson-Sparck Jones IDF (Definition 3.1.1)**

IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))

**Bayesian Posterior (Theorem 4.1.3)**

P(rel | s) = (L * p) / (L * p + (1 - L)(1 - p))

where L = sigmoid(alpha * (s - beta))

**Probabilistic AND (Theorem 5.1.1)**

P(A and B) = P(A) * P(B)

**Probabilistic OR (Theorem 5.2.1)**

P(A or B) = 1 - (1 - P(A))(1 - P(B))

**Cross-Entropy Loss (Definition 8.1.1)**

L = -1/N * sum(y log p + (1 - y) log(1 - p))

where p = sigmoid(alpha * (s - beta))

## Requirements

- Rust toolchain (edition 2021)
- No external crates
