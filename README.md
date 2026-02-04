# Bayesian BM25 (Rust + Python)

Use this package to score text with BM25, convert scores into calibrated probabilities (Bayesian BM25), and fuse lexical + vector signals with proper probabilistic AND/OR. It also ships a reproducible experiment suite so you can verify the expected numerical properties.

- PyPI package: `bayesian_bm25_rs`
- Import name: `bayesian_bm25`

## Install (Python)

```
pip install bayesian_bm25_rs
```

## Quick Usage (Python)

### 1) Default corpus + queries

```
import bayesian_bm25 as bb

corpus = bb.build_default_corpus()
docs = corpus.documents()
queries = bb.build_default_queries()

bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
score = bm25.score(queries[0].terms, docs[0])
print("score0", score)
```

### 2) Your own corpus

```
import bayesian_bm25 as bb

corpus = bb.Corpus()
corpus.add_document("d1", "neural networks for ranking", [0.1] * 8)
corpus.add_document("d2", "bm25 is a strong baseline", [0.2] * 8)
corpus.build_index()  # required before creating scorers

bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
print(bm25.idf("bm25"))
```

### 3) Bayesian calibration + hybrid fusion

```
import bayesian_bm25 as bb

corpus = bb.build_default_corpus()
docs = corpus.documents()
queries = bb.build_default_queries()

bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
bayes = bb.BayesianBM25Scorer(bm25, 1.0, 0.5)
vector = bb.VectorScorer()
hybrid = bb.HybridScorer(bayes, vector)

q = queries[0]
prob_or = hybrid.score_or(q.terms, q.embedding, docs[0])
prob_and = hybrid.score_and(q.terms, q.embedding, docs[0])
print("OR", prob_or, "AND", prob_and)
```

## Run the Experiments (Python)

```
import bayesian_bm25 as bb

results = bb.run_experiments()
print(all(r.passed for r in results))
```

Each result has `name`, `passed`, and `details` fields.

## Run the Experiments (Rust)

```
cargo run --bin run_experiments
```

## Build (Rust)

```
make build
```

## Pyodide / WASM

Pyodide builds are supported. See `docs/pyodide.md` for the tested toolchain setup and wheel build flow.

## What’s Included

- **BM25** (standard + rewritten form)
- **Bayesian BM25** (sigmoid likelihood, informative priors)
- **Vector similarity** (cosine)
- **Hybrid fusion** (probabilistic AND/OR, naive sum, RRF)
- **Parameter learning** (cross-entropy gradient descent)
- **10 validation experiments** covering bounds, monotonicity, stability, and convergence

## Packaging for PyPI

Build a wheel with maturin:

```
python -m pip install maturin
maturin build --release
```

This produces platform wheels under `target/wheels/` (native) and `dist/` (Pyodide, when built via `pyodide build`).
