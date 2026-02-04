# bb25 (Bayesian BM25)

bb25 is a fast, self-contained BM25 + Bayesian calibration implementation with a minimal Python API. It also includes a small reference corpus and experiment suite so you can validate the expected numerical properties.

- PyPI package name: `bb25`
- Python import name: `bb25`

## Install

```
pip install bb25
```

## Quick start

### Use the built-in corpus and queries

```
import bb25 as bb

corpus = bb.build_default_corpus()
docs = corpus.documents()
queries = bb.build_default_queries()

bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
score = bm25.score(queries[0].terms, docs[0])
print("score0", score)
```

### Build your own corpus

```
import bb25 as bb

corpus = bb.Corpus()
corpus.add_document("d1", "neural networks for ranking", [0.1] * 8)
corpus.add_document("d2", "bm25 is a strong baseline", [0.2] * 8)
corpus.build_index()  # must be called before creating scorers

bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
print(bm25.idf("bm25"))
```

### Bayesian calibration + hybrid fusion

```
import bb25 as bb

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

## Run the experiments

```
import bb25 as bb

results = bb.run_experiments()
print(all(r.passed for r in results))
```

## Sample script

See `docs/sample_usage.py` for an end-to-end example using BM25, Bayesian calibration, and hybrid fusion.

## Benchmarks (BM25 vs Bayesian)

See `benchmarks/README.md` for a lightweight runner that compares BM25 and Bayesian BM25 on your own corpora.

## Build from source (Rust)

```
make build
```

## PyPI publishing

Build a wheel with maturin:

```
python -m pip install maturin
maturin build --release
```

For Pyodide builds, see `docs/pyodide.md`.
