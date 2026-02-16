# bb25 (Bayesian BM25)

bb25 is a fast, self-contained BM25 + Bayesian calibration implementation with a minimal Python API. It also includes a small reference corpus and experiment suite so you can validate the expected numerical properties.

> **Original author's implementation**: The paper author (Jaepil Jeong, Cognica) maintains the reference Python implementation at [cognica-io/bayesian-bm25](https://github.com/cognica-io/bayesian-bm25). That library focuses on production-ready score-to-probability conversion with BM25 ranking order preservation, auto parameter estimation, online learning, and log-odds conjunction for hybrid fusion. If you need a drop-in probability transform for an existing search system, use the original. bb25 is a Rust-core experimental validation that prioritizes performance and end-to-end reproducibility of the paper's claims.

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

## English Benchmark (SQuAD, 100 validation queries)

This is where BB25 shines: Bayesian Hybrid beats the classic BM25 Hybrid.

| Method               | NDCG@10       | MRR@10   | Notes                                |
| -------------------- | ------------ | -------- | ------------------------------------ |
| **WS (BB25+Dense)**  | **0.9149** | **0.8850** | **SOTA!**                |
| WS (BM25+Dense)      | 0.9051       | 0.8717   |                                      |
| RRF (BM25+Dense)     | 0.8874       | 0.8483   | RRF underperforms weighted sum       |

# Conclusion

"Bayesian BM25 (bb25) has demonstrated the potential to outperform classic BM25 in hybrid search."

On the English dataset (SQuAD), combining bb25 with Dense (BGE-M3) achieves higher performance than the BM25 + Dense baseline (+1.0%p NDCG). This suggests the probabilistic score from bb25 blends more smoothly with vector scores (less scale mismatch than a simple weighted sum).

Original paper and implementations:

- **Paper**: [Bayesian BM25: A Probabilistic Framework for Hybrid Text and Vector Search](https://www.researchgate.net/publication/400212695_Bayesian_BM25_A_Probabilistic_Framework_for_Hybrid_Text_and_Vector_Search)
- **Author's reference implementation (Python)**: [cognica-io/bayesian-bm25](https://github.com/cognica-io/bayesian-bm25)
- **This implementation (Rust + Python bindings)**: [instructkr/bb25](https://github.com/instructkr/bb25)

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
