# Benchmarks (BM25 vs Bayesian BM25)

This folder provides a lightweight benchmark runner to compare classic BM25 and Bayesian BM25 on your own corpora. It is designed to be simple and reproducible, not fast.

## File formats

### `docs.jsonl`
Each line is a JSON object with:

- `doc_id` (string)
- `text` (string)
- `embedding` (optional, list of floats; ignored by this runner)

Example:

```
{"doc_id":"d1","text":"machine learning basics"}
{"doc_id":"d2","text":"bm25 baseline retrieval"}
```

### `queries.jsonl`
Each line is a JSON object with:

- `query_id` (string)
- `text` (string)
- `terms` (optional list of pre-tokenized strings)
- `embedding` (optional list of floats; ignored by this runner)

Example:

```
{"query_id":"q1","text":"bm25 ranking"}
{"query_id":"q2","text":"bayesian calibration"}
```

### `qrels.tsv`
Whitespace-separated rows:

```
query_id  doc_id  relevance
```

Example:

```
q1 d2 1
q1 d1 0
q2 d1 1
```

You can also use `qrels.jsonl` with fields `query_id`, `doc_id`, `relevance`.

## Run

```
python benchmarks/run_benchmark.py \
  --docs docs.jsonl \
  --queries queries.jsonl \
  --qrels qrels.tsv
```

Optional parameters:

- `--bm25-k1`, `--bm25-b`
- `--alpha`, `--beta` (Bayesian calibration)
- `--cutoffs` (comma-separated list, default: 5,10,20,100)
- `--output-json results.json`

## Korean tokenization note

The built-in tokenizer is a simple "split on non-alphanumeric" rule. For Korean (or any language without whitespace tokenization), pre-tokenize the text yourself and join tokens with spaces before writing `docs.jsonl` / `queries.jsonl`. You can also pass pre-tokenized terms using the `terms` field in `queries.jsonl`.

## Expected output

Tab-separated summary (one row per scorer):

```
scorer  queries  elapsed_s  ndcg@5  map@5  mrr@5  ndcg@10  map@10  mrr@10  ...
```
