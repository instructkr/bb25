#!/usr/bin/env python3
"""BM25 vs Bayesian BM25 benchmark runner.

Evaluates ranking quality and probability calibration across:
  1. Raw BM25 (baseline)
  2. Bayesian BM25 (fixed params)
  3. Bayesian BM25 (batch-fitted params via BayesianProbabilityTransform)
  4. Hybrid OR / AND (when embeddings available)
  5. Balanced log-odds fusion
  6. Gated log-odds fusion (relu, swish)
  7. Learned-weight log-odds fusion (LearnableLogOddsWeights)
  8. Attention-weighted log-odds fusion (AttentionLogOddsWeights)
  9. Calibration diagnostics (ECE, Brier, reliability diagram)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import bb25 as bb


@dataclass
class DocRecord:
    doc_id: str
    text: str
    embedding: List[float] = field(default_factory=list)


@dataclass
class QueryRecord:
    query_id: str
    text: str
    terms: Optional[List[str]] = None
    embedding: Optional[List[float]] = None


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield json.loads(line)


def load_docs(path: Path, id_field: str, text_field: str, embedding_field: Optional[str]) -> List[DocRecord]:
    docs: List[DocRecord] = []
    for row in load_jsonl(path):
        doc_id = str(row[id_field])
        text = str(row[text_field])
        embedding: List[float] = []
        if embedding_field and embedding_field in row and row[embedding_field] is not None:
            embedding = [float(x) for x in row[embedding_field]]
        docs.append(DocRecord(doc_id=doc_id, text=text, embedding=embedding))
    return docs


def load_queries(
    path: Path,
    id_field: str,
    text_field: str,
    terms_field: Optional[str],
    embedding_field: Optional[str],
) -> List[QueryRecord]:
    queries: List[QueryRecord] = []
    for row in load_jsonl(path):
        query_id = str(row[id_field])
        text = str(row[text_field])
        terms = None
        if terms_field and terms_field in row and row[terms_field] is not None:
            terms = [str(t) for t in row[terms_field]]
        embedding = None
        if embedding_field and embedding_field in row and row[embedding_field] is not None:
            embedding = [float(x) for x in row[embedding_field]]
        queries.append(QueryRecord(query_id=query_id, text=text, terms=terms, embedding=embedding))
    return queries


def load_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    if path.suffix == ".jsonl":
        rows = load_jsonl(path)
        for row in rows:
            qid = str(row["query_id"])
            did = str(row["doc_id"])
            rel = float(row.get("relevance", 1.0))
            qrels.setdefault(qid, {})[did] = rel
        return qrels

    with path.open("r", encoding="utf-8") as handle:
        first_line = True
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                continue
            if first_line:
                first_line = False
                try:
                    float(parts[2])
                except ValueError:
                    continue
            qid, did, rel_str = parts[0], parts[1], parts[2]
            rel = float(rel_str)
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def parse_cutoffs(raw: str) -> List[int]:
    items = [int(x.strip()) for x in raw.split(",") if x.strip()]
    unique = sorted(set([x for x in items if x > 0]))
    return unique


def encode_embeddings(
    docs: List[DocRecord],
    queries: List[QueryRecord],
    model_name: str,
    batch_size: int,
) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    doc_texts = [doc.text for doc in docs]
    print(f"Encoding {len(doc_texts)} documents...")
    doc_embs = model.encode(doc_texts, batch_size=batch_size, show_progress_bar=True)
    for i, doc in enumerate(docs):
        doc.embedding = doc_embs[i].tolist()

    query_texts = [q.text for q in queries]
    print(f"Encoding {len(query_texts)} queries...")
    query_embs = model.encode(query_texts, batch_size=batch_size, show_progress_bar=True)
    for i, q in enumerate(queries):
        q.embedding = query_embs[i].tolist()


def build_corpus(docs: List[DocRecord]) -> bb.Corpus:
    corpus = bb.Corpus(None)
    for doc in docs:
        corpus.add_document(doc.doc_id, doc.text, doc.embedding)
    corpus.build_index()
    return corpus


def rank_docs(scores: List[Tuple[str, float]]) -> List[str]:
    return [doc_id for doc_id, _ in sorted(scores, key=lambda item: (-item[1], item[0]))]


def average_precision_at_k(ranked: List[str], rel_map: Dict[str, float], k: int) -> float:
    if not rel_map:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if rel_map.get(doc_id, 0.0) > 0:
            hits += 1
            precision_sum += hits / idx
    denom = min(len([r for r in rel_map.values() if r > 0]), k)
    if denom == 0:
        return 0.0
    return precision_sum / denom


def dcg_at_k(ranked: List[str], rel_map: Dict[str, float], k: int) -> float:
    score = 0.0
    for idx, doc_id in enumerate(ranked[:k], start=1):
        rel = rel_map.get(doc_id, 0.0)
        if rel <= 0:
            continue
        score += (2 ** rel - 1) / math.log2(idx + 1)
    return score


def ndcg_at_k(ranked: List[str], rel_map: Dict[str, float], k: int) -> float:
    if not rel_map:
        return 0.0
    ideal_rels = sorted([r for r in rel_map.values() if r > 0], reverse=True)
    ideal_dcg = 0.0
    for idx, rel in enumerate(ideal_rels[:k], start=1):
        ideal_dcg += (2 ** rel - 1) / math.log2(idx + 1)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked, rel_map, k) / ideal_dcg


def mrr_at_k(ranked: List[str], rel_map: Dict[str, float], k: int) -> float:
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if rel_map.get(doc_id, 0.0) > 0:
            return 1.0 / idx
    return 0.0


def evaluate(
    queries: List[QueryRecord],
    docs: List[bb.Document],
    scorer_name: str,
    score_fn,
    qrels: Dict[str, Dict[str, float]],
    tokenizer: bb.Tokenizer,
    cutoffs: List[int],
) -> Dict[str, float]:
    metrics = {f"map@{k}": 0.0 for k in cutoffs}
    metrics.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    metrics.update({f"mrr@{k}": 0.0 for k in cutoffs})

    counted = 0
    start = time.perf_counter()
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)
        scores = [(doc.id, score_fn(terms, doc)) for doc in docs]
        ranked = rank_docs(scores)
        for k in cutoffs:
            metrics[f"map@{k}"] += average_precision_at_k(ranked, rel_map, k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(ranked, rel_map, k)
            metrics[f"mrr@{k}"] += mrr_at_k(ranked, rel_map, k)
        counted += 1

    elapsed = time.perf_counter() - start
    if counted == 0:
        return {"scorer": scorer_name, "queries": 0, "elapsed_s": elapsed}

    for key in list(metrics.keys()):
        metrics[key] /= counted
    metrics["scorer"] = scorer_name
    metrics["queries"] = counted
    metrics["elapsed_s"] = elapsed
    return metrics


def evaluate_hybrid(
    queries: List[QueryRecord],
    docs: List[bb.Document],
    scorer_name: str,
    score_fn,
    qrels: Dict[str, Dict[str, float]],
    tokenizer: bb.Tokenizer,
    cutoffs: List[int],
) -> Dict[str, float]:
    metrics = {f"map@{k}": 0.0 for k in cutoffs}
    metrics.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    metrics.update({f"mrr@{k}": 0.0 for k in cutoffs})

    counted = 0
    start = time.perf_counter()
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        if query.embedding is None:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)
        scores = [(doc.id, score_fn(terms, query.embedding, doc)) for doc in docs]
        ranked = rank_docs(scores)
        for k in cutoffs:
            metrics[f"map@{k}"] += average_precision_at_k(ranked, rel_map, k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(ranked, rel_map, k)
            metrics[f"mrr@{k}"] += mrr_at_k(ranked, rel_map, k)
        counted += 1

    elapsed = time.perf_counter() - start
    if counted == 0:
        return {"scorer": scorer_name, "queries": 0, "elapsed_s": elapsed}

    for key in list(metrics.keys()):
        metrics[key] /= counted
    metrics["scorer"] = scorer_name
    metrics["queries"] = counted
    metrics["elapsed_s"] = elapsed
    return metrics


def evaluate_balanced_fusion(
    queries: List[QueryRecord],
    docs: List[bb.Document],
    bayes: bb.BayesianBM25Scorer,
    vector: bb.VectorScorer,
    qrels: Dict[str, Dict[str, float]],
    tokenizer: bb.Tokenizer,
    cutoffs: List[int],
    weight: float = 0.5,
) -> Dict[str, float]:
    metrics = {f"map@{k}": 0.0 for k in cutoffs}
    metrics.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    metrics.update({f"mrr@{k}": 0.0 for k in cutoffs})

    doc_ids = [doc.id for doc in docs]
    counted = 0
    start = time.perf_counter()
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        if query.embedding is None:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)

        sparse_probs = [bayes.score(terms, doc) for doc in docs]
        dense_sims = [vector.score(query.embedding, doc) for doc in docs]
        fused = bb.balanced_log_odds_fusion(sparse_probs, dense_sims, weight)

        scores = list(zip(doc_ids, fused))
        ranked = rank_docs(scores)
        for k in cutoffs:
            metrics[f"map@{k}"] += average_precision_at_k(ranked, rel_map, k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(ranked, rel_map, k)
            metrics[f"mrr@{k}"] += mrr_at_k(ranked, rel_map, k)
        counted += 1

    elapsed = time.perf_counter() - start
    if counted == 0:
        return {"scorer": "balanced_fusion", "queries": 0, "elapsed_s": elapsed}

    for key in list(metrics.keys()):
        metrics[key] /= counted
    metrics["scorer"] = "balanced_fusion"
    metrics["queries"] = counted
    metrics["elapsed_s"] = elapsed
    return metrics


def evaluate_gated_fusion(
    queries: List[QueryRecord],
    docs: List[bb.Document],
    bayes: bb.BayesianBM25Scorer,
    vector: bb.VectorScorer,
    qrels: Dict[str, Dict[str, float]],
    tokenizer: bb.Tokenizer,
    cutoffs: List[int],
    gating: str,
) -> Dict[str, float]:
    metrics = {f"map@{k}": 0.0 for k in cutoffs}
    metrics.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    metrics.update({f"mrr@{k}": 0.0 for k in cutoffs})

    counted = 0
    start = time.perf_counter()
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        if query.embedding is None:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)

        doc_scores: List[Tuple[str, float]] = []
        for doc in docs:
            sparse_prob = bayes.score(terms, doc)
            dense_sim = vector.score(query.embedding, doc)
            dense_prob = bb.cosine_to_probability(dense_sim)
            fused = bb.log_odds_conjunction(
                [sparse_prob, dense_prob], gating=gating,
            )
            doc_scores.append((doc.id, fused))

        ranked = rank_docs(doc_scores)
        for k in cutoffs:
            metrics[f"map@{k}"] += average_precision_at_k(ranked, rel_map, k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(ranked, rel_map, k)
            metrics[f"mrr@{k}"] += mrr_at_k(ranked, rel_map, k)
        counted += 1

    elapsed = time.perf_counter() - start
    scorer_name = f"gated_{gating}"
    if counted == 0:
        return {"scorer": scorer_name, "queries": 0, "elapsed_s": elapsed}

    for key in list(metrics.keys()):
        metrics[key] /= counted
    metrics["scorer"] = scorer_name
    metrics["queries"] = counted
    metrics["elapsed_s"] = elapsed
    return metrics


def evaluate_learned_weights_fusion(
    queries: List[QueryRecord],
    docs: List[bb.Document],
    bayes: bb.BayesianBM25Scorer,
    vector: bb.VectorScorer,
    qrels: Dict[str, Dict[str, float]],
    tokenizer: bb.Tokenizer,
    cutoffs: List[int],
) -> Dict[str, float]:
    # Phase 1: collect training data from all queries with qrels
    train_probs: List[List[float]] = []
    train_labels: List[float] = []
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        if query.embedding is None:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)
        for doc in docs:
            did = doc.id
            if did not in rel_map:
                continue
            sparse_prob = bayes.score(terms, doc)
            dense_sim = vector.score(query.embedding, doc)
            dense_prob = bb.cosine_to_probability(dense_sim)
            train_probs.append([sparse_prob, dense_prob])
            train_labels.append(1.0 if rel_map[did] > 0 else 0.0)

    if len(train_probs) < 4:
        return {"scorer": "learned_weights", "queries": 0, "elapsed_s": 0.0}

    # Phase 2: learn weights
    learner = bb.LearnableLogOddsWeights(2)
    learner.fit(train_probs, train_labels)
    weights = learner.weights

    # Phase 3: evaluate with learned weights
    metrics = {f"map@{k}": 0.0 for k in cutoffs}
    metrics.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    metrics.update({f"mrr@{k}": 0.0 for k in cutoffs})

    counted = 0
    start = time.perf_counter()
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        if query.embedding is None:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)

        doc_scores: List[Tuple[str, float]] = []
        for doc in docs:
            sparse_prob = bayes.score(terms, doc)
            dense_sim = vector.score(query.embedding, doc)
            dense_prob = bb.cosine_to_probability(dense_sim)
            fused = bb.log_odds_conjunction(
                [sparse_prob, dense_prob], weights=weights,
            )
            doc_scores.append((doc.id, fused))

        ranked = rank_docs(doc_scores)
        for k in cutoffs:
            metrics[f"map@{k}"] += average_precision_at_k(ranked, rel_map, k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(ranked, rel_map, k)
            metrics[f"mrr@{k}"] += mrr_at_k(ranked, rel_map, k)
        counted += 1

    elapsed = time.perf_counter() - start
    if counted == 0:
        return {"scorer": "learned_weights", "queries": 0, "elapsed_s": elapsed}

    for key in list(metrics.keys()):
        metrics[key] /= counted
    metrics["scorer"] = "learned_weights"
    metrics["queries"] = counted
    metrics["elapsed_s"] = elapsed
    metrics["weights"] = weights
    return metrics


def evaluate_attention_fusion(
    queries: List[QueryRecord],
    docs: List[bb.Document],
    bayes: bb.BayesianBM25Scorer,
    vector: bb.VectorScorer,
    qrels: Dict[str, Dict[str, float]],
    tokenizer: bb.Tokenizer,
    cutoffs: List[int],
) -> Dict[str, float]:
    """Evaluate AttentionLogOddsWeights fusion.

    Uses query length as a single query feature so the attention mechanism
    can learn query-dependent signal weights.
    """
    n_signals = 2
    n_query_features = 1  # query length

    # Phase 1: collect training data
    train_probs: List[List[float]] = []
    train_labels: List[float] = []
    train_features: List[float] = []
    train_query_ids: List[int] = []
    qid_map: Dict[str, int] = {}

    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        if query.embedding is None:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)
        qlen = float(len(terms))
        if query.query_id not in qid_map:
            qid_map[query.query_id] = len(qid_map)
        qid_int = qid_map[query.query_id]

        for doc in docs:
            did = doc.id
            if did not in rel_map:
                continue
            sparse_prob = bayes.score(terms, doc)
            dense_sim = vector.score(query.embedding, doc)
            dense_prob = bb.cosine_to_probability(dense_sim)
            train_probs.append([sparse_prob, dense_prob])
            train_labels.append(1.0 if rel_map[did] > 0 else 0.0)
            train_features.append(qlen)
            train_query_ids.append(qid_int)

    if len(train_probs) < 4:
        return {"scorer": "attention", "queries": 0, "elapsed_s": 0.0}

    # Phase 2: fit attention model
    n_samples = len(train_probs)
    flat_probs = [p for pair in train_probs for p in pair]
    attn = bb.AttentionLogOddsWeights(n_signals, n_query_features)
    attn.fit(
        flat_probs, train_labels, train_features,
        n_samples,
        query_ids=train_query_ids,
    )

    # Phase 3: evaluate
    metrics = {f"map@{k}": 0.0 for k in cutoffs}
    metrics.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    metrics.update({f"mrr@{k}": 0.0 for k in cutoffs})

    counted = 0
    start = time.perf_counter()
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        if query.embedding is None:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)
        qlen = float(len(terms))

        # Score all docs: collect probs matrix for this query
        doc_ids: List[str] = []
        query_probs: List[float] = []
        for doc in docs:
            sparse_prob = bayes.score(terms, doc)
            dense_sim = vector.score(query.embedding, doc)
            dense_prob = bb.cosine_to_probability(dense_sim)
            doc_ids.append(doc.id)
            query_probs.extend([sparse_prob, dense_prob])

        n_docs_q = len(doc_ids)
        # combine(probs, m=n_rows, query_features, m_q=n_feature_rows)
        # Pass m_q=1 so the single query feature row is broadcast to all docs
        query_features = [qlen]
        fused_scores = attn.combine(
            query_probs, n_docs_q,
            query_features, 1,
            use_averaged=True,
        )

        doc_scores = list(zip(doc_ids, fused_scores))
        ranked = rank_docs(doc_scores)
        for k in cutoffs:
            metrics[f"map@{k}"] += average_precision_at_k(ranked, rel_map, k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(ranked, rel_map, k)
            metrics[f"mrr@{k}"] += mrr_at_k(ranked, rel_map, k)
        counted += 1

    elapsed = time.perf_counter() - start
    if counted == 0:
        return {"scorer": "attention", "queries": 0, "elapsed_s": elapsed}

    for key in list(metrics.keys()):
        metrics[key] /= counted
    metrics["scorer"] = "attention"
    metrics["queries"] = counted
    metrics["elapsed_s"] = elapsed
    return metrics


def evaluate_fitted_bayesian(
    queries: List[QueryRecord],
    docs: List[bb.Document],
    bm25: bb.BM25Scorer,
    qrels: Dict[str, Dict[str, float]],
    tokenizer: bb.Tokenizer,
    cutoffs: List[int],
    alpha: float,
    beta: float,
) -> Dict[str, float]:
    # Phase 1: collect (score, label) pairs for fitting
    train_scores: List[float] = []
    train_labels: List[float] = []
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)
        for doc in docs:
            did = doc.id
            if did not in rel_map:
                continue
            raw_score = bm25.score(terms, doc)
            if raw_score > 0.0:
                train_scores.append(raw_score)
                train_labels.append(1.0 if rel_map[did] > 0 else 0.0)

    if len(train_scores) < 4:
        return {"scorer": "bayesian_fitted", "queries": 0, "elapsed_s": 0.0}

    # Phase 2: fit BayesianProbabilityTransform
    transform = bb.BayesianProbabilityTransform(alpha=alpha, beta=beta)
    transform.fit(train_scores, train_labels)
    fitted_alpha = transform.alpha
    fitted_beta = transform.beta

    # Phase 3: evaluate with fitted params
    fitted_bayes = bb.BayesianBM25Scorer(bm25, fitted_alpha, fitted_beta)

    metrics = {f"map@{k}": 0.0 for k in cutoffs}
    metrics.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    metrics.update({f"mrr@{k}": 0.0 for k in cutoffs})

    all_probs: List[float] = []
    all_labels: List[float] = []
    counted = 0
    start = time.perf_counter()
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)
        scores = [(doc.id, fitted_bayes.score(terms, doc)) for doc in docs]
        ranked = rank_docs(scores)
        for k in cutoffs:
            metrics[f"map@{k}"] += average_precision_at_k(ranked, rel_map, k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(ranked, rel_map, k)
            metrics[f"mrr@{k}"] += mrr_at_k(ranked, rel_map, k)

        for did, prob in scores:
            if did in rel_map and prob > 0.0:
                all_probs.append(prob)
                all_labels.append(1.0 if rel_map[did] > 0 else 0.0)
        counted += 1

    elapsed = time.perf_counter() - start
    if counted == 0:
        return {"scorer": "bayesian_fitted", "queries": 0, "elapsed_s": elapsed}

    for key in list(metrics.keys()):
        metrics[key] /= counted
    metrics["scorer"] = "bayesian_fitted"
    metrics["queries"] = counted
    metrics["elapsed_s"] = elapsed
    metrics["fitted_alpha"] = fitted_alpha
    metrics["fitted_beta"] = fitted_beta

    if all_probs:
        metrics["ece"] = bb.expected_calibration_error(all_probs, all_labels)
        metrics["brier"] = bb.brier_score(all_probs, all_labels)

    return metrics


def collect_calibration_data(
    queries: List[QueryRecord],
    docs: List[bb.Document],
    scorer_name: str,
    score_fn,
    qrels: Dict[str, Dict[str, float]],
    tokenizer: bb.Tokenizer,
) -> Dict[str, float]:
    all_probs: List[float] = []
    all_labels: List[float] = []
    for query in queries:
        rel_map = qrels.get(query.query_id, {})
        if not rel_map:
            continue
        terms = query.terms or tokenizer.tokenize(query.text)
        for doc in docs:
            did = doc.id
            if did not in rel_map:
                continue
            prob = score_fn(terms, doc)
            if prob > 0.0:
                all_probs.append(prob)
                all_labels.append(1.0 if rel_map[did] > 0 else 0.0)

    if not all_probs:
        return {"scorer": scorer_name}

    report = bb.calibration_report(all_probs, all_labels)
    return {
        "scorer": scorer_name,
        "n_samples": report.n_samples,
        "ece": report.ece,
        "brier": report.brier,
    }


def format_table(results: List[Dict[str, float]], cutoffs: List[int]) -> str:
    headers = ["scorer", "queries", "elapsed_s"]
    for k in cutoffs:
        headers.extend([f"ndcg@{k}", f"map@{k}", f"mrr@{k}"])
    lines = ["\t".join(headers)]
    for row in results:
        values = []
        for h in headers:
            val = row.get(h, 0.0)
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("\t".join(values))
    return "\n".join(lines)


def format_calibration_table(calibration_results: List[Dict[str, float]]) -> str:
    headers = ["scorer", "n_samples", "ece", "brier"]
    lines = ["\t".join(headers)]
    for row in calibration_results:
        values = []
        for h in headers:
            val = row.get(h, "")
            if isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append("\t".join(values))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 vs Bayesian BM25 benchmark runner")
    parser.add_argument("--docs", type=Path, required=True, help="JSONL docs with doc_id + text")
    parser.add_argument("--queries", type=Path, required=True, help="JSONL queries with query_id + text")
    parser.add_argument("--qrels", type=Path, required=True, help="TSV (qid did rel) or JSONL qrels")
    parser.add_argument("--doc-id", default="doc_id")
    parser.add_argument("--doc-text", default="text")
    parser.add_argument("--doc-embedding", default=None)
    parser.add_argument("--query-id", default="query_id")
    parser.add_argument("--query-text", default="text")
    parser.add_argument("--query-terms", default=None)
    parser.add_argument("--query-embedding", default=None)
    parser.add_argument("--embedding-model", default=None,
                        help="sentence-transformers model name (e.g. all-MiniLM-L6-v2)")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--cutoffs", default="5,10,20,100")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cutoffs = parse_cutoffs(args.cutoffs)

    docs = load_docs(args.docs, args.doc_id, args.doc_text, args.doc_embedding)
    if args.max_docs:
        docs = docs[: args.max_docs]
    queries = load_queries(
        args.queries,
        args.query_id,
        args.query_text,
        args.query_terms,
        args.query_embedding,
    )
    if args.max_queries:
        queries = queries[: args.max_queries]

    if args.embedding_model:
        encode_embeddings(docs, queries, args.embedding_model, args.embedding_batch_size)

    qrels = load_qrels(args.qrels)

    corpus = build_corpus(docs)
    bm25 = bb.BM25Scorer(corpus, args.bm25_k1, args.bm25_b)
    bayes = bb.BayesianBM25Scorer(bm25, args.alpha, args.beta)

    tokenizer = bb.Tokenizer()
    doc_objs = corpus.documents()

    has_embeddings = any(
        q.embedding is not None and len(q.embedding) > 0 for q in queries
    )

    vector = bb.VectorScorer()
    hybrid_or = bb.HybridScorer(bayes, vector)
    hybrid_and = bb.HybridScorer(bayes, vector)

    # -----------------------------------------------------------------------
    # Ranking evaluation
    # -----------------------------------------------------------------------
    results = []
    results.append(
        evaluate(
            queries, doc_objs, "bm25",
            lambda terms, doc: bm25.score(terms, doc),
            qrels, tokenizer, cutoffs,
        )
    )
    results.append(
        evaluate(
            queries, doc_objs, "bayesian",
            lambda terms, doc: bayes.score(terms, doc),
            qrels, tokenizer, cutoffs,
        )
    )

    # Bayesian with batch-fitted parameters
    results.append(
        evaluate_fitted_bayesian(
            queries, doc_objs, bm25, qrels, tokenizer, cutoffs,
            args.alpha, args.beta,
        )
    )

    if has_embeddings:
        results.append(
            evaluate_hybrid(
                queries, doc_objs, "hybrid_or",
                hybrid_or.score_or, qrels, tokenizer, cutoffs,
            )
        )
        results.append(
            evaluate_hybrid(
                queries, doc_objs, "hybrid_and",
                hybrid_and.score_and, qrels, tokenizer, cutoffs,
            )
        )
        results.append(
            evaluate_balanced_fusion(
                queries, doc_objs, bayes, vector,
                qrels, tokenizer, cutoffs,
            )
        )

        # Gated fusion variants
        for gating in ("relu", "swish"):
            results.append(
                evaluate_gated_fusion(
                    queries, doc_objs, bayes, vector,
                    qrels, tokenizer, cutoffs, gating,
                )
            )

        # Learned-weight fusion
        results.append(
            evaluate_learned_weights_fusion(
                queries, doc_objs, bayes, vector,
                qrels, tokenizer, cutoffs,
            )
        )

        # Attention-weighted fusion
        results.append(
            evaluate_attention_fusion(
                queries, doc_objs, bayes, vector,
                qrels, tokenizer, cutoffs,
            )
        )

    # -----------------------------------------------------------------------
    # Ranking results table
    # -----------------------------------------------------------------------
    print("=== Ranking Metrics ===")
    table = format_table(results, cutoffs)
    print(table)

    # -----------------------------------------------------------------------
    # Calibration diagnostics
    # -----------------------------------------------------------------------
    print("\n=== Calibration Metrics ===")
    calibration_results = []
    calibration_results.append(
        collect_calibration_data(
            queries, doc_objs, "bayesian",
            lambda terms, doc: bayes.score(terms, doc),
            qrels, tokenizer,
        )
    )

    # Calibration for fitted Bayesian (if it ran)
    fitted_row = next((r for r in results if r.get("scorer") == "bayesian_fitted"), None)
    if fitted_row and "ece" in fitted_row:
        calibration_results.append({
            "scorer": "bayesian_fitted",
            "n_samples": fitted_row.get("n_samples", ""),
            "ece": fitted_row["ece"],
            "brier": fitted_row["brier"],
        })

    cal_table = format_calibration_table(calibration_results)
    print(cal_table)

    # Reliability diagram for the Bayesian scorer
    cal_data = calibration_results[0]
    if "n_samples" in cal_data and cal_data["n_samples"]:
        all_probs: List[float] = []
        all_labels: List[float] = []
        for query in queries:
            rel_map = qrels.get(query.query_id, {})
            if not rel_map:
                continue
            terms = query.terms or tokenizer.tokenize(query.text)
            for doc in doc_objs:
                did = doc.id
                if did not in rel_map:
                    continue
                prob = bayes.score(terms, doc)
                if prob > 0.0:
                    all_probs.append(prob)
                    all_labels.append(1.0 if rel_map[did] > 0 else 0.0)

        if all_probs:
            report = bb.calibration_report(all_probs, all_labels)
            print(f"\n{report.summary()}")

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    if args.output_json:
        payload = {
            "cutoffs": cutoffs,
            "results": results,
            "calibration": calibration_results,
        }
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
