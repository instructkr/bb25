#!/usr/bin/env python3
"""Simple BM25 vs Bayesian BM25 benchmark runner."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import bb25 as bb


@dataclass
class DocRecord:
    doc_id: str
    text: str
    embedding: List[float]


@dataclass
class QueryRecord:
    query_id: str
    text: str
    terms: Optional[List[str]]
    embedding: Optional[List[float]]


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
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            qid, did, rel_str = parts[0], parts[1], parts[2]
            rel = float(rel_str)
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def parse_cutoffs(raw: str) -> List[int]:
    items = [int(x.strip()) for x in raw.split(",") if x.strip()]
    unique = sorted(set([x for x in items if x > 0]))
    return unique


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

    results = []
    results.append(
        evaluate(
            queries,
            doc_objs,
            "bm25",
            lambda terms, doc: bm25.score(terms, doc),
            qrels,
            tokenizer,
            cutoffs,
        )
    )
    results.append(
        evaluate(
            queries,
            doc_objs,
            "bayesian",
            lambda terms, doc: bayes.score(terms, doc),
            qrels,
            tokenizer,
            cutoffs,
        )
    )

    if has_embeddings:
        query_embedding_map = {}
        for q in queries:
            if q.embedding:
                query_embedding_map[q.query_id] = q.embedding

        def make_hybrid_fn(scorer_method):
            def fn(terms, doc):
                qid = None
                for q in queries:
                    q_terms = q.terms or tokenizer.tokenize(q.text)
                    if q_terms == list(terms):
                        qid = q.query_id
                        break
                emb = query_embedding_map.get(qid, [0.0] * len(doc.embedding))
                return scorer_method(terms, emb, doc)
            return fn

        results.append(
            evaluate(
                queries,
                doc_objs,
                "hybrid_or",
                make_hybrid_fn(hybrid_or.score_or),
                qrels,
                tokenizer,
                cutoffs,
            )
        )
        results.append(
            evaluate(
                queries,
                doc_objs,
                "hybrid_and",
                make_hybrid_fn(hybrid_and.score_and),
                qrels,
                tokenizer,
                cutoffs,
            )
        )

    table = format_table(results, cutoffs)
    print(table)

    if args.output_json:
        payload = {"cutoffs": cutoffs, "results": results}
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
