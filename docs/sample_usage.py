import bb25 as bb


def main() -> None:
    corpus = bb.build_default_corpus()
    docs = corpus.documents()
    queries = bb.build_default_queries()

    bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
    bayes = bb.BayesianBM25Scorer(bm25, 1.0, 0.5)
    vector = bb.VectorScorer()
    hybrid = bb.HybridScorer(bayes, vector)

    q = queries[0]
    doc = docs[0]

    print("=== Basic Scoring ===")
    print("query:", q.text)
    print("bm25:", bm25.score(q.terms, doc))
    print("bayes:", bayes.score(q.terms, doc))
    print("or:", hybrid.score_or(q.terms, q.embedding, doc))
    print("and:", hybrid.score_and(q.terms, q.embedding, doc))

    # -----------------------------------------------------------------------
    # GELU gating and generalized Swish
    # -----------------------------------------------------------------------
    print("\n=== Gated Fusion ===")
    sparse_prob = bayes.score(q.terms, doc)
    dense_prob = bb.cosine_to_probability(vector.score(q.embedding, doc))
    probs = [sparse_prob, dense_prob]

    for gating in ("none", "relu", "swish", "gelu"):
        fused = bb.log_odds_conjunction(probs, gating=gating)
        print(f"  log_odds({gating}): {fused:.6f}")

    # Generalized Swish with custom beta
    for beta in (0.5, 1.0, 2.0, 5.0):
        fused = bb.log_odds_conjunction(probs, gating="swish", gating_beta=beta)
        print(f"  swish(beta={beta}): {fused:.6f}")

    # -----------------------------------------------------------------------
    # TemporalBayesianTransform
    # -----------------------------------------------------------------------
    print("\n=== Temporal Transform ===")
    temporal = bb.TemporalBayesianTransform(
        alpha=1.0, beta=0.0, decay_half_life=50.0,
    )
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    labels = [0.0, 0.0, 1.0, 1.0, 1.0]
    timestamps = [0, 10, 20, 30, 100]

    temporal.fit(scores, labels, timestamps=timestamps)
    print(f"  fitted alpha: {temporal.alpha:.4f}")
    print(f"  fitted beta: {temporal.beta:.4f}")

    temporal.update([3.5], [1.0])
    print(f"  after update, timestamp: {temporal.timestamp}")

    # -----------------------------------------------------------------------
    # Neural score calibration
    # -----------------------------------------------------------------------
    print("\n=== Neural Score Calibration ===")

    # Platt (sigmoid) calibration
    platt = bb.PlattCalibrator(a=1.0, b=0.0)
    cal_scores = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    cal_labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    platt.fit(cal_scores, cal_labels)
    print(f"  Platt a={platt.a:.4f}, b={platt.b:.4f}")
    calibrated = platt.calibrate(cal_scores)
    print(f"  Platt calibrated: {[f'{p:.4f}' for p in calibrated]}")

    # Isotonic calibration
    isotonic = bb.IsotonicCalibrator()
    isotonic.fit(cal_scores, cal_labels)
    calibrated_iso = isotonic.calibrate(cal_scores)
    print(f"  Isotonic calibrated: {[f'{p:.4f}' for p in calibrated_iso]}")

    # -----------------------------------------------------------------------
    # BlockMaxIndex (BMW)
    # -----------------------------------------------------------------------
    print("\n=== BlockMaxIndex ===")
    bmw = bb.BlockMaxIndex(block_size=3)
    score_matrix = [
        [1.0, 3.0, 2.0, 5.0, 4.0, 0.5],  # term 0
        [0.5, 0.2, 0.8, 0.1, 0.9, 0.3],  # term 1
    ]
    bmw.build(score_matrix)
    print(f"  block_size={bmw.block_size}, n_blocks={bmw.n_blocks}")
    for term in range(2):
        for block in range(bmw.n_blocks):
            ub = bmw.block_upper_bound(term, block)
            print(f"  term {term}, block {block}: upper_bound={ub:.4f}")

    # -----------------------------------------------------------------------
    # LearnableLogOddsWeights with base_rate
    # -----------------------------------------------------------------------
    print("\n=== LearnableLogOddsWeights with base_rate ===")
    for br in (None, 0.1, 0.5, 0.9):
        learner = bb.LearnableLogOddsWeights(2, base_rate=br)
        fused = learner.combine([0.7, 0.8])
        print(f"  base_rate={br}: combine([0.7, 0.8]) = {fused:.6f}")

    # -----------------------------------------------------------------------
    # AttentionLogOddsWeights with seed, base_rate, pruning
    # -----------------------------------------------------------------------
    print("\n=== AttentionLogOddsWeights with seed/base_rate/prune ===")
    attn = bb.AttentionLogOddsWeights(2, 1, seed=42, base_rate=0.3)
    print(f"  base_rate={attn.base_rate}")

    # Pruning
    candidate_probs = [0.01, 0.01, 0.99, 0.99, 0.5, 0.5]
    candidate_qf = [1.0, 1.0, 1.0]
    survivors, fused_probs = attn.prune(
        candidate_probs, 3, candidate_qf, 3, threshold=0.3,
    )
    print(f"  prune(threshold=0.3): survivors={survivors}, fused={[f'{p:.4f}' for p in fused_probs]}")

    # -----------------------------------------------------------------------
    # MultiHeadAttentionLogOddsWeights
    # -----------------------------------------------------------------------
    print("\n=== MultiHeadAttentionLogOddsWeights ===")
    mh = bb.MultiHeadAttentionLogOddsWeights(
        n_heads=4, n_signals=2, n_query_features=1,
    )
    print(f"  n_heads={mh.n_heads}")
    result = mh.combine([0.8, 0.9], 1, [1.0], 1)
    print(f"  combine([0.8, 0.9]): {result[0]:.6f}")


if __name__ == "__main__":
    main()
