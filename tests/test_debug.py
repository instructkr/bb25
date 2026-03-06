import unittest

import bb25 as bb


class TestFusionDebuggerCreation(unittest.TestCase):
    def test_default(self):
        d = bb.FusionDebugger()
        self.assertIsNotNone(d)

    def test_with_params(self):
        d = bb.FusionDebugger(alpha=2.0, beta=1.0, base_rate=0.1)
        self.assertIsNotNone(d)


class TestTraceBM25(unittest.TestCase):
    def test_basic_trace(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        trace = d.trace_bm25(2.0, 5.0, 0.5)
        self.assertAlmostEqual(trace.raw_score, 2.0)
        self.assertAlmostEqual(trace.tf, 5.0)
        self.assertAlmostEqual(trace.doc_len_ratio, 0.5)
        self.assertGreater(trace.likelihood, 0.5)
        self.assertGreater(trace.posterior, 0.0)
        self.assertLess(trace.posterior, 1.0)
        self.assertAlmostEqual(trace.alpha, 1.0)
        self.assertAlmostEqual(trace.beta, 0.5)

    def test_prior_values(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        trace = d.trace_bm25(2.0, 5.0, 0.5)
        # tf_prior for tf=5: 0.2 + 0.7 * 0.5 = 0.55
        self.assertAlmostEqual(trace.tf_prior, 0.55)
        # norm_prior at ratio=0.5 peaks at 0.9
        self.assertAlmostEqual(trace.norm_prior, 0.9)

    def test_base_rate_trace(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5, base_rate=0.1)
        trace = d.trace_bm25(2.0, 5.0, 0.5)
        self.assertAlmostEqual(trace.base_rate, 0.1)
        self.assertIsNotNone(trace.logit_base_rate)


class TestTraceVector(unittest.TestCase):
    def test_basic(self):
        d = bb.FusionDebugger()
        trace = d.trace_vector(0.8)
        self.assertAlmostEqual(trace.cosine_score, 0.8)
        self.assertAlmostEqual(trace.probability, 0.9)
        self.assertGreater(trace.logit_probability, 0.0)

    def test_negative_cosine(self):
        d = bb.FusionDebugger()
        trace = d.trace_vector(-0.5)
        self.assertLess(trace.probability, 0.5)


class TestTraceNot(unittest.TestCase):
    def test_complement(self):
        d = bb.FusionDebugger()
        trace = d.trace_not(0.8, "BM25")
        self.assertAlmostEqual(trace.input_probability, 0.8)
        self.assertEqual(trace.input_name, "BM25")
        self.assertAlmostEqual(trace.complement, 0.2, places=5)

    def test_logit_sign_flip(self):
        d = bb.FusionDebugger()
        trace = d.trace_not(0.8, "signal")
        self.assertAlmostEqual(trace.logit_input, -trace.logit_complement, places=5)


class TestTraceFusion(unittest.TestCase):
    def test_log_odds(self):
        d = bb.FusionDebugger()
        trace = d.trace_fusion([0.8, 0.7])
        self.assertEqual(trace.method, "log_odds")
        self.assertIsNotNone(trace.logits)
        self.assertIsNotNone(trace.mean_logit)
        self.assertIsNotNone(trace.n_alpha_scale)
        self.assertGreater(trace.fused_probability, 0.0)

    def test_prob_and(self):
        d = bb.FusionDebugger()
        trace = d.trace_fusion([0.8, 0.7], method="prob_and")
        self.assertEqual(trace.method, "prob_and")
        self.assertIsNotNone(trace.log_probs)
        self.assertAlmostEqual(trace.fused_probability, 0.8 * 0.7, places=5)

    def test_prob_or(self):
        d = bb.FusionDebugger()
        trace = d.trace_fusion([0.8, 0.7], method="prob_or")
        self.assertEqual(trace.method, "prob_or")
        self.assertIsNotNone(trace.complements)
        expected = 1.0 - (1.0 - 0.8) * (1.0 - 0.7)
        self.assertAlmostEqual(trace.fused_probability, expected, places=5)

    def test_prob_not(self):
        d = bb.FusionDebugger()
        trace = d.trace_fusion([0.8, 0.7], method="prob_not")
        self.assertEqual(trace.method, "prob_not")
        expected = (1.0 - 0.8) * (1.0 - 0.7)
        self.assertAlmostEqual(trace.fused_probability, expected, places=5)

    def test_weighted_log_odds(self):
        d = bb.FusionDebugger()
        trace = d.trace_fusion([0.8, 0.7], weights=[0.6, 0.4])
        self.assertEqual(trace.method, "log_odds")
        self.assertIsNotNone(trace.weights)
        self.assertEqual(len(trace.weights), 2)

    def test_custom_names(self):
        d = bb.FusionDebugger()
        trace = d.trace_fusion([0.8, 0.7], names=["sparse", "dense"])
        self.assertEqual(trace.signal_names, ["sparse", "dense"])


class TestTraceDocument(unittest.TestCase):
    def test_bm25_only(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        trace = d.trace_document(bm25_score=2.0, tf=5.0, doc_len_ratio=0.5, doc_id="d01")
        self.assertEqual(trace.doc_id, "d01")
        self.assertGreater(trace.final_probability, 0.0)

    def test_vector_only(self):
        d = bb.FusionDebugger()
        trace = d.trace_document(cosine_score=0.8, doc_id="d02")
        self.assertEqual(trace.doc_id, "d02")
        self.assertGreater(trace.final_probability, 0.0)

    def test_hybrid(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        trace = d.trace_document(
            bm25_score=2.0, tf=5.0, doc_len_ratio=0.5,
            cosine_score=0.8, doc_id="d01"
        )
        self.assertGreater(trace.final_probability, 0.0)

    def test_no_signals_raises(self):
        d = bb.FusionDebugger()
        with self.assertRaises(ValueError):
            d.trace_document()


class TestCompare(unittest.TestCase):
    def test_basic_comparison(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        ta = d.trace_document(bm25_score=3.0, tf=5.0, doc_len_ratio=0.5, doc_id="d01")
        tb = d.trace_document(bm25_score=1.0, tf=2.0, doc_len_ratio=0.8, doc_id="d02")
        cmp = d.compare(ta, tb)
        self.assertIsNotNone(cmp.dominant_signal)
        self.assertGreater(len(cmp.signal_deltas), 0)

    def test_crossover_detection(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        # d01: strong BM25, weak vector
        ta = d.trace_document(
            bm25_score=5.0, tf=8.0, doc_len_ratio=0.5,
            cosine_score=0.1, doc_id="d01"
        )
        # d02: weak BM25, strong vector
        tb = d.trace_document(
            bm25_score=0.5, tf=1.0, doc_len_ratio=0.8,
            cosine_score=0.95, doc_id="d02"
        )
        cmp = d.compare(ta, tb)
        # Should detect crossover between signals
        self.assertIsNotNone(cmp.crossover_stage)


class TestFormatting(unittest.TestCase):
    def test_format_trace(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        trace = d.trace_document(bm25_score=2.0, tf=5.0, doc_len_ratio=0.5, doc_id="d01")
        text = d.format_trace(trace)
        self.assertIn("Document: d01", text)
        self.assertIn("BM25", text)
        self.assertIn("final=", text)

    def test_format_trace_non_verbose(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        trace = d.trace_document(bm25_score=2.0, tf=5.0, doc_len_ratio=0.5, doc_id="d01")
        text = d.format_trace(trace, verbose=False)
        self.assertIn("Document: d01", text)
        # Non-verbose should not have logit details
        self.assertNotIn("logit(posterior)", text)

    def test_format_summary(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        trace = d.trace_document(bm25_score=2.0, tf=5.0, doc_len_ratio=0.5, doc_id="d01")
        summary = d.format_summary(trace)
        self.assertIn("d01", summary)
        self.assertIn("Fused=", summary)

    def test_format_comparison(self):
        d = bb.FusionDebugger(alpha=1.0, beta=0.5)
        ta = d.trace_document(bm25_score=3.0, tf=5.0, doc_len_ratio=0.5, doc_id="d01")
        tb = d.trace_document(bm25_score=1.0, tf=2.0, doc_len_ratio=0.8, doc_id="d02")
        cmp = d.compare(ta, tb)
        text = d.format_comparison(cmp)
        self.assertIn("Comparison:", text)
        self.assertIn("Dominant signal:", text)
        self.assertIn("Rank order:", text)


if __name__ == "__main__":
    unittest.main()
