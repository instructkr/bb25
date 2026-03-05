import unittest

import bb25 as bb


class TestGating(unittest.TestCase):
    """Test gating parameter in log_odds_conjunction."""

    def test_no_gating_default(self):
        result = bb.log_odds_conjunction([0.8, 0.8])
        self.assertGreater(result, 0.5)

    def test_relu_gating(self):
        # Relu zeros out negative logits (prob < 0.5)
        result_none = bb.log_odds_conjunction([0.8, 0.3], gating="none")
        result_relu = bb.log_odds_conjunction([0.8, 0.3], gating="relu")
        # With relu, the negative logit from 0.3 becomes 0, so result should be higher
        self.assertGreater(result_relu, result_none)

    def test_swish_gating(self):
        result_none = bb.log_odds_conjunction([0.8, 0.3], gating="none")
        result_swish = bb.log_odds_conjunction([0.8, 0.3], gating="swish")
        # Swish is softer than relu, should still increase relative to none
        self.assertGreater(result_swish, result_none)

    def test_gating_with_weights(self):
        probs = [0.8, 0.3]
        weights = [0.6, 0.4]
        result = bb.log_odds_conjunction(probs, weights=weights, gating="relu")
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_invalid_gating(self):
        with self.assertRaises(ValueError):
            bb.log_odds_conjunction([0.8, 0.8], gating="invalid")


class TestLearnableLogOddsWeights(unittest.TestCase):
    """Test LearnableLogOddsWeights."""

    def test_creation(self):
        w = bb.LearnableLogOddsWeights(3)
        self.assertEqual(w.n_signals, 3)
        self.assertAlmostEqual(w.alpha, 0.0)
        # Uniform initialization
        weights = w.weights
        self.assertEqual(len(weights), 3)
        for wi in weights:
            self.assertAlmostEqual(wi, 1.0 / 3, places=6)

    def test_combine(self):
        w = bb.LearnableLogOddsWeights(2)
        result = w.combine([0.8, 0.7])
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_fit(self):
        w = bb.LearnableLogOddsWeights(2)
        probs = [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
        labels = [1.0, 0.0, 1.0, 0.0]
        w.fit(probs, labels)
        # First signal should have higher weight after training
        weights = w.weights
        self.assertGreater(weights[0], weights[1])

    def test_update(self):
        w = bb.LearnableLogOddsWeights(2)
        for _ in range(20):
            w.update([[0.9, 0.1]], [1.0])
            w.update([[0.1, 0.9]], [0.0])
        # Averaged weights should be available
        avg = w.averaged_weights
        self.assertEqual(len(avg), 2)


class TestAttentionLogOddsWeights(unittest.TestCase):
    """Test AttentionLogOddsWeights."""

    def test_creation(self):
        a = bb.AttentionLogOddsWeights(2, 3)
        self.assertEqual(a.n_signals, 2)
        self.assertEqual(a.n_query_features, 3)
        self.assertAlmostEqual(a.alpha, 0.5)
        self.assertFalse(a.normalize)

    def test_combine(self):
        a = bb.AttentionLogOddsWeights(2, 3)
        probs = [0.8, 0.7]
        qf = [1.0, 0.0, 0.5]
        result = a.combine(probs, 1, qf, 1)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0], 0.0)
        self.assertLess(result[0], 1.0)

    def test_combine_batched(self):
        a = bb.AttentionLogOddsWeights(2, 3)
        # 3 candidates, 2 signals each
        probs = [0.8, 0.7, 0.6, 0.5, 0.9, 0.1]
        qf = [1.0, 0.0, 0.5]  # single query
        result = a.combine(probs, 3, qf, 1)
        self.assertEqual(len(result), 3)

    def test_fit(self):
        a = bb.AttentionLogOddsWeights(2, 2)
        # 4 samples, 2 signals each
        probs = [0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.8]
        labels = [1.0, 0.0, 1.0, 0.0]
        qf = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        a.fit(probs, labels, qf, 4)

    def test_normalize(self):
        a = bb.AttentionLogOddsWeights(2, 3, normalize=True)
        self.assertTrue(a.normalize)
        probs = [0.8, 0.7, 0.6, 0.5]
        qf = [1.0, 0.0, 0.5]
        result = a.combine(probs, 2, qf, 1)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
