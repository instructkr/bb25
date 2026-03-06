import unittest

import bb25 as bb


class TestBayesianProbabilityTransform(unittest.TestCase):
    """Test BayesianProbabilityTransform."""

    def test_basic_creation(self):
        t = bb.BayesianProbabilityTransform()
        self.assertAlmostEqual(t.alpha, 1.0)
        self.assertAlmostEqual(t.beta, 0.0)
        self.assertIsNone(t.base_rate)

    def test_custom_params(self):
        t = bb.BayesianProbabilityTransform(alpha=2.0, beta=1.5, base_rate=0.1)
        self.assertAlmostEqual(t.alpha, 2.0)
        self.assertAlmostEqual(t.beta, 1.5)
        self.assertAlmostEqual(t.base_rate, 0.1)

    def test_likelihood(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        # At score=0 with beta=0: sigmoid(0) = 0.5
        self.assertAlmostEqual(t.likelihood(0.0), 0.5, places=6)
        # Positive score -> probability > 0.5
        self.assertGreater(t.likelihood(1.0), 0.5)
        # Negative score -> probability < 0.5
        self.assertLess(t.likelihood(-1.0), 0.5)

    def test_tf_prior(self):
        self.assertAlmostEqual(bb.BayesianProbabilityTransform.tf_prior(0.0), 0.2)
        self.assertAlmostEqual(bb.BayesianProbabilityTransform.tf_prior(10.0), 0.9)
        # Saturates at tf=10
        self.assertAlmostEqual(
            bb.BayesianProbabilityTransform.tf_prior(20.0), 0.9
        )

    def test_norm_prior(self):
        # Peaks at ratio=0.5
        self.assertAlmostEqual(bb.BayesianProbabilityTransform.norm_prior(0.5), 0.9)
        # Falls at extremes
        p_at_0 = bb.BayesianProbabilityTransform.norm_prior(0.0)
        self.assertAlmostEqual(p_at_0, 0.3)

    def test_composite_prior(self):
        prior = bb.BayesianProbabilityTransform.composite_prior(5.0, 0.5)
        self.assertGreaterEqual(prior, 0.1)
        self.assertLessEqual(prior, 0.9)

    def test_posterior(self):
        # High likelihood + high prior -> high posterior
        p = bb.BayesianProbabilityTransform.posterior(0.9, 0.8)
        self.assertGreater(p, 0.9)

        # With base_rate
        p_br = bb.BayesianProbabilityTransform.posterior(0.9, 0.8, base_rate=0.1)
        self.assertLess(p_br, p)

    def test_score_to_probability(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        p = t.score_to_probability(1.0, 5.0, 0.5)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_invalid_base_rate(self):
        with self.assertRaises(ValueError):
            bb.BayesianProbabilityTransform(base_rate=0.0)
        with self.assertRaises(ValueError):
            bb.BayesianProbabilityTransform(base_rate=1.0)


class TestWAND(unittest.TestCase):
    """Test WAND upper bound."""

    def test_wand_upper_bound(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        ub = t.wand_upper_bound(5.0)
        self.assertGreater(ub, 0.0)
        self.assertLessEqual(ub, 1.0)

    def test_wand_monotonic(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        ub_low = t.wand_upper_bound(1.0)
        ub_high = t.wand_upper_bound(5.0)
        self.assertGreater(ub_high, ub_low)

    def test_wand_custom_p_max(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        ub_default = t.wand_upper_bound(3.0)
        ub_low = t.wand_upper_bound(3.0, p_max=0.5)
        self.assertGreater(ub_default, ub_low)


class TestFitBalanced(unittest.TestCase):
    """Test batch fitting with balanced mode."""

    def test_fit_balanced(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = [0.5, 1.0, 2.0, 3.0, 0.1, 0.2]
        labels = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        t.fit(scores, labels, mode="balanced")
        self.assertNotAlmostEqual(t.alpha, 1.0, places=2)

    def test_fit_convergence(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = [0.1, 0.5, 1.5, 3.0, 4.0, 5.0]
        labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        t.fit(scores, labels, learning_rate=0.1, max_iterations=2000)
        # High scores should get high probability
        self.assertGreater(t.likelihood(5.0), 0.8)
        # Low scores should get low probability
        self.assertLess(t.likelihood(0.1), 0.3)


class TestPriorAware(unittest.TestCase):
    """Test prior-aware training mode."""

    def test_fit_prior_aware(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = [0.5, 1.0, 2.0, 3.0]
        labels = [0.0, 0.0, 1.0, 1.0]
        tfs = [1.0, 2.0, 5.0, 8.0]
        dlrs = [0.5, 0.7, 0.4, 0.6]
        t.fit(scores, labels, mode="prior_aware", tfs=tfs, doc_len_ratios=dlrs)


class TestOnlineUpdate(unittest.TestCase):
    """Test online SGD updates."""

    def test_update_online(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        for _ in range(10):
            t.update([2.0], [1.0])
            t.update([0.1], [0.0])
        self.assertIsNotNone(t.averaged_alpha)
        self.assertIsNotNone(t.averaged_beta)

    def test_update_moves_params(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        alpha_before = t.alpha
        for _ in range(50):
            t.update([5.0], [1.0])
            t.update([0.0], [0.0])
        # Params should have moved
        self.assertNotAlmostEqual(t.alpha, alpha_before, places=2)

    def test_averaged_params_smooth(self):
        t = bb.BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        raw_alphas = []
        avg_alphas = []
        for _ in range(100):
            t.update([3.0], [1.0])
            t.update([0.1], [0.0])
            raw_alphas.append(t.alpha)
            avg_alphas.append(t.averaged_alpha)
        # Averaged alpha should have less variance than raw alpha
        raw_diffs = [abs(raw_alphas[i+1] - raw_alphas[i]) for i in range(len(raw_alphas)-1)]
        avg_diffs = [abs(avg_alphas[i+1] - avg_alphas[i]) for i in range(len(avg_alphas)-1)]
        raw_var = sum(d * d for d in raw_diffs) / len(raw_diffs)
        avg_var = sum(d * d for d in avg_diffs) / len(avg_diffs)
        self.assertGreater(raw_var, avg_var)


if __name__ == "__main__":
    unittest.main()
