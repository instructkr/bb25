import unittest

import bb25 as bb


class TestBrierScore(unittest.TestCase):
    def test_perfect_predictions(self):
        probs = [1.0, 0.0, 1.0, 0.0]
        labels = [1.0, 0.0, 1.0, 0.0]
        self.assertAlmostEqual(bb.brier_score(probs, labels), 0.0)

    def test_worst_predictions(self):
        probs = [0.0, 1.0]
        labels = [1.0, 0.0]
        self.assertAlmostEqual(bb.brier_score(probs, labels), 1.0)

    def test_constant_prediction(self):
        probs = [0.5, 0.5, 0.5, 0.5]
        labels = [1.0, 0.0, 1.0, 0.0]
        self.assertAlmostEqual(bb.brier_score(probs, labels), 0.25)


class TestECE(unittest.TestCase):
    def test_perfect_calibration(self):
        probs = [0.0, 1.0]
        labels = [0.0, 1.0]
        ece = bb.expected_calibration_error(probs, labels)
        self.assertAlmostEqual(ece, 0.0, places=5)

    def test_ece_non_negative(self):
        probs = [0.1, 0.4, 0.6, 0.9]
        labels = [0.0, 0.0, 1.0, 1.0]
        ece = bb.expected_calibration_error(probs, labels)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


class TestReliabilityDiagram(unittest.TestCase):
    def test_basic(self):
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        bins = bb.reliability_diagram(probs, labels, 5)
        self.assertGreater(len(bins), 0)
        for avg_pred, avg_actual, count in bins:
            self.assertGreaterEqual(count, 1)
            self.assertGreaterEqual(avg_pred, 0.0)
            self.assertLessEqual(avg_pred, 1.0)

    def test_custom_bins(self):
        probs = [0.1, 0.9]
        labels = [0.0, 1.0]
        bins = bb.reliability_diagram(probs, labels, 2)
        self.assertEqual(len(bins), 2)


class TestCalibrationReport(unittest.TestCase):
    def test_report_fields(self):
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        report = bb.calibration_report(probs, labels)
        self.assertEqual(report.n_samples, 6)
        self.assertEqual(report.n_bins, 10)
        self.assertGreaterEqual(report.ece, 0.0)
        self.assertGreaterEqual(report.brier, 0.0)

    def test_summary_format(self):
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        report = bb.calibration_report(probs, labels)
        summary = report.summary()
        self.assertIn("Calibration Report", summary)
        self.assertIn("ECE", summary)
        self.assertIn("Brier", summary)
        self.assertIn("Reliability Diagram", summary)


if __name__ == "__main__":
    unittest.main()
