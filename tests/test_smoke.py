import unittest

import bayesian_bm25 as bb


class SmokeTests(unittest.TestCase):
    def test_run_experiments(self):
        results = bb.run_experiments()
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r.passed for r in results))

    def test_default_builders(self):
        corpus = bb.build_default_corpus()
        queries = bb.build_default_queries()
        self.assertEqual(corpus.n, 20)
        self.assertEqual(len(queries), 7)

    def test_scoring(self):
        corpus = bb.build_default_corpus()
        doc = corpus.get_document("d01")
        bm25 = bb.BM25Scorer(corpus)
        bayes = bb.BayesianBM25Scorer(bm25)
        score = bayes.score(["machine", "learning"], doc)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
