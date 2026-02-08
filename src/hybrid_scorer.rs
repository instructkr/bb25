use std::rc::Rc;

use crate::bayesian_scorer::BayesianBM25Scorer;
use crate::math_utils::{logit, safe_log, safe_prob, sigmoid, EPSILON};
use crate::vector_scorer::VectorScorer;
use crate::corpus::Document;

pub struct HybridScorer {
    bayesian: Rc<BayesianBM25Scorer>,
    vector: Rc<VectorScorer>,
    alpha: f64,
}

impl HybridScorer {
    pub fn new(bayesian: Rc<BayesianBM25Scorer>, vector: Rc<VectorScorer>, alpha: f64) -> Self {
        Self { bayesian, vector, alpha }
    }

    pub fn probabilistic_and(&self, probs: &[f64]) -> f64 {
        if probs.is_empty() {
            return 0.0;
        }
        let n = probs.len();
        if n == 1 {
            return safe_prob(probs[0]);
        }

        // Stage 1: Geometric mean in log-space
        let mut log_sum = 0.0;
        for p in probs {
            let p = safe_prob(*p);
            log_sum += safe_log(p);
        }
        let geo_mean = (log_sum / n as f64).exp();

        // Stage 2: Log-odds transformation with agreement bonus
        let l_adjusted = logit(geo_mean) + self.alpha * (n as f64).ln();

        // Stage 3: Return to probability space
        sigmoid(l_adjusted)
    }

    pub fn probabilistic_or(&self, probs: &[f64]) -> f64 {
        let mut log_complement_sum = 0.0;
        for p in probs {
            let p = safe_prob(*p);
            log_complement_sum += safe_log(1.0 - p);
        }
        1.0 - log_complement_sum.exp()
    }

    pub fn score_and(
        &self,
        query_terms: &[String],
        query_embedding: &[f64],
        doc: &Document,
    ) -> f64 {
        let bayesian_prob = self.bayesian.score(query_terms, doc);
        let vector_prob = self.vector.score(query_embedding, doc);
        if bayesian_prob < EPSILON && vector_prob < EPSILON {
            return 0.0;
        }
        self.probabilistic_and(&[bayesian_prob, vector_prob])
    }

    pub fn score_or(
        &self,
        query_terms: &[String],
        query_embedding: &[f64],
        doc: &Document,
    ) -> f64 {
        let bayesian_prob = self.bayesian.score(query_terms, doc);
        let vector_prob = self.vector.score(query_embedding, doc);
        self.probabilistic_or(&[bayesian_prob, vector_prob])
    }

    pub fn naive_sum(&self, scores: &[f64]) -> f64 {
        scores.iter().sum()
    }

    pub fn rrf_score(&self, ranks: &[usize], k: usize) -> f64 {
        ranks
            .iter()
            .map(|rank| 1.0 / (k as f64 + *rank as f64))
            .sum()
    }
}
