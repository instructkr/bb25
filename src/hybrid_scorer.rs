use std::rc::Rc;

use crate::bayesian_scorer::BayesianBM25Scorer;
use crate::corpus::Document;
use crate::fusion;
use crate::math_utils::EPSILON;
use crate::vector_scorer::VectorScorer;

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
        fusion::log_odds_conjunction(probs, Some(self.alpha), None, fusion::Gating::NoGating)
    }

    pub fn probabilistic_or(&self, probs: &[f64]) -> f64 {
        fusion::prob_or(probs)
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
