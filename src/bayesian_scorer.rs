use std::rc::Rc;

use crate::bm25_scorer::BM25Scorer;
use crate::corpus::Document;
use crate::math_utils::{clamp, safe_log, safe_prob, sigmoid};

pub struct BayesianBM25Scorer {
    bm25: Rc<BM25Scorer>,
    alpha: f64,
    beta: f64,
}

impl BayesianBM25Scorer {
    pub fn new(bm25: Rc<BM25Scorer>, alpha: f64, beta: f64) -> Self {
        Self { bm25, alpha, beta }
    }

    pub fn likelihood(&self, score: f64) -> f64 {
        sigmoid(self.alpha * (score - self.beta))
    }

    pub fn tf_prior(&self, tf: usize) -> f64 {
        0.2 + 0.7 * (tf as f64 / 10.0).min(1.0)
    }

    /// Document length normalization prior (Eq. 26).
    ///
    /// Symmetric bell curve centered at ratio=0.5:
    /// P_norm = 0.3 + 0.6 * (1 - min(1, |ratio - 0.5| * 2))
    /// Peaks at 0.9 when doc_length/avg_doc_length = 0.5,
    /// falls to 0.3 at extremes.
    pub fn norm_prior(&self, doc_length: usize, avg_doc_length: f64) -> f64 {
        if avg_doc_length < 1.0 {
            return 0.5;
        }
        let ratio = doc_length as f64 / avg_doc_length;
        0.3 + 0.6 * (1.0 - ((ratio - 0.5).abs() * 2.0).min(1.0))
    }

    pub fn composite_prior(&self, tf: usize, doc_length: usize, avg_doc_length: f64) -> f64 {
        let p_tf = self.tf_prior(tf);
        let p_norm = self.norm_prior(doc_length, avg_doc_length);
        clamp(0.7 * p_tf + 0.3 * p_norm, 0.1, 0.9)
    }

    pub fn posterior(&self, score: f64, prior: f64) -> f64 {
        let mut lik = self.likelihood(score);
        lik = safe_prob(lik);
        let prior = safe_prob(prior);
        let numerator = lik * prior;
        let denominator = numerator + (1.0 - lik) * (1.0 - prior);
        numerator / denominator
    }

    pub fn score_term(&self, term: &str, doc: &Document) -> f64 {
        let raw_score = self.bm25.score_term_standard(term, doc);
        if raw_score == 0.0 {
            return 0.0;
        }
        let tf = *doc.term_freq.get(term).unwrap_or(&0);
        let prior = self.composite_prior(tf, doc.length, self.bm25.avgdl());
        self.posterior(raw_score, prior)
    }

    pub fn score(&self, query_terms: &[String], doc: &Document) -> f64 {
        let mut log_complement_sum = 0.0;
        let mut has_match = false;

        for term in query_terms {
            let p = self.score_term(term, doc);
            if p > 0.0 {
                has_match = true;
                let p = safe_prob(p);
                log_complement_sum += safe_log(1.0 - p);
            }
        }

        if !has_match {
            return 0.0;
        }

        1.0 - log_complement_sum.exp()
    }
}
