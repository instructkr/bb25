use std::rc::Rc;

use crate::bm25_scorer::BM25Scorer;
use crate::corpus::Document;
use crate::fusion;
use crate::math_utils::{clamp, safe_prob, sigmoid};

pub struct BayesianBM25Scorer {
    bm25: Rc<BM25Scorer>,
    alpha: f64,
    beta: f64,
    base_rate: Option<f64>,
}

impl BayesianBM25Scorer {
    pub fn new(bm25: Rc<BM25Scorer>, alpha: f64, beta: f64, base_rate: Option<f64>) -> Self {
        if let Some(br) = base_rate {
            assert!(
                br > 0.0 && br < 1.0,
                "base_rate must be in (0, 1), got {}",
                br
            );
        }
        Self { bm25, alpha, beta, base_rate }
    }

    pub fn base_rate(&self) -> Option<f64> {
        self.base_rate
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

    /// Two-step Bayesian posterior update (Remark 4.4.5).
    ///
    /// Step 1: Standard Bayes update with likelihood and prior.
    /// Step 2 (if base_rate is set): Second Bayes update using base_rate as
    /// a corpus-level prior, adjusting the posterior toward the base rate.
    pub fn posterior(&self, score: f64, prior: f64) -> f64 {
        let lik = safe_prob(self.likelihood(score));
        let prior = safe_prob(prior);

        // Step 1: standard Bayes update
        let numerator = lik * prior;
        let denominator = numerator + (1.0 - lik) * (1.0 - prior);
        let p1 = numerator / denominator;

        // Step 2: base rate adjustment
        match self.base_rate {
            Some(br) => {
                let num2 = p1 * br;
                let den2 = num2 + (1.0 - p1) * (1.0 - br);
                num2 / den2
            }
            None => p1,
        }
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
        let posteriors: Vec<f64> = query_terms
            .iter()
            .map(|term| self.score_term(term, doc))
            .filter(|&p| p > 0.0)
            .collect();

        if posteriors.is_empty() {
            return 0.0;
        }

        fusion::prob_or(&posteriors)
    }
}
