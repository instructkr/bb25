use crate::attention_weights::AttentionLogOddsWeights;
use crate::math_utils::{logit, safe_prob, sigmoid};

/// Multi-head attention over log-odds signals.
///
/// Each head independently computes query-dependent attention weights
/// over the same probability signals. The final fused result averages
/// the per-head outputs in log-odds space before applying sigmoid.
pub struct MultiHeadAttentionLogOddsWeights {
    n_heads: usize,
    heads: Vec<AttentionLogOddsWeights>,
}

impl MultiHeadAttentionLogOddsWeights {
    /// Create a new multi-head attention module.
    ///
    /// Each head is initialized with a different seed (0..n_heads-1).
    pub fn new(
        n_heads: usize,
        n_signals: usize,
        n_query_features: usize,
        alpha: f64,
        normalize: bool,
    ) -> Self {
        assert!(n_heads >= 1, "n_heads must be >= 1, got {}", n_heads);
        let heads = (0..n_heads)
            .map(|h| {
                AttentionLogOddsWeights::new(
                    n_signals,
                    n_query_features,
                    alpha,
                    normalize,
                    h as u64,
                    None,
                )
            })
            .collect();
        Self { n_heads, heads }
    }

    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    pub fn heads(&self) -> &[AttentionLogOddsWeights] {
        &self.heads
    }

    /// Combine probability signals by averaging per-head outputs in log-odds space.
    ///
    /// probs: flat array of shape (m * n_signals)
    /// query_features: flat array of shape (m_q * n_query_features)
    /// Returns Vec of length m.
    pub fn combine(
        &self,
        probs: &[f64],
        m: usize,
        query_features: &[f64],
        m_q: usize,
        use_averaged: bool,
    ) -> Vec<f64> {
        let head_results: Vec<Vec<f64>> = self
            .heads
            .iter()
            .map(|head| head.combine(probs, m, query_features, m_q, use_averaged))
            .collect();

        let mut results = vec![0.0; m];
        let n_h = self.n_heads as f64;
        for i in 0..m {
            let avg_logit: f64 = head_results
                .iter()
                .map(|hr| logit(safe_prob(hr[i])))
                .sum::<f64>()
                / n_h;
            results[i] = sigmoid(avg_logit);
        }
        results
    }

    /// Batch gradient descent to train all heads.
    pub fn fit(
        &mut self,
        probs: &[f64],
        labels: &[f64],
        query_features: &[f64],
        m: usize,
        query_ids: Option<&[usize]>,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
    ) {
        for head in self.heads.iter_mut() {
            head.fit(
                probs,
                labels,
                query_features,
                m,
                query_ids,
                learning_rate,
                max_iterations,
                tolerance,
            );
        }
    }

    /// Online SGD update for all heads.
    pub fn update(
        &mut self,
        probs: &[f64],
        labels: &[f64],
        query_features: &[f64],
        m: usize,
        learning_rate: f64,
        momentum: f64,
        decay_tau: f64,
        max_grad_norm: f64,
        avg_decay: f64,
    ) {
        for head in self.heads.iter_mut() {
            head.update(
                probs,
                labels,
                query_features,
                m,
                learning_rate,
                momentum,
                decay_tau,
                max_grad_norm,
                avg_decay,
            );
        }
    }

    /// Compute upper bounds by averaging per-head upper bound log-odds.
    pub fn compute_upper_bounds(
        &self,
        upper_bound_probs: &[f64],
        m: usize,
        query_features: &[f64],
        m_q: usize,
        use_averaged: bool,
    ) -> Vec<f64> {
        let head_ubs: Vec<Vec<f64>> = self
            .heads
            .iter()
            .map(|head| {
                head.compute_upper_bounds(upper_bound_probs, m, query_features, m_q, use_averaged)
            })
            .collect();

        let mut results = vec![0.0; m];
        let n_h = self.n_heads as f64;
        for i in 0..m {
            let avg_logit: f64 = head_ubs
                .iter()
                .map(|ub| logit(safe_prob(ub[i])))
                .sum::<f64>()
                / n_h;
            results[i] = sigmoid(avg_logit);
        }
        results
    }

    /// Prune candidates using multi-head upper bounds.
    ///
    /// Returns (surviving_indices, fused_probabilities).
    pub fn prune(
        &self,
        probs: &[f64],
        m: usize,
        query_features: &[f64],
        m_q: usize,
        threshold: f64,
        upper_bound_probs: Option<&[f64]>,
        use_averaged: bool,
    ) -> (Vec<usize>, Vec<f64>) {
        let n = if !self.heads.is_empty() {
            self.heads[0].n_signals()
        } else {
            return (Vec::new(), Vec::new());
        };
        let nqf = if !self.heads.is_empty() {
            self.heads[0].n_query_features()
        } else {
            return (Vec::new(), Vec::new());
        };

        let ub_probs = upper_bound_probs.unwrap_or(probs);
        let upper_bounds =
            self.compute_upper_bounds(ub_probs, m, query_features, m_q, use_averaged);

        let surviving: Vec<usize> = (0..m)
            .filter(|&i| upper_bounds[i] >= threshold)
            .collect();

        if surviving.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let survivor_m = surviving.len();
        let mut survivor_probs = vec![0.0; survivor_m * n];
        for (si, &orig_i) in surviving.iter().enumerate() {
            for j in 0..n {
                survivor_probs[si * n + j] = probs[orig_i * n + j];
            }
        }

        let mut survivor_qf = vec![0.0; survivor_m * nqf];
        for (si, &orig_i) in surviving.iter().enumerate() {
            let qi = orig_i.min(m_q - 1);
            for k in 0..nqf {
                survivor_qf[si * nqf + k] = query_features[qi * nqf + k];
            }
        }

        let fused = self.combine(&survivor_probs, survivor_m, &survivor_qf, survivor_m, use_averaged);
        (surviving, fused)
    }
}
