use crate::fusion::{log_odds_conjunction, Gating};
use crate::math_utils::{logit, min_max_normalize, safe_prob, sigmoid, softmax_rows};

/// Query-dependent signal weighting via attention (Paper 2, Section 8).
///
/// Computes per-signal softmax attention weights from query features:
/// w_i(q) = softmax(W @ features + b)[i], then combines probability
/// signals via weighted log-odds conjunction.
pub struct AttentionLogOddsWeights {
    n_signals: usize,
    n_query_features: usize,
    alpha: f64,
    normalize: bool,
    // W: (n_signals, n_query_features) stored row-major
    w_matrix: Vec<f64>,
    // b: (n_signals,)
    bias: Vec<f64>,
    // Online learning state
    n_updates: usize,
    grad_w_ema: Vec<f64>,
    grad_b_ema: Vec<f64>,
    // Polyak averaging
    w_avg: Vec<f64>,
    b_avg: Vec<f64>,
    // Base rate
    base_rate: Option<f64>,
    logit_base_rate: Option<f64>,
}

impl AttentionLogOddsWeights {
    /// Create new attention weights with Xavier initialization.
    pub fn new(
        n_signals: usize,
        n_query_features: usize,
        alpha: f64,
        normalize: bool,
        seed: u64,
        base_rate: Option<f64>,
    ) -> Self {
        assert!(n_signals >= 1, "n_signals must be >= 1, got {}", n_signals);
        assert!(
            n_query_features >= 1,
            "n_query_features must be >= 1, got {}",
            n_query_features
        );
        if let Some(br) = base_rate {
            assert!(
                br > 0.0 && br < 1.0,
                "base_rate must be in (0, 1), got {}",
                br
            );
        }

        let logit_br = base_rate.map(|br| {
            let br = safe_prob(br);
            logit(br)
        });

        // Xavier-style initialization using a simple PRNG
        let scale = 1.0 / (n_query_features as f64).sqrt();
        let total = n_signals * n_query_features;
        let w_matrix = simple_normal_init(total, scale, seed);

        Self {
            n_signals,
            n_query_features,
            alpha,
            normalize,
            w_matrix: w_matrix.clone(),
            bias: vec![0.0; n_signals],
            n_updates: 0,
            grad_w_ema: vec![0.0; total],
            grad_b_ema: vec![0.0; n_signals],
            w_avg: w_matrix,
            b_avg: vec![0.0; n_signals],
            base_rate,
            logit_base_rate: logit_br,
        }
    }

    pub fn base_rate(&self) -> Option<f64> {
        self.base_rate
    }

    pub fn n_signals(&self) -> usize {
        self.n_signals
    }

    pub fn n_query_features(&self) -> usize {
        self.n_query_features
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn normalize(&self) -> bool {
        self.normalize
    }

    /// Weight matrix W of shape (n_signals, n_query_features).
    pub fn weights_matrix(&self) -> Vec<f64> {
        self.w_matrix.clone()
    }

    /// Compute softmax attention weights from query features.
    ///
    /// query_features: flat array of shape (m * n_query_features)
    /// Returns flat array of shape (m * n_signals)
    fn compute_weights(&self, query_features: &[f64], m: usize, use_averaged: bool) -> Vec<f64> {
        let n = self.n_signals;
        let nqf = self.n_query_features;
        let w = if use_averaged { &self.w_avg } else { &self.w_matrix };
        let b = if use_averaged { &self.b_avg } else { &self.bias };

        // z = query_features @ W^T + b
        // query_features: (m, nqf), W: (n, nqf), result: (m, n)
        let mut z = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut val = b[col];
                for k in 0..nqf {
                    val += query_features[row * nqf + k] * w[col * nqf + k];
                }
                z[row * n + col] = val;
            }
        }

        softmax_rows(&z, n)
    }

    /// Per-column min-max normalization on logit array (m rows, n_signals cols).
    fn normalize_logits_columns(&self, x: &mut [f64], m: usize) {
        let n = self.n_signals;
        for col in 0..n {
            let column: Vec<f64> = (0..m).map(|row| x[row * n + col]).collect();
            let normalized = min_max_normalize(&column);
            for row in 0..m {
                x[row * n + col] = normalized[row];
            }
        }
    }

    /// Combine probability signals via query-dependent weighted log-odds.
    ///
    /// probs: flat array of shape (m * n_signals) for m candidates
    /// query_features: flat array of shape (m_q * n_query_features)
    /// If m_q < m, the last query feature row is broadcast.
    pub fn combine(
        &self,
        probs: &[f64],
        m: usize,
        query_features: &[f64],
        m_q: usize,
        use_averaged: bool,
    ) -> Vec<f64> {
        let n = self.n_signals;
        let weights = self.compute_weights(query_features, m_q, use_averaged);

        let lbr = self.logit_base_rate.unwrap_or(0.0);

        if m == 1 && !self.normalize {
            if self.logit_base_rate.is_some() {
                let scale = (n as f64).powf(self.alpha);
                let w_flat: Vec<f64> = (0..n).map(|j| weights[j]).collect();
                let l_weighted: f64 = w_flat
                    .iter()
                    .zip(probs.iter())
                    .map(|(&wi, &p)| wi * logit(safe_prob(p)))
                    .sum();
                return vec![sigmoid(scale * l_weighted + lbr)];
            }
            let w_flat: Vec<f64> = (0..n).map(|j| weights[j]).collect();
            let row_probs: Vec<f64> = (0..n).map(|j| probs[j]).collect();
            return vec![log_odds_conjunction(&row_probs, Some(self.alpha), Some(&w_flat), Gating::NoGating)];
        }

        if self.normalize {
            let scale = (n as f64).powf(self.alpha);
            let mut x: Vec<f64> = probs.iter().map(|&p| logit(safe_prob(p))).collect();
            self.normalize_logits_columns(&mut x, m);

            let mut results = vec![0.0; m];
            for i in 0..m {
                let wi_row = (i).min(m_q - 1);
                let mut l_weighted = 0.0;
                for j in 0..n {
                    l_weighted += weights[wi_row * n + j] * x[i * n + j];
                }
                results[i] = sigmoid(scale * l_weighted + lbr);
            }
            return results;
        }

        // Batched: each row has its own query-dependent weights
        let mut results = vec![0.0; m];
        for i in 0..m {
            let wi_row = (i).min(m_q - 1);
            if self.logit_base_rate.is_some() {
                let scale = (n as f64).powf(self.alpha);
                let mut l_weighted = 0.0;
                for j in 0..n {
                    l_weighted += weights[wi_row * n + j] * logit(safe_prob(probs[i * n + j]));
                }
                results[i] = sigmoid(scale * l_weighted + lbr);
            } else {
                let w_slice: Vec<f64> = (0..n).map(|j| weights[wi_row * n + j]).collect();
                let row_probs: Vec<f64> = (0..n).map(|j| probs[i * n + j]).collect();
                results[i] = log_odds_conjunction(&row_probs, Some(self.alpha), Some(&w_slice), Gating::NoGating);
            }
        }
        results
    }

    /// Batch gradient descent on BCE loss to learn W and b.
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
        let n = self.n_signals;
        let nqf = self.n_query_features;
        let scale = (n as f64).powf(self.alpha);

        // Compute logits of input signals
        let mut x: Vec<f64> = probs.iter().map(|&p| logit(safe_prob(p))).collect();

        if self.normalize {
            if let Some(qids) = query_ids {
                // Per-query group normalization
                let mut unique_ids: Vec<usize> = qids.to_vec();
                unique_ids.sort_unstable();
                unique_ids.dedup();
                for &qid in &unique_ids {
                    let indices: Vec<usize> = (0..m).filter(|&i| qids[i] == qid).collect();
                    let group_m = indices.len();
                    for col in 0..n {
                        let column: Vec<f64> = indices.iter().map(|&i| x[i * n + col]).collect();
                        let normalized = min_max_normalize(&column);
                        for (idx, &i) in indices.iter().enumerate() {
                            x[i * n + col] = normalized[idx];
                        }
                    }
                    let _ = group_m;
                }
            } else {
                self.normalize_logits_columns(&mut x, m);
            }
        }

        for _ in 0..max_iterations {
            // Compute per-sample attention weights: z = qf @ W^T + b
            let mut z = vec![0.0; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut val = self.bias[col];
                    for k in 0..nqf {
                        val += query_features[row * nqf + k] * self.w_matrix[col * nqf + k];
                    }
                    z[row * n + col] = val;
                }
            }
            let w = softmax_rows(&z, n);

            // Compute predictions and gradients
            let mut grad_w = vec![0.0; n * nqf];
            let mut grad_b = vec![0.0; n];

            let lbr = self.logit_base_rate.unwrap_or(0.0);
            for i in 0..m {
                let x_bar_w: f64 = (0..n).map(|j| w[i * n + j] * x[i * n + j]).sum();
                let p = sigmoid(scale * x_bar_w + lbr);
                let error = p - labels[i];

                // grad_z_j = scale * error * w_j * (x_j - x_bar_w)
                for j in 0..n {
                    let gz = scale * error * w[i * n + j] * (x[i * n + j] - x_bar_w);
                    // dL/dW_jk = gz * qf_k
                    for k in 0..nqf {
                        grad_w[j * nqf + k] += gz * query_features[i * nqf + k];
                    }
                    grad_b[j] += gz;
                }
            }

            // Average over samples
            let m_f = m as f64;
            let old_w = self.w_matrix.clone();
            let old_b = self.bias.clone();

            for idx in 0..grad_w.len() {
                grad_w[idx] /= m_f;
                self.w_matrix[idx] -= learning_rate * grad_w[idx];
            }
            for j in 0..n {
                grad_b[j] /= m_f;
                self.bias[j] -= learning_rate * grad_b[j];
            }

            // Check convergence
            let max_change_w = old_w
                .iter()
                .zip(self.w_matrix.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let max_change_b = old_b
                .iter()
                .zip(self.bias.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            if max_change_w.max(max_change_b) < tolerance {
                break;
            }
        }

        // Reset online state
        self.n_updates = 0;
        self.grad_w_ema = vec![0.0; n * nqf];
        self.grad_b_ema = vec![0.0; n];
        self.w_avg = self.w_matrix.clone();
        self.b_avg = self.bias.clone();
    }

    /// Online SGD update from a single observation or mini-batch.
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
        let n = self.n_signals;
        let nqf = self.n_query_features;
        let scale = (n as f64).powf(self.alpha);

        let mut x: Vec<f64> = probs.iter().map(|&p| logit(safe_prob(p))).collect();

        if self.normalize && m > 1 {
            self.normalize_logits_columns(&mut x, m);
        }

        // Compute attention weights
        let mut z = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut val = self.bias[col];
                for k in 0..nqf {
                    val += query_features[row * nqf + k] * self.w_matrix[col * nqf + k];
                }
                z[row * n + col] = val;
            }
        }
        let w = softmax_rows(&z, n);

        // Compute gradients
        let mut grad_w = vec![0.0; n * nqf];
        let mut grad_b = vec![0.0; n];

        let lbr = self.logit_base_rate.unwrap_or(0.0);
        for i in 0..m {
            let x_bar_w: f64 = (0..n).map(|j| w[i * n + j] * x[i * n + j]).sum();
            let p = sigmoid(scale * x_bar_w + lbr);
            let error = p - labels[i];

            for j in 0..n {
                let gz = scale * error * w[i * n + j] * (x[i * n + j] - x_bar_w);
                for k in 0..nqf {
                    grad_w[j * nqf + k] += gz * query_features[i * nqf + k];
                }
                grad_b[j] += gz;
            }
        }

        let m_f = m as f64;
        for idx in 0..grad_w.len() {
            grad_w[idx] /= m_f;
        }
        for j in 0..n {
            grad_b[j] /= m_f;
        }

        // EMA smoothing
        for idx in 0..grad_w.len() {
            self.grad_w_ema[idx] =
                momentum * self.grad_w_ema[idx] + (1.0 - momentum) * grad_w[idx];
        }
        for j in 0..n {
            self.grad_b_ema[j] =
                momentum * self.grad_b_ema[j] + (1.0 - momentum) * grad_b[j];
        }

        // Bias correction
        self.n_updates += 1;
        let correction = 1.0 - momentum.powi(self.n_updates as i32);
        let mut corrected_w: Vec<f64> = self.grad_w_ema.iter().map(|&g| g / correction).collect();
        let mut corrected_b: Vec<f64> = self.grad_b_ema.iter().map(|&g| g / correction).collect();

        // L2 gradient clipping (joint norm)
        let grad_norm = (corrected_w.iter().map(|&g| g * g).sum::<f64>()
            + corrected_b.iter().map(|&g| g * g).sum::<f64>())
        .sqrt();
        if grad_norm > max_grad_norm {
            let clip_scale = max_grad_norm / grad_norm;
            for g in corrected_w.iter_mut() {
                *g *= clip_scale;
            }
            for g in corrected_b.iter_mut() {
                *g *= clip_scale;
            }
        }

        // Learning rate decay
        let effective_lr = learning_rate / (1.0 + self.n_updates as f64 / decay_tau);

        for idx in 0..self.w_matrix.len() {
            self.w_matrix[idx] -= effective_lr * corrected_w[idx];
        }
        for j in 0..n {
            self.bias[j] -= effective_lr * corrected_b[j];
        }

        // Polyak averaging
        for idx in 0..self.w_matrix.len() {
            self.w_avg[idx] =
                avg_decay * self.w_avg[idx] + (1.0 - avg_decay) * self.w_matrix[idx];
        }
        for j in 0..n {
            self.b_avg[j] = avg_decay * self.b_avg[j] + (1.0 - avg_decay) * self.bias[j];
        }
    }

    /// Compute upper bounds on fused probabilities using per-signal upper bound probs.
    ///
    /// upper_bound_probs: flat array of shape (m * n_signals)
    /// Returns a Vec of length m with upper bound fused probabilities.
    pub fn compute_upper_bounds(
        &self,
        upper_bound_probs: &[f64],
        m: usize,
        query_features: &[f64],
        m_q: usize,
        use_averaged: bool,
    ) -> Vec<f64> {
        let n = self.n_signals;
        let scale = (n as f64).powf(self.alpha);
        let weights = self.compute_weights(query_features, m_q, use_averaged);
        let lbr = self.logit_base_rate.unwrap_or(0.0);

        let mut results = vec![0.0; m];
        for i in 0..m {
            let wi_row = (i).min(m_q - 1);
            let mut l_weighted = 0.0;
            for j in 0..n {
                l_weighted += weights[wi_row * n + j] * logit(safe_prob(upper_bound_probs[i * n + j]));
            }
            results[i] = sigmoid(scale * l_weighted + lbr);
        }
        results
    }

    /// Prune candidates whose upper bound fused probability is below threshold.
    ///
    /// Returns (surviving_indices, fused_probabilities) for candidates that survive pruning.
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
        let n = self.n_signals;

        // Compute upper bounds to determine surviving candidates
        let ub_probs = upper_bound_probs.unwrap_or(probs);
        let upper_bounds = self.compute_upper_bounds(ub_probs, m, query_features, m_q, use_averaged);

        // Filter by threshold
        let surviving: Vec<usize> = (0..m)
            .filter(|&i| upper_bounds[i] >= threshold)
            .collect();

        // Compute actual fused probabilities for survivors
        let survivor_m = surviving.len();
        if survivor_m == 0 {
            return (Vec::new(), Vec::new());
        }

        // Gather survivor probs into a flat array
        let mut survivor_probs = vec![0.0; survivor_m * n];
        for (si, &orig_i) in surviving.iter().enumerate() {
            for j in 0..n {
                survivor_probs[si * n + j] = probs[orig_i * n + j];
            }
        }

        // Gather survivor query features
        let nqf = self.n_query_features;
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

/// Simple PRNG-based normal initialization for reproducibility.
///
/// Uses a linear congruential generator + Box-Muller transform.
fn simple_normal_init(n: usize, scale: f64, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_add(1);
    let mut result = Vec::with_capacity(n);

    let pairs = (n + 1) / 2;
    for _ in 0..pairs {
        // LCG
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-15); // avoid log(0)

        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;

        // Box-Muller
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        result.push(r * theta.cos() * scale);
        result.push(r * theta.sin() * scale);
    }

    result.truncate(n);
    result
}
