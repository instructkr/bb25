use crate::fusion::{log_odds_conjunction, Gating};
use crate::math_utils::{logit, safe_prob, sigmoid, softmax};

/// Learnable per-signal reliability weights for log-odds conjunction (Remark 5.3.2).
///
/// Learns weights that map from the Naive Bayes uniform initialization
/// (w_i = 1/n) to per-signal reliability weights via softmax parameterization.
///
/// The gradient dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)
/// is Hebbian: the product of pre-synaptic activity (signal deviation
/// from weighted mean) and post-synaptic error (prediction minus label).
pub struct LearnableLogOddsWeights {
    n_signals: usize,
    alpha: f64,
    logits: Vec<f64>,
    n_updates: usize,
    grad_logits_ema: Vec<f64>,
    weights_avg: Vec<f64>,
}

impl LearnableLogOddsWeights {
    pub fn new(n_signals: usize, alpha: f64) -> Self {
        assert!(n_signals >= 1, "n_signals must be >= 1, got {}", n_signals);
        let uniform = 1.0 / n_signals as f64;
        Self {
            n_signals,
            alpha,
            logits: vec![0.0; n_signals],
            n_updates: 0,
            grad_logits_ema: vec![0.0; n_signals],
            weights_avg: vec![uniform; n_signals],
        }
    }

    pub fn n_signals(&self) -> usize {
        self.n_signals
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Current weights: softmax of internal logits.
    pub fn weights(&self) -> Vec<f64> {
        softmax(&self.logits)
    }

    /// Polyak-averaged weights for stable inference.
    pub fn averaged_weights(&self) -> Vec<f64> {
        self.weights_avg.clone()
    }

    /// Combine probability signals via weighted log-odds conjunction.
    pub fn combine(&self, probs: &[f64], use_averaged: bool) -> f64 {
        let w = if use_averaged {
            self.weights_avg.clone()
        } else {
            self.weights()
        };
        log_odds_conjunction(probs, Some(self.alpha), Some(&w), Gating::NoGating)
    }

    /// Batch gradient descent on BCE loss to learn weights.
    pub fn fit(
        &mut self,
        probs: &[Vec<f64>],
        labels: &[f64],
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
    ) {
        let m = probs.len();
        let n = self.n_signals;
        let scale = (n as f64).powf(self.alpha);

        // Precompute log-odds of input signals
        let x: Vec<Vec<f64>> = probs
            .iter()
            .map(|row| {
                assert_eq!(row.len(), n, "probs row length {} != n_signals {}", row.len(), n);
                row.iter().map(|&p| logit(safe_prob(p))).collect()
            })
            .collect();

        for _ in 0..max_iterations {
            let w = softmax(&self.logits);

            let mut grad_logits = vec![0.0; n];

            for i in 0..m {
                // Weighted mean log-odds
                let x_bar_w: f64 = w.iter().zip(x[i].iter()).map(|(&wj, &xj)| wj * xj).sum();

                // Predicted probability
                let p = sigmoid(scale * x_bar_w);
                let error = p - labels[i];

                // Gradient for each logit z_j
                for j in 0..n {
                    grad_logits[j] += scale * error * w[j] * (x[i][j] - x_bar_w);
                }
            }

            // Average over samples
            let mut max_change = 0.0_f64;
            for j in 0..n {
                grad_logits[j] /= m as f64;
                let delta = learning_rate * grad_logits[j];
                self.logits[j] -= delta;
                max_change = max_change.max(delta.abs());
            }

            if max_change < tolerance {
                break;
            }
        }

        // Reset online state
        self.n_updates = 0;
        self.grad_logits_ema = vec![0.0; n];
        self.weights_avg = softmax(&self.logits);
    }

    /// Online SGD update from a single observation or mini-batch.
    pub fn update(
        &mut self,
        probs: &[Vec<f64>],
        labels: &[f64],
        learning_rate: f64,
        momentum: f64,
        decay_tau: f64,
        max_grad_norm: f64,
        avg_decay: f64,
    ) {
        let m = probs.len();
        let n = self.n_signals;
        let scale = (n as f64).powf(self.alpha);
        let w = softmax(&self.logits);

        let mut grad_logits = vec![0.0; n];

        for i in 0..m {
            assert_eq!(probs[i].len(), n);
            let x: Vec<f64> = probs[i].iter().map(|&p| logit(safe_prob(p))).collect();
            let x_bar_w: f64 = w.iter().zip(x.iter()).map(|(&wj, &xj)| wj * xj).sum();
            let p = sigmoid(scale * x_bar_w);
            let error = p - labels[i];

            for j in 0..n {
                grad_logits[j] += scale * error * w[j] * (x[j] - x_bar_w);
            }
        }

        // Average over mini-batch
        for j in 0..n {
            grad_logits[j] /= m as f64;
        }

        // EMA smoothing
        for j in 0..n {
            self.grad_logits_ema[j] =
                momentum * self.grad_logits_ema[j] + (1.0 - momentum) * grad_logits[j];
        }

        // Bias correction
        self.n_updates += 1;
        let correction = 1.0 - momentum.powi(self.n_updates as i32);
        let mut corrected: Vec<f64> = self.grad_logits_ema.iter().map(|&g| g / correction).collect();

        // L2 gradient clipping
        let grad_norm: f64 = corrected.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if grad_norm > max_grad_norm {
            let clip_scale = max_grad_norm / grad_norm;
            for g in corrected.iter_mut() {
                *g *= clip_scale;
            }
        }

        // Learning rate decay
        let effective_lr = learning_rate / (1.0 + self.n_updates as f64 / decay_tau);

        for j in 0..n {
            self.logits[j] -= effective_lr * corrected[j];
        }

        // Polyak averaging of weights in the simplex
        let raw_weights = softmax(&self.logits);
        for j in 0..n {
            self.weights_avg[j] = avg_decay * self.weights_avg[j] + (1.0 - avg_decay) * raw_weights[j];
        }
    }
}
