use crate::math_utils::{clamp, safe_prob, sigmoid, EPSILON};

const ALPHA_MIN: f64 = 0.01;

/// Training mode for parameter learning (C1/C2/C3 conditions).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum TrainingMode {
    /// C1: Train on sigmoid likelihood pred = sigmoid(alpha*(s-beta)).
    #[default]
    Balanced,
    /// C2: Train on full Bayesian posterior with composite prior.
    PriorAware,
    /// C3: Same training as balanced, but at inference prior=0.5.
    PriorFree,
}

/// Transforms raw BM25 scores into calibrated probabilities.
///
/// Implements sigmoid likelihood + composite prior + Bayesian posterior
/// with optional base_rate correction (two-step Bayes update).
///
/// Supports batch fitting (gradient descent) and online learning
/// (SGD with EMA, Polyak averaging, gradient clipping).
pub struct BayesianProbabilityTransform {
    pub alpha: f64,
    pub beta: f64,
    pub base_rate: Option<f64>,
    #[allow(dead_code)]
    logit_base_rate: Option<f64>,
    training_mode: TrainingMode,
    n_updates: usize,
    grad_alpha_ema: f64,
    grad_beta_ema: f64,
    alpha_avg: f64,
    beta_avg: f64,
}

impl BayesianProbabilityTransform {
    pub fn new(alpha: f64, beta: f64, base_rate: Option<f64>) -> Self {
        if let Some(br) = base_rate {
            assert!(
                br > 0.0 && br < 1.0,
                "base_rate must be in (0, 1), got {}",
                br
            );
        }
        let logit_br = base_rate.map(|br| {
            let br = clamp(br, EPSILON, 1.0 - EPSILON);
            (br / (1.0 - br)).ln()
        });
        Self {
            alpha,
            beta,
            base_rate,
            logit_base_rate: logit_br,
            training_mode: TrainingMode::Balanced,
            n_updates: 0,
            grad_alpha_ema: 0.0,
            grad_beta_ema: 0.0,
            alpha_avg: alpha,
            beta_avg: beta,
        }
    }

    /// EMA-averaged alpha for stable inference after online updates.
    pub fn averaged_alpha(&self) -> f64 {
        self.alpha_avg
    }

    /// EMA-averaged beta for stable inference after online updates.
    pub fn averaged_beta(&self) -> f64 {
        self.beta_avg
    }

    /// Current training mode.
    pub fn training_mode(&self) -> TrainingMode {
        self.training_mode
    }

    /// Sigmoid likelihood: sigma(alpha * (score - beta)).
    pub fn likelihood(&self, score: f64) -> f64 {
        sigmoid(self.alpha * (score - self.beta))
    }

    /// Term-frequency prior: 0.2 + 0.7 * min(1, tf / 10).
    pub fn tf_prior(tf: f64) -> f64 {
        0.2 + 0.7 * (tf / 10.0).min(1.0)
    }

    /// Document-length normalization prior (Eq. 26).
    ///
    /// P_norm = 0.3 + 0.6 * (1 - min(1, |doc_len_ratio - 0.5| * 2))
    pub fn norm_prior(doc_len_ratio: f64) -> f64 {
        0.3 + 0.6 * (1.0 - ((doc_len_ratio - 0.5).abs() * 2.0).min(1.0))
    }

    /// Composite prior: clamp(0.7 * P_tf + 0.3 * P_norm, 0.1, 0.9).
    pub fn composite_prior(tf: f64, doc_len_ratio: f64) -> f64 {
        let p_tf = Self::tf_prior(tf);
        let p_norm = Self::norm_prior(doc_len_ratio);
        clamp(0.7 * p_tf + 0.3 * p_norm, 0.1, 0.9)
    }

    /// Bayesian posterior via two-step Bayes update.
    ///
    /// Without base_rate: P = L*p / (L*p + (1-L)*(1-p))
    /// With base_rate: second Bayes update using base_rate as corpus-level prior.
    pub fn posterior(likelihood_val: f64, prior: f64, base_rate: Option<f64>) -> f64 {
        let l = safe_prob(likelihood_val);
        let p = safe_prob(prior);
        let numerator = l * p;
        let denominator = numerator + (1.0 - l) * (1.0 - p);
        let mut result = safe_prob(numerator / denominator);

        if let Some(br) = base_rate {
            let num_br = result * br;
            let den_br = num_br + (1.0 - result) * (1.0 - br);
            result = safe_prob(num_br / den_br);
        }

        result
    }

    /// Full pipeline: BM25 score -> calibrated probability.
    pub fn score_to_probability(
        &self,
        score: f64,
        tf: f64,
        doc_len_ratio: f64,
    ) -> f64 {
        let l_val = self.likelihood(score);

        let prior = if self.training_mode == TrainingMode::PriorFree {
            0.5
        } else {
            Self::composite_prior(tf, doc_len_ratio)
        };

        Self::posterior(l_val, prior, self.base_rate)
    }

    /// WAND upper bound for safe document pruning (Theorem 6.1.2).
    pub fn wand_upper_bound(&self, bm25_upper_bound: f64, p_max: f64) -> f64 {
        let l_max = self.likelihood(bm25_upper_bound);
        Self::posterior(l_max, p_max, self.base_rate)
    }

    /// Batch gradient descent to learn alpha and beta (Algorithm 8.3.1).
    pub fn fit(
        &mut self,
        scores: &[f64],
        labels: &[f64],
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
        mode: TrainingMode,
        tfs: Option<&[f64]>,
        doc_len_ratios: Option<&[f64]>,
    ) {
        if mode == TrainingMode::PriorAware {
            assert!(
                tfs.is_some() && doc_len_ratios.is_some(),
                "tfs and doc_len_ratios are required when mode is PriorAware"
            );
        }

        let priors: Option<Vec<f64>> = if mode == TrainingMode::PriorAware {
            let tfs = tfs.unwrap();
            let dlrs = doc_len_ratios.unwrap();
            Some(
                tfs.iter()
                    .zip(dlrs.iter())
                    .map(|(&tf, &dlr)| Self::composite_prior(tf, dlr))
                    .collect(),
            )
        } else {
            Option::None
        };

        let mut alpha = self.alpha;
        let mut beta = self.beta;
        let n = scores.len() as f64;

        for _ in 0..max_iterations {
            let (grad_alpha, grad_beta) = if mode == TrainingMode::PriorAware {
                let priors_ref = priors.as_ref().unwrap();
                compute_prior_aware_gradients(scores, labels, priors_ref, alpha, beta, n)
            } else {
                compute_balanced_gradients(scores, labels, alpha, beta, n)
            };

            let new_alpha = alpha - learning_rate * grad_alpha;
            let new_beta = beta - learning_rate * grad_beta;

            if (new_alpha - alpha).abs() < tolerance && (new_beta - beta).abs() < tolerance {
                alpha = new_alpha;
                beta = new_beta;
                break;
            }

            alpha = new_alpha;
            beta = new_beta;
        }

        self.alpha = alpha;
        self.beta = beta;
        self.training_mode = mode;
        self.n_updates = 0;
        self.grad_alpha_ema = 0.0;
        self.grad_beta_ema = 0.0;
        self.alpha_avg = alpha;
        self.beta_avg = beta;
    }

    /// Online SGD update from a single observation or mini-batch.
    pub fn update(
        &mut self,
        scores: &[f64],
        labels: &[f64],
        learning_rate: f64,
        momentum: f64,
        decay_tau: f64,
        max_grad_norm: f64,
        avg_decay: f64,
        mode: Option<TrainingMode>,
        tfs: Option<&[f64]>,
        doc_len_ratios: Option<&[f64]>,
    ) {
        let effective_mode = mode.unwrap_or(self.training_mode);
        if effective_mode == TrainingMode::PriorAware {
            assert!(
                tfs.is_some() && doc_len_ratios.is_some(),
                "tfs and doc_len_ratios are required when mode is PriorAware"
            );
        }

        let n = scores.len() as f64;

        let (grad_alpha, grad_beta) = if effective_mode == TrainingMode::PriorAware {
            let tfs = tfs.unwrap();
            let dlrs = doc_len_ratios.unwrap();
            let priors: Vec<f64> = tfs
                .iter()
                .zip(dlrs.iter())
                .map(|(&tf, &dlr)| Self::composite_prior(tf, dlr))
                .collect();
            compute_prior_aware_gradients(scores, labels, &priors, self.alpha, self.beta, n)
        } else {
            compute_balanced_gradients(scores, labels, self.alpha, self.beta, n)
        };

        if mode.is_some() {
            self.training_mode = effective_mode;
        }

        // EMA smoothing
        self.grad_alpha_ema = momentum * self.grad_alpha_ema + (1.0 - momentum) * grad_alpha;
        self.grad_beta_ema = momentum * self.grad_beta_ema + (1.0 - momentum) * grad_beta;

        // Bias correction
        self.n_updates += 1;
        let correction = 1.0 - momentum.powi(self.n_updates as i32);
        let mut corrected_alpha = self.grad_alpha_ema / correction;
        let mut corrected_beta = self.grad_beta_ema / correction;

        // Gradient clipping
        let grad_norm = (corrected_alpha * corrected_alpha + corrected_beta * corrected_beta).sqrt();
        if grad_norm > max_grad_norm {
            let scale = max_grad_norm / grad_norm;
            corrected_alpha *= scale;
            corrected_beta *= scale;
        }

        // Learning rate decay
        let effective_lr = learning_rate / (1.0 + self.n_updates as f64 / decay_tau);

        self.alpha -= effective_lr * corrected_alpha;
        self.beta -= effective_lr * corrected_beta;

        // Alpha must stay positive
        if self.alpha < ALPHA_MIN {
            self.alpha = ALPHA_MIN;
        }

        // Polyak parameter averaging
        self.alpha_avg = avg_decay * self.alpha_avg + (1.0 - avg_decay) * self.alpha;
        self.beta_avg = avg_decay * self.beta_avg + (1.0 - avg_decay) * self.beta;
    }
}

/// Compute gradients for balanced/prior_free training mode.
fn compute_balanced_gradients(
    scores: &[f64],
    labels: &[f64],
    alpha: f64,
    beta: f64,
    n: f64,
) -> (f64, f64) {
    let mut grad_alpha = 0.0;
    let mut grad_beta = 0.0;
    for (&s, &y) in scores.iter().zip(labels.iter()) {
        let l = safe_prob(sigmoid(alpha * (s - beta)));
        let error = l - y;
        grad_alpha += error * (s - beta);
        grad_beta += error * (-alpha);
    }
    (grad_alpha / n, grad_beta / n)
}

/// Compute gradients for prior_aware training mode.
fn compute_prior_aware_gradients(
    scores: &[f64],
    labels: &[f64],
    priors: &[f64],
    alpha: f64,
    beta: f64,
    n: f64,
) -> (f64, f64) {
    let mut grad_alpha = 0.0;
    let mut grad_beta = 0.0;
    for (i, (&s, &y)) in scores.iter().zip(labels.iter()).enumerate() {
        let l = safe_prob(sigmoid(alpha * (s - beta)));
        let p = priors[i];
        let denom = l * p + (1.0 - l) * (1.0 - p);
        let predicted = safe_prob(l * p / denom);

        let dp_dl = p * (1.0 - p) / (denom * denom);
        let dl_dalpha = l * (1.0 - l) * (s - beta);
        let dl_dbeta = -l * (1.0 - l) * alpha;

        let error = predicted - y;
        grad_alpha += error * dp_dl * dl_dalpha;
        grad_beta += error * dp_dl * dl_dbeta;
    }
    (grad_alpha / n, grad_beta / n)
}
