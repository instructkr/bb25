use crate::math_utils::{logit, safe_prob, sigmoid};

/// Maps cosine similarity [-1, 1] to probability (0, 1).
pub fn cosine_to_probability(score: f64) -> f64 {
    safe_prob((1.0 + score) / 2.0)
}

/// Probabilistic complement: P(not A) = 1 - P(A).
pub fn prob_not(prob: f64) -> f64 {
    safe_prob(1.0 - safe_prob(prob))
}

/// Probabilistic AND via product rule in log-space.
///
/// P(A1 AND A2 AND ... AND An) = product(p_i), computed as exp(sum(ln(p_i)))
/// for numerical stability.
pub fn prob_and(probs: &[f64]) -> f64 {
    let log_sum: f64 = probs.iter().map(|&p| safe_prob(p).ln()).sum();
    log_sum.exp()
}

/// Probabilistic OR via complement rule in log-space.
///
/// P(A1 OR A2 OR ... OR An) = 1 - product(1 - p_i), computed as
/// 1 - exp(sum(ln(1 - p_i))) for numerical stability.
pub fn prob_or(probs: &[f64]) -> f64 {
    let log_complement_sum: f64 = probs
        .iter()
        .map(|&p| (1.0 - safe_prob(p)).ln())
        .sum();
    1.0 - log_complement_sum.exp()
}

/// Log-odds conjunction (paper Eq. 20/23, Theorem 8.3).
///
/// Unweighted (weights=None):
///   sigmoid(mean(logit(p_i)) * n^alpha)
///   Default alpha = 0.5
///
/// Weighted (weights=Some):
///   sigmoid(n^alpha * sum(w_i * logit(p_i)))
///   Default alpha = 0.0
///   Requires: all w_i >= 0, sum(w_i) = 1.0
pub fn log_odds_conjunction(
    probs: &[f64],
    alpha: Option<f64>,
    weights: Option<&[f64]>,
) -> f64 {
    if probs.is_empty() {
        return 0.5;
    }
    let n = probs.len() as f64;

    match weights {
        None => {
            let effective_alpha = alpha.unwrap_or(0.5);
            let logit_sum: f64 = probs.iter().map(|&p| logit(safe_prob(p))).sum();
            let l_bar = logit_sum / n;
            sigmoid(l_bar * n.powf(effective_alpha))
        }
        Some(w) => {
            assert_eq!(
                w.len(),
                probs.len(),
                "weights length must match probs length"
            );
            assert!(
                w.iter().all(|&wi| wi >= 0.0),
                "all weights must be non-negative"
            );
            assert!(
                (w.iter().sum::<f64>() - 1.0).abs() < 1e-6,
                "weights must sum to 1.0"
            );

            let effective_alpha = alpha.unwrap_or(0.0);
            let weighted_logit_sum: f64 = probs
                .iter()
                .zip(w.iter())
                .map(|(&p, &wi)| wi * logit(safe_prob(p)))
                .sum();
            sigmoid(n.powf(effective_alpha) * weighted_logit_sum)
        }
    }
}

/// Balanced log-odds fusion for hybrid sparse-dense retrieval.
///
/// Converts both score vectors to logit-space, min-max normalizes each,
/// then linearly blends: weight * dense_norm + (1 - weight) * sparse_norm.
pub fn balanced_log_odds_fusion(
    sparse_probs: &[f64],
    dense_similarities: &[f64],
    weight: f64,
) -> Vec<f64> {
    let n = sparse_probs.len();
    let logit_sparse: Vec<f64> = sparse_probs
        .iter()
        .map(|&p| logit(safe_prob(p)))
        .collect();
    let logit_dense: Vec<f64> = dense_similarities
        .iter()
        .map(|&s| logit(cosine_to_probability(s)))
        .collect();

    let sparse_norm = min_max_normalize(&logit_sparse);
    let dense_norm = min_max_normalize(&logit_dense);

    (0..n)
        .map(|i| weight * dense_norm[i] + (1.0 - weight) * sparse_norm[i])
        .collect()
}

fn min_max_normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range < 1e-12 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|&v| (v - min_val) / range).collect()
}
