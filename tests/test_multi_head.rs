use bayesian_bm25::attention_weights::AttentionLogOddsWeights;
use bayesian_bm25::multi_head_attention::MultiHeadAttentionLogOddsWeights;
use bayesian_bm25::safe_prob;

#[test]
fn single_head_matches_attention_seed_0() {
    let mh = MultiHeadAttentionLogOddsWeights::new(1, 3, 2, 0.5, false);
    let single = AttentionLogOddsWeights::new(3, 2, 0.5, false, 0, None);

    let probs = vec![0.6, 0.7, 0.8];
    let qf = vec![1.0, 0.5];

    let mh_result = mh.combine(&probs, 1, &qf, 1, false);
    let single_result = single.combine(&probs, 1, &qf, 1, false);

    assert!(
        (mh_result[0] - single_result[0]).abs() < 1e-10,
        "Single head MH ({}) should match seed=0 ({})", mh_result[0], single_result[0]
    );
}

#[test]
fn batched_combine_correct_shape() {
    let mh = MultiHeadAttentionLogOddsWeights::new(4, 3, 2, 0.5, false);
    let m = 5;
    let probs = vec![0.6; m * 3];
    let qf = vec![1.0; m * 2];

    let results = mh.combine(&probs, m, &qf, m, false);
    assert_eq!(results.len(), m);
    for &r in &results {
        assert!(r > 0.0 && r < 1.0, "Result {} not in (0, 1)", r);
    }
}

#[test]
fn fit_reduces_bce_loss() {
    let mut mh = MultiHeadAttentionLogOddsWeights::new(2, 3, 2, 0.5, false);

    let m = 10;
    let n_signals = 3;
    let n_qf = 2;

    let mut probs = vec![0.0; m * n_signals];
    let mut labels = vec![0.0; m];
    let mut qf = vec![0.0; m * n_qf];

    for i in 0..m {
        let is_positive = i >= m / 2;
        labels[i] = if is_positive { 1.0 } else { 0.0 };
        qf[i * n_qf] = if is_positive { 1.0 } else { 0.0 };
        qf[i * n_qf + 1] = 0.5;
        for j in 0..n_signals {
            probs[i * n_signals + j] = if is_positive { 0.7 + 0.05 * j as f64 } else { 0.2 + 0.05 * j as f64 };
        }
    }

    let initial_preds = mh.combine(&probs, m, &qf, m, false);
    let initial_loss: f64 = initial_preds.iter().zip(labels.iter())
        .map(|(&p, &y)| { let p = safe_prob(p); -(y * p.ln() + (1.0 - y) * (1.0 - p).ln()) })
        .sum::<f64>() / m as f64;

    mh.fit(&probs, &labels, &qf, m, None, 0.1, 200, 1e-8);

    let final_preds = mh.combine(&probs, m, &qf, m, false);
    let final_loss: f64 = final_preds.iter().zip(labels.iter())
        .map(|(&p, &y)| { let p = safe_prob(p); -(y * p.ln() + (1.0 - y) * (1.0 - p).ln()) })
        .sum::<f64>() / m as f64;

    assert!(final_loss < initial_loss, "Fit should reduce BCE loss: initial={}, final={}", initial_loss, final_loss);
}

#[test]
fn compute_upper_bounds_in_unit_interval() {
    let mh = MultiHeadAttentionLogOddsWeights::new(3, 2, 2, 0.5, false);
    let m = 4;
    let probs = vec![0.8, 0.9, 0.7, 0.85, 0.6, 0.75, 0.95, 0.88];
    let qf = vec![1.0, 0.5, 0.8, 0.3, 1.2, 0.7, 0.9, 0.4];

    let ubs = mh.compute_upper_bounds(&probs, m, &qf, m, false);
    assert_eq!(ubs.len(), m);
    for &ub in &ubs {
        assert!(ub > 0.0 && ub < 1.0, "Upper bound {} not in (0, 1)", ub);
    }
}
