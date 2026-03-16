use bayesian_bm25::fusion::{log_odds_conjunction, Gating};
use bayesian_bm25::learnable_weights::LearnableLogOddsWeights;

#[test]
fn base_rate_shifts_combine_output() {
    let w_none = LearnableLogOddsWeights::new(3, 0.0, None);
    let w_low = LearnableLogOddsWeights::new(3, 0.0, Some(0.1));
    let w_high = LearnableLogOddsWeights::new(3, 0.0, Some(0.9));

    let probs = vec![0.6, 0.7, 0.8];
    let r_none = w_none.combine(&probs, false);
    let r_low = w_low.combine(&probs, false);
    let r_high = w_high.combine(&probs, false);

    assert!(r_low < r_none, "base_rate=0.1 ({}) should be < None ({})", r_low, r_none);
    assert!(r_high > r_none, "base_rate=0.9 ({}) should be > None ({})", r_high, r_none);
}

#[test]
fn base_rate_none_matches_log_odds_conjunction() {
    let w = LearnableLogOddsWeights::new(3, 0.5, None);
    let probs = vec![0.6, 0.7, 0.8];

    let result = w.combine(&probs, false);
    let expected = log_odds_conjunction(&probs, Some(0.5), Some(&w.weights()), Gating::NoGating);
    assert!(
        (result - expected).abs() < 1e-10,
        "base_rate=None result ({}) should match log_odds_conjunction ({})", result, expected
    );
}
