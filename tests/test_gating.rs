use bayesian_bm25::fusion::{log_odds_conjunction, Gating};

#[test]
fn gelu_gating_produces_values_in_open_unit_interval() {
    let probs = vec![0.7, 0.8, 0.9];
    let result = log_odds_conjunction(&probs, Some(0.5), None, Gating::Gelu);
    assert!(result > 0.0 && result < 1.0, "GELU result {} not in (0, 1)", result);

    let probs2 = vec![0.3, 0.6, 0.9];
    let result2 = log_odds_conjunction(&probs2, Some(0.5), None, Gating::Gelu);
    assert!(result2 > 0.0 && result2 < 1.0, "GELU result {} not in (0, 1)", result2);
}

#[test]
fn gelu_matches_generalized_swish_1702() {
    let probs = vec![0.6, 0.75, 0.85, 0.55];
    let gelu_result = log_odds_conjunction(&probs, Some(0.5), None, Gating::Gelu);
    let gs_result = log_odds_conjunction(&probs, Some(0.5), None, Gating::GeneralizedSwish(1.702));
    assert!(
        (gelu_result - gs_result).abs() < 1e-10,
        "GELU ({}) and GeneralizedSwish(1.702) ({}) differ by more than 1e-10",
        gelu_result, gs_result
    );
}

#[test]
fn generalized_swish_different_betas_produce_different_results() {
    let probs = vec![0.7, 0.8, 0.65];
    let r1 = log_odds_conjunction(&probs, Some(0.5), None, Gating::GeneralizedSwish(0.5));
    let r2 = log_odds_conjunction(&probs, Some(0.5), None, Gating::GeneralizedSwish(2.0));
    let r3 = log_odds_conjunction(&probs, Some(0.5), None, Gating::GeneralizedSwish(5.0));
    assert!((r1 - r2).abs() > 1e-6, "beta=0.5 and beta=2.0 should differ");
    assert!((r2 - r3).abs() > 1e-6, "beta=2.0 and beta=5.0 should differ");
}

#[test]
fn gelu_ordering_for_agreeing_signals() {
    let probs = vec![0.8, 0.85, 0.9];
    let swish = log_odds_conjunction(&probs, Some(0.5), None, Gating::Swish);
    let gelu = log_odds_conjunction(&probs, Some(0.5), None, Gating::Gelu);
    let relu = log_odds_conjunction(&probs, Some(0.5), None, Gating::Relu);
    assert!(swish < gelu, "Expected swish ({}) < gelu ({})", swish, gelu);
    assert!(gelu < relu, "Expected gelu ({}) < relu ({})", gelu, relu);
}
