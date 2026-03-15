use bayesian_bm25::attention_weights::AttentionLogOddsWeights;

#[test]
fn seed_produces_different_initializations() {
    let w0 = AttentionLogOddsWeights::new(3, 2, 0.5, false, 0, None);
    let w42 = AttentionLogOddsWeights::new(3, 2, 0.5, false, 42, None);

    let m0 = w0.weights_matrix();
    let m42 = w42.weights_matrix();

    let any_diff = m0.iter().zip(m42.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(any_diff, "Seeds 0 and 42 should produce different weight matrices");
}

#[test]
fn compute_upper_bounds_in_unit_interval() {
    let w = AttentionLogOddsWeights::new(3, 2, 0.5, false, 0, None);
    let m = 4;
    let probs = vec![
        0.6, 0.7, 0.8,
        0.5, 0.6, 0.7,
        0.8, 0.9, 0.95,
        0.55, 0.65, 0.75,
    ];
    let qf = vec![1.0, 0.5, 0.8, 0.3, 1.2, 0.7, 0.9, 0.4];

    let ubs = w.compute_upper_bounds(&probs, m, &qf, m, false);
    assert_eq!(ubs.len(), m);
    for &ub in &ubs {
        assert!(ub > 0.0 && ub < 1.0, "Upper bound {} not in (0, 1)", ub);
    }
}

#[test]
fn prune_filters_correctly() {
    let w = AttentionLogOddsWeights::new(2, 1, 0.5, false, 0, None);
    let m = 5;
    let probs = vec![
        0.01, 0.01,
        0.99, 0.99,
        0.5, 0.5,
        0.99, 0.99,
        0.01, 0.01,
    ];
    let qf = vec![1.0; m];

    let (survivors, fused) = w.prune(&probs, m, &qf, m, 0.5, None, false);

    assert!(survivors.contains(&1), "High-probability candidate 1 should survive");
    assert!(survivors.contains(&3), "High-probability candidate 3 should survive");
    assert!(!survivors.contains(&0), "Low-probability candidate 0 should be pruned");
    assert!(!survivors.contains(&4), "Low-probability candidate 4 should be pruned");
    assert_eq!(survivors.len(), fused.len());
}
