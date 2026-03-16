use bayesian_bm25::probability::{TemporalBayesianTransform, TrainingMode};

#[test]
fn temporal_transform_constructor_properties() {
    let t = TemporalBayesianTransform::new(1.5, 0.3, None, 50.0);
    assert!((t.decay_half_life() - 50.0).abs() < 1e-10);
    assert_eq!(t.timestamp(), 0);
    assert!((t.transform.alpha - 1.5).abs() < 1e-10);
    assert!((t.transform.beta - 0.3).abs() < 1e-10);
}

#[test]
fn temporal_transform_fit_without_timestamps_changes_params() {
    let mut t = TemporalBayesianTransform::new(1.0, 0.0, None, 100.0);
    let original_alpha = t.transform.alpha;
    let original_beta = t.transform.beta;

    let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0];
    t.fit(&scores, &labels, None, 0.1, 500, 1e-8, TrainingMode::Balanced, None, None);

    assert!(
        (t.transform.alpha - original_alpha).abs() > 1e-6
            || (t.transform.beta - original_beta).abs() > 1e-6,
        "Fit should change parameters"
    );
}

#[test]
fn temporal_transform_fit_with_timestamps_short_halflife() {
    let mut t_short = TemporalBayesianTransform::new(1.0, 0.0, None, 1.0);
    let mut t_long = TemporalBayesianTransform::new(1.0, 0.0, None, 10000.0);

    let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let labels = vec![1.0, 1.0, 0.0, 0.0, 1.0];
    let timestamps = vec![0, 1, 2, 3, 100];

    t_short.fit(&scores, &labels, Some(&timestamps), 0.1, 500, 1e-8, TrainingMode::Balanced, None, None);
    t_long.fit(&scores, &labels, Some(&timestamps), 0.1, 500, 1e-8, TrainingMode::Balanced, None, None);

    let diff_beta = (t_short.transform.beta - t_long.transform.beta).abs();
    let diff_alpha = (t_short.transform.alpha - t_long.transform.alpha).abs();
    assert!(
        diff_beta > 1e-3 || diff_alpha > 1e-3,
        "Short and long half-life should produce different parameters (beta diff: {}, alpha diff: {})",
        diff_beta, diff_alpha
    );
}

#[test]
fn temporal_transform_update_increments_timestamp() {
    let mut t = TemporalBayesianTransform::new(1.0, 0.0, None, 100.0);
    assert_eq!(t.timestamp(), 0);

    t.update(&[3.0], &[1.0], 0.01, 0.9, 1000.0, 1.0, 0.995, None, None, None);
    assert_eq!(t.timestamp(), 1);

    t.update(&[2.0], &[0.0], 0.01, 0.9, 1000.0, 1.0, 0.995, None, None, None);
    assert_eq!(t.timestamp(), 2);
}
