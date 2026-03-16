use bayesian_bm25::calibration::{IsotonicCalibrator, PlattCalibrator};
use bayesian_bm25::sigmoid;

#[test]
fn platt_calibrator_default_params() {
    let cal = PlattCalibrator::new(1.0, 0.0);
    assert!((cal.a - 1.0).abs() < 1e-10);
    assert!((cal.b - 0.0).abs() < 1e-10);
}

#[test]
fn platt_calibrator_calibrate_zero_is_sigmoid_zero() {
    let cal = PlattCalibrator::new(1.0, 0.0);
    let result = cal.calibrate(0.0);
    let expected = sigmoid(0.0);
    assert!((result - expected).abs() < 1e-10, "calibrate(0.0) = {} but sigmoid(0.0) = {}", result, expected);
}

#[test]
fn platt_calibrator_fit_changes_params() {
    let mut cal = PlattCalibrator::new(1.0, 0.0);
    let scores = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let labels = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    cal.fit(&scores, &labels, 0.1, 1000, 1e-8);
    assert!((cal.a - 1.0).abs() > 1e-4 || (cal.b - 0.0).abs() > 1e-4, "Fit should change Platt parameters");
}

#[test]
fn platt_calibrator_monotonicity() {
    let mut cal = PlattCalibrator::new(1.0, 0.0);
    let scores = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    cal.fit(&scores, &labels, 0.1, 1000, 1e-8);

    let calibrated = cal.calibrate_batch(&scores);
    for i in 1..calibrated.len() {
        assert!(
            calibrated[i] >= calibrated[i - 1] - 1e-10,
            "Platt calibration should be monotonic at index {}", i
        );
    }
}

#[test]
fn isotonic_calibrator_fit_produces_monotonic_output() {
    let mut cal = IsotonicCalibrator::new();
    let scores = vec![0.1, 0.5, 0.3, 0.8, 0.6, 0.9, 0.2, 0.7, 0.4, 1.0];
    let labels = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    cal.fit(&scores, &labels);

    let test_scores: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
    let calibrated = cal.calibrate_batch(&test_scores);
    for i in 1..calibrated.len() {
        assert!(
            calibrated[i] >= calibrated[i - 1] - 1e-10,
            "Isotonic calibration should be monotonic at index {}", i
        );
    }
}

#[test]
#[should_panic(expected = "has not been fitted")]
fn isotonic_calibrator_calibrate_before_fit_panics() {
    let cal = IsotonicCalibrator::new();
    let _ = cal.calibrate(0.5);
}
