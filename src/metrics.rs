/// Calibration metrics for evaluating probability quality.

/// A single bin in a reliability diagram: (avg_predicted, avg_actual, count).
pub type ReliabilityBin = (f64, f64, usize);

/// Expected Calibration Error (ECE).
///
/// Measures how well predicted probabilities match actual relevance rates.
/// Lower is better. Perfect calibration = 0.
pub fn expected_calibration_error(
    probabilities: &[f64],
    labels: &[f64],
    n_bins: usize,
) -> f64 {
    let total = probabilities.len() as f64;
    let mut ece = 0.0;

    for bin_idx in 0..n_bins {
        let lo = bin_idx as f64 / n_bins as f64;
        let hi = (bin_idx + 1) as f64 / n_bins as f64;

        let mut sum_prob = 0.0;
        let mut sum_label = 0.0;
        let mut count = 0usize;

        for (i, &p) in probabilities.iter().enumerate() {
            let in_bin = if bin_idx == 0 {
                p >= lo && p <= hi
            } else {
                p > lo && p <= hi
            };
            if in_bin {
                sum_prob += p;
                sum_label += labels[i];
                count += 1;
            }
        }

        if count == 0 {
            continue;
        }

        let avg_prob = sum_prob / count as f64;
        let avg_label = sum_label / count as f64;
        ece += (count as f64 / total) * (avg_prob - avg_label).abs();
    }

    ece
}

/// Brier score: mean squared error between probabilities and labels.
///
/// Decomposes into calibration + discrimination. Lower is better.
pub fn brier_score(probabilities: &[f64], labels: &[f64]) -> f64 {
    let n = probabilities.len() as f64;
    probabilities
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| (p - y) * (p - y))
        .sum::<f64>()
        / n
}

/// Compute reliability diagram data: (avg_predicted, avg_actual, count) per bin.
///
/// Perfect calibration means avg_predicted == avg_actual for every bin.
pub fn reliability_diagram(
    probabilities: &[f64],
    labels: &[f64],
    n_bins: usize,
) -> Vec<ReliabilityBin> {
    let mut bins = Vec::new();

    for bin_idx in 0..n_bins {
        let lo = bin_idx as f64 / n_bins as f64;
        let hi = (bin_idx + 1) as f64 / n_bins as f64;

        let mut sum_prob = 0.0;
        let mut sum_label = 0.0;
        let mut count = 0usize;

        for (i, &p) in probabilities.iter().enumerate() {
            let in_bin = if bin_idx == 0 {
                p >= lo && p <= hi
            } else {
                p > lo && p <= hi
            };
            if in_bin {
                sum_prob += p;
                sum_label += labels[i];
                count += 1;
            }
        }

        if count > 0 {
            bins.push((sum_prob / count as f64, sum_label / count as f64, count));
        }
    }

    bins
}

/// One-call calibration diagnostic report.
pub struct CalibrationReport {
    pub ece: f64,
    pub brier: f64,
    pub reliability: Vec<ReliabilityBin>,
    pub n_samples: usize,
    pub n_bins: usize,
}

impl CalibrationReport {
    /// Formatted text summary of calibration metrics.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "Calibration Report".to_string(),
            "==================".to_string(),
            format!("  Samples : {}", self.n_samples),
            format!("  Bins    : {}", self.n_bins),
            format!("  ECE     : {:.6}", self.ece),
            format!("  Brier   : {:.6}", self.brier),
            String::new(),
            "  Reliability Diagram".to_string(),
            "  -------------------".to_string(),
            format!("  {:>10}  {:>10}  {:>6}", "Predicted", "Actual", "Count"),
        ];
        for &(avg_pred, avg_actual, count) in &self.reliability {
            lines.push(format!(
                "  {:>10.4}  {:>10.4}  {:>6}",
                avg_pred, avg_actual, count
            ));
        }
        lines.join("\n")
    }
}

/// Compute a full calibration diagnostic report in one call.
pub fn calibration_report(
    probabilities: &[f64],
    labels: &[f64],
    n_bins: usize,
) -> CalibrationReport {
    CalibrationReport {
        ece: expected_calibration_error(probabilities, labels, n_bins),
        brier: brier_score(probabilities, labels),
        reliability: reliability_diagram(probabilities, labels, n_bins),
        n_samples: probabilities.len(),
        n_bins,
    }
}
