use crate::math_utils::{safe_prob, sigmoid};

/// Platt scaling calibrator: P(y=1|s) = sigmoid(a*s + b).
///
/// Learns parameters a and b via gradient descent on binary cross-entropy loss.
pub struct PlattCalibrator {
    pub a: f64,
    pub b: f64,
}

impl PlattCalibrator {
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// Fit Platt scaling parameters from scores and binary labels.
    pub fn fit(
        &mut self,
        scores: &[f64],
        labels: &[f64],
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
    ) {
        let n = scores.len() as f64;
        let mut a = self.a;
        let mut b = self.b;

        for _ in 0..max_iterations {
            let mut grad_a = 0.0;
            let mut grad_b = 0.0;

            for (&s, &y) in scores.iter().zip(labels.iter()) {
                let p = safe_prob(sigmoid(a * s + b));
                let error = p - y;
                grad_a += error * s;
                grad_b += error;
            }

            grad_a /= n;
            grad_b /= n;

            let new_a = a - learning_rate * grad_a;
            let new_b = b - learning_rate * grad_b;

            if (new_a - a).abs() < tolerance && (new_b - b).abs() < tolerance {
                a = new_a;
                b = new_b;
                break;
            }

            a = new_a;
            b = new_b;
        }

        self.a = a;
        self.b = b;
    }

    /// Calibrate a single score.
    pub fn calibrate(&self, score: f64) -> f64 {
        sigmoid(self.a * score + self.b)
    }

    /// Calibrate a batch of scores.
    pub fn calibrate_batch(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate(s)).collect()
    }
}

/// Isotonic regression calibrator using the Pool Adjacent Violators Algorithm (PAVA).
///
/// Fits a non-decreasing step function from scores to probabilities,
/// then uses binary search with linear interpolation for prediction.
pub struct IsotonicCalibrator {
    x_breakpoints: Option<Vec<f64>>,
    y_breakpoints: Option<Vec<f64>>,
}

impl IsotonicCalibrator {
    pub fn new() -> Self {
        Self {
            x_breakpoints: None,
            y_breakpoints: None,
        }
    }

    /// Fit isotonic regression using the PAVA algorithm.
    ///
    /// Sorts (score, label) pairs by score, then merges adjacent blocks
    /// that violate the non-decreasing constraint.
    pub fn fit(&mut self, scores: &[f64], labels: &[f64]) {
        assert_eq!(
            scores.len(),
            labels.len(),
            "scores and labels must have the same length"
        );
        if scores.is_empty() {
            self.x_breakpoints = Some(Vec::new());
            self.y_breakpoints = Some(Vec::new());
            return;
        }

        // Sort by score
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_x: Vec<f64> = indices.iter().map(|&i| scores[i]).collect();
        let sorted_y: Vec<f64> = indices.iter().map(|&i| labels[i]).collect();

        // PAVA: maintain blocks of (sum_y, count, representative_x_start, representative_x_end)
        let mut block_sum: Vec<f64> = sorted_y.iter().copied().collect();
        let mut block_count: Vec<f64> = vec![1.0; sorted_y.len()];
        let mut block_x_start: Vec<f64> = sorted_x.clone();
        let mut block_x_end: Vec<f64> = sorted_x.clone();
        let mut n_blocks = sorted_y.len();

        // Pool adjacent violators
        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            let mut new_sum = Vec::new();
            let mut new_count = Vec::new();
            let mut new_x_start = Vec::new();
            let mut new_x_end = Vec::new();

            while i < n_blocks {
                let mut s = block_sum[i];
                let mut c = block_count[i];
                let xs = block_x_start[i];
                let mut xe = block_x_end[i];

                // Merge forward while violating non-decreasing constraint
                while i + 1 < n_blocks && s / c > block_sum[i + 1] / block_count[i + 1] {
                    i += 1;
                    s += block_sum[i];
                    c += block_count[i];
                    xe = block_x_end[i];
                    changed = true;
                }

                new_sum.push(s);
                new_count.push(c);
                new_x_start.push(xs);
                new_x_end.push(xe);
                i += 1;
            }

            block_sum = new_sum;
            block_count = new_count;
            block_x_start = new_x_start;
            block_x_end = new_x_end;
            n_blocks = block_sum.len();
        }

        // Build breakpoints: use midpoint of each block's x range as the representative x
        let mut x_bp = Vec::with_capacity(n_blocks);
        let mut y_bp = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            x_bp.push((block_x_start[i] + block_x_end[i]) / 2.0);
            y_bp.push(block_sum[i] / block_count[i]);
        }

        self.x_breakpoints = Some(x_bp);
        self.y_breakpoints = Some(y_bp);
    }

    /// Calibrate a single score using binary search and linear interpolation.
    pub fn calibrate(&self, score: f64) -> f64 {
        let x_bp = match &self.x_breakpoints {
            Some(bp) => bp,
            None => panic!("IsotonicCalibrator has not been fitted"),
        };
        let y_bp = self.y_breakpoints.as_ref().unwrap();

        if x_bp.is_empty() {
            return 0.5;
        }

        if x_bp.len() == 1 {
            return y_bp[0];
        }

        // Clamp to boundary values
        if score <= x_bp[0] {
            return y_bp[0];
        }
        if score >= x_bp[x_bp.len() - 1] {
            return y_bp[y_bp.len() - 1];
        }

        // Binary search for the interval
        let mut lo = 0;
        let mut hi = x_bp.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if x_bp[mid] <= score {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Linear interpolation between breakpoints
        let x0 = x_bp[lo];
        let x1 = x_bp[hi];
        let y0 = y_bp[lo];
        let y1 = y_bp[hi];

        let range = x1 - x0;
        if range.abs() < 1e-15 {
            return (y0 + y1) / 2.0;
        }

        let t = (score - x0) / range;
        y0 + t * (y1 - y0)
    }

    /// Calibrate a batch of scores.
    pub fn calibrate_batch(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate(s)).collect()
    }
}
