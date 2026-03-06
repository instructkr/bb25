pub const EPSILON: f64 = 1e-10;

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let ez = (-x).exp();
        1.0 / (1.0 + ez)
    } else {
        let ez = x.exp();
        ez / (1.0 + ez)
    }
}

pub fn safe_log(p: f64) -> f64 {
    p.max(EPSILON).ln()
}

pub fn logit(p: f64) -> f64 {
    let p = clamp(p, EPSILON, 1.0 - EPSILON);
    (p / (1.0 - p)).ln()
}

pub fn safe_prob(p: f64) -> f64 {
    clamp(p, EPSILON, 1.0 - EPSILON)
}

pub fn clamp(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

pub fn vector_magnitude(v: &[f64]) -> f64 {
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mag_a = vector_magnitude(a);
    let mag_b = vector_magnitude(b);
    if mag_a < EPSILON || mag_b < EPSILON {
        return 0.0;
    }
    dot_product(a, b) / (mag_a * mag_b)
}

/// Numerically stable softmax over a 1D slice.
///
/// Shifts by max to prevent overflow, then normalizes.
pub fn softmax(z: &[f64]) -> Vec<f64> {
    if z.is_empty() {
        return Vec::new();
    }
    let max_z = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_z: Vec<f64> = z.iter().map(|&v| (v - max_z).exp()).collect();
    let sum: f64 = exp_z.iter().sum();
    exp_z.iter().map(|&e| e / sum).collect()
}

/// Row-wise softmax over a 2D array stored as a flat slice.
///
/// Each row of `n_cols` elements gets an independent softmax.
pub fn softmax_rows(z: &[f64], n_cols: usize) -> Vec<f64> {
    let n_rows = z.len() / n_cols;
    let mut result = vec![0.0; z.len()];
    for r in 0..n_rows {
        let start = r * n_cols;
        let end = start + n_cols;
        let row = &z[start..end];
        let sm = softmax(row);
        result[start..end].copy_from_slice(&sm);
    }
    result
}

/// Min-max normalize a slice to [0, 1]. Returns zeros if range is negligible.
pub fn min_max_normalize(values: &[f64]) -> Vec<f64> {
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
