use std::collections::HashMap;

use crate::fusion::{cosine_to_probability, prob_not};
use crate::math_utils::{logit, safe_prob, sigmoid};
use crate::probability::BayesianProbabilityTransform;

/// Trace of a single BM25 signal through the full probability pipeline.
#[derive(Clone, Debug)]
pub struct BM25SignalTrace {
    pub raw_score: f64,
    pub tf: f64,
    pub doc_len_ratio: f64,
    pub likelihood: f64,
    pub tf_prior: f64,
    pub norm_prior: f64,
    pub composite_prior: f64,
    pub logit_likelihood: f64,
    pub logit_prior: f64,
    pub logit_base_rate: Option<f64>,
    pub posterior: f64,
    pub alpha: f64,
    pub beta: f64,
    pub base_rate: Option<f64>,
}

/// Trace of a cosine similarity through probability conversion.
#[derive(Clone, Debug)]
pub struct VectorSignalTrace {
    pub cosine_score: f64,
    pub probability: f64,
    pub logit_probability: f64,
}

/// Trace of a probabilistic NOT (complement) operation.
#[derive(Clone, Debug)]
pub struct NotTrace {
    pub input_probability: f64,
    pub input_name: String,
    pub complement: f64,
    pub logit_input: f64,
    pub logit_complement: f64,
}

/// Trace of the combination step for multiple probability signals.
#[derive(Clone, Debug)]
pub struct FusionTrace {
    pub signal_probabilities: Vec<f64>,
    pub signal_names: Vec<String>,
    pub method: String,
    // Log-odds intermediates
    pub logits: Option<Vec<f64>>,
    pub mean_logit: Option<f64>,
    pub alpha: Option<f64>,
    pub n_alpha_scale: Option<f64>,
    pub scaled_logit: Option<f64>,
    pub weights: Option<Vec<f64>>,
    // prob_and intermediates
    pub log_probs: Option<Vec<f64>>,
    pub log_prob_sum: Option<f64>,
    // prob_or/prob_not intermediates
    pub complements: Option<Vec<f64>>,
    pub log_complements: Option<Vec<f64>>,
    pub log_complement_sum: Option<f64>,
    // Output
    pub fused_probability: f64,
}

/// Signal type enum for document traces.
#[derive(Clone, Debug)]
pub enum SignalTrace {
    BM25(BM25SignalTrace),
    Vector(VectorSignalTrace),
}

/// Complete trace for one document across all signals and fusion.
#[derive(Clone, Debug)]
pub struct DocumentTrace {
    pub doc_id: Option<String>,
    pub signals: Vec<(String, SignalTrace)>,
    pub fusion: FusionTrace,
    pub final_probability: f64,
}

/// Comparison of two document traces explaining rank differences.
#[derive(Clone, Debug)]
pub struct ComparisonResult {
    pub doc_a: DocumentTrace,
    pub doc_b: DocumentTrace,
    pub signal_deltas: Vec<(String, f64)>,
    pub dominant_signal: String,
    pub crossover_stage: Option<String>,
}

/// Traces intermediate values through the Bayesian BM25 fusion pipeline.
pub struct FusionDebugger {
    transform: BayesianProbabilityTransform,
}

impl FusionDebugger {
    pub fn new(transform: BayesianProbabilityTransform) -> Self {
        Self { transform }
    }

    pub fn transform(&self) -> &BayesianProbabilityTransform {
        &self.transform
    }

    /// Trace a single BM25 score through the full probability pipeline.
    pub fn trace_bm25(
        &self,
        score: f64,
        tf: f64,
        doc_len_ratio: f64,
    ) -> BM25SignalTrace {
        let t = &self.transform;

        let likelihood_val = t.likelihood(score);
        let tf_prior_val = BayesianProbabilityTransform::tf_prior(tf);
        let norm_prior_val = BayesianProbabilityTransform::norm_prior(doc_len_ratio);
        let composite_prior_val = BayesianProbabilityTransform::composite_prior(tf, doc_len_ratio);
        let posterior_val = BayesianProbabilityTransform::posterior(
            likelihood_val,
            composite_prior_val,
            t.base_rate,
        );

        let logit_likelihood_val = logit(likelihood_val);
        let logit_prior_val = logit(composite_prior_val);
        let logit_base_rate_val = t.base_rate.map(|br| logit(safe_prob(br)));

        BM25SignalTrace {
            raw_score: score,
            tf,
            doc_len_ratio,
            likelihood: likelihood_val,
            tf_prior: tf_prior_val,
            norm_prior: norm_prior_val,
            composite_prior: composite_prior_val,
            logit_likelihood: logit_likelihood_val,
            logit_prior: logit_prior_val,
            logit_base_rate: logit_base_rate_val,
            posterior: posterior_val,
            alpha: t.alpha,
            beta: t.beta,
            base_rate: t.base_rate,
        }
    }

    /// Trace a cosine similarity through probability conversion.
    pub fn trace_vector(&self, cosine_score: f64) -> VectorSignalTrace {
        let prob_val = cosine_to_probability(cosine_score);
        let logit_val = logit(prob_val);

        VectorSignalTrace {
            cosine_score,
            probability: prob_val,
            logit_probability: logit_val,
        }
    }

    /// Trace a probabilistic NOT (complement) operation.
    pub fn trace_not(&self, probability: f64, name: &str) -> NotTrace {
        let complement = prob_not(probability);
        let logit_in = logit(safe_prob(probability));
        let logit_out = logit(safe_prob(complement));

        NotTrace {
            input_probability: probability,
            input_name: name.to_string(),
            complement,
            logit_input: logit_in,
            logit_complement: logit_out,
        }
    }

    /// Trace the fusion of multiple probability signals.
    pub fn trace_fusion(
        &self,
        probabilities: &[f64],
        names: Option<&[String]>,
        method: &str,
        alpha: Option<f64>,
        weights: Option<&[f64]>,
    ) -> FusionTrace {
        let n = probabilities.len();
        let signal_names: Vec<String> = match names {
            Some(ns) => ns.to_vec(),
            None => (0..n).map(|i| format!("signal_{}", i)).collect(),
        };

        let probs: Vec<f64> = probabilities.iter().map(|&p| safe_prob(p)).collect();

        match method {
            "log_odds" => self.trace_log_odds(&probs, &signal_names, alpha, weights),
            "prob_and" => self.trace_prob_and(&probs, &signal_names),
            "prob_or" => self.trace_prob_or(&probs, &signal_names),
            "prob_not" => self.trace_prob_not_fusion(&probs, &signal_names),
            _ => panic!("method must be 'log_odds', 'prob_and', 'prob_or', or 'prob_not', got '{}'", method),
        }
    }

    fn trace_log_odds(
        &self,
        probs: &[f64],
        names: &[String],
        alpha: Option<f64>,
        weights: Option<&[f64]>,
    ) -> FusionTrace {
        let n = probs.len();
        let logits_arr: Vec<f64> = probs.iter().map(|&p| logit(p)).collect();

        if let Some(w) = weights {
            let effective_alpha = alpha.unwrap_or(0.0);
            let n_alpha_scale = (n as f64).powf(effective_alpha);
            let weighted_logit: f64 = w.iter().zip(logits_arr.iter()).map(|(&wi, &li)| wi * li).sum();
            let scaled = n_alpha_scale * weighted_logit;
            let fused = sigmoid(scaled);

            return FusionTrace {
                signal_probabilities: probs.to_vec(),
                signal_names: names.to_vec(),
                method: "log_odds".to_string(),
                logits: Some(logits_arr),
                mean_logit: Some(weighted_logit),
                alpha: Some(effective_alpha),
                n_alpha_scale: Some(n_alpha_scale),
                scaled_logit: Some(scaled),
                weights: Some(w.to_vec()),
                log_probs: None,
                log_prob_sum: None,
                complements: None,
                log_complements: None,
                log_complement_sum: None,
                fused_probability: fused,
            };
        }

        let effective_alpha = alpha.unwrap_or(0.5);
        let mean_logit_val: f64 = logits_arr.iter().sum::<f64>() / n as f64;
        let n_alpha_scale = (n as f64).powf(effective_alpha);
        let scaled = mean_logit_val * n_alpha_scale;
        let fused = sigmoid(scaled);

        FusionTrace {
            signal_probabilities: probs.to_vec(),
            signal_names: names.to_vec(),
            method: "log_odds".to_string(),
            logits: Some(logits_arr),
            mean_logit: Some(mean_logit_val),
            alpha: Some(effective_alpha),
            n_alpha_scale: Some(n_alpha_scale),
            scaled_logit: Some(scaled),
            weights: None,
            log_probs: None,
            log_prob_sum: None,
            complements: None,
            log_complements: None,
            log_complement_sum: None,
            fused_probability: fused,
        }
    }

    fn trace_prob_and(&self, probs: &[f64], names: &[String]) -> FusionTrace {
        let log_probs: Vec<f64> = probs.iter().map(|&p| p.ln()).collect();
        let log_sum: f64 = log_probs.iter().sum();
        let fused = log_sum.exp();

        FusionTrace {
            signal_probabilities: probs.to_vec(),
            signal_names: names.to_vec(),
            method: "prob_and".to_string(),
            logits: None,
            mean_logit: None,
            alpha: None,
            n_alpha_scale: None,
            scaled_logit: None,
            weights: None,
            log_probs: Some(log_probs),
            log_prob_sum: Some(log_sum),
            complements: None,
            log_complements: None,
            log_complement_sum: None,
            fused_probability: fused,
        }
    }

    fn trace_prob_or(&self, probs: &[f64], names: &[String]) -> FusionTrace {
        let comps: Vec<f64> = probs.iter().map(|&p| 1.0 - p).collect();
        let log_comps: Vec<f64> = comps.iter().map(|&c| c.ln()).collect();
        let log_sum: f64 = log_comps.iter().sum();
        let fused = 1.0 - log_sum.exp();

        FusionTrace {
            signal_probabilities: probs.to_vec(),
            signal_names: names.to_vec(),
            method: "prob_or".to_string(),
            logits: None,
            mean_logit: None,
            alpha: None,
            n_alpha_scale: None,
            scaled_logit: None,
            weights: None,
            log_probs: None,
            log_prob_sum: None,
            complements: Some(comps),
            log_complements: Some(log_comps),
            log_complement_sum: Some(log_sum),
            fused_probability: fused,
        }
    }

    fn trace_prob_not_fusion(&self, probs: &[f64], names: &[String]) -> FusionTrace {
        let comps: Vec<f64> = probs.iter().map(|&p| 1.0 - p).collect();
        let log_comps: Vec<f64> = comps.iter().map(|&c| c.ln()).collect();
        let log_sum: f64 = log_comps.iter().sum();
        let fused = log_sum.exp();

        FusionTrace {
            signal_probabilities: probs.to_vec(),
            signal_names: names.to_vec(),
            method: "prob_not".to_string(),
            logits: None,
            mean_logit: None,
            alpha: None,
            n_alpha_scale: None,
            scaled_logit: None,
            weights: None,
            log_probs: None,
            log_prob_sum: None,
            complements: Some(comps),
            log_complements: Some(log_comps),
            log_complement_sum: Some(log_sum),
            fused_probability: fused,
        }
    }

    /// Full pipeline trace for one document (convenience method).
    pub fn trace_document(
        &self,
        bm25_score: Option<f64>,
        tf: Option<f64>,
        doc_len_ratio: Option<f64>,
        cosine_score: Option<f64>,
        method: &str,
        alpha: Option<f64>,
        weights: Option<&[f64]>,
        doc_id: Option<&str>,
    ) -> DocumentTrace {
        let mut signals: Vec<(String, SignalTrace)> = Vec::new();
        let mut probs: Vec<f64> = Vec::new();
        let mut names: Vec<String> = Vec::new();

        if let Some(bm25) = bm25_score {
            let tf_val = tf.expect("tf is required when bm25_score is provided");
            let dlr_val = doc_len_ratio.expect("doc_len_ratio is required when bm25_score is provided");
            let trace = self.trace_bm25(bm25, tf_val, dlr_val);
            probs.push(trace.posterior);
            names.push("BM25".to_string());
            signals.push(("BM25".to_string(), SignalTrace::BM25(trace)));
        }

        if let Some(cos) = cosine_score {
            let trace = self.trace_vector(cos);
            probs.push(trace.probability);
            names.push("Vector".to_string());
            signals.push(("Vector".to_string(), SignalTrace::Vector(trace)));
        }

        assert!(!probs.is_empty(), "At least one of bm25_score or cosine_score must be provided");

        let fusion_trace = self.trace_fusion(
            &probs,
            Some(&names),
            method,
            alpha,
            weights,
        );

        DocumentTrace {
            doc_id: doc_id.map(|s| s.to_string()),
            signals,
            fusion: fusion_trace.clone(),
            final_probability: fusion_trace.fused_probability,
        }
    }

    /// Compare two document traces to explain rank differences.
    pub fn compare(
        &self,
        trace_a: &DocumentTrace,
        trace_b: &DocumentTrace,
    ) -> ComparisonResult {
        // Collect all unique signal names preserving order
        let mut all_names: Vec<String> = Vec::new();
        let mut seen = HashMap::new();
        for (name, _) in &trace_a.signals {
            if seen.insert(name.clone(), ()).is_none() {
                all_names.push(name.clone());
            }
        }
        for (name, _) in &trace_b.signals {
            if seen.insert(name.clone(), ()).is_none() {
                all_names.push(name.clone());
            }
        }

        let mut signal_deltas: Vec<(String, f64)> = Vec::new();
        for name in &all_names {
            let prob_a = signal_probability(trace_a, name);
            let prob_b = signal_probability(trace_b, name);
            signal_deltas.push((name.clone(), prob_a - prob_b));
        }

        // Dominant signal: largest absolute delta
        let dominant = signal_deltas
            .iter()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(name, _)| name.clone())
            .unwrap_or_default();

        // Crossover detection
        let fused_delta = trace_a.final_probability - trace_b.final_probability;
        let mut crossover_stage: Option<String> = None;
        for (name, delta) in &signal_deltas {
            if name == &dominant {
                continue;
            }
            if fused_delta != 0.0
                && *delta != 0.0
                && ((fused_delta > 0.0 && *delta < 0.0) || (fused_delta < 0.0 && *delta > 0.0))
            {
                crossover_stage = Some(name.clone());
                break;
            }
        }

        ComparisonResult {
            doc_a: trace_a.clone(),
            doc_b: trace_b.clone(),
            signal_deltas,
            dominant_signal: dominant,
            crossover_stage,
        }
    }

    /// Format a document trace as human-readable text.
    pub fn format_trace(&self, trace: &DocumentTrace, verbose: bool) -> String {
        let mut lines: Vec<String> = Vec::new();
        let doc_label = trace.doc_id.as_deref().unwrap_or("unknown");
        lines.push(format!("Document: {}", doc_label));

        for (name, sig) in &trace.signals {
            match sig {
                SignalTrace::BM25(s) => {
                    lines.push(format!(
                        "  [{}] raw={:.2} -> likelihood={:.3} (alpha={:.2}, beta={:.2})",
                        name, s.raw_score, s.likelihood, s.alpha, s.beta
                    ));
                    lines.push(format!("         tf={:.0} -> tf_prior={:.3}", s.tf, s.tf_prior));
                    lines.push(format!(
                        "         dl_ratio={:.2} -> norm_prior={:.3}",
                        s.doc_len_ratio, s.norm_prior
                    ));
                    lines.push(format!("         composite_prior={:.3}", s.composite_prior));
                    if let Some(br) = s.base_rate {
                        let posterior_no_br = BayesianProbabilityTransform::posterior(
                            s.likelihood,
                            s.composite_prior,
                            None,
                        );
                        lines.push(format!("         posterior={:.3}", posterior_no_br));
                        lines.push(format!(
                            "         with base_rate={:.3}: posterior={:.3}",
                            br, s.posterior
                        ));
                    } else {
                        lines.push(format!("         posterior={:.3}", s.posterior));
                    }
                    if verbose {
                        lines.push(format!(
                            "         logit(posterior)={:.3}",
                            logit(safe_prob(s.posterior))
                        ));
                    }
                    lines.push(String::new());
                }
                SignalTrace::Vector(s) => {
                    lines.push(format!(
                        "  [{}] cosine={:.3} -> prob={:.3}",
                        name, s.cosine_score, s.probability
                    ));
                    if verbose {
                        lines.push(format!("           logit(prob)={:.3}", s.logit_probability));
                    }
                    lines.push(String::new());
                }
            }
        }

        // Fusion
        let f = &trace.fusion;
        let alpha_str = f.alpha.map_or(String::new(), |a| format!(", alpha={}", a));
        let n_str = format!(", n={}", f.signal_probabilities.len());
        lines.push(format!("  [Fusion] method={}{}{}", f.method, alpha_str, n_str));

        if verbose {
            if let Some(ref logits) = f.logits {
                let s: Vec<String> = logits.iter().map(|v| format!("{:.3}", v)).collect();
                lines.push(format!("           logits=[{}]", s.join(", ")));
            }
            if let Some(ml) = f.mean_logit {
                lines.push(format!("           mean_logit={:.3}", ml));
            }
            if let (Some(nas), Some(sl)) = (f.n_alpha_scale, f.scaled_logit) {
                lines.push(format!("           n^alpha={:.3}, scaled={:.3}", nas, sl));
            }
            if let Some(ref w) = f.weights {
                let s: Vec<String> = w.iter().map(|v| format!("{:.3}", v)).collect();
                lines.push(format!("           weights=[{}]", s.join(", ")));
            }
            if let Some(ref lp) = f.log_probs {
                let s: Vec<String> = lp.iter().map(|v| format!("{:.3}", v)).collect();
                lines.push(format!("           ln(P)=[{}]", s.join(", ")));
                if let Some(lps) = f.log_prob_sum {
                    lines.push(format!("           sum(ln(P))={:.3}", lps));
                }
            }
            if let Some(ref c) = f.complements {
                let s: Vec<String> = c.iter().map(|v| format!("{:.3}", v)).collect();
                lines.push(format!("           1-P=[{}]", s.join(", ")));
            }
            if let Some(ref lc) = f.log_complements {
                let s: Vec<String> = lc.iter().map(|v| format!("{:.3}", v)).collect();
                lines.push(format!("           ln(1-P)=[{}]", s.join(", ")));
                if let Some(lcs) = f.log_complement_sum {
                    lines.push(format!("           sum(ln(1-P))={:.3}", lcs));
                }
            }
        }

        lines.push(format!("           -> final={:.3}", f.fused_probability));
        lines.join("\n")
    }

    /// Compact one-line summary of a document trace.
    pub fn format_summary(&self, trace: &DocumentTrace) -> String {
        let doc_label = trace.doc_id.as_deref().unwrap_or("unknown");
        let mut parts: Vec<String> = Vec::new();
        for (_, sig) in &trace.signals {
            match sig {
                SignalTrace::BM25(s) => parts.push(format!("BM25={:.3}", s.posterior)),
                SignalTrace::Vector(s) => parts.push(format!("Vec={:.3}", s.probability)),
            }
        }

        let f = &trace.fusion;
        let alpha_str = f.alpha.map_or(String::new(), |a| format!(", alpha={}", a));
        format!(
            "{}: {} -> Fused={:.3} ({}{})",
            doc_label,
            parts.join(" "),
            f.fused_probability,
            f.method,
            alpha_str
        )
    }

    /// Format a comparison result as human-readable text.
    pub fn format_comparison(&self, comparison: &ComparisonResult) -> String {
        let a = &comparison.doc_a;
        let b = &comparison.doc_b;
        let a_label = a.doc_id.as_deref().unwrap_or("doc_a");
        let b_label = b.doc_id.as_deref().unwrap_or("doc_b");

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("Comparison: {} vs {}", a_label, b_label));

        lines.push(format!(
            "  {:<12} {:>8}  {:>8}  {:>8}   dominant",
            "Signal", a_label, b_label, "delta"
        ));

        for (name, delta) in &comparison.signal_deltas {
            let prob_a = signal_probability(a, name);
            let prob_b = signal_probability(b, name);
            let dominant_marker = if name == &comparison.dominant_signal {
                "   <-- largest"
            } else {
                ""
            };
            lines.push(format!(
                "  {:<12} {:>8.3}  {:>8.3}  {:>+8.3}{}",
                name, prob_a, prob_b, delta, dominant_marker
            ));
        }

        let fused_delta = a.final_probability - b.final_probability;
        lines.push(format!(
            "  {:<12} {:>8.3}  {:>8.3}  {:>+8.3}",
            "Fused", a.final_probability, b.final_probability, fused_delta
        ));
        lines.push(String::new());

        if fused_delta > 0.0 {
            lines.push(format!(
                "  Rank order: {} > {} (by {:+.3})",
                a_label, b_label, fused_delta
            ));
        } else if fused_delta < 0.0 {
            lines.push(format!(
                "  Rank order: {} > {} (by +{:.3})",
                b_label, a_label, fused_delta.abs()
            ));
        } else {
            lines.push("  Rank order: tied".to_string());
        }

        let dom = &comparison.dominant_signal;
        let dom_delta = comparison
            .signal_deltas
            .iter()
            .find(|(n, _)| n == dom)
            .map(|(_, d)| *d)
            .unwrap_or(0.0);
        let favored = if dom_delta >= 0.0 { a_label } else { b_label };
        lines.push(format!(
            "  Dominant signal: {} ({:+.3} in {}'s favor)",
            dom, dom_delta, favored
        ));

        if let Some(ref cross) = comparison.crossover_stage {
            let cross_delta = comparison
                .signal_deltas
                .iter()
                .find(|(n, _)| n == cross)
                .map(|(_, d)| *d)
                .unwrap_or(0.0);
            let cross_favored = if cross_delta >= 0.0 { a_label } else { b_label };
            lines.push(format!(
                "  Note: {} favored {}, but {} signal outweighed it",
                cross, cross_favored, dom
            ));
        }

        lines.join("\n")
    }
}

/// Extract the final probability from a signal within a document trace.
fn signal_probability(trace: &DocumentTrace, name: &str) -> f64 {
    for (n, sig) in &trace.signals {
        if n == name {
            return match sig {
                SignalTrace::BM25(s) => s.posterior,
                SignalTrace::Vector(s) => s.probability,
            };
        }
    }
    0.5 // neutral if signal missing
}
