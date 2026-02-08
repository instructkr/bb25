use std::collections::HashMap;
use std::rc::Rc;

use crate::bayesian_scorer::BayesianBM25Scorer;
use crate::bm25_scorer::BM25Scorer;
use crate::corpus::{Corpus, Document};
use crate::hybrid_scorer::HybridScorer;
use crate::math_utils::{safe_log, sigmoid, EPSILON};
use crate::parameter_learner::ParameterLearner;
use crate::vector_scorer::VectorScorer;

#[derive(Clone)]
pub struct Query {
    pub text: String,
    pub terms: Vec<String>,
    pub embedding: Option<Vec<f64>>,
    pub relevant: Vec<String>,
}

impl Query {
    pub fn new(
        text: &str,
        terms: &[&str],
        embedding: Option<Vec<f64>>,
        relevant: &[&str],
    ) -> Self {
        Self {
            text: text.to_string(),
            terms: terms.iter().map(|t| t.to_string()).collect(),
            embedding,
            relevant: relevant.iter().map(|r| r.to_string()).collect(),
        }
    }
}

pub struct ExperimentRunner {
    corpus: Rc<Corpus>,
    queries: Vec<Query>,
    bm25: Rc<BM25Scorer>,
    bayesian: Rc<BayesianBM25Scorer>,
    vector: Rc<VectorScorer>,
    hybrid: HybridScorer,
}

impl ExperimentRunner {
    pub fn new(corpus: Rc<Corpus>, queries: Vec<Query>, k1: f64, b: f64) -> Self {
        let bm25 = Rc::new(BM25Scorer::new(Rc::clone(&corpus), k1, b));
        let bayesian = Rc::new(BayesianBM25Scorer::new(Rc::clone(&bm25), 1.0, 0.5));
        let vector = Rc::new(VectorScorer::new());
        let hybrid = HybridScorer::new(Rc::clone(&bayesian), Rc::clone(&vector), 0.5);

        Self {
            corpus,
            queries,
            bm25,
            bayesian,
            vector,
            hybrid,
        }
    }

    pub fn run_all(&self) -> Vec<(String, bool, String)> {
        let experiments: Vec<(&str, fn(&ExperimentRunner) -> (bool, String))> = vec![
            ("1. BM25 Formula Equivalence", ExperimentRunner::exp1_formula_equivalence),
            ("2. Score Calibration", ExperimentRunner::exp2_score_calibration),
            ("3. Monotonicity Preservation", ExperimentRunner::exp3_monotonicity),
            ("4. Prior Bounds", ExperimentRunner::exp4_prior_bounds),
            ("5. IDF Properties", ExperimentRunner::exp5_idf_properties),
            ("6. Hybrid Search Quality", ExperimentRunner::exp6_hybrid_quality),
            ("7. Naive vs RRF vs Bayesian", ExperimentRunner::exp7_method_comparison),
            ("8. Log-space Numerical Stability", ExperimentRunner::exp8_numerical_stability),
            ("9. Parameter Learning Convergence", ExperimentRunner::exp9_parameter_learning),
            ("10. Conjunction/Disjunction Bounds", ExperimentRunner::exp10_conjunction_disjunction),
        ];

        experiments
            .into_iter()
            .map(|(name, func)| {
                let (passed, details) = func(self);
                (name.to_string(), passed, details)
            })
            .collect()
    }

    fn exp1_formula_equivalence(&self) -> (bool, String) {
        let mut max_diff = 0.0;
        let mut comparisons = 0usize;

        for query in &self.queries {
            for doc in self.corpus.documents() {
                for term in &query.terms {
                    let s1 = self.bm25.score_term_standard(term, doc);
                    let s2 = self.bm25.score_term_rewritten(term, doc);
                    let diff = (s1 - s2).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    comparisons += 1;
                }
            }
        }

        let passed = max_diff < 1e-10;
        let details = format!("max_diff={:.2e} across {} comparisons", max_diff, comparisons);
        (passed, details)
    }

    fn exp2_score_calibration(&self) -> (bool, String) {
        let mut all_in_range = true;
        let mut ordering_preserved = true;
        let mut violations: Vec<String> = Vec::new();

        for query in &self.queries {
            let mut bm25_scores: Vec<(String, f64)> = Vec::new();
            let mut bayesian_scores: Vec<(String, f64)> = Vec::new();

            for doc in self.corpus.documents() {
                let raw = self.bm25.score(&query.terms, doc);
                let calibrated = self.bayesian.score(&query.terms, doc);

                bm25_scores.push((doc.id.clone(), raw));
                bayesian_scores.push((doc.id.clone(), calibrated));

                if calibrated < -EPSILON || calibrated > 1.0 + EPSILON {
                    all_in_range = false;
                    violations.push(format!("doc={} calibrated={:.6}", doc.id, calibrated));
                }
            }

            bm25_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let bayesian_map: HashMap<String, f64> = bayesian_scores
                .into_iter()
                .map(|(id, score)| (id, score))
                .collect();

            for i in 0..bm25_scores.len().saturating_sub(1) {
                let (id_a, score_a) = (&bm25_scores[i].0, bm25_scores[i].1);
                let (id_b, score_b) = (&bm25_scores[i + 1].0, bm25_scores[i + 1].1);
                if score_a > score_b + EPSILON {
                    let ba = bayesian_map.get(id_a).copied().unwrap_or(0.0);
                    let bb = bayesian_map.get(id_b).copied().unwrap_or(0.0);
                    if ba < bb - EPSILON {
                        ordering_preserved = false;
                        violations.push(format!(
                            "query={}: BM25({})={:.4} > BM25({})={:.4} but Bayesian({})={:.6} < Bayesian({})={:.6}",
                            query.text, id_a, score_a, id_b, score_b, id_a, ba, id_b, bb
                        ));
                    }
                }
            }
        }

        let mut parts = vec![
            format!("range=[0,1]: {}", all_in_range),
            format!("ordering: {}", ordering_preserved),
        ];
        if !violations.is_empty() {
            let preview = violations.into_iter().take(3).collect::<Vec<_>>().join("; ");
            parts.push(format!("violations: {}", preview));
        }

        let passed = all_in_range && ordering_preserved;
        (passed, parts.join(", "))
    }

    fn exp3_monotonicity(&self) -> (bool, String) {
        let mut passed = true;
        let mut details_parts: Vec<String> = Vec::new();
        let mut terms_tested = 0usize;

        for term in self.corpus.df.keys() {
            let mut matching_docs: Vec<&Document> = self
                .corpus
                .documents()
                .iter()
                .filter(|doc| doc.term_freq.get(term).copied().unwrap_or(0) > 0)
                .collect();

            if matching_docs.len() < 2 {
                continue;
            }

            matching_docs.sort_by_key(|doc| doc.term_freq.get(term).copied().unwrap_or(0));

            for pair in matching_docs.windows(2) {
                let d1 = pair[0];
                let d2 = pair[1];
                let tf1 = d1.term_freq.get(term).copied().unwrap_or(0);
                let tf2 = d2.term_freq.get(term).copied().unwrap_or(0);
                if tf1 == tf2 {
                    continue;
                }
                if (d1.length as isize - d2.length as isize).abs() <= 3 {
                    let s1 = self.bayesian.score_term(term, d1);
                    let s2 = self.bayesian.score_term(term, d2);
                    terms_tested += 1;
                    if s1 > s2 + EPSILON {
                        passed = false;
                        details_parts.push(format!(
                            "term={}: tf({})={} > tf({})={} but score {:.4} > {:.4}",
                            term, d1.id, tf1, d2.id, tf2, s1, s2
                        ));
                    }
                }
            }
        }

        let mut synthetic_passed = true;
        for raw_score in [0.1, 0.5, 1.0, 2.0, 5.0] {
            for prior in [0.2, 0.5, 0.8] {
                let p1 = self.bayesian.posterior(raw_score, prior);
                let p2 = self.bayesian.posterior(raw_score + 0.1, prior);
                if p2 < p1 - EPSILON {
                    synthetic_passed = false;
                }
            }
        }

        passed = passed && synthetic_passed;
        let mut detail = format!(
            "terms_tested={}, synthetic_monotonic={}",
            terms_tested, synthetic_passed
        );
        if !details_parts.is_empty() {
            let preview = details_parts.into_iter().take(3).collect::<Vec<_>>().join("; ");
            detail.push_str(", violations: ");
            detail.push_str(&preview);
        }

        (passed, detail)
    }

    fn exp4_prior_bounds(&self) -> (bool, String) {
        let mut all_bounded = true;
        let mut min_prior: f64 = 1.0;
        let mut max_prior: f64 = 0.0;
        let mut violations: Vec<String> = Vec::new();

        for doc in self.corpus.documents() {
            for (term, tf) in &doc.term_freq {
                let prior = self
                    .bayesian
                    .composite_prior(*tf, doc.length, self.corpus.avgdl);
                min_prior = min_prior.min(prior);
                max_prior = max_prior.max(prior);
                if prior < 0.1 - EPSILON || prior > 0.9 + EPSILON {
                    all_bounded = false;
                    violations.push(format!(
                        "doc={} term={} prior={:.6}",
                        doc.id, term, prior
                    ));
                }
            }
        }

        let mut detail = format!("range=[{:.4}, {:.4}]", min_prior, max_prior);
        if !violations.is_empty() {
            let preview = violations.into_iter().take(3).collect::<Vec<_>>().join("; ");
            detail.push_str(", violations: ");
            detail.push_str(&preview);
        }

        (all_bounded, detail)
    }

    fn exp5_idf_properties(&self) -> (bool, String) {
        let mut all_terms: Vec<String> = self.corpus.df.keys().cloned().collect();
        all_terms.sort();

        let idf_values: HashMap<String, f64> = all_terms
            .iter()
            .map(|t| (t.clone(), self.bm25.idf(t)))
            .collect();

        let mut non_neg_ok = true;
        for term in &all_terms {
            let df_t = *self.corpus.df.get(term).unwrap_or(&0) as f64;
            if df_t <= self.corpus.n as f64 / 2.0 {
                if idf_values.get(term).copied().unwrap_or(0.0) < -EPSILON {
                    non_neg_ok = false;
                }
            }
        }

        let mut df_idf_pairs: Vec<(usize, f64)> = all_terms
            .iter()
            .map(|t| (*self.corpus.df.get(t).unwrap_or(&0), idf_values[t]))
            .collect();
        df_idf_pairs.sort_by_key(|(df, _)| *df);

        let mut monotonic_ok = true;
        for pair in df_idf_pairs.windows(2) {
            let (df1, idf1) = pair[0];
            let (df2, idf2) = pair[1];
            if df1 < df2 && idf1 < idf2 - EPSILON {
                monotonic_ok = false;
            }
        }

        let mut bound_ok = true;
        for query in &self.queries {
            for term in &query.terms {
                let ub = self.bm25.upper_bound(term);
                for doc in self.corpus.documents() {
                    let actual = self.bm25.score_term_standard(term, doc);
                    if actual > ub + EPSILON {
                        bound_ok = false;
                    }
                }
            }
        }

        let detail = format!(
            "non_neg={}, monotonic={}, upper_bound={}",
            non_neg_ok, monotonic_ok, bound_ok
        );
        (non_neg_ok && monotonic_ok && bound_ok, detail)
    }

    fn exp6_hybrid_quality(&self) -> (bool, String) {
        let mut passed = true;
        let mut tests = 0usize;
        let mut violations: Vec<String> = Vec::new();

        for query in &self.queries {
            let Some(embedding) = &query.embedding else { continue };
            for doc in self.corpus.documents() {
                let bayesian_p = self.bayesian.score(&query.terms, doc);
                let vector_p = self.vector.score(embedding, doc);
                let probs = [bayesian_p, vector_p];

                let and_score = self.hybrid.probabilistic_and(&probs);
                let or_score = self.hybrid.probabilistic_or(&probs);

                tests += 1;

                // Log-odds conjunction: AND <= OR
                if and_score > or_score + EPSILON {
                    passed = false;
                    violations.push(format!(
                        "AND={:.6} > OR={:.6} (doc={})",
                        and_score, or_score, doc.id
                    ));
                }

                // Agreement amplification: both > 0.5 => AND > geometric mean
                if bayesian_p > 0.5 && vector_p > 0.5 {
                    let geo_mean = (bayesian_p * vector_p).sqrt();
                    if and_score < geo_mean - EPSILON {
                        passed = false;
                        violations.push(format!(
                            "no amplification: AND={:.6} < geo_mean={:.6} (doc={})",
                            and_score, geo_mean, doc.id
                        ));
                    }
                }

                // Irrelevance preservation: both < 0.5 => AND < 0.5
                if bayesian_p < 0.5 && vector_p < 0.5 {
                    if and_score > 0.5 + EPSILON {
                        passed = false;
                        violations.push(format!(
                            "irrelevance violated: AND={:.6} > 0.5 (doc={})",
                            and_score, doc.id
                        ));
                    }
                }

                // OR >= max(p_i)
                let max_p = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if or_score < max_p - EPSILON {
                    passed = false;
                    violations.push(format!(
                        "OR={:.6} < max={:.6} (doc={})",
                        or_score, max_p, doc.id
                    ));
                }
            }
        }

        let mut detail = format!("tests={}", tests);
        if !violations.is_empty() {
            let preview = violations.into_iter().take(3).collect::<Vec<_>>().join("; ");
            detail.push_str(", violations: ");
            detail.push_str(&preview);
        }

        (passed, detail)
    }

    fn exp7_method_comparison(&self) -> (bool, String) {
        let mut results_table: Vec<(String, HashMap<String, Vec<String>>)> = Vec::new();

        for query in &self.queries {
            let Some(embedding) = &query.embedding else { continue };
            let mut doc_scores: Vec<HashMap<String, f64>> = Vec::new();
            let mut id_list: Vec<String> = Vec::new();

            for doc in self.corpus.documents() {
                let bm25_raw = self.bm25.score(&query.terms, doc);
                let bayesian_p = self.bayesian.score(&query.terms, doc);
                let vector_p = self.vector.score(embedding, doc);
                let hybrid_or = self.hybrid.score_or(&query.terms, embedding, doc);
                let hybrid_and = self.hybrid.score_and(&query.terms, embedding, doc);
                let naive = self.hybrid.naive_sum(&[bm25_raw, vector_p]);

                let mut scores = HashMap::new();
                scores.insert("bm25".to_string(), bm25_raw);
                scores.insert("bayesian".to_string(), bayesian_p);
                scores.insert("vector".to_string(), vector_p);
                scores.insert("hybrid_or".to_string(), hybrid_or);
                scores.insert("hybrid_and".to_string(), hybrid_and);
                scores.insert("naive".to_string(), naive);
                scores.insert("rrf".to_string(), 0.0);
                doc_scores.push(scores);
                id_list.push(doc.id.clone());
            }

            let mut bm25_ranked: Vec<(String, f64)> = id_list
                .iter()
                .zip(doc_scores.iter())
                .map(|(id, scores)| (id.clone(), scores["bm25"]))
                .collect();
            bm25_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut vector_ranked: Vec<(String, f64)> = id_list
                .iter()
                .zip(doc_scores.iter())
                .map(|(id, scores)| (id.clone(), scores["vector"]))
                .collect();
            vector_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let bm25_rank: HashMap<String, usize> = bm25_ranked
                .iter()
                .enumerate()
                .map(|(i, (id, _))| (id.clone(), i + 1))
                .collect();
            let vector_rank: HashMap<String, usize> = vector_ranked
                .iter()
                .enumerate()
                .map(|(i, (id, _))| (id.clone(), i + 1))
                .collect();

            for (idx, id) in id_list.iter().enumerate() {
                let rrf = self
                    .hybrid
                    .rrf_score(&[bm25_rank[id], vector_rank[id]], 60);
                doc_scores[idx].insert("rrf".to_string(), rrf);
            }

            let mut top5: HashMap<String, Vec<String>> = HashMap::new();
            for method in ["bm25", "bayesian", "hybrid_or", "hybrid_and", "naive", "rrf"] {
                let mut ranked: Vec<(String, f64)> = id_list
                    .iter()
                    .zip(doc_scores.iter())
                    .map(|(id, scores)| (id.clone(), scores[method]))
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let ids = ranked.into_iter().take(5).map(|(id, _)| id).collect();
                top5.insert(method.to_string(), ids);
            }

            results_table.push((query.text.clone(), top5));
        }

        let mut detail_lines: Vec<String> = Vec::new();
        for (query_text, top5) in &results_table {
            detail_lines.push(format!("query='{}':", query_text));
            for method in ["bm25", "bayesian", "hybrid_or", "naive", "rrf"] {
                let ids = top5.get(method).cloned().unwrap_or_default();
                detail_lines.push(format!("  {} top5: {:?}", method, ids));
            }
        }

        (true, detail_lines.join("\n"))
    }

    fn exp8_numerical_stability(&self) -> (bool, String) {
        let mut passed = true;
        let mut tests: Vec<String> = Vec::new();

        let extreme_probs = [1e-15, 1e-10, 1e-5, 0.001, 0.5, 0.999, 1.0 - 1e-10];
        for p in extreme_probs {
            let and_result = self.hybrid.probabilistic_and(&[p, p]);
            if and_result < -EPSILON || and_result > 1.0 + EPSILON {
                passed = false;
            }
            if and_result.is_nan() || and_result.is_infinite() {
                passed = false;
            }
            tests.push(format!("AND({:.2e}, {:.2e})={:.2e}", p, p, and_result));

            let or_result = self.hybrid.probabilistic_or(&[p, p]);
            if or_result < -EPSILON || or_result > 1.0 + EPSILON {
                passed = false;
            }
            if or_result.is_nan() || or_result.is_infinite() {
                passed = false;
            }
            tests.push(format!("OR({:.2e}, {:.2e})={:.2e}", p, p, or_result));
        }

        for x in [-700.0, -100.0, -1.0, 0.0, 1.0, 100.0, 700.0] {
            let s = sigmoid(x);
            if s < 0.0 || s > 1.0 || s.is_nan() || s.is_infinite() {
                passed = false;
            }
            tests.push(format!("sigmoid({:.0})={:.6}", x, s));
        }

        for p in [0.0, 1e-300, 1e-15, 0.5, 1.0] {
            let result = safe_log(p);
            if result.is_nan() || result.is_infinite() {
                passed = false;
            }
            tests.push(format!("safe_log({:.2e})={:.2}", p, result));
        }

        let preview = tests.iter().take(10).cloned().collect::<Vec<_>>().join("; ");
        let detail = format!("{} ... ({} total tests)", preview, tests.len());
        (passed, detail)
    }

    fn exp9_parameter_learning(&self) -> (bool, String) {
        let query = &self.queries[0];
        let relevant_ids: std::collections::HashSet<String> =
            query.relevant.iter().cloned().collect();

        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for doc in self.corpus.documents() {
            let raw_score = self.bm25.score(&query.terms, doc);
            scores.push(raw_score);
            labels.push(if relevant_ids.contains(&doc.id) { 1.0 } else { 0.0 });
        }

        let learner = ParameterLearner::new(0.1, 500, 1e-8);
        let result = learner.learn(&scores, &labels);

        let loss_history = &result.loss_history;
        let loss_decreased = loss_history.last().unwrap_or(&0.0) < &loss_history[0];
        let alpha_positive = result.alpha > 0.0;
        let decreasing_steps = loss_history
            .windows(2)
            .filter(|pair| pair[1] <= pair[0] + EPSILON)
            .count();
        let mostly_decreasing = decreasing_steps >= (loss_history.len() - 1) * 8 / 10;

        let passed = loss_decreased && alpha_positive && mostly_decreasing;
        let detail = format!(
            "alpha={:.4}, beta={:.4}, initial_loss={:.4}, final_loss={:.4}, decreasing_steps={}/{}, converged={}",
            result.alpha,
            result.beta,
            loss_history[0],
            loss_history.last().unwrap_or(&0.0),
            decreasing_steps,
            loss_history.len().saturating_sub(1),
            result.converged,
        );

        (passed, detail)
    }

    fn exp10_conjunction_disjunction(&self) -> (bool, String) {
        let mut passed = true;
        let mut tests = 0usize;
        let mut violations: Vec<String> = Vec::new();

        let test_probs = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99];
        for p1 in test_probs {
            for p2 in test_probs {
                let probs = [p1, p2];
                let and_result = self.hybrid.probabilistic_and(&probs);
                let or_result = self.hybrid.probabilistic_or(&probs);
                tests += 1;

                // AND <= OR always
                if and_result > or_result + EPSILON {
                    passed = false;
                    violations.push(format!(
                        "AND({:.2},{:.2})={:.6} > OR={:.6}",
                        p1, p2, and_result, or_result
                    ));
                }

                // Agreement amplification: both > 0.5 => AND > geo_mean
                if p1 > 0.5 && p2 > 0.5 {
                    let geo_mean = (p1 * p2).sqrt();
                    if and_result < geo_mean - EPSILON {
                        passed = false;
                        violations.push(format!(
                            "AND({:.2},{:.2})={:.6} < geo_mean={:.6}",
                            p1, p2, and_result, geo_mean
                        ));
                    }
                }

                // Irrelevance preservation: both < 0.5 => AND < 0.5
                if p1 < 0.5 && p2 < 0.5 {
                    if and_result > 0.5 + EPSILON {
                        passed = false;
                        violations.push(format!(
                            "AND({:.2},{:.2})={:.6} > 0.5",
                            p1, p2, and_result
                        ));
                    }
                }

                // OR >= max(p_i) always
                let max_p = p1.max(p2);
                if or_result < max_p - EPSILON {
                    passed = false;
                    violations.push(format!(
                        "OR({:.2},{:.2})={:.6} < max={:.2}",
                        p1, p2, or_result, max_p
                    ));
                }
            }
        }

        // Identity for single signal: AND(p) = p
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let and_single = self.hybrid.probabilistic_and(&[p]);
            tests += 1;
            if (and_single - p).abs() > EPSILON {
                passed = false;
                violations.push(format!(
                    "identity: AND({:.1})={:.6} != {:.1}",
                    p, and_single, p
                ));
            }
        }

        // 3-signal tests
        for p1 in [0.1, 0.5, 0.9] {
            for p2 in [0.2, 0.6] {
                for p3 in [0.3, 0.8] {
                    let probs = [p1, p2, p3];
                    let and_result = self.hybrid.probabilistic_and(&probs);
                    let or_result = self.hybrid.probabilistic_or(&probs);
                    tests += 1;

                    // AND <= OR
                    if and_result > or_result + EPSILON {
                        passed = false;
                    }

                    // OR >= max
                    if or_result < probs.iter().copied().fold(f64::NEG_INFINITY, f64::max) - EPSILON {
                        passed = false;
                    }

                    // All above 0.5 => AND > geo_mean
                    if probs.iter().all(|&p| p > 0.5) {
                        let geo_mean = (p1 * p2 * p3).powf(1.0 / 3.0);
                        if and_result < geo_mean - EPSILON {
                            passed = false;
                        }
                    }
                }
            }
        }

        let mut detail = format!("tests={}", tests);
        if !violations.is_empty() {
            let preview = violations.into_iter().take(3).collect::<Vec<_>>().join("; ");
            detail.push_str(", violations: ");
            detail.push_str(&preview);
        }

        (passed, detail)
    }
}
