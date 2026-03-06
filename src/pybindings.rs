use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::bayesian_scorer::BayesianBM25Scorer;
use crate::bm25_scorer::BM25Scorer;
use crate::corpus::{Corpus as CoreCorpus, Document};
use crate::defaults::{build_default_corpus, build_default_queries};
use crate::experiments::{ExperimentRunner, Query};
use crate::fusion;
use crate::hybrid_scorer::HybridScorer;
use crate::parameter_learner::{ParameterLearner, ParameterLearnerResult};
use crate::tokenizer::Tokenizer;
use crate::vector_scorer::VectorScorer;
use crate::probability::BayesianProbabilityTransform;
use crate::learnable_weights::LearnableLogOddsWeights;
use crate::attention_weights::AttentionLogOddsWeights;
use crate::metrics;
use crate::debug::{FusionDebugger, BM25SignalTrace, VectorSignalTrace, FusionTrace, DocumentTrace, ComparisonResult, NotTrace};

fn parse_gating(gating: Option<&str>) -> PyResult<fusion::Gating> {
    match gating {
        None | Some("none") => Ok(fusion::Gating::NoGating),
        Some("relu") => Ok(fusion::Gating::Relu),
        Some("swish") => Ok(fusion::Gating::Swish),
        Some(other) => Err(PyValueError::new_err(format!(
            "gating must be 'none', 'relu', or 'swish', got '{}'",
            other
        ))),
    }
}

fn parse_training_mode(mode: Option<&str>) -> PyResult<crate::probability::TrainingMode> {
    match mode {
        None | Some("balanced") => Ok(crate::probability::TrainingMode::Balanced),
        Some("prior_aware") => Ok(crate::probability::TrainingMode::PriorAware),
        Some("prior_free") => Ok(crate::probability::TrainingMode::PriorFree),
        Some(other) => Err(PyValueError::new_err(format!(
            "mode must be 'balanced', 'prior_aware', or 'prior_free', got '{}'",
            other
        ))),
    }
}

#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new() -> Self {
        Self {
            inner: Tokenizer::new(),
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        self.inner.tokenize(text)
    }
}

#[pyclass(name = "Document")]
pub struct PyDocument {
    inner: Document,
}

#[pymethods]
impl PyDocument {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn embedding(&self) -> Vec<f64> {
        self.inner.embedding.clone()
    }

    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens.clone()
    }

    #[getter]
    fn length(&self) -> usize {
        self.inner.length
    }

    #[getter]
    fn term_freq(&self) -> HashMap<String, usize> {
        self.inner.term_freq.clone()
    }
}

#[pyclass(unsendable, name = "Corpus")]
pub struct PyCorpus {
    inner: RefCell<Option<CoreCorpus>>,
    shared: RefCell<Option<Rc<CoreCorpus>>>,
    built: Cell<bool>,
}

impl PyCorpus {
    fn shared_corpus(&self) -> PyResult<Rc<CoreCorpus>> {
        if let Some(shared) = self.shared.borrow().as_ref() {
            return Ok(Rc::clone(shared));
        }

        if !self.built.get() {
            return Err(PyRuntimeError::new_err(
                "Corpus.build_index() must be called before creating scorers",
            ));
        }

        let mut inner = self.inner.borrow_mut();
        let Some(corpus) = inner.take() else {
            return Err(PyRuntimeError::new_err(
                "Corpus is already frozen and cannot be shared",
            ));
        };

        let rc = Rc::new(corpus);
        *self.shared.borrow_mut() = Some(Rc::clone(&rc));
        Ok(rc)
    }

    fn with_corpus<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&CoreCorpus) -> PyResult<R>,
    {
        if let Some(shared) = self.shared.borrow().as_ref() {
            return f(shared);
        }
        if let Some(inner) = self.inner.borrow().as_ref() {
            return f(inner);
        }
        Err(PyRuntimeError::new_err("Corpus is unavailable"))
    }
}

#[pymethods]
impl PyCorpus {
    #[new]
    #[pyo3(signature = (_tokenizer=None))]
    fn new(_tokenizer: Option<&PyTokenizer>) -> Self {
        let core = CoreCorpus::new(Tokenizer::new());
        Self {
            inner: RefCell::new(Some(core)),
            shared: RefCell::new(None),
            built: Cell::new(false),
        }
    }

    fn add_document(&self, doc_id: &str, text: &str, embedding: Vec<f64>) -> PyResult<()> {
        if self.shared.borrow().is_some() {
            return Err(PyRuntimeError::new_err(
                "Corpus is frozen and cannot be modified",
            ));
        }
        let mut inner = self.inner.borrow_mut();
        let Some(corpus) = inner.as_mut() else {
            return Err(PyRuntimeError::new_err("Corpus is unavailable"));
        };
        corpus.add_document(doc_id, text, embedding);
        Ok(())
    }

    fn build_index(&self) -> PyResult<()> {
        if self.shared.borrow().is_some() {
            return Err(PyRuntimeError::new_err(
                "Corpus is frozen and cannot be rebuilt",
            ));
        }
        let mut inner = self.inner.borrow_mut();
        let Some(corpus) = inner.as_mut() else {
            return Err(PyRuntimeError::new_err("Corpus is unavailable"));
        };
        corpus.build_index();
        self.built.set(true);
        Ok(())
    }

    fn get_document(&self, doc_id: &str) -> PyResult<PyDocument> {
        self.with_corpus(|corpus| {
            let doc = corpus
                .get_document(doc_id)
                .cloned()
                .ok_or_else(|| PyValueError::new_err("Document not found"))?;
            Ok(PyDocument { inner: doc })
        })
    }

    fn documents(&self) -> PyResult<Vec<PyDocument>> {
        self.with_corpus(|corpus| {
            Ok(corpus
                .documents()
                .iter()
                .cloned()
                .map(|doc| PyDocument { inner: doc })
                .collect())
        })
    }

    #[getter]
    fn n(&self) -> PyResult<usize> {
        self.with_corpus(|corpus| Ok(corpus.n))
    }

    #[getter]
    fn avgdl(&self) -> PyResult<f64> {
        self.with_corpus(|corpus| Ok(corpus.avgdl))
    }

    #[getter]
    fn df(&self) -> PyResult<HashMap<String, usize>> {
        self.with_corpus(|corpus| Ok(corpus.df.clone()))
    }
}

#[pyclass(unsendable, name = "BM25Scorer")]
pub struct PyBM25Scorer {
    inner: Rc<BM25Scorer>,
}

#[pymethods]
impl PyBM25Scorer {
    #[new]
    #[pyo3(signature = (corpus, k1=None, b=None))]
    fn new(corpus: &PyCorpus, k1: Option<f64>, b: Option<f64>) -> PyResult<Self> {
        let corpus = corpus.shared_corpus()?;
        Ok(Self {
            inner: Rc::new(BM25Scorer::new(
                corpus,
                k1.unwrap_or(1.2),
                b.unwrap_or(0.75),
            )),
        })
    }

    fn idf(&self, term: &str) -> f64 {
        self.inner.idf(term)
    }

    fn score_term_standard(&self, term: &str, doc: &PyDocument) -> f64 {
        self.inner.score_term_standard(term, &doc.inner)
    }

    fn score_term_rewritten(&self, term: &str, doc: &PyDocument) -> f64 {
        self.inner.score_term_rewritten(term, &doc.inner)
    }

    fn score(&self, query_terms: Vec<String>, doc: &PyDocument) -> f64 {
        self.inner.score(&query_terms, &doc.inner)
    }

    fn upper_bound(&self, term: &str) -> f64 {
        self.inner.upper_bound(term)
    }
}

#[pyclass(unsendable, name = "BayesianBM25Scorer")]
pub struct PyBayesianBM25Scorer {
    inner: Rc<BayesianBM25Scorer>,
}

#[pymethods]
impl PyBayesianBM25Scorer {
    #[new]
    #[pyo3(signature = (bm25, alpha=None, beta=None, base_rate=None))]
    fn new(bm25: &PyBM25Scorer, alpha: Option<f64>, beta: Option<f64>, base_rate: Option<f64>) -> Self {
        Self {
            inner: Rc::new(BayesianBM25Scorer::new(
                Rc::clone(&bm25.inner),
                alpha.unwrap_or(1.0),
                beta.unwrap_or(0.5),
                base_rate,
            )),
        }
    }

    fn likelihood(&self, score: f64) -> f64 {
        self.inner.likelihood(score)
    }

    fn tf_prior(&self, tf: usize) -> f64 {
        self.inner.tf_prior(tf)
    }

    fn norm_prior(&self, doc_length: usize, avg_doc_length: f64) -> f64 {
        self.inner.norm_prior(doc_length, avg_doc_length)
    }

    fn composite_prior(&self, tf: usize, doc_length: usize, avg_doc_length: f64) -> f64 {
        self.inner.composite_prior(tf, doc_length, avg_doc_length)
    }

    fn posterior(&self, score: f64, prior: f64) -> f64 {
        self.inner.posterior(score, prior)
    }

    fn score_term(&self, term: &str, doc: &PyDocument) -> f64 {
        self.inner.score_term(term, &doc.inner)
    }

    fn score(&self, query_terms: Vec<String>, doc: &PyDocument) -> f64 {
        self.inner.score(&query_terms, &doc.inner)
    }

    #[getter]
    fn base_rate(&self) -> Option<f64> {
        self.inner.base_rate()
    }
}

#[pyclass(unsendable, name = "VectorScorer")]
pub struct PyVectorScorer {
    inner: Rc<VectorScorer>,
}

#[pymethods]
impl PyVectorScorer {
    #[new]
    fn new() -> Self {
        Self {
            inner: Rc::new(VectorScorer::new()),
        }
    }

    fn score_to_probability(&self, sim: f64) -> f64 {
        self.inner.score_to_probability(sim)
    }

    fn score(&self, query_embedding: Vec<f64>, doc: &PyDocument) -> f64 {
        self.inner.score(&query_embedding, &doc.inner)
    }
}

#[pyclass(unsendable, name = "HybridScorer")]
pub struct PyHybridScorer {
    inner: HybridScorer,
}

#[pymethods]
impl PyHybridScorer {
    #[new]
    #[pyo3(signature = (bayesian, vector, alpha=None))]
    fn new(bayesian: &PyBayesianBM25Scorer, vector: &PyVectorScorer, alpha: Option<f64>) -> Self {
        Self {
            inner: HybridScorer::new(
                Rc::clone(&bayesian.inner),
                Rc::clone(&vector.inner),
                alpha.unwrap_or(0.5),
            ),
        }
    }

    fn probabilistic_and(&self, probs: Vec<f64>) -> f64 {
        self.inner.probabilistic_and(&probs)
    }

    fn probabilistic_or(&self, probs: Vec<f64>) -> f64 {
        self.inner.probabilistic_or(&probs)
    }

    fn score_and(&self, query_terms: Vec<String>, query_embedding: Vec<f64>, doc: &PyDocument) -> f64 {
        self.inner.score_and(&query_terms, &query_embedding, &doc.inner)
    }

    fn score_or(&self, query_terms: Vec<String>, query_embedding: Vec<f64>, doc: &PyDocument) -> f64 {
        self.inner.score_or(&query_terms, &query_embedding, &doc.inner)
    }

    fn naive_sum(&self, scores: Vec<f64>) -> f64 {
        self.inner.naive_sum(&scores)
    }

    fn rrf_score(&self, ranks: Vec<usize>, k: Option<usize>) -> f64 {
        self.inner.rrf_score(&ranks, k.unwrap_or(60))
    }
}

#[pyclass(name = "ParameterLearner")]
pub struct PyParameterLearner {
    inner: ParameterLearner,
}

#[pymethods]
impl PyParameterLearner {
    #[new]
    #[pyo3(signature = (learning_rate=None, max_iterations=None, tolerance=None))]
    fn new(learning_rate: Option<f64>, max_iterations: Option<usize>, tolerance: Option<f64>) -> Self {
        Self {
            inner: ParameterLearner::new(
                learning_rate.unwrap_or(0.01),
                max_iterations.unwrap_or(1000),
                tolerance.unwrap_or(1e-6),
            ),
        }
    }

    fn cross_entropy_loss(&self, scores: Vec<f64>, labels: Vec<f64>, alpha: f64, beta: f64) -> PyResult<f64> {
        if scores.len() != labels.len() {
            return Err(PyValueError::new_err("scores and labels must have same length"));
        }
        Ok(self.inner.cross_entropy_loss(&scores, &labels, alpha, beta))
    }

    fn learn(&self, scores: Vec<f64>, labels: Vec<f64>) -> PyResult<PyParameterLearnerResult> {
        if scores.len() != labels.len() {
            return Err(PyValueError::new_err("scores and labels must have same length"));
        }
        let result = self.inner.learn(&scores, &labels);
        Ok(PyParameterLearnerResult::from_result(&result))
    }
}

#[pyclass(name = "ParameterLearnerResult")]
pub struct PyParameterLearnerResult {
    #[pyo3(get)]
    alpha: f64,
    #[pyo3(get)]
    beta: f64,
    #[pyo3(get)]
    loss_history: Vec<f64>,
    #[pyo3(get)]
    converged: bool,
}

impl PyParameterLearnerResult {
    fn from_result(result: &ParameterLearnerResult) -> Self {
        Self {
            alpha: result.alpha,
            beta: result.beta,
            loss_history: result.loss_history.clone(),
            converged: result.converged,
        }
    }
}

#[pyclass(name = "Query")]
pub struct PyQuery {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    terms: Vec<String>,
    #[pyo3(get)]
    embedding: Option<Vec<f64>>,
    #[pyo3(get)]
    relevant: Vec<String>,
}

#[pymethods]
impl PyQuery {
    #[new]
    #[pyo3(signature = (text, terms, embedding=None, relevant=None))]
    fn new(
        text: &str,
        terms: Vec<String>,
        embedding: Option<Vec<f64>>,
        relevant: Option<Vec<String>>,
    ) -> Self {
        Self {
            text: text.to_string(),
            terms,
            embedding,
            relevant: relevant.unwrap_or_default(),
        }
    }
}

impl PyQuery {
    fn clone_inner(&self) -> Query {
        Query {
            text: self.text.clone(),
            terms: self.terms.clone(),
            embedding: self.embedding.clone(),
            relevant: self.relevant.clone(),
        }
    }
}

#[pyclass(name = "ExperimentResult")]
pub struct PyExperimentResult {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    passed: bool,
    #[pyo3(get)]
    details: String,
}

#[pyclass(unsendable, name = "ExperimentRunner")]
pub struct PyExperimentRunner {
    inner: ExperimentRunner,
}

#[pymethods]
impl PyExperimentRunner {
    #[new]
    #[pyo3(signature = (corpus, queries, k1=None, b=None))]
    fn new(corpus: &PyCorpus, queries: Vec<Py<PyQuery>>, k1: Option<f64>, b: Option<f64>) -> PyResult<Self> {
        let corpus = corpus.shared_corpus()?;
        let mut query_list = Vec::with_capacity(queries.len());
        Python::attach(|py| {
            for q in &queries {
                let q_ref = q.borrow(py);
                query_list.push(q_ref.clone_inner());
            }
        });

        Ok(Self {
            inner: ExperimentRunner::new(
                corpus,
                query_list,
                k1.unwrap_or(1.2),
                b.unwrap_or(0.75),
            ),
        })
    }

    fn run_all(&self) -> Vec<PyExperimentResult> {
        self.inner
            .run_all()
            .into_iter()
            .map(|(name, passed, details)| PyExperimentResult { name, passed, details })
            .collect()
    }
}

#[pyfunction(name = "build_default_corpus")]
fn build_default_corpus_py() -> PyCorpus {
    let core = build_default_corpus();
    PyCorpus {
        inner: RefCell::new(None),
        shared: RefCell::new(Some(Rc::new(core))),
        built: Cell::new(true),
    }
}

#[pyfunction(name = "build_default_queries")]
fn build_default_queries_py(py: Python) -> PyResult<Vec<Py<PyQuery>>> {
    let mut out = Vec::new();
    for q in build_default_queries() {
        out.push(Py::new(
            py,
            PyQuery {
                text: q.text,
                terms: q.terms,
                embedding: q.embedding,
                relevant: q.relevant,
            },
        )?);
    }
    Ok(out)
}

#[pyfunction(name = "prob_not")]
fn prob_not_py(prob: f64) -> f64 {
    fusion::prob_not(prob)
}

#[pyfunction(name = "prob_and")]
fn prob_and_py(probs: Vec<f64>) -> f64 {
    fusion::prob_and(&probs)
}

#[pyfunction(name = "prob_or")]
fn prob_or_py(probs: Vec<f64>) -> f64 {
    fusion::prob_or(&probs)
}

#[pyfunction(name = "cosine_to_probability")]
fn cosine_to_probability_py(score: f64) -> f64 {
    fusion::cosine_to_probability(score)
}

#[pyfunction(name = "log_odds_conjunction")]
#[pyo3(signature = (probs, alpha=None, weights=None, gating=None))]
fn log_odds_conjunction_py(
    probs: Vec<f64>,
    alpha: Option<f64>,
    weights: Option<Vec<f64>>,
    gating: Option<&str>,
) -> PyResult<f64> {
    if let Some(ref w) = weights {
        if w.len() != probs.len() {
            return Err(PyValueError::new_err(
                "weights length must match probs length",
            ));
        }
        if !w.iter().all(|&wi| wi >= 0.0) {
            return Err(PyValueError::new_err("all weights must be non-negative"));
        }
        if (w.iter().sum::<f64>() - 1.0).abs() >= 1e-6 {
            return Err(PyValueError::new_err("weights must sum to 1.0"));
        }
    }
    let g = parse_gating(gating)?;
    Ok(fusion::log_odds_conjunction(
        &probs,
        alpha,
        weights.as_deref(),
        g,
    ))
}

#[pyfunction(name = "balanced_log_odds_fusion")]
#[pyo3(signature = (sparse_probs, dense_similarities, weight=None))]
fn balanced_log_odds_fusion_py(
    sparse_probs: Vec<f64>,
    dense_similarities: Vec<f64>,
    weight: Option<f64>,
) -> PyResult<Vec<f64>> {
    if sparse_probs.len() != dense_similarities.len() {
        return Err(PyValueError::new_err(
            "sparse_probs and dense_similarities must have the same length",
        ));
    }
    Ok(fusion::balanced_log_odds_fusion(
        &sparse_probs,
        &dense_similarities,
        weight.unwrap_or(0.5),
    ))
}

// ---------------------------------------------------------------------------
// BayesianProbabilityTransform
// ---------------------------------------------------------------------------

#[pyclass(unsendable, name = "BayesianProbabilityTransform")]
pub struct PyBayesianProbabilityTransform {
    inner: RefCell<BayesianProbabilityTransform>,
}

#[pymethods]
impl PyBayesianProbabilityTransform {
    #[new]
    #[pyo3(signature = (alpha=None, beta=None, base_rate=None))]
    fn new(alpha: Option<f64>, beta: Option<f64>, base_rate: Option<f64>) -> PyResult<Self> {
        if let Some(br) = base_rate {
            if br <= 0.0 || br >= 1.0 {
                return Err(PyValueError::new_err(format!(
                    "base_rate must be in (0, 1), got {}",
                    br
                )));
            }
        }
        Ok(Self {
            inner: RefCell::new(BayesianProbabilityTransform::new(
                alpha.unwrap_or(1.0),
                beta.unwrap_or(0.0),
                base_rate,
            )),
        })
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.borrow().alpha
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.inner.borrow().beta
    }

    #[getter]
    fn base_rate(&self) -> Option<f64> {
        self.inner.borrow().base_rate
    }

    #[getter]
    fn averaged_alpha(&self) -> f64 {
        self.inner.borrow().averaged_alpha()
    }

    #[getter]
    fn averaged_beta(&self) -> f64 {
        self.inner.borrow().averaged_beta()
    }

    fn likelihood(&self, score: f64) -> f64 {
        self.inner.borrow().likelihood(score)
    }

    #[staticmethod]
    fn tf_prior(tf: f64) -> f64 {
        BayesianProbabilityTransform::tf_prior(tf)
    }

    #[staticmethod]
    fn norm_prior(doc_len_ratio: f64) -> f64 {
        BayesianProbabilityTransform::norm_prior(doc_len_ratio)
    }

    #[staticmethod]
    fn composite_prior(tf: f64, doc_len_ratio: f64) -> f64 {
        BayesianProbabilityTransform::composite_prior(tf, doc_len_ratio)
    }

    #[staticmethod]
    #[pyo3(signature = (likelihood_val, prior, base_rate=None))]
    fn posterior(likelihood_val: f64, prior: f64, base_rate: Option<f64>) -> f64 {
        BayesianProbabilityTransform::posterior(likelihood_val, prior, base_rate)
    }

    fn score_to_probability(&self, score: f64, tf: f64, doc_len_ratio: f64) -> f64 {
        self.inner.borrow().score_to_probability(score, tf, doc_len_ratio)
    }

    #[pyo3(signature = (bm25_upper_bound, p_max=None))]
    fn wand_upper_bound(&self, bm25_upper_bound: f64, p_max: Option<f64>) -> f64 {
        self.inner.borrow().wand_upper_bound(bm25_upper_bound, p_max.unwrap_or(0.9))
    }

    #[pyo3(signature = (scores, labels, learning_rate=None, max_iterations=None, tolerance=None, mode=None, tfs=None, doc_len_ratios=None))]
    fn fit(
        &self,
        scores: Vec<f64>,
        labels: Vec<f64>,
        learning_rate: Option<f64>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
        mode: Option<&str>,
        tfs: Option<Vec<f64>>,
        doc_len_ratios: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let m = parse_training_mode(mode)?;
        self.inner.borrow_mut().fit(
            &scores,
            &labels,
            learning_rate.unwrap_or(0.01),
            max_iterations.unwrap_or(1000),
            tolerance.unwrap_or(1e-6),
            m,
            tfs.as_deref(),
            doc_len_ratios.as_deref(),
        );
        Ok(())
    }

    #[pyo3(signature = (score, label, learning_rate=None, momentum=None, decay_tau=None, max_grad_norm=None, avg_decay=None, mode=None, tf=None, doc_len_ratio=None))]
    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        score: Vec<f64>,
        label: Vec<f64>,
        learning_rate: Option<f64>,
        momentum: Option<f64>,
        decay_tau: Option<f64>,
        max_grad_norm: Option<f64>,
        avg_decay: Option<f64>,
        mode: Option<&str>,
        tf: Option<Vec<f64>>,
        doc_len_ratio: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let m = if mode.is_some() {
            Some(parse_training_mode(mode)?)
        } else {
            None
        };
        self.inner.borrow_mut().update(
            &score,
            &label,
            learning_rate.unwrap_or(0.01),
            momentum.unwrap_or(0.9),
            decay_tau.unwrap_or(1000.0),
            max_grad_norm.unwrap_or(1.0),
            avg_decay.unwrap_or(0.995),
            m,
            tf.as_deref(),
            doc_len_ratio.as_deref(),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LearnableLogOddsWeights
// ---------------------------------------------------------------------------

#[pyclass(unsendable, name = "LearnableLogOddsWeights")]
pub struct PyLearnableLogOddsWeights {
    inner: RefCell<LearnableLogOddsWeights>,
}

#[pymethods]
impl PyLearnableLogOddsWeights {
    #[new]
    #[pyo3(signature = (n_signals, alpha=None))]
    fn new(n_signals: usize, alpha: Option<f64>) -> PyResult<Self> {
        if n_signals < 1 {
            return Err(PyValueError::new_err(format!(
                "n_signals must be >= 1, got {}",
                n_signals
            )));
        }
        Ok(Self {
            inner: RefCell::new(LearnableLogOddsWeights::new(
                n_signals,
                alpha.unwrap_or(0.0),
            )),
        })
    }

    #[getter]
    fn n_signals(&self) -> usize {
        self.inner.borrow().n_signals()
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.borrow().alpha()
    }

    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.borrow().weights()
    }

    #[getter]
    fn averaged_weights(&self) -> Vec<f64> {
        self.inner.borrow().averaged_weights()
    }

    #[pyo3(signature = (probs, use_averaged=None))]
    fn combine(&self, probs: Vec<f64>, use_averaged: Option<bool>) -> f64 {
        self.inner.borrow().combine(&probs, use_averaged.unwrap_or(false))
    }

    #[pyo3(signature = (probs, labels, learning_rate=None, max_iterations=None, tolerance=None))]
    fn fit(
        &self,
        probs: Vec<Vec<f64>>,
        labels: Vec<f64>,
        learning_rate: Option<f64>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> PyResult<()> {
        self.inner.borrow_mut().fit(
            &probs,
            &labels,
            learning_rate.unwrap_or(0.01),
            max_iterations.unwrap_or(1000),
            tolerance.unwrap_or(1e-6),
        );
        Ok(())
    }

    #[pyo3(signature = (probs, label, learning_rate=None, momentum=None, decay_tau=None, max_grad_norm=None, avg_decay=None))]
    fn update(
        &self,
        probs: Vec<Vec<f64>>,
        label: Vec<f64>,
        learning_rate: Option<f64>,
        momentum: Option<f64>,
        decay_tau: Option<f64>,
        max_grad_norm: Option<f64>,
        avg_decay: Option<f64>,
    ) -> PyResult<()> {
        self.inner.borrow_mut().update(
            &probs,
            &label,
            learning_rate.unwrap_or(0.01),
            momentum.unwrap_or(0.9),
            decay_tau.unwrap_or(1000.0),
            max_grad_norm.unwrap_or(1.0),
            avg_decay.unwrap_or(0.995),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AttentionLogOddsWeights
// ---------------------------------------------------------------------------

#[pyclass(unsendable, name = "AttentionLogOddsWeights")]
pub struct PyAttentionLogOddsWeights {
    inner: RefCell<AttentionLogOddsWeights>,
}

#[pymethods]
impl PyAttentionLogOddsWeights {
    #[new]
    #[pyo3(signature = (n_signals, n_query_features, alpha=None, normalize=None))]
    fn new(
        n_signals: usize,
        n_query_features: usize,
        alpha: Option<f64>,
        normalize: Option<bool>,
    ) -> PyResult<Self> {
        if n_signals < 1 {
            return Err(PyValueError::new_err(format!(
                "n_signals must be >= 1, got {}",
                n_signals
            )));
        }
        if n_query_features < 1 {
            return Err(PyValueError::new_err(format!(
                "n_query_features must be >= 1, got {}",
                n_query_features
            )));
        }
        Ok(Self {
            inner: RefCell::new(AttentionLogOddsWeights::new(
                n_signals,
                n_query_features,
                alpha.unwrap_or(0.5),
                normalize.unwrap_or(false),
            )),
        })
    }

    #[getter]
    fn n_signals(&self) -> usize {
        self.inner.borrow().n_signals()
    }

    #[getter]
    fn n_query_features(&self) -> usize {
        self.inner.borrow().n_query_features()
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.borrow().alpha()
    }

    #[getter]
    fn normalize(&self) -> bool {
        self.inner.borrow().normalize()
    }

    #[getter]
    fn weights_matrix(&self) -> Vec<f64> {
        self.inner.borrow().weights_matrix()
    }

    #[pyo3(signature = (probs, m, query_features, m_q, use_averaged=None))]
    fn combine(
        &self,
        probs: Vec<f64>,
        m: usize,
        query_features: Vec<f64>,
        m_q: usize,
        use_averaged: Option<bool>,
    ) -> Vec<f64> {
        self.inner.borrow().combine(
            &probs,
            m,
            &query_features,
            m_q,
            use_averaged.unwrap_or(false),
        )
    }

    #[pyo3(signature = (probs, labels, query_features, m, query_ids=None, learning_rate=None, max_iterations=None, tolerance=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        probs: Vec<f64>,
        labels: Vec<f64>,
        query_features: Vec<f64>,
        m: usize,
        query_ids: Option<Vec<usize>>,
        learning_rate: Option<f64>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> PyResult<()> {
        self.inner.borrow_mut().fit(
            &probs,
            &labels,
            &query_features,
            m,
            query_ids.as_deref(),
            learning_rate.unwrap_or(0.01),
            max_iterations.unwrap_or(1000),
            tolerance.unwrap_or(1e-6),
        );
        Ok(())
    }

    #[pyo3(signature = (probs, labels, query_features, m, learning_rate=None, momentum=None, decay_tau=None, max_grad_norm=None, avg_decay=None))]
    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        probs: Vec<f64>,
        labels: Vec<f64>,
        query_features: Vec<f64>,
        m: usize,
        learning_rate: Option<f64>,
        momentum: Option<f64>,
        decay_tau: Option<f64>,
        max_grad_norm: Option<f64>,
        avg_decay: Option<f64>,
    ) -> PyResult<()> {
        self.inner.borrow_mut().update(
            &probs,
            &labels,
            &query_features,
            m,
            learning_rate.unwrap_or(0.01),
            momentum.unwrap_or(0.9),
            decay_tau.unwrap_or(1000.0),
            max_grad_norm.unwrap_or(1.0),
            avg_decay.unwrap_or(0.995),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Calibration Metrics
// ---------------------------------------------------------------------------

#[pyclass(name = "CalibrationReport")]
pub struct PyCalibrationReport {
    #[pyo3(get)]
    ece: f64,
    #[pyo3(get)]
    brier: f64,
    #[pyo3(get)]
    reliability: Vec<(f64, f64, usize)>,
    #[pyo3(get)]
    n_samples: usize,
    #[pyo3(get)]
    n_bins: usize,
}

#[pymethods]
impl PyCalibrationReport {
    fn summary(&self) -> String {
        let report = metrics::CalibrationReport {
            ece: self.ece,
            brier: self.brier,
            reliability: self.reliability.clone(),
            n_samples: self.n_samples,
            n_bins: self.n_bins,
        };
        report.summary()
    }
}

#[pyfunction(name = "expected_calibration_error")]
#[pyo3(signature = (probabilities, labels, n_bins=None))]
fn expected_calibration_error_py(
    probabilities: Vec<f64>,
    labels: Vec<f64>,
    n_bins: Option<usize>,
) -> f64 {
    metrics::expected_calibration_error(&probabilities, &labels, n_bins.unwrap_or(10))
}

#[pyfunction(name = "brier_score")]
fn brier_score_py(probabilities: Vec<f64>, labels: Vec<f64>) -> f64 {
    metrics::brier_score(&probabilities, &labels)
}

#[pyfunction(name = "reliability_diagram")]
#[pyo3(signature = (probabilities, labels, n_bins=None))]
fn reliability_diagram_py(
    probabilities: Vec<f64>,
    labels: Vec<f64>,
    n_bins: Option<usize>,
) -> Vec<(f64, f64, usize)> {
    metrics::reliability_diagram(&probabilities, &labels, n_bins.unwrap_or(10))
}

#[pyfunction(name = "calibration_report")]
#[pyo3(signature = (probabilities, labels, n_bins=None))]
fn calibration_report_py(
    probabilities: Vec<f64>,
    labels: Vec<f64>,
    n_bins: Option<usize>,
) -> PyCalibrationReport {
    let r = metrics::calibration_report(&probabilities, &labels, n_bins.unwrap_or(10));
    PyCalibrationReport {
        ece: r.ece,
        brier: r.brier,
        reliability: r.reliability,
        n_samples: r.n_samples,
        n_bins: r.n_bins,
    }
}

// ---------------------------------------------------------------------------
// FusionDebugger
// ---------------------------------------------------------------------------

#[pyclass(name = "BM25SignalTrace")]
pub struct PyBM25SignalTrace {
    #[pyo3(get)]
    raw_score: f64,
    #[pyo3(get)]
    tf: f64,
    #[pyo3(get)]
    doc_len_ratio: f64,
    #[pyo3(get)]
    likelihood: f64,
    #[pyo3(get)]
    tf_prior: f64,
    #[pyo3(get)]
    norm_prior: f64,
    #[pyo3(get)]
    composite_prior: f64,
    #[pyo3(get)]
    logit_likelihood: f64,
    #[pyo3(get)]
    logit_prior: f64,
    #[pyo3(get)]
    logit_base_rate: Option<f64>,
    #[pyo3(get)]
    posterior: f64,
    #[pyo3(get)]
    alpha: f64,
    #[pyo3(get)]
    beta: f64,
    #[pyo3(get)]
    base_rate: Option<f64>,
}

impl PyBM25SignalTrace {
    fn from_core(t: &BM25SignalTrace) -> Self {
        Self {
            raw_score: t.raw_score,
            tf: t.tf,
            doc_len_ratio: t.doc_len_ratio,
            likelihood: t.likelihood,
            tf_prior: t.tf_prior,
            norm_prior: t.norm_prior,
            composite_prior: t.composite_prior,
            logit_likelihood: t.logit_likelihood,
            logit_prior: t.logit_prior,
            logit_base_rate: t.logit_base_rate,
            posterior: t.posterior,
            alpha: t.alpha,
            beta: t.beta,
            base_rate: t.base_rate,
        }
    }
}

#[pyclass(name = "VectorSignalTrace")]
pub struct PyVectorSignalTrace {
    #[pyo3(get)]
    cosine_score: f64,
    #[pyo3(get)]
    probability: f64,
    #[pyo3(get)]
    logit_probability: f64,
}

impl PyVectorSignalTrace {
    fn from_core(t: &VectorSignalTrace) -> Self {
        Self {
            cosine_score: t.cosine_score,
            probability: t.probability,
            logit_probability: t.logit_probability,
        }
    }
}

#[pyclass(name = "NotTrace")]
pub struct PyNotTrace {
    #[pyo3(get)]
    input_probability: f64,
    #[pyo3(get)]
    input_name: String,
    #[pyo3(get)]
    complement: f64,
    #[pyo3(get)]
    logit_input: f64,
    #[pyo3(get)]
    logit_complement: f64,
}

impl PyNotTrace {
    fn from_core(t: &NotTrace) -> Self {
        Self {
            input_probability: t.input_probability,
            input_name: t.input_name.clone(),
            complement: t.complement,
            logit_input: t.logit_input,
            logit_complement: t.logit_complement,
        }
    }
}

#[pyclass(name = "FusionTrace")]
pub struct PyFusionTrace {
    #[pyo3(get)]
    signal_probabilities: Vec<f64>,
    #[pyo3(get)]
    signal_names: Vec<String>,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    logits: Option<Vec<f64>>,
    #[pyo3(get)]
    mean_logit: Option<f64>,
    #[pyo3(get)]
    alpha: Option<f64>,
    #[pyo3(get)]
    n_alpha_scale: Option<f64>,
    #[pyo3(get)]
    scaled_logit: Option<f64>,
    #[pyo3(get)]
    weights: Option<Vec<f64>>,
    #[pyo3(get)]
    log_probs: Option<Vec<f64>>,
    #[pyo3(get)]
    log_prob_sum: Option<f64>,
    #[pyo3(get)]
    complements: Option<Vec<f64>>,
    #[pyo3(get)]
    log_complements: Option<Vec<f64>>,
    #[pyo3(get)]
    log_complement_sum: Option<f64>,
    #[pyo3(get)]
    fused_probability: f64,
}

impl PyFusionTrace {
    fn from_core(t: &FusionTrace) -> Self {
        Self {
            signal_probabilities: t.signal_probabilities.clone(),
            signal_names: t.signal_names.clone(),
            method: t.method.clone(),
            logits: t.logits.clone(),
            mean_logit: t.mean_logit,
            alpha: t.alpha,
            n_alpha_scale: t.n_alpha_scale,
            scaled_logit: t.scaled_logit,
            weights: t.weights.clone(),
            log_probs: t.log_probs.clone(),
            log_prob_sum: t.log_prob_sum,
            complements: t.complements.clone(),
            log_complements: t.log_complements.clone(),
            log_complement_sum: t.log_complement_sum,
            fused_probability: t.fused_probability,
        }
    }
}

#[pyclass(name = "DocumentTrace")]
pub struct PyDocumentTrace {
    inner: DocumentTrace,
}

#[pymethods]
impl PyDocumentTrace {
    #[getter]
    fn doc_id(&self) -> Option<String> {
        self.inner.doc_id.clone()
    }

    #[getter]
    fn final_probability(&self) -> f64 {
        self.inner.final_probability
    }

    #[getter]
    fn fusion(&self) -> PyFusionTrace {
        PyFusionTrace::from_core(&self.inner.fusion)
    }
}

#[pyclass(name = "ComparisonResult")]
pub struct PyComparisonResult {
    inner: ComparisonResult,
}

#[pymethods]
impl PyComparisonResult {
    #[getter]
    fn signal_deltas(&self) -> Vec<(String, f64)> {
        self.inner.signal_deltas.clone()
    }

    #[getter]
    fn dominant_signal(&self) -> String {
        self.inner.dominant_signal.clone()
    }

    #[getter]
    fn crossover_stage(&self) -> Option<String> {
        self.inner.crossover_stage.clone()
    }
}

#[pyclass(name = "FusionDebugger")]
pub struct PyFusionDebugger {
    inner: FusionDebugger,
}

#[pymethods]
impl PyFusionDebugger {
    #[new]
    #[pyo3(signature = (alpha=None, beta=None, base_rate=None))]
    fn new(alpha: Option<f64>, beta: Option<f64>, base_rate: Option<f64>) -> PyResult<Self> {
        if let Some(br) = base_rate {
            if br <= 0.0 || br >= 1.0 {
                return Err(PyValueError::new_err(format!(
                    "base_rate must be in (0, 1), got {}",
                    br
                )));
            }
        }
        let transform = BayesianProbabilityTransform::new(
            alpha.unwrap_or(1.0),
            beta.unwrap_or(0.0),
            base_rate,
        );
        Ok(Self {
            inner: FusionDebugger::new(transform),
        })
    }

    fn trace_bm25(&self, score: f64, tf: f64, doc_len_ratio: f64) -> PyBM25SignalTrace {
        PyBM25SignalTrace::from_core(&self.inner.trace_bm25(score, tf, doc_len_ratio))
    }

    fn trace_vector(&self, cosine_score: f64) -> PyVectorSignalTrace {
        PyVectorSignalTrace::from_core(&self.inner.trace_vector(cosine_score))
    }

    #[pyo3(signature = (probability, name=None))]
    fn trace_not(&self, probability: f64, name: Option<&str>) -> PyNotTrace {
        PyNotTrace::from_core(&self.inner.trace_not(probability, name.unwrap_or("signal")))
    }

    #[pyo3(signature = (probabilities, names=None, method=None, alpha=None, weights=None))]
    fn trace_fusion(
        &self,
        probabilities: Vec<f64>,
        names: Option<Vec<String>>,
        method: Option<&str>,
        alpha: Option<f64>,
        weights: Option<Vec<f64>>,
    ) -> PyFusionTrace {
        let trace = self.inner.trace_fusion(
            &probabilities,
            names.as_deref(),
            method.unwrap_or("log_odds"),
            alpha,
            weights.as_deref(),
        );
        PyFusionTrace::from_core(&trace)
    }

    #[pyo3(signature = (bm25_score=None, tf=None, doc_len_ratio=None, cosine_score=None, method=None, alpha=None, weights=None, doc_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn trace_document(
        &self,
        bm25_score: Option<f64>,
        tf: Option<f64>,
        doc_len_ratio: Option<f64>,
        cosine_score: Option<f64>,
        method: Option<&str>,
        alpha: Option<f64>,
        weights: Option<Vec<f64>>,
        doc_id: Option<&str>,
    ) -> PyResult<PyDocumentTrace> {
        if bm25_score.is_none() && cosine_score.is_none() {
            return Err(PyValueError::new_err(
                "At least one of bm25_score or cosine_score must be provided",
            ));
        }
        let trace = self.inner.trace_document(
            bm25_score,
            tf,
            doc_len_ratio,
            cosine_score,
            method.unwrap_or("log_odds"),
            alpha,
            weights.as_deref(),
            doc_id,
        );
        Ok(PyDocumentTrace { inner: trace })
    }

    fn compare(&self, trace_a: &PyDocumentTrace, trace_b: &PyDocumentTrace) -> PyComparisonResult {
        let result = self.inner.compare(&trace_a.inner, &trace_b.inner);
        PyComparisonResult { inner: result }
    }

    #[pyo3(signature = (trace, verbose=None))]
    fn format_trace(&self, trace: &PyDocumentTrace, verbose: Option<bool>) -> String {
        self.inner.format_trace(&trace.inner, verbose.unwrap_or(true))
    }

    fn format_summary(&self, trace: &PyDocumentTrace) -> String {
        self.inner.format_summary(&trace.inner)
    }

    fn format_comparison(&self, comparison: &PyComparisonResult) -> String {
        self.inner.format_comparison(&comparison.inner)
    }
}

// ---------------------------------------------------------------------------

#[pyfunction(name = "run_experiments")]
fn run_experiments_py() -> Vec<PyExperimentResult> {
    let corpus = Rc::new(build_default_corpus());
    let queries = build_default_queries();
    let runner = ExperimentRunner::new(corpus, queries, 1.2, 0.75);
    runner
        .run_all()
        .into_iter()
        .map(|(name, passed, details)| PyExperimentResult { name, passed, details })
        .collect()
}

#[pymodule]
fn bb25(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyDocument>()?;
    m.add_class::<PyCorpus>()?;
    m.add_class::<PyBM25Scorer>()?;
    m.add_class::<PyBayesianBM25Scorer>()?;
    m.add_class::<PyVectorScorer>()?;
    m.add_class::<PyHybridScorer>()?;
    m.add_class::<PyParameterLearner>()?;
    m.add_class::<PyParameterLearnerResult>()?;
    m.add_class::<PyQuery>()?;
    m.add_class::<PyExperimentResult>()?;
    m.add_class::<PyExperimentRunner>()?;

    // Probability transform
    m.add_class::<PyBayesianProbabilityTransform>()?;

    // Learnable weights
    m.add_class::<PyLearnableLogOddsWeights>()?;
    m.add_class::<PyAttentionLogOddsWeights>()?;

    // Calibration metrics
    m.add_class::<PyCalibrationReport>()?;

    // Debug/trace types
    m.add_class::<PyBM25SignalTrace>()?;
    m.add_class::<PyVectorSignalTrace>()?;
    m.add_class::<PyNotTrace>()?;
    m.add_class::<PyFusionTrace>()?;
    m.add_class::<PyDocumentTrace>()?;
    m.add_class::<PyComparisonResult>()?;
    m.add_class::<PyFusionDebugger>()?;

    // Functions
    m.add_function(wrap_pyfunction!(build_default_corpus_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_default_queries_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_experiments_py, m)?)?;
    m.add_function(wrap_pyfunction!(prob_not_py, m)?)?;
    m.add_function(wrap_pyfunction!(prob_and_py, m)?)?;
    m.add_function(wrap_pyfunction!(prob_or_py, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_to_probability_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_odds_conjunction_py, m)?)?;
    m.add_function(wrap_pyfunction!(balanced_log_odds_fusion_py, m)?)?;
    m.add_function(wrap_pyfunction!(expected_calibration_error_py, m)?)?;
    m.add_function(wrap_pyfunction!(brier_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(reliability_diagram_py, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_report_py, m)?)?;

    m.add("__all__", vec![
        // Core types
        "Tokenizer",
        "Document",
        "Corpus",
        "BM25Scorer",
        "BayesianBM25Scorer",
        "VectorScorer",
        "HybridScorer",
        "ParameterLearner",
        "ParameterLearnerResult",
        "Query",
        "ExperimentResult",
        "ExperimentRunner",
        // Probability transform
        "BayesianProbabilityTransform",
        // Learnable weights
        "LearnableLogOddsWeights",
        "AttentionLogOddsWeights",
        // Calibration
        "CalibrationReport",
        // Debug
        "BM25SignalTrace",
        "VectorSignalTrace",
        "NotTrace",
        "FusionTrace",
        "DocumentTrace",
        "ComparisonResult",
        "FusionDebugger",
        // Functions
        "build_default_corpus",
        "build_default_queries",
        "run_experiments",
        "prob_not",
        "prob_and",
        "prob_or",
        "cosine_to_probability",
        "log_odds_conjunction",
        "balanced_log_odds_fusion",
        "expected_calibration_error",
        "brier_score",
        "reliability_diagram",
        "calibration_report",
    ])?;

    Ok(())
}
