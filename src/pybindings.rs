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
use crate::hybrid_scorer::HybridScorer;
use crate::parameter_learner::{ParameterLearner, ParameterLearnerResult};
use crate::tokenizer::Tokenizer;
use crate::vector_scorer::VectorScorer;

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
    fn new(bm25: &PyBM25Scorer, alpha: Option<f64>, beta: Option<f64>) -> Self {
        Self {
            inner: Rc::new(BayesianBM25Scorer::new(
                Rc::clone(&bm25.inner),
                alpha.unwrap_or(1.0),
                beta.unwrap_or(0.5),
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

    m.add_function(wrap_pyfunction!(build_default_corpus_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_default_queries_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_experiments_py, m)?)?;

    m.add("__all__", vec![
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
        "build_default_corpus",
        "build_default_queries",
        "run_experiments",
    ])?;

    Ok(())
}
