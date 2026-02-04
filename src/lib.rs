pub mod math_utils;
pub mod tokenizer;
pub mod corpus;
pub mod bm25_scorer;
pub mod bayesian_scorer;
pub mod vector_scorer;
pub mod hybrid_scorer;
pub mod parameter_learner;
pub mod experiments;
pub mod defaults;

#[cfg(feature = "python")]
mod pybindings;

pub use math_utils::{
    clamp,
    cosine_similarity,
    dot_product,
    safe_log,
    safe_prob,
    sigmoid,
    vector_magnitude,
    EPSILON,
};

pub use tokenizer::Tokenizer;
pub use corpus::{Corpus, Document};
pub use bm25_scorer::BM25Scorer;
pub use bayesian_scorer::BayesianBM25Scorer;
pub use vector_scorer::VectorScorer;
pub use hybrid_scorer::HybridScorer;
pub use parameter_learner::{ParameterLearner, ParameterLearnerResult};
pub use experiments::{ExperimentRunner, Query};
pub use defaults::{build_default_corpus, build_default_queries};
