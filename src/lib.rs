pub mod math_utils;
pub mod tokenizer;
pub mod corpus;
pub mod bm25_scorer;
pub mod bayesian_scorer;
pub mod vector_scorer;
pub mod hybrid_scorer;
pub mod fusion;
pub mod parameter_learner;
pub mod experiments;
pub mod defaults;
pub mod probability;
pub mod learnable_weights;
pub mod attention_weights;
pub mod multi_head_attention;
pub mod calibration;
pub mod block_max_index;
pub mod metrics;
pub mod debug;

#[cfg(feature = "python")]
mod pybindings;

pub use math_utils::{
    clamp,
    cosine_similarity,
    dot_product,
    logit,
    min_max_normalize,
    safe_log,
    safe_prob,
    sigmoid,
    softmax,
    softmax_rows,
    vector_magnitude,
    EPSILON,
};

pub use fusion::{
    balanced_log_odds_fusion,
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
    prob_not,
    prob_or,
    Gating,
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
pub use probability::{BayesianProbabilityTransform, TemporalBayesianTransform, TrainingMode};
pub use learnable_weights::LearnableLogOddsWeights;
pub use attention_weights::AttentionLogOddsWeights;
pub use multi_head_attention::MultiHeadAttentionLogOddsWeights;
pub use calibration::{PlattCalibrator, IsotonicCalibrator};
pub use block_max_index::BlockMaxIndex;
pub use metrics::{
    brier_score,
    calibration_report,
    expected_calibration_error,
    reliability_diagram,
    CalibrationReport,
};
pub use debug::{
    BM25SignalTrace,
    ComparisonResult,
    DocumentTrace,
    FusionDebugger,
    FusionTrace,
    NotTrace,
    SignalTrace,
    VectorSignalTrace,
};
