use crate::probability::BayesianProbabilityTransform;

/// Block-max index for efficient top-k retrieval with early termination.
///
/// Stores precomputed block-level maximum scores for each term,
/// enabling safe pruning of document blocks that cannot contribute
/// to the top-k results.
pub struct BlockMaxIndex {
    block_size: usize,
    block_maxes: Option<Vec<Vec<f64>>>, // [n_terms][n_blocks]
    n_docs: usize,
    n_terms: usize,
}

impl BlockMaxIndex {
    /// Create a new block-max index with the specified block size.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size >= 1, "block_size must be >= 1, got {}", block_size);
        Self {
            block_size,
            block_maxes: None,
            n_docs: 0,
            n_terms: 0,
        }
    }

    /// Build block-max structures from a score matrix.
    ///
    /// score_matrix: &[Vec<f64>] where score_matrix[term][doc] gives the
    /// score of the document for that term.
    pub fn build(&mut self, score_matrix: &[Vec<f64>]) {
        if score_matrix.is_empty() {
            self.block_maxes = Some(Vec::new());
            self.n_terms = 0;
            self.n_docs = 0;
            return;
        }

        self.n_terms = score_matrix.len();
        self.n_docs = score_matrix[0].len();
        let n_blocks = (self.n_docs + self.block_size - 1) / self.block_size;

        let mut all_maxes = Vec::with_capacity(self.n_terms);

        for term_scores in score_matrix {
            assert_eq!(
                term_scores.len(),
                self.n_docs,
                "All term score vectors must have the same length"
            );

            let mut term_block_maxes = Vec::with_capacity(n_blocks);

            for block_id in 0..n_blocks {
                let start = block_id * self.block_size;
                let end = (start + self.block_size).min(self.n_docs);
                let max_val = term_scores[start..end]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                term_block_maxes.push(max_val);
            }

            all_maxes.push(term_block_maxes);
        }

        self.block_maxes = Some(all_maxes);
    }

    /// Get the block-level upper bound score for a given term and block.
    pub fn block_upper_bound(&self, term_idx: usize, block_id: usize) -> f64 {
        match &self.block_maxes {
            Some(maxes) => {
                assert!(
                    term_idx < maxes.len(),
                    "term_idx {} out of range (n_terms={})",
                    term_idx,
                    maxes.len()
                );
                assert!(
                    block_id < maxes[term_idx].len(),
                    "block_id {} out of range (n_blocks={})",
                    block_id,
                    maxes[term_idx].len()
                );
                maxes[term_idx][block_id]
            }
            None => panic!("BlockMaxIndex has not been built"),
        }
    }

    /// Compute a Bayesian upper bound probability for a term in a block.
    ///
    /// Uses the block-max score as the BM25 upper bound and the given
    /// p_max as the prior upper bound, then delegates to the transform's
    /// wand_upper_bound method.
    pub fn bayesian_block_upper_bound(
        &self,
        term_idx: usize,
        block_id: usize,
        transform: &BayesianProbabilityTransform,
        p_max: f64,
    ) -> f64 {
        let bm25_ub = self.block_upper_bound(term_idx, block_id);
        transform.wand_upper_bound(bm25_ub, p_max)
    }

    /// Block size used for this index.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of blocks (computed from n_docs and block_size).
    pub fn n_blocks(&self) -> usize {
        if self.n_docs == 0 {
            return 0;
        }
        (self.n_docs + self.block_size - 1) / self.block_size
    }
}
