use bayesian_bm25::block_max_index::BlockMaxIndex;
use bayesian_bm25::probability::BayesianProbabilityTransform;

#[test]
fn block_max_index_block_count() {
    let cases: Vec<(usize, usize, usize)> = vec![
        (10, 4, 3),    // ceil(10/4) = 3
        (8, 4, 2),     // ceil(8/4) = 2
        (1, 128, 1),   // ceil(1/128) = 1
        (256, 128, 2), // ceil(256/128) = 2
        (129, 128, 2), // ceil(129/128) = 2
    ];
    for (n_docs, block_size, expected_blocks) in cases {
        let mut idx = BlockMaxIndex::new(block_size);
        idx.build(&[vec![0.0; n_docs]]);
        assert_eq!(idx.n_blocks(), expected_blocks, "n_docs={}, block_size={}", n_docs, block_size);
    }
}

#[test]
fn block_upper_bound_geq_all_scores_in_block() {
    let block_size = 3;
    let mut idx = BlockMaxIndex::new(block_size);
    let scores = vec![vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.5, 7.0, 6.0, 1.5]];
    idx.build(&scores);

    let ub0 = idx.block_upper_bound(0, 0);
    assert!((ub0 - 3.0).abs() < 1e-10);
    for &s in &scores[0][0..3] {
        assert!(ub0 >= s, "block_upper_bound {} < score {}", ub0, s);
    }

    let ub1 = idx.block_upper_bound(0, 1);
    assert!((ub1 - 5.0).abs() < 1e-10);

    let ub2 = idx.block_upper_bound(0, 2);
    assert!((ub2 - 7.0).abs() < 1e-10);
}

#[test]
fn bayesian_block_upper_bound_geq_actual_probability() {
    let block_size = 3;
    let mut idx = BlockMaxIndex::new(block_size);
    let scores = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
    idx.build(&scores);

    let transform = BayesianProbabilityTransform::new(1.0, 0.5, None);
    let p_max = 0.9;

    for block_id in 0..2 {
        let bayesian_ub = idx.bayesian_block_upper_bound(0, block_id, &transform, p_max);
        assert!(bayesian_ub > 0.0 && bayesian_ub <= 1.0);

        let start = block_id * block_size;
        let end = (start + block_size).min(scores[0].len());
        for doc_idx in start..end {
            let actual_prob = transform.score_to_probability(scores[0][doc_idx], 1.0, 0.5);
            assert!(
                bayesian_ub >= actual_prob - 1e-10,
                "bayesian_block_upper_bound {} < actual prob {} for doc {}", bayesian_ub, actual_prob, doc_idx
            );
        }
    }
}
