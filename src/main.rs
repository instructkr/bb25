use std::rc::Rc;

use bayesian_bm25::{build_default_corpus, build_default_queries, ExperimentRunner};

fn main() {
    let corpus = Rc::new(build_default_corpus());
    let queries = build_default_queries();
    let query_count = queries.len();
    let runner = ExperimentRunner::new(Rc::clone(&corpus), queries, 1.2, 0.75);
    let results = runner.run_all();

    println!("{}", "=".repeat(72));
    println!("Bayesian BM25 Experimental Validation");
    println!("{}", "=".repeat(72));
    println!();
    println!(
        "Corpus: {} documents, avgdl={:.1}, vocabulary={} terms",
        corpus.n,
        corpus.avgdl,
        corpus.df.len()
    );
    println!("Queries: {}", query_count);
    println!();

    let mut all_passed = true;
    for (name, passed, details) in results {
        let status = if passed { "PASS" } else { "FAIL" };
        if !passed {
            all_passed = false;
        }
        println!("{}", "-".repeat(72));
        println!("[{}] {}", status, name);
        for line in details.lines() {
            println!("       {}", line);
        }
        println!();
    }

    println!("{}", "=".repeat(72));
    if all_passed {
        println!("All 10 experiments PASSED.");
    } else {
        println!("Some experiments FAILED.");
    }
    println!("{}", "=".repeat(72));
}
