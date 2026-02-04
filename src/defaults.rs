use crate::corpus::Corpus;
use crate::experiments::Query;
use crate::tokenizer::Tokenizer;

struct DocumentDef {
    id: &'static str,
    text: &'static str,
    embedding: [f64; 8],
}

fn build_documents() -> Vec<DocumentDef> {
    vec![
        DocumentDef {
            id: "d01",
            text: "Machine learning algorithms learn patterns from data using statistical methods",
            embedding: [0.9, 0.3, 0.1, 0.0, 0.1, 0.0, 0.4, 0.1],
        },
        DocumentDef {
            id: "d02",
            text: "Deep learning neural networks require large training datasets for supervised learning",
            embedding: [0.8, 0.9, 0.1, 0.0, 0.0, 0.1, 0.2, 0.3],
        },
        DocumentDef {
            id: "d03",
            text: "Unsupervised learning discovers hidden structure in unlabeled data",
            embedding: [0.9, 0.4, 0.0, 0.0, 0.1, 0.0, 0.3, 0.2],
        },
        DocumentDef {
            id: "d04",
            text: "Reinforcement learning agents maximize cumulative reward through exploration",
            embedding: [0.8, 0.5, 0.0, 0.0, 0.0, 0.1, 0.3, 0.0],
        },
        DocumentDef {
            id: "d05",
            text: "Transfer learning adapts pre-trained models to new domains with limited data",
            embedding: [0.9, 0.7, 0.1, 0.0, 0.0, 0.0, 0.2, 0.3],
        },
        DocumentDef {
            id: "d06",
            text: "Information retrieval systems search and rank documents by relevance to queries",
            embedding: [0.1, 0.0, 0.9, 0.8, 0.0, 0.0, 0.2, 0.1],
        },
        DocumentDef {
            id: "d07",
            text: "BM25 is a bag of words retrieval function that ranks documents based on term frequency",
            embedding: [0.1, 0.0, 0.8, 0.9, 0.0, 0.0, 0.3, 0.0],
        },
        DocumentDef {
            id: "d08",
            text: "TF-IDF weighting reflects how important a word is to a document in a collection",
            embedding: [0.1, 0.0, 0.8, 0.7, 0.0, 0.0, 0.2, 0.0],
        },
        DocumentDef {
            id: "d09",
            text: "Query expansion improves search recall by adding related terms to the original query",
            embedding: [0.2, 0.0, 0.9, 0.6, 0.0, 0.0, 0.1, 0.1],
        },
        DocumentDef {
            id: "d10",
            text: "Relevance feedback uses explicit user judgments to improve retrieval performance",
            embedding: [0.2, 0.0, 0.8, 0.7, 0.0, 0.0, 0.2, 0.0],
        },
        DocumentDef {
            id: "d11",
            text: "Relational databases store data in tables with SQL as the query language",
            embedding: [0.0, 0.0, 0.1, 0.0, 0.9, 0.2, 0.0, 0.0],
        },
        DocumentDef {
            id: "d12",
            text: "NoSQL databases provide flexible schema design for unstructured data storage",
            embedding: [0.0, 0.0, 0.1, 0.0, 0.9, 0.3, 0.0, 0.0],
        },
        DocumentDef {
            id: "d13",
            text: "Database indexing structures like B-trees accelerate data retrieval operations",
            embedding: [0.0, 0.0, 0.3, 0.1, 0.9, 0.1, 0.0, 0.0],
        },
        DocumentDef {
            id: "d14",
            text: "Transaction processing ensures ACID properties for reliable data operations",
            embedding: [0.0, 0.0, 0.0, 0.0, 0.9, 0.3, 0.0, 0.0],
        },
        DocumentDef {
            id: "d15",
            text: "Distributed databases partition data across multiple nodes for scalability",
            embedding: [0.0, 0.0, 0.1, 0.0, 0.8, 0.9, 0.0, 0.0],
        },
        DocumentDef {
            id: "d16",
            text: "Vector search uses embedding similarity to find semantically related documents",
            embedding: [0.3, 0.3, 0.7, 0.5, 0.1, 0.0, 0.2, 0.9],
        },
        DocumentDef {
            id: "d17",
            text: "Hybrid search combines lexical matching with vector similarity for better retrieval",
            embedding: [0.2, 0.2, 0.8, 0.6, 0.0, 0.0, 0.3, 0.8],
        },
        DocumentDef {
            id: "d18",
            text: "Bayesian probability provides a framework for updating beliefs with new evidence",
            embedding: [0.3, 0.1, 0.2, 0.2, 0.0, 0.0, 0.9, 0.1],
        },
        DocumentDef {
            id: "d19",
            text: "Probabilistic models estimate relevance scores using statistical inference methods",
            embedding: [0.4, 0.1, 0.5, 0.4, 0.0, 0.0, 0.8, 0.2],
        },
        DocumentDef {
            id: "d20",
            text: "Cosine similarity measures the angle between two vectors in high-dimensional space",
            embedding: [0.2, 0.1, 0.3, 0.2, 0.0, 0.0, 0.3, 0.9],
        },
    ]
}

pub fn build_default_queries() -> Vec<Query> {
    vec![
        Query::new(
            "machine learning",
            &["machine", "learning"],
            Some(vec![0.9, 0.5, 0.1, 0.0, 0.0, 0.0, 0.3, 0.2]),
            &["d01", "d02", "d03", "d04", "d05"],
        ),
        Query::new(
            "Bayesian probability",
            &["bayesian", "probability"],
            Some(vec![0.3, 0.1, 0.2, 0.2, 0.0, 0.0, 0.9, 0.1]),
            &["d18", "d19"],
        ),
        Query::new(
            "search",
            &["search"],
            Some(vec![0.1, 0.0, 0.9, 0.6, 0.0, 0.0, 0.1, 0.3]),
            &["d06", "d09", "d16", "d17"],
        ),
        Query::new(
            "transaction processing",
            &["transaction", "processing"],
            Some(vec![0.0, 0.0, 0.0, 0.0, 0.9, 0.3, 0.0, 0.0]),
            &["d14"],
        ),
        Query::new(
            "data",
            &["data"],
            Some(vec![0.4, 0.2, 0.3, 0.1, 0.4, 0.2, 0.2, 0.2]),
            &["d01", "d03", "d05", "d11", "d12", "d13", "d14", "d15"],
        ),
        Query::new(
            "vector search embeddings",
            &["vector", "search", "embeddings"],
            Some(vec![0.2, 0.2, 0.7, 0.4, 0.0, 0.0, 0.2, 0.9]),
            &["d16", "d17", "d20"],
        ),
        Query::new(
            "retrieval augmented generation",
            &["retrieval", "augmented", "generation"],
            Some(vec![0.4, 0.4, 0.7, 0.5, 0.0, 0.0, 0.2, 0.4]),
            &["d06", "d07", "d10", "d17"],
        ),
    ]
}

pub fn build_default_corpus() -> Corpus {
    let tokenizer = Tokenizer::new();
    let mut corpus = Corpus::new(tokenizer);
    for doc in build_documents() {
        corpus.add_document(doc.id, doc.text, doc.embedding.to_vec());
    }
    corpus.build_index();
    corpus
}
