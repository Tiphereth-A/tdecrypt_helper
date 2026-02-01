//! TF-IDF-based similar segment search using cosine similarity.

use scirs2_text::{cosine_similarity, TfidfVectorizer, Vectorizer, WhitespaceTokenizer};

use super::state::DecryptionApp;

impl DecryptionApp {
    /// Finds segments similar to `target_idx` using TF-IDF and cosine similarity.
    ///
    /// Stores results in `self.similar_popup`.
    pub(super) fn compute_similar_segments(&mut self, target_idx: usize) {
        if target_idx >= self.project.segments.len() {
            return;
        }

        // Collect all documents as joined token strings
        let documents: Vec<String> = self
            .project
            .segments
            .iter()
            .map(|seg| {
                seg.tokens
                    .iter()
                    .map(|t| t.original.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();

        // Create vectorizer with whitespace tokenizer to preserve pre-tokenized words
        // This is important for handling special Unicode characters correctly
        let tokenizer = Box::new(WhitespaceTokenizer::new());
        let mut vectorizer =
            TfidfVectorizer::with_tokenizer(tokenizer, false, true, Some("l2".to_string()));

        // Fit and transform documents
        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let matrix = match vectorizer.fit_transform(&doc_refs) {
            Ok(m) => m,
            Err(_) => return,
        };

        // Get target document vector
        let target_vector = match matrix.row(target_idx).into_owned() {
            vec => vec,
        };

        // Compute similarities with all other documents
        let mut scores: Vec<(usize, f64)> = Vec::new();
        for (idx, _) in self.project.segments.iter().enumerate() {
            if idx == target_idx {
                continue;
            }

            let doc_vector = matrix.row(idx);
            match cosine_similarity(target_vector.view(), doc_vector) {
                Ok(sim) => {
                    if sim > 0.0 {
                        scores.push((idx, sim));
                    }
                }
                Err(_) => continue,
            }
        }

        // Sort by similarity descending and take top 5
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(5);

        self.similar_popup = Some((target_idx, scores));
    }
}
