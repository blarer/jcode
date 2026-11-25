use super::*;

/// Keep the most lexically relevant memories for Jev judgement. Small stores
/// pass through unchanged. Memories with no term overlap are dropped only when
/// the store is larger than the candidate budget.
pub(crate) fn prefilter_for_jev(entries: Vec<MemoryEntry>, query: &str) -> Vec<MemoryEntry> {
    if entries.len() <= MAX_JEV_RECALL_CANDIDATES {
        return entries;
    }
    let ranked = bm25_rank(&entries, query, MAX_JEV_RECALL_CANDIDATES);
    let mut entries: Vec<Option<MemoryEntry>> = entries.into_iter().map(Some).collect();
    ranked
        .into_iter()
        .filter_map(|(idx, _)| entries[idx].take())
        .collect()
}

/// Rank memories by BM25 over their normalized search text.
///
/// Returns `(entry_index, score)` pairs sorted by score desc, truncated to
/// `limit`. Memories with zero query-term overlap are dropped.
pub(super) fn bm25_rank(
    entries: &[MemoryEntry],
    query_text: &str,
    limit: usize,
) -> Vec<(usize, f32)> {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    // Conversational queries ("why does it do that, is it fixed?") are mostly
    // function words that match every memory and push the useful terms out of
    // the candidate budget. They are dropped from the query side only, so a
    // query made of nothing but stop words still matches as before.
    // Measured 2026-11-24 over the ~2,800-memory home store at the 72-memory
    // budget: the right memory survived for 21/28 real held-out queries before
    // this, 23/28 after; 39/40 on a separate set either way.
    let all_terms: Vec<String> = normalize_search_text(query_text)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let content_terms: Vec<String> = all_terms
        .iter()
        .filter(|t| !is_query_stop_word(t))
        .cloned()
        .collect();
    let q_terms = if content_terms.is_empty() {
        all_terms
    } else {
        content_terms
    };
    if q_terms.is_empty() {
        return Vec::new();
    }
    let q_set: std::collections::HashSet<&String> = q_terms.iter().collect();

    // Tokenize each doc once; compute df and doc lengths.
    let docs: Vec<Vec<String>> = entries
        .iter()
        .map(|e| {
            e.searchable_text()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect()
        })
        .collect();

    let n = docs.len().max(1) as f32;
    let avgdl = docs.iter().map(|d| d.len()).sum::<usize>() as f32 / n;
    let mut df: std::collections::HashMap<&str, f32> = std::collections::HashMap::new();
    for doc in &docs {
        let unique: std::collections::HashSet<&str> = doc.iter().map(|s| s.as_str()).collect();
        for t in unique {
            *df.entry(t).or_insert(0.0) += 1.0;
        }
    }

    let mut scored: Vec<(usize, f32)> = Vec::new();
    for (idx, doc) in docs.iter().enumerate() {
        if doc.is_empty() {
            continue;
        }
        let dl = doc.len() as f32;
        let mut tf: std::collections::HashMap<&str, f32> = std::collections::HashMap::new();
        for t in doc {
            *tf.entry(t.as_str()).or_insert(0.0) += 1.0;
        }
        let mut score = 0.0f32;
        for term in &q_set {
            let Some(&f) = tf.get(term.as_str()) else {
                continue;
            };
            let n_q = *df.get(term.as_str()).unwrap_or(&0.0);
            if n_q == 0.0 {
                continue;
            }
            let idf = (((n - n_q + 0.5) / (n_q + 0.5)) + 1.0).ln();
            let denom = f + K1 * (1.0 - B + B * dl / avgdl);
            score += idf * (f * (K1 + 1.0)) / denom;
        }
        if score > 0.0 {
            scored.push((idx, score));
        }
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(limit);
    scored
}

/// English function words and chat filler that carry no topic. Query side only.
fn is_query_stop_word(term: &str) -> bool {
    const STOP: &[&str] = &[
        "a", "about", "actually", "again", "all", "also", "am", "an", "and", "any", "are", "as",
        "at", "be", "because", "been", "bro", "but", "by", "can", "could", "did", "do", "does",
        "doing", "dont", "don't", "for", "from", "get", "go", "going", "gonna", "had", "has",
        "have", "he", "her", "here", "him", "his", "how", "i", "i'm", "if", "in", "into", "is",
        "it", "it's", "its", "just", "know", "let", "like", "make", "me", "mean", "more", "my",
        "need", "no", "not", "now", "of", "okay", "ok", "on", "one", "or", "our", "out", "please",
        "really", "right", "say", "see", "she", "should", "so", "some", "still", "than", "that",
        "the", "their", "them", "then", "there", "these", "they", "thing", "things", "think",
        "this", "those", "to", "too", "um", "up", "us", "use", "very", "wait", "want", "was", "we",
        "were", "what", "what's", "when", "where", "which", "while", "who", "why", "will", "with",
        "would", "yeah", "yes", "you", "your",
    ];
    STOP.contains(&term)
}
