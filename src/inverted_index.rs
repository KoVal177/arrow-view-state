//! Token-based inverted index for text search.
//!
//! Build from a string column with a tokenizer, then search by query string
//! or prefix. Feature-gated behind `inverted-index`.

use std::collections::HashMap;

use arrow_array::Array;
use arrow_array::StringArray;
use roaring::RoaringBitmap;

use crate::error::IndexError;
use crate::filter::FilterIndex;

/// A trait for splitting text into searchable tokens.
pub trait Tokenizer: Send + Sync {
    /// Tokenize text into a vector of token string slices.
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str>;
}

/// Splits on whitespace.
pub struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split_whitespace().collect()
    }
}

/// Extracts character n-grams.
pub struct NgramTokenizer {
    /// N-gram size (in characters).
    pub n: usize,
}

impl Tokenizer for NgramTokenizer {
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        if text.len() < self.n {
            return vec![text];
        }
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut tokens = Vec::new();
        for window in chars.windows(self.n) {
            let start = window[0].0;
            let end_char = window.last().expect("non-empty window");
            let end = end_char.0 + end_char.1.len_utf8();
            tokens.push(&text[start..end]);
        }
        tokens
    }
}

/// Inverted index mapping tokens to physical row IDs.
pub struct InvertedIndex {
    map: HashMap<String, RoaringBitmap>,
    total_rows: u32,
}

impl InvertedIndex {
    /// Build from a string array.
    ///
    /// Each non-null value is tokenized; tokens are lowercased and mapped to
    /// row IDs.
    ///
    /// # Errors
    /// Returns [`IndexError::TooManyRows`] if array length > `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn build(array: &StringArray, tokenizer: &dyn Tokenizer) -> Result<Self, IndexError> {
        let n = array.len();
        if n as u64 > u64::from(u32::MAX) {
            return Err(IndexError::TooManyRows(n as u64));
        }

        let mut map: HashMap<String, RoaringBitmap> = HashMap::new();

        for i in 0..n {
            if array.is_null(i) {
                continue;
            }
            let text = array.value(i);
            let tokens = tokenizer.tokenize(text);
            for token in tokens {
                let lower = token.to_lowercase();
                map.entry(lower).or_default().insert(i as u32);
            }
        }

        Ok(Self {
            map,
            total_rows: n as u32,
        })
    }

    /// Search for rows containing ALL tokens in the query (AND semantics).
    ///
    /// The query is tokenized and lowercased. Returns the intersection of all
    /// token bitmaps.
    pub fn search(&self, query: &str, tokenizer: &dyn Tokenizer) -> FilterIndex {
        let tokens = tokenizer.tokenize(query);
        if tokens.is_empty() {
            return FilterIndex::from_ids(std::iter::empty::<u32>());
        }

        let mut result: Option<RoaringBitmap> = None;
        for token in tokens {
            let lower = token.to_lowercase();
            match self.map.get(&lower) {
                Some(bitmap) => {
                    result = Some(match result {
                        Some(existing) => existing & bitmap,
                        None => bitmap.clone(),
                    });
                }
                None => {
                    return FilterIndex::from_ids(std::iter::empty::<u32>());
                }
            }
        }

        match result {
            Some(bitmap) => FilterIndex::from_bitmap(bitmap),
            None => FilterIndex::from_ids(std::iter::empty::<u32>()),
        }
    }

    /// Search for rows whose tokens start with the given prefix.
    ///
    /// Returns the union of all token bitmaps where the token starts with
    /// `prefix` (lowercased).
    pub fn search_prefix(&self, prefix: &str) -> FilterIndex {
        let lower = prefix.to_lowercase();
        let mut result = RoaringBitmap::new();
        for (token, bitmap) in &self.map {
            if token.starts_with(&lower) {
                result |= bitmap;
            }
        }
        FilterIndex::from_bitmap(result)
    }

    /// Number of distinct tokens indexed.
    pub fn token_count(&self) -> usize {
        self.map.len()
    }

    /// Total rows indexed.
    pub fn total_rows(&self) -> u32 {
        self.total_rows
    }
}
