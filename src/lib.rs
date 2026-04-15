//! High-performance columnar view-state engine for Apache Arrow.
//!
//! # Architecture
//!
//! Pure indexing engine — no I/O, no column names, no pagination.
//!
//! ## Core types
//! - [`PermutationIndex`] — sorted permutation: virtual row → physical row ID.
//! - [`FilterIndex`] — sparse filter with set algebra.
//! - [`PhysicalSelection`] — physical row IDs for late materialisation.
//!
//! ## Builders
//! - [`SortBuilder`] — streaming multi-column sort builder.
//!
//! ## Optional features
//! - `evaluate` — SIMD predicate evaluation → `FilterIndex`.
//! - `hash-index` — O(1) equality lookup index.
//! - `inverted-index` — token-based text search index.
//! - `persist` — save/load indices via mmap persistence.

// ── Modules ────────────────────────────────────────────────────────────────

pub mod error;
pub mod filter;
pub mod permutation;
pub mod scalar_value;
pub mod selection;
pub mod sort_builder;

pub(crate) mod storage;

#[cfg(feature = "mmap")]
pub(crate) mod mmap_builder;

#[cfg(feature = "evaluate")]
pub mod evaluate;

#[cfg(feature = "hash-index")]
pub mod hash_index;

#[cfg(feature = "inverted-index")]
pub mod inverted_index;

#[cfg(feature = "persist")]
pub mod persist;

#[cfg(test)]
mod tests;

// ── Re-exports ─────────────────────────────────────────────────────────────

pub use error::IndexError;
pub use filter::FilterIndex;
pub use permutation::PermutationIndex;
pub use scalar_value::OwnedScalar;
pub use selection::PhysicalSelection;
pub use sort_builder::SortBuilder;

#[cfg(feature = "evaluate")]
pub use evaluate::{ScalarPredicate, ScalarValue, evaluate, evaluate_batches};

#[cfg(feature = "hash-index")]
pub use hash_index::HashIndex;

#[cfg(feature = "inverted-index")]
pub use inverted_index::{InvertedIndex, NgramTokenizer, Tokenizer, WhitespaceTokenizer};

#[cfg(feature = "persist")]
pub use persist::{load_filter, load_permutation, save_filter, save_permutation};
