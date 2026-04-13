//! High-performance columnar permutation index and filter engine for Apache Arrow.
//!
//! # Architecture
//!
//! This crate is a pure indexing engine — no application state, no I/O, no
//! networking, no column names, no pagination. Consumers own their own
//! view/session state.
//!
//! ## Core types
//!
//! - [`PermutationIndex`] — a sorted permutation mapping virtual rows to
//!   physical row IDs. The single core type. Apply to any Arrow array with
//!   `arrow_select::take`.
//! - [`FilterIndex`] — a sparse filter backed by Roaring Bitmap, with set
//!   algebra (`intersection`, `union`, `negate`).
//!
//! ## Builder functions
//!
//! - [`build`] — multi-column sort from `&[ArrayRef]` + `&[SortOptions]`.
//! - [`refine`] — stable tiebreaker refinement of an existing permutation.

pub mod builder;
pub mod error;
pub mod filter;
pub mod permutation;

#[cfg(test)]
mod tests;

// ── Public re-exports ──────────────────────────────────────────────────────────

pub use builder::{build, refine};
pub use error::IndexError;
pub use filter::FilterIndex;
pub use permutation::PermutationIndex;
