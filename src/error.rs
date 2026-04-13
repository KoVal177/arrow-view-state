//! Error types for the permutation index engine.

use arrow_schema::ArrowError;

/// Errors during index build, refinement, or application.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// No columns were provided to `build()` or `refine()`.
    #[error("columns slice must not be empty")]
    EmptyColumns,

    /// Row count exceeds the `u32::MAX` physical row ID limit.
    #[error("row count {0} exceeds u32::MAX ({max})", max = u32::MAX)]
    TooManyRows(u64),

    /// Columns have different lengths, or refinement columns don't match existing permutation.
    #[error("length mismatch: expected {expected} rows, got {actual}")]
    LengthMismatch {
        /// Expected row count.
        expected: u64,
        /// Actual row count of the mismatched column.
        actual: u64,
    },

    /// Arrow row-encoding failed (usually an unsupported data type).
    #[error("row encoding failed: {0}")]
    RowEncodingFailed(#[from] ArrowError),

    /// Arrow `take` kernel failed during `PermutationIndex::apply`.
    #[error("take failed: {0}")]
    TakeFailed(ArrowError),
}
