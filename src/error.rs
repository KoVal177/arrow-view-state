//! Error types for the permutation index engine.

use arrow_schema::ArrowError;

/// Errors during index build or application.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// No columns or no rows were provided.
    #[error("columns slice must not be empty")]
    EmptyColumns,

    /// Row count exceeds the `u32::MAX` physical row ID limit.
    #[error("row count {0} exceeds u32::MAX ({max})", max = u32::MAX)]
    TooManyRows(u64),

    /// Columns have different lengths, or sort-column count doesn't match fields.
    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Expected count.
        expected: u64,
        /// Actual count.
        actual: u64,
    },

    /// Arrow row-encoding failed (usually an unsupported data type).
    #[error("row encoding failed: {0}")]
    RowEncodingFailed(#[from] ArrowError),

    /// I/O error during mmap storage creation.
    #[cfg(feature = "mmap")]
    #[error("mmap storage error: {0}")]
    MmapError(std::io::Error),

    /// Predicate evaluation kernel failed (e.g. type mismatch between array and scalar).
    #[cfg(feature = "evaluate")]
    #[error("predicate evaluation failed: {0}")]
    PredicateEvalFailed(String),

    /// Persistence I/O or format error.
    #[cfg(feature = "persist")]
    #[error("persistence error: {0}")]
    PersistError(String),

    /// Column data type is not supported for this index type.
    #[error("unsupported data type for indexing: {0}")]
    UnsupportedType(String),
}
