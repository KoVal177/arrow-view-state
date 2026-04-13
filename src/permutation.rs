//! `PermutationIndex` — the core sorted permutation type.
//!
//! `permutation[virtual_row]` = physical row ID in the source.
//! Apply to any Arrow array with [`PermutationIndex::apply`] (backed by `arrow_select::take`).

use arrow_array::{ArrayRef, UInt32Array};
use arrow_select::take::take;

use crate::error::IndexError;

/// A sorted permutation index over a columnar dataset.
///
/// `permutation[virtual_row]` = physical row ID in the source.
/// Apply to any Arrow array with `arrow_select::take`.
#[derive(Debug, Clone)]
pub struct PermutationIndex {
    permutation: UInt32Array,
}

impl PermutationIndex {
    /// Identity permutation — natural (unsorted) order: `[0, 1, 2, ..., n-1]`.
    ///
    /// # Errors
    ///
    /// Returns `IndexError::TooManyRows` if `n > u32::MAX`.
    pub fn natural(n: u64) -> Result<Self, IndexError> {
        if n > u64::from(u32::MAX) {
            return Err(IndexError::TooManyRows(n));
        }
        #[allow(clippy::cast_possible_truncation)]
        let permutation = UInt32Array::from_iter_values(0..n as u32);
        Ok(Self { permutation })
    }

    /// Construct from a pre-built `UInt32Array`.
    ///
    /// The caller guarantees that the array represents a valid permutation
    /// (each value in `0..n` appears exactly once). This is not validated
    /// at runtime for performance — the builder functions guarantee it.
    pub(crate) fn from_array(permutation: UInt32Array) -> Self {
        Self { permutation }
    }

    /// Number of rows in the permutation.
    pub fn len(&self) -> u64 {
        self.permutation.len() as u64
    }

    /// Whether the permutation is empty.
    pub fn is_empty(&self) -> bool {
        self.permutation.is_empty()
    }

    /// Borrow the underlying `UInt32Array` — the selection vector.
    ///
    /// Each element is a physical row ID. The position in the array is the
    /// virtual (sorted) row index.
    pub fn indices(&self) -> &UInt32Array {
        &self.permutation
    }

    /// Consume and return the underlying `UInt32Array`.
    pub fn into_indices(self) -> UInt32Array {
        self.permutation
    }

    /// Apply this permutation to an Arrow array, reordering it.
    ///
    /// Equivalent to `arrow_select::take(array, self.indices(), None)`.
    ///
    /// # Errors
    ///
    /// Returns `IndexError::TakeFailed` if the take kernel fails (e.g., index
    /// out of bounds for the given array).
    pub fn apply(&self, array: &dyn arrow_array::Array) -> Result<ArrayRef, IndexError> {
        take(array, &self.permutation, None).map_err(IndexError::TakeFailed)
    }

    /// Slice the permutation to a subrange (zero-copy).
    ///
    /// Returns a new `PermutationIndex` covering `[offset..offset+length]`.
    ///
    /// # Panics
    ///
    /// Panics if `offset + length > self.len()`.
    #[must_use]
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        Self {
            permutation: self.permutation.slice(offset, length),
        }
    }
}
