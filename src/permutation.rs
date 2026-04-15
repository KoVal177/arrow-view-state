//! `PermutationIndex` — the core sorted permutation type.
//!
//! `permutation[virtual_row]` = physical row ID in the source.
//! Use [`PermutationIndex::read_range`] for windowed access.

use arrow_array::UInt32Array;

use crate::error::IndexError;
use crate::selection::PhysicalSelection;
use crate::storage::PermutationStorage;

/// A sorted permutation index over a columnar dataset.
///
/// `permutation[virtual_row]` = physical row ID in the source.
///
/// Internal storage is either in-memory (`UInt32Array`) or memory-mapped
/// (temp file), chosen automatically based on row count and feature flags.
#[derive(Debug)]
pub struct PermutationIndex {
    pub(crate) storage: PermutationStorage,
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
        Ok(Self {
            storage: PermutationStorage::InMemory(permutation),
        })
    }

    /// Create a `PermutationIndex` from an externally-constructed `UInt32Array`.
    ///
    /// The caller guarantees that the array represents a valid permutation.
    /// (each value in `0..n` appears exactly once). This is not validated
    /// at runtime for performance — the builder functions guarantee it.
    pub(crate) fn from_array(permutation: UInt32Array) -> Self {
        Self {
            storage: PermutationStorage::InMemory(permutation),
        }
    }

    /// Construct from a pre-built [`PermutationStorage`].
    pub(crate) fn from_storage(storage: PermutationStorage) -> Self {
        Self { storage }
    }

    /// Number of rows in the permutation.
    pub fn len(&self) -> u64 {
        self.storage.len() as u64
    }

    /// Whether the permutation is empty.
    pub fn is_empty(&self) -> bool {
        self.storage.len() == 0
    }

    /// Read row IDs for virtual range `[start, end)`.
    ///
    /// This is the primary access method — works for both in-memory and mmap.
    /// Clamped to `[0, len)`.
    pub fn read_range(&self, start: usize, end: usize) -> Vec<u32> {
        self.storage.read_range(start, end)
    }

    /// Consume and return all row IDs as a `Vec<u32>`.
    pub fn into_vec(self) -> Vec<u32> {
        self.storage.into_vec()
    }

    /// Slice the permutation to a subrange.
    ///
    /// Returns a new in-memory `PermutationIndex` covering `[offset..offset+length]`.
    ///
    /// # Panics
    ///
    /// Panics if `offset + length > self.len()`.
    #[must_use]
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        match &self.storage {
            PermutationStorage::InMemory(arr) => Self {
                storage: PermutationStorage::InMemory(arr.slice(offset, length)),
            },
            #[cfg(feature = "mmap")]
            PermutationStorage::Mmap { .. } => {
                let ids = self.storage.read_range(offset, offset + length);
                Self::from_array(UInt32Array::from(ids))
            }
            #[cfg(feature = "persist")]
            PermutationStorage::MmapPersisted { .. } => {
                let ids = self.storage.read_range(offset, offset + length);
                Self::from_array(UInt32Array::from(ids))
            }
        }
    }

    /// Convert a virtual row range to a physical selection for late materialisation.
    ///
    /// Given virtual rows `[start, end)`, returns the corresponding physical
    /// row IDs as sorted, merged ranges suitable for Parquet `RowSelection`.
    ///
    /// Also returns the virtual-order IDs (the raw `read_range` output) that
    /// callers need for reordering materialised rows back to sort order.
    ///
    /// # Returns
    ///
    /// `(physical_selection, virtual_ids)` where:
    /// - `physical_selection` has ranges in ascending physical order
    /// - `virtual_ids[i]` = physical row ID at virtual position `start + i`
    pub fn to_physical_selection(&self, start: usize, end: usize) -> (PhysicalSelection, Vec<u32>) {
        let virtual_ids = self.read_range(start, end);
        let selection = PhysicalSelection::from_ids(virtual_ids.iter().copied());
        (selection, virtual_ids)
    }
}
