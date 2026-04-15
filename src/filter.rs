//! `FilterIndex` — sparse filter backed by Roaring Bitmap.
//!
//! Stores which physical row IDs pass the active filter. Composes with
//! `PermutationIndex` via [`FilterIndex::apply_to_permutation`] to produce
//! a filtered, sorted selection vector.

use arrow_array::{BooleanArray, UInt32Array};
use roaring::RoaringBitmap;

use crate::permutation::PermutationIndex;

/// A filter over physical row IDs, backed by a Roaring Bitmap.
///
/// Construct from matching row IDs or from an Arrow `BooleanArray`,
/// then apply to a `PermutationIndex` to get a filtered selection vector.
#[derive(Debug, Clone)]
pub struct FilterIndex {
    mask: RoaringBitmap,
}

impl FilterIndex {
    /// Construct from an iterator of matching physical row IDs.
    pub fn from_ids(ids: impl IntoIterator<Item = u32>) -> Self {
        Self {
            mask: ids.into_iter().collect(),
        }
    }

    /// Construct from an Arrow `BooleanArray` where `true` = passes filter.
    ///
    /// Null values are treated as `false` (do not pass filter).
    pub fn from_boolean_array(arr: &BooleanArray) -> Self {
        let mask: RoaringBitmap = arr
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                #[allow(clippy::cast_possible_truncation)]
                v.unwrap_or(false).then_some(i as u32)
            })
            .collect();
        Self { mask }
    }

    /// Number of rows passing the filter.
    pub fn len(&self) -> u64 {
        self.mask.len()
    }

    /// Whether no rows pass the filter.
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Check whether a specific physical row ID passes the filter.
    pub fn contains(&self, id: u32) -> bool {
        self.mask.contains(id)
    }

    /// Apply this filter to a sorted permutation, returning a new `PermutationIndex`
    /// containing only the permuted rows that pass the filter, in sort order.
    ///
    /// This is the core composition operation:
    /// `sorted_and_filtered = filter.apply_to_permutation(&sorted)`
    ///
    /// Processes in chunks to avoid reading entire mmap indices at once.
    /// Time: O(n) where n = permutation length, with O(1) membership checks
    /// via the Roaring Bitmap.
    #[allow(clippy::cast_possible_truncation)] // len ≤ u32::MAX by construction
    pub fn apply_to_permutation(&self, permutation: &PermutationIndex) -> PermutationIndex {
        const CHUNK: usize = 65_536;
        let len = permutation.len() as usize;
        let mut values = Vec::new();

        let mut offset = 0;
        while offset < len {
            let end = (offset + CHUNK).min(len);
            let chunk = permutation.read_range(offset, end);
            for &id in &chunk {
                if self.mask.contains(id) {
                    values.push(id);
                }
            }
            offset = end;
        }

        PermutationIndex::from_array(UInt32Array::from(values))
    }

    // ── Set algebra ────────────────────────────────────────────────────

    /// Intersect two filters (AND semantics).
    #[must_use]
    pub fn intersection(&self, other: &FilterIndex) -> FilterIndex {
        FilterIndex {
            mask: &self.mask & &other.mask,
        }
    }

    /// Union two filters (OR semantics).
    #[must_use]
    pub fn union(&self, other: &FilterIndex) -> FilterIndex {
        FilterIndex {
            mask: &self.mask | &other.mask,
        }
    }

    /// Negate filter within a universe of `total_rows` physical rows.
    ///
    /// Returns a filter containing all rows in `[0, total_rows)` that are
    /// NOT in the current filter. Uses `insert_range` for O(containers)
    /// universe construction instead of O(elements) iteration.
    #[must_use]
    pub fn negate(&self, total_rows: u32) -> FilterIndex {
        let mut universe = RoaringBitmap::new();
        universe.insert_range(0..total_rows);
        FilterIndex {
            mask: universe - &self.mask,
        }
    }

    /// Set difference: rows in `self` but not in `other`.
    ///
    /// Equivalent to `self ∩ ¬other`, but without allocating the negation.
    #[must_use]
    pub fn difference(&self, other: &FilterIndex) -> FilterIndex {
        FilterIndex {
            mask: &self.mask - &other.mask,
        }
    }

    /// Convert to a `BooleanArray` of length `total_rows`.
    ///
    /// `true` at position `i` means physical row `i` passes the filter.
    pub fn into_boolean_array(&self, total_rows: u32) -> BooleanArray {
        let values: Vec<bool> = (0..total_rows).map(|i| self.mask.contains(i)).collect();
        BooleanArray::from(values)
    }

    /// Iterate over matching physical row IDs in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.mask.iter()
    }

    /// Access the underlying bitmap (for persistence and index types).
    #[allow(dead_code)] // used by feature-gated modules
    pub(crate) fn bitmap(&self) -> &RoaringBitmap {
        &self.mask
    }

    /// Construct from a raw bitmap (for persistence and index types).
    #[allow(dead_code)] // used by feature-gated modules
    pub(crate) fn from_bitmap(mask: RoaringBitmap) -> Self {
        Self { mask }
    }

    /// Construct from a borrowed bitmap (clone).
    #[allow(dead_code)] // used by feature-gated modules
    pub(crate) fn from_bitmap_ref(bitmap: &RoaringBitmap) -> Self {
        Self {
            mask: bitmap.clone(),
        }
    }
}
