//! Physical row selection for late materialisation.
//!
//! [`PhysicalSelection`] is a set of non-overlapping, ascending physical row ID
//! ranges. It is the output of [`PermutationIndex::to_physical_selection`] and
//! can be trivially converted to `parquet::arrow::arrow_reader::RowSelection`
//! by downstream consumers (the conversion lives outside this crate to avoid
//! a parquet dependency).

use std::collections::HashMap;
use std::ops::Range;

/// A set of physical row ID ranges, sorted ascending and non-overlapping.
///
/// Each range is `start..end` (exclusive end) in physical row ID space.
/// Consumers convert this to their reader's row selection type.
///
/// # Invariants
///
/// - Ranges are sorted by start.
/// - No two ranges overlap or are adjacent (they are merged).
/// - Each range is non-empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalSelection {
    ranges: Vec<Range<u32>>,
}

impl PhysicalSelection {
    /// Create from pre-sorted, non-overlapping, non-empty ranges.
    ///
    /// # Panics
    ///
    /// Debug-asserts that ranges are sorted, non-overlapping, and non-empty.
    #[allow(dead_code)] // used by downstream crates and future evaluate module
    pub(crate) fn from_sorted_ranges(ranges: Vec<Range<u32>>) -> Self {
        debug_assert!(
            ranges.iter().all(|r| r.start < r.end),
            "all ranges must be non-empty"
        );
        debug_assert!(
            ranges.windows(2).all(|w| w[0].end <= w[1].start),
            "ranges must be sorted and non-overlapping"
        );
        Self { ranges }
    }

    /// Build from an unsorted iterator of physical row IDs.
    ///
    /// Sorts the IDs, deduplicates, and merges into contiguous ranges.
    pub fn from_ids(ids: impl IntoIterator<Item = u32>) -> Self {
        let mut sorted: Vec<u32> = ids.into_iter().collect();
        sorted.sort_unstable();
        sorted.dedup();

        let mut ranges = Vec::new();
        let mut iter = sorted.into_iter();

        if let Some(first) = iter.next() {
            let mut start = first;
            let mut end = first + 1;

            for id in iter {
                if id == end {
                    end += 1;
                } else {
                    ranges.push(start..end);
                    start = id;
                    end = id + 1;
                }
            }
            ranges.push(start..end);
        }

        Self { ranges }
    }

    /// The physical ranges (sorted, non-overlapping).
    pub fn ranges(&self) -> &[Range<u32>] {
        &self.ranges
    }

    /// Total number of selected physical rows.
    pub fn row_count(&self) -> usize {
        self.ranges.iter().map(|r| (r.end - r.start) as usize).sum()
    }

    /// Whether no rows are selected.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Iterate over individual physical row IDs in ascending order.
    pub fn iter_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.ranges.iter().flat_map(Clone::clone)
    }

    /// Mapping from ascending physical order back to virtual (sorted) order.
    ///
    /// Returns a `Vec<usize>` of length `row_count()` where `result[i]` is
    /// the 0-based offset within the virtual range that physical row `i`
    /// corresponds to. This is needed to reorder materialised rows back to
    /// sort order after a Parquet read (which returns rows in physical order).
    pub fn virtual_order_map(&self, virtual_ids: &[u32]) -> Vec<usize> {
        let phys_to_virtual: HashMap<u32, usize> = virtual_ids
            .iter()
            .enumerate()
            .map(|(vi, &pi)| (pi, vi))
            .collect();

        self.iter_ids()
            .map(|phys_id| phys_to_virtual[&phys_id])
            .collect()
    }
}
