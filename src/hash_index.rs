//! Pre-grouped hash index for O(1) equality lookups.
//!
//! Build once from a column, then query by value to get a [`FilterIndex`]
//! without scanning the column again. Feature-gated behind `hash-index`.

use std::collections::HashMap;

use arrow_array::{Array, RecordBatch};
use roaring::RoaringBitmap;

use crate::error::IndexError;
use crate::filter::FilterIndex;
use crate::scalar_value::{OwnedScalar, extract_scalar};

/// A hash-based index over a single column.
///
/// Maps each distinct value to a [`RoaringBitmap`] of physical row IDs.
pub struct HashIndex {
    map: HashMap<OwnedScalar, RoaringBitmap>,
    total_rows: u32,
}

impl HashIndex {
    /// Build from a single Arrow array (full column).
    ///
    /// Iterates once, grouping row IDs by value.
    ///
    /// # Errors
    /// - [`IndexError::TooManyRows`] if array length > `u32::MAX`.
    /// - [`IndexError::UnsupportedType`] if the array type is not indexable.
    #[allow(clippy::cast_possible_truncation)]
    pub fn build(array: &dyn Array) -> Result<Self, IndexError> {
        let n = array.len();
        if n as u64 > u64::from(u32::MAX) {
            return Err(IndexError::TooManyRows(n as u64));
        }

        let mut map: HashMap<OwnedScalar, RoaringBitmap> = HashMap::new();

        for i in 0..n {
            let scalar = extract_scalar(array, i)
                .ok_or_else(|| IndexError::UnsupportedType(format!("{:?}", array.data_type())))?;
            map.entry(scalar).or_default().insert(i as u32);
        }

        Ok(Self {
            map,
            total_rows: n as u32,
        })
    }

    /// Build from a stream of `(batch, column_index)` pairs.
    ///
    /// Assigns global physical IDs across batches.
    ///
    /// # Errors
    /// - [`IndexError::TooManyRows`] if total rows > `u32::MAX`.
    /// - [`IndexError::UnsupportedType`] if the column type is not indexable.
    #[allow(clippy::cast_possible_truncation)]
    pub fn build_batches(
        batches: impl Iterator<Item = (RecordBatch, usize)>,
    ) -> Result<Self, IndexError> {
        let mut map: HashMap<OwnedScalar, RoaringBitmap> = HashMap::new();
        let mut offset: u64 = 0;

        for (batch, col_idx) in batches {
            let array = batch.column(col_idx);
            let n = array.len() as u64;
            if offset + n > u64::from(u32::MAX) {
                return Err(IndexError::TooManyRows(offset + n));
            }
            for i in 0..array.len() {
                let scalar = extract_scalar(array.as_ref(), i).ok_or_else(|| {
                    IndexError::UnsupportedType(format!("{:?}", array.data_type()))
                })?;
                map.entry(scalar)
                    .or_default()
                    .insert((offset + i as u64) as u32);
            }
            offset += n;
        }

        Ok(Self {
            map,
            total_rows: offset as u32,
        })
    }

    /// Lookup all rows matching a single value. O(1).
    pub fn lookup(&self, value: &OwnedScalar) -> FilterIndex {
        match self.map.get(value) {
            Some(bitmap) => FilterIndex::from_bitmap_ref(bitmap),
            None => FilterIndex::from_ids(std::iter::empty::<u32>()),
        }
    }

    /// Lookup all rows matching any of the given values (union). O(values).
    pub fn lookup_many(&self, values: &[OwnedScalar]) -> FilterIndex {
        let mut result = RoaringBitmap::new();
        for value in values {
            if let Some(bitmap) = self.map.get(value) {
                result |= bitmap;
            }
        }
        FilterIndex::from_bitmap(result)
    }

    /// Number of distinct values indexed.
    pub fn distinct_count(&self) -> usize {
        self.map.len()
    }

    /// Total rows indexed.
    pub fn total_rows(&self) -> u32 {
        self.total_rows
    }

    /// Iterate over distinct values and their row counts.
    pub fn value_counts(&self) -> impl Iterator<Item = (&OwnedScalar, u64)> {
        self.map.iter().map(|(k, v)| (k, v.len()))
    }
}
