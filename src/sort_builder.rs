//! Streaming sort index builder.
//!
//! Accepts `RecordBatch`es one at a time via [`SortBuilder::push`], encodes sort keys with
//! `RowConverter`, and produces a [`PermutationIndex`] on [`SortBuilder::finish`].

use arrow_array::{RecordBatch, UInt32Array};
use arrow_row::{RowConverter, SortField};
use tracing::debug;

use crate::error::IndexError;
use crate::permutation::PermutationIndex;
use crate::storage::PermutationStorage;

// Re-export `SortField` for caller convenience (it's from `arrow_row`).
pub use arrow_row::SortField as ArrowSortField;

/// Streaming sort index builder.
///
/// Create with [`SortBuilder::new`], feed batches via [`SortBuilder::push`],
/// then call [`SortBuilder::finish`] to produce the sorted [`PermutationIndex`].
///
/// # Example
///
/// ```ignore
/// let mut builder = SortBuilder::new(vec![
///     SortField::new(DataType::Int64),
///     SortField::new_with_options(DataType::Utf8, SortOptions { descending: true, .. }),
/// ])?;
///
/// for batch in record_batch_stream {
///     builder.push(&batch, &[0, 2])?;  // sort by columns 0 and 2
/// }
///
/// let index = builder.finish()?;
/// ```
#[derive(Debug)]
pub struct SortBuilder {
    converter: RowConverter,
    /// `(encoded_row_bytes, global_physical_id)` pairs accumulated across pushes.
    entries: Vec<(Box<[u8]>, u32)>,
    fields_len: usize,
    global_offset: u64,
}

impl SortBuilder {
    /// Create a new builder for a sort with the given fields.
    ///
    /// `fields` defines the sort key schema and direction. One [`SortField`] per
    /// sort column, in priority order (first = primary, second = tiebreaker, …).
    ///
    /// # Errors
    ///
    /// - [`IndexError::EmptyColumns`] if `fields` is empty.
    /// - [`IndexError::RowEncodingFailed`] if the `RowConverter` cannot be created.
    pub fn new(fields: Vec<SortField>) -> Result<Self, IndexError> {
        if fields.is_empty() {
            return Err(IndexError::EmptyColumns);
        }
        let fields_len = fields.len();
        let converter = RowConverter::new(fields)?;
        Ok(Self {
            converter,
            entries: Vec::new(),
            fields_len,
            global_offset: 0,
        })
    }

    /// Ingest one [`RecordBatch`].
    ///
    /// `sort_columns` are the column indices within `batch` that correspond
    /// to the [`SortField`]s passed to [`SortBuilder::new`]. Must have the same length.
    ///
    /// The batch's sort columns are encoded into fixed-width row keys; the
    /// batch itself is NOT retained.
    ///
    /// # Errors
    ///
    /// - [`IndexError::LengthMismatch`] if `sort_columns.len() != fields.len()`.
    /// - [`IndexError::TooManyRows`] if cumulative rows exceed `u32::MAX`.
    /// - [`IndexError::RowEncodingFailed`] on encoding failure.
    pub fn push(&mut self, batch: &RecordBatch, sort_columns: &[usize]) -> Result<(), IndexError> {
        if sort_columns.len() != self.fields_len {
            return Err(IndexError::LengthMismatch {
                expected: self.fields_len as u64,
                actual: sort_columns.len() as u64,
            });
        }

        let n = batch.num_rows();
        if n == 0 {
            return Ok(());
        }

        let new_total = self.global_offset + n as u64;
        if new_total > u64::from(u32::MAX) {
            return Err(IndexError::TooManyRows(new_total));
        }

        let columns: Vec<_> = sort_columns
            .iter()
            .map(|&idx| batch.column(idx).clone())
            .collect();

        let rows = self.converter.convert_columns(&columns)?;

        self.entries.reserve(n);
        for i in 0..n {
            let row = rows.row(i);
            let bytes: Box<[u8]> = row.as_ref().into();
            let global_id = u32::try_from(self.global_offset + i as u64)
                .map_err(|_| IndexError::TooManyRows(self.global_offset + i as u64))?;
            self.entries.push((bytes, global_id));
        }

        self.global_offset = new_total;
        Ok(())
    }

    /// Total rows ingested so far.
    pub fn rows_ingested(&self) -> u64 {
        self.global_offset
    }

    /// Consume the builder, run parallel argsort, and produce a [`PermutationIndex`].
    ///
    /// # Errors
    ///
    /// - [`IndexError::EmptyColumns`] if no rows were pushed.
    /// - [`IndexError::MmapError`] if mmap storage creation fails (when `mmap` feature is enabled).
    pub fn finish(self) -> Result<PermutationIndex, IndexError> {
        if self.entries.is_empty() {
            return Err(IndexError::EmptyColumns);
        }

        let n = self.entries.len();
        debug!(rows = n, "finalising sort index");

        let mut ids: Vec<u32> = self.entries.iter().map(|(_, id)| *id).collect();
        let entries = &self.entries;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            ids.par_sort_unstable_by(|&a_id, &b_id| {
                entries[a_id as usize].0.cmp(&entries[b_id as usize].0)
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            ids.sort_unstable_by(|&a_id, &b_id| {
                entries[a_id as usize].0.cmp(&entries[b_id as usize].0)
            });
        }

        let storage = make_storage(ids, n as u64)?;
        Ok(PermutationIndex::from_storage(storage))
    }
}

/// Route to mmap or in-memory storage based on row count and feature flags.
fn make_storage(ids: Vec<u32>, n: u64) -> Result<PermutationStorage, IndexError> {
    #[cfg(feature = "mmap")]
    if n > crate::mmap_builder::MMAP_THRESHOLD {
        return crate::mmap_builder::write_mmap(&ids);
    }
    let _ = n;
    Ok(PermutationStorage::InMemory(UInt32Array::from(ids)))
}
