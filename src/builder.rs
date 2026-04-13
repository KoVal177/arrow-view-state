//! Multi-column sort index builder.
//!
//! - `build`  — full sort from one or more columns. Uses `RowConverter` + parallel argsort.
//! - `refine` — stable refinement: adds tiebreaker column(s) to an existing permutation.
//!
//! This crate has no I/O. Callers read column arrays and pass them in.

use arrow_array::{ArrayRef, UInt32Array};
use arrow_row::{RowConverter, Rows, SortField};
use arrow_schema::SortOptions;
use tracing::debug;

use crate::error::IndexError;
use crate::permutation::PermutationIndex;

/// Build a permutation index sorting by one or more columns simultaneously.
///
/// `columns` and `options` must have equal length and at least one element.
/// All columns must have the same length, and that length must not exceed `u32::MAX`.
///
/// # Errors
///
/// - `IndexError::EmptyColumns` if `columns` is empty.
/// - `IndexError::TooManyRows` if any column exceeds `u32::MAX` rows.
/// - `IndexError::LengthMismatch` if columns have different lengths.
/// - `IndexError::RowEncodingFailed` on Arrow `RowConverter` failure.
///
/// # Panics
///
/// Panics if `columns.len() != options.len()` — this is a programmer error.
pub fn build(
    columns: &[ArrayRef],
    options: &[SortOptions],
) -> Result<PermutationIndex, IndexError> {
    validate_inputs(columns, options)?;

    let n = columns[0].len();
    debug!(columns = columns.len(), rows = n, "building sort index");

    let encoded = encode_columns(columns, options)?;
    let permutation = argsort_rows(&encoded, n);
    Ok(PermutationIndex::from_array(permutation))
}

/// Stable refinement: add tiebreaker column(s) to an existing permutation.
///
/// The existing permutation order is preserved for rows where the new
/// columns compare equal (stable sort semantics).
///
/// # Errors
///
/// - `IndexError::EmptyColumns` if `columns` is empty.
/// - `IndexError::LengthMismatch` if new columns don't match existing permutation length.
/// - `IndexError::RowEncodingFailed` on Arrow `RowConverter` failure.
///
/// # Panics
///
/// Panics if `columns.len() != options.len()` — this is a programmer error.
pub fn refine(
    existing: PermutationIndex,
    columns: &[ArrayRef],
    options: &[SortOptions],
) -> Result<PermutationIndex, IndexError> {
    validate_inputs(columns, options)?;

    let existing_len = existing.len();
    let new_len = columns[0].len() as u64;
    if new_len != existing_len {
        return Err(IndexError::LengthMismatch {
            expected: existing_len,
            actual: new_len,
        });
    }

    debug!(
        new_columns = columns.len(),
        existing_rows = existing_len,
        "refining sort index"
    );

    let encoded = encode_columns(columns, options)?;
    let mut ids: Vec<u32> = existing.into_indices().values().to_vec();

    // Stable sort: preserves prior column ordering for equal new-column values.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        ids.par_sort_by(|&i, &j| encoded.row(i as usize).cmp(&encoded.row(j as usize)));
    }
    #[cfg(not(feature = "parallel"))]
    {
        ids.sort_by(|&i, &j| encoded.row(i as usize).cmp(&encoded.row(j as usize)));
    }

    Ok(PermutationIndex::from_array(UInt32Array::from(ids)))
}

// ── Private helpers ────────────────────────────────────────────────────────────

fn validate_inputs(columns: &[ArrayRef], options: &[SortOptions]) -> Result<(), IndexError> {
    if columns.is_empty() {
        return Err(IndexError::EmptyColumns);
    }
    assert!(
        columns.len() == options.len(),
        "columns.len() ({}) must equal options.len() ({})",
        columns.len(),
        options.len(),
    );

    let n = columns[0].len() as u64;
    if n > u64::from(u32::MAX) {
        return Err(IndexError::TooManyRows(n));
    }
    for col in columns.iter().skip(1) {
        if col.len() as u64 != n {
            return Err(IndexError::LengthMismatch {
                expected: n,
                actual: col.len() as u64,
            });
        }
    }
    Ok(())
}

fn encode_columns(columns: &[ArrayRef], options: &[SortOptions]) -> Result<Rows, IndexError> {
    let fields: Vec<SortField> = columns
        .iter()
        .zip(options.iter())
        .map(|(col, &opts)| SortField::new_with_options(col.data_type().clone(), opts))
        .collect();
    let converter = RowConverter::new(fields)?;
    converter.convert_columns(columns).map_err(IndexError::from)
}

/// Parallel argsort over row-encoded byte slices.
#[allow(clippy::cast_possible_truncation)] // n already guarded ≤ u32::MAX by validate_inputs
fn argsort_rows(rows: &Rows, n: usize) -> UInt32Array {
    let mut ids: Vec<u32> = (0..n as u32).collect();

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        ids.par_sort_unstable_by(|&i, &j| rows.row(i as usize).cmp(&rows.row(j as usize)));
    }
    #[cfg(not(feature = "parallel"))]
    {
        ids.sort_unstable_by(|&i, &j| rows.row(i as usize).cmp(&rows.row(j as usize)));
    }

    UInt32Array::from(ids)
}
