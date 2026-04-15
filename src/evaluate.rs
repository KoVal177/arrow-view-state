//! SIMD predicate evaluation for Arrow arrays.
//!
//! Evaluates typed predicates against Arrow arrays using `arrow_ord` comparison
//! kernels. Returns [`FilterIndex`] bitmaps.
//!
//! Feature-gated behind `evaluate`.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, RecordBatch, Scalar, StringArray, TimestampMicrosecondArray,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_ord::cmp;

use crate::error::IndexError;
use crate::filter::FilterIndex;

/// A scalar predicate to evaluate against a single column.
#[derive(Debug, Clone)]
pub enum ScalarPredicate {
    /// Equal to a scalar value.
    Eq(ScalarValue),
    /// Not equal to a scalar value.
    NotEq(ScalarValue),
    /// Less than a scalar value.
    Lt(ScalarValue),
    /// Less than or equal to a scalar value.
    Lte(ScalarValue),
    /// Greater than a scalar value.
    Gt(ScalarValue),
    /// Greater than or equal to a scalar value.
    Gte(ScalarValue),
    /// Value is null.
    IsNull,
    /// Value is not null.
    IsNotNull,
    /// Value is in the range `[lo, hi]` (inclusive both ends).
    Between {
        /// Lower bound (inclusive).
        lo: ScalarValue,
        /// Upper bound (inclusive).
        hi: ScalarValue,
    },
}

/// An owned scalar value for predicate comparison.
///
/// Covers the types commonly used for filtering. Extend as needed.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    /// Boolean scalar.
    Bool(bool),
    /// Signed 8-bit integer.
    Int8(i8),
    /// Signed 16-bit integer.
    Int16(i16),
    /// Signed 32-bit integer.
    Int32(i32),
    /// Signed 64-bit integer.
    Int64(i64),
    /// Unsigned 8-bit integer.
    UInt8(u8),
    /// Unsigned 16-bit integer.
    UInt16(u16),
    /// Unsigned 32-bit integer.
    UInt32(u32),
    /// Unsigned 64-bit integer.
    UInt64(u64),
    /// 32-bit float.
    Float32(f32),
    /// 64-bit float.
    Float64(f64),
    /// UTF-8 string.
    Utf8(String),
    /// Date as days since epoch.
    Date32(i32),
    /// Timestamp as microseconds since epoch.
    TimestampMicros(i64),
}

impl ScalarValue {
    /// Convert to a single-element `ArrayRef` for use as a datum.
    fn to_array(&self) -> ArrayRef {
        match self {
            Self::Bool(v) => Arc::new(BooleanArray::from(vec![*v])),
            Self::Int8(v) => Arc::new(Int8Array::from(vec![*v])),
            Self::Int16(v) => Arc::new(Int16Array::from(vec![*v])),
            Self::Int32(v) => Arc::new(Int32Array::from(vec![*v])),
            Self::Int64(v) => Arc::new(Int64Array::from(vec![*v])),
            Self::UInt8(v) => Arc::new(UInt8Array::from(vec![*v])),
            Self::UInt16(v) => Arc::new(UInt16Array::from(vec![*v])),
            Self::UInt32(v) => Arc::new(UInt32Array::from(vec![*v])),
            Self::UInt64(v) => Arc::new(UInt64Array::from(vec![*v])),
            Self::Float32(v) => Arc::new(Float32Array::from(vec![*v])),
            Self::Float64(v) => Arc::new(Float64Array::from(vec![*v])),
            Self::Utf8(v) => Arc::new(StringArray::from(vec![v.as_str()])),
            Self::Date32(v) => Arc::new(Date32Array::from(vec![*v])),
            Self::TimestampMicros(v) => Arc::new(TimestampMicrosecondArray::from(vec![*v])),
        }
    }

    /// Create a `Scalar` datum for `arrow_ord` comparisons.
    fn to_scalar(&self) -> Scalar<ArrayRef> {
        Scalar::new(self.to_array())
    }
}

#[allow(clippy::needless_pass_by_value)]
fn map_err(e: arrow_schema::ArrowError) -> IndexError {
    IndexError::PredicateEvalFailed(e.to_string())
}

/// Build a `BooleanArray` where `true` = row is null.
fn is_null_mask(column: &ArrayRef) -> BooleanArray {
    (0..column.len())
        .map(|i| Some(column.is_null(i)))
        .collect::<BooleanArray>()
}

/// Build a `BooleanArray` where `true` = row is not null.
fn is_not_null_mask(column: &ArrayRef) -> BooleanArray {
    (0..column.len())
        .map(|i| Some(column.is_valid(i)))
        .collect::<BooleanArray>()
}

/// Element-wise AND of two `BooleanArray`s.
fn bool_and(a: &BooleanArray, b: &BooleanArray) -> BooleanArray {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| match (x, y) {
            (Some(true), Some(true)) => Some(true),
            (Some(false), _) | (_, Some(false)) => Some(false),
            _ => None,
        })
        .collect::<BooleanArray>()
}

/// Evaluate a scalar predicate against a column array.
///
/// Returns a [`FilterIndex`] containing the physical row IDs where the predicate
/// is true. Null values are treated as non-matching (false).
///
/// # Errors
///
/// Returns [`IndexError::PredicateEvalFailed`] if the arrow kernel fails
/// (e.g. type mismatch between array and scalar).
pub fn evaluate(column: &ArrayRef, predicate: &ScalarPredicate) -> Result<FilterIndex, IndexError> {
    let mask: BooleanArray = match predicate {
        ScalarPredicate::Eq(v) => cmp::eq(column, &v.to_scalar()).map_err(map_err)?,
        ScalarPredicate::NotEq(v) => cmp::neq(column, &v.to_scalar()).map_err(map_err)?,
        ScalarPredicate::Lt(v) => cmp::lt(column, &v.to_scalar()).map_err(map_err)?,
        ScalarPredicate::Lte(v) => cmp::lt_eq(column, &v.to_scalar()).map_err(map_err)?,
        ScalarPredicate::Gt(v) => cmp::gt(column, &v.to_scalar()).map_err(map_err)?,
        ScalarPredicate::Gte(v) => cmp::gt_eq(column, &v.to_scalar()).map_err(map_err)?,
        ScalarPredicate::IsNull => is_null_mask(column),
        ScalarPredicate::IsNotNull => is_not_null_mask(column),
        ScalarPredicate::Between { lo, hi } => {
            let gte = cmp::gt_eq(column, &lo.to_scalar()).map_err(map_err)?;
            let lte = cmp::lt_eq(column, &hi.to_scalar()).map_err(map_err)?;
            bool_and(&gte, &lte)
        }
    };

    Ok(FilterIndex::from_boolean_array(&mask))
}

/// Evaluate a predicate across multiple batches, producing a single [`FilterIndex`].
///
/// Each batch contributes physical row IDs offset by the cumulative row count
/// of all previous batches. This is the natural companion to
/// [`SortBuilder::push`](crate::sort_builder::SortBuilder::push).
///
/// # Errors
///
/// - [`IndexError::TooManyRows`] if cumulative rows exceed `u32::MAX`.
/// - [`IndexError::PredicateEvalFailed`] on kernel error.
#[allow(clippy::cast_possible_truncation)]
pub fn evaluate_batches(
    batches: impl Iterator<Item = (RecordBatch, usize)>,
    predicate: &ScalarPredicate,
) -> Result<FilterIndex, IndexError> {
    let mut combined = FilterIndex::from_ids(std::iter::empty::<u32>());
    let mut offset: u64 = 0;

    for (batch, col_idx) in batches {
        let n = batch.num_rows() as u64;
        if offset + n > u64::from(u32::MAX) {
            return Err(IndexError::TooManyRows(offset + n));
        }

        let column = batch.column(col_idx).clone();
        let local_filter = evaluate(&column, predicate)?;

        let offset_u32 = offset as u32;
        let global_ids = local_filter.iter().map(|id| id + offset_u32);
        let global_filter = FilterIndex::from_ids(global_ids);

        combined = combined.union(&global_filter);
        offset += n;
    }

    Ok(combined)
}
