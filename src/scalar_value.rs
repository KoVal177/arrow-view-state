//! Owned, hashable scalar value for indexing.

use std::hash::{Hash, Hasher};

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, BooleanArray, Date32Array, Int8Array, Int16Array, Int32Array, Int64Array,
    LargeStringArray, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
#[cfg(feature = "hash-index")]
use arrow_array::{Float32Array, Float64Array};
use arrow_schema::DataType;

/// An owned scalar value that implements `Hash + Eq` for use as a hash map key.
///
/// Covers the common Arrow column types. Floats use `ordered_float::OrderedFloat`
/// for correct hashing (NaN == NaN, total ordering).
#[derive(Debug, Clone)]
pub enum OwnedScalar {
    /// SQL NULL.
    Null,
    /// Boolean.
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
    /// 32-bit float (ordered for hashing).
    #[cfg(feature = "hash-index")]
    Float32(ordered_float::OrderedFloat<f32>),
    /// 64-bit float (ordered for hashing).
    #[cfg(feature = "hash-index")]
    Float64(ordered_float::OrderedFloat<f64>),
    /// UTF-8 string.
    Utf8(String),
    /// Large UTF-8 string.
    LargeUtf8(String),
    /// Days since epoch.
    Date32(i32),
    /// Microseconds since epoch.
    TimestampMicros(i64),
}

impl PartialEq for OwnedScalar {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int8(a), Self::Int8(b)) => a == b,
            (Self::Int16(a), Self::Int16(b)) => a == b,
            (Self::Int32(a), Self::Int32(b)) => a == b,
            (Self::Int64(a), Self::Int64(b)) => a == b,
            (Self::UInt8(a), Self::UInt8(b)) => a == b,
            (Self::UInt16(a), Self::UInt16(b)) => a == b,
            (Self::UInt32(a), Self::UInt32(b)) => a == b,
            (Self::UInt64(a), Self::UInt64(b)) => a == b,
            #[cfg(feature = "hash-index")]
            (Self::Float32(a), Self::Float32(b)) => a == b,
            #[cfg(feature = "hash-index")]
            (Self::Float64(a), Self::Float64(b)) => a == b,
            (Self::Utf8(a), Self::Utf8(b)) => a == b,
            (Self::LargeUtf8(a), Self::LargeUtf8(b)) => a == b,
            (Self::Date32(a), Self::Date32(b)) => a == b,
            (Self::TimestampMicros(a), Self::TimestampMicros(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for OwnedScalar {}

impl Hash for OwnedScalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Null => {}
            Self::Bool(v) => v.hash(state),
            Self::Int8(v) => v.hash(state),
            Self::Int16(v) => v.hash(state),
            Self::Int32(v) => v.hash(state),
            Self::Int64(v) => v.hash(state),
            Self::UInt8(v) => v.hash(state),
            Self::UInt16(v) => v.hash(state),
            Self::UInt32(v) => v.hash(state),
            Self::UInt64(v) => v.hash(state),
            #[cfg(feature = "hash-index")]
            Self::Float32(v) => v.hash(state),
            #[cfg(feature = "hash-index")]
            Self::Float64(v) => v.hash(state),
            Self::Utf8(v) => v.hash(state),
            Self::LargeUtf8(v) => v.hash(state),
            Self::Date32(v) => v.hash(state),
            Self::TimestampMicros(v) => v.hash(state),
        }
    }
}

/// Extract an [`OwnedScalar`] at position `i` from an Arrow array.
///
/// Returns `OwnedScalar::Null` for null values. Returns `None` if the data
/// type is not supported.
pub fn extract_scalar(array: &dyn Array, i: usize) -> Option<OwnedScalar> {
    if array.is_null(i) {
        return Some(OwnedScalar::Null);
    }
    match array.data_type() {
        DataType::Boolean => {
            let a = array.as_any().downcast_ref::<BooleanArray>()?;
            Some(OwnedScalar::Bool(a.value(i)))
        }
        DataType::Int8 => {
            let a = array.as_any().downcast_ref::<Int8Array>()?;
            Some(OwnedScalar::Int8(a.value(i)))
        }
        DataType::Int16 => {
            let a = array.as_any().downcast_ref::<Int16Array>()?;
            Some(OwnedScalar::Int16(a.value(i)))
        }
        DataType::Int32 => {
            let a = array.as_any().downcast_ref::<Int32Array>()?;
            Some(OwnedScalar::Int32(a.value(i)))
        }
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>()?;
            Some(OwnedScalar::Int64(a.value(i)))
        }
        DataType::UInt8 => {
            let a = array.as_any().downcast_ref::<UInt8Array>()?;
            Some(OwnedScalar::UInt8(a.value(i)))
        }
        DataType::UInt16 => {
            let a = array.as_any().downcast_ref::<UInt16Array>()?;
            Some(OwnedScalar::UInt16(a.value(i)))
        }
        DataType::UInt32 => {
            let a = array.as_any().downcast_ref::<UInt32Array>()?;
            Some(OwnedScalar::UInt32(a.value(i)))
        }
        DataType::UInt64 => {
            let a = array.as_any().downcast_ref::<UInt64Array>()?;
            Some(OwnedScalar::UInt64(a.value(i)))
        }
        #[cfg(feature = "hash-index")]
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>()?;
            Some(OwnedScalar::Float32(ordered_float::OrderedFloat(
                a.value(i),
            )))
        }
        #[cfg(feature = "hash-index")]
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>()?;
            Some(OwnedScalar::Float64(ordered_float::OrderedFloat(
                a.value(i),
            )))
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>()?;
            Some(OwnedScalar::Utf8(a.value(i).to_owned()))
        }
        DataType::LargeUtf8 => {
            let a = array.as_any().downcast_ref::<LargeStringArray>()?;
            Some(OwnedScalar::LargeUtf8(a.value(i).to_owned()))
        }
        DataType::Date32 => {
            let a = array.as_any().downcast_ref::<Date32Array>()?;
            Some(OwnedScalar::Date32(a.value(i)))
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            let a = array.as_primitive::<arrow_array::types::TimestampMicrosecondType>();
            Some(OwnedScalar::TimestampMicros(a.value(i)))
        }
        _ => None,
    }
}
