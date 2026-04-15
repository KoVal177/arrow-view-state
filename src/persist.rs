//! Persistence for permutation and filter indices.
//!
//! Indices are saved to named files and loaded via memory mapping for
//! zero-copy access. Feature-gated behind `persist`.

use std::io::Write;
use std::path::Path;

use crate::error::IndexError;
use crate::filter::FilterIndex;
use crate::permutation::PermutationIndex;
use crate::storage::PermutationStorage;

const PERM_MAGIC: &[u8; 8] = b"AVSPI\x00\x02\x00";
const FILTER_MAGIC: &[u8; 8] = b"AVSFI\x00\x02\x00";

#[allow(clippy::needless_pass_by_value)]
fn persist_io(e: std::io::Error) -> IndexError {
    IndexError::PersistError(e.to_string())
}

/// Save a permutation index to a file.
///
/// Writes the raw `u32` array in little-endian format. The file can be
/// loaded with [`load_permutation`] for zero-copy mmap access.
///
/// # File format
/// - Bytes 0..8: magic `b"AVSPI\x00\x02\x00"`
/// - Bytes 8..16: row count as u64 LE
/// - Bytes 16..16+4*n: u32 LE permutation array
///
/// # Errors
/// Returns `PersistError` on I/O failure.
pub fn save_permutation(index: &PermutationIndex, path: &Path) -> Result<(), IndexError> {
    #[allow(clippy::cast_possible_truncation)]
    let ids = index.read_range(0, index.len() as usize);
    let n = ids.len() as u64;

    let mut file = std::fs::File::create(path).map_err(persist_io)?;
    let mut writer = std::io::BufWriter::new(&mut file);

    writer.write_all(PERM_MAGIC).map_err(persist_io)?;
    writer.write_all(&n.to_le_bytes()).map_err(persist_io)?;
    for &id in &ids {
        writer.write_all(&id.to_le_bytes()).map_err(persist_io)?;
    }
    writer.flush().map_err(persist_io)?;
    drop(writer);
    file.sync_all().map_err(persist_io)?;

    Ok(())
}

/// Load a permutation index from a file via memory mapping.
///
/// The file must have been written by [`save_permutation`].
/// The returned `PermutationIndex` uses `MmapPersisted` storage.
///
/// # Errors
/// Returns `PersistError` on I/O failure, invalid magic, or truncated file.
#[allow(unsafe_code)]
pub fn load_permutation(path: &Path) -> Result<PermutationIndex, IndexError> {
    let file = std::fs::File::open(path).map_err(persist_io)?;

    // SAFETY: file is opened read-only; we only read from the mmap.
    let map = unsafe { memmap2::MmapOptions::new().map(&file).map_err(persist_io)? };

    if map.len() < 16 {
        return Err(IndexError::PersistError("file too small for header".into()));
    }
    if &map[0..8] != PERM_MAGIC {
        return Err(IndexError::PersistError("invalid magic bytes".into()));
    }

    let n = u64::from_le_bytes(
        map[8..16]
            .try_into()
            .map_err(|_| IndexError::PersistError("row count read failed".into()))?,
    );
    #[allow(clippy::cast_possible_truncation)]
    let expected_len = 16 + n as usize * 4;
    if map.len() < expected_len {
        return Err(IndexError::PersistError(format!(
            "file truncated: expected {expected_len} bytes, got {}",
            map.len()
        )));
    }

    #[allow(clippy::cast_possible_truncation)]
    let len = n as usize;

    Ok(PermutationIndex::from_storage(
        PermutationStorage::MmapPersisted {
            map,
            len,
            path: path.to_path_buf(),
        },
    ))
}

/// Save a filter index to a file.
///
/// Uses Roaring Bitmap's portable serialisation format wrapped in a header.
///
/// # File format
/// - Bytes 0..8: magic `b"AVSFI\x00\x02\x00"`
/// - Bytes 8..16: reserved (zero)
/// - Bytes 16..: Roaring Bitmap portable serialisation
///
/// # Errors
/// Returns `PersistError` on I/O failure.
pub fn save_filter(index: &FilterIndex, path: &Path) -> Result<(), IndexError> {
    let mut file = std::fs::File::create(path).map_err(persist_io)?;
    let mut writer = std::io::BufWriter::new(&mut file);

    writer.write_all(FILTER_MAGIC).map_err(persist_io)?;
    writer.write_all(&[0u8; 8]).map_err(persist_io)?; // reserved

    index
        .bitmap()
        .serialize_into(&mut writer)
        .map_err(|e| IndexError::PersistError(format!("bitmap serialisation: {e}")))?;

    writer.flush().map_err(persist_io)?;
    drop(writer);
    file.sync_all().map_err(persist_io)?;
    Ok(())
}

/// Load a filter index from a file.
///
/// The file must have been written by [`save_filter`].
///
/// # Errors
/// Returns `PersistError` on I/O failure, invalid magic, or truncated file.
pub fn load_filter(path: &Path) -> Result<FilterIndex, IndexError> {
    let data = std::fs::read(path).map_err(persist_io)?;

    if data.len() < 16 {
        return Err(IndexError::PersistError("file too small for header".into()));
    }
    if &data[0..8] != FILTER_MAGIC {
        return Err(IndexError::PersistError("invalid magic bytes".into()));
    }

    let bitmap = roaring::RoaringBitmap::deserialize_from(&data[16..])
        .map_err(|e| IndexError::PersistError(format!("bitmap deserialisation: {e}")))?;

    Ok(FilterIndex::from_bitmap(bitmap))
}
