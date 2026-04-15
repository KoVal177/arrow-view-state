//! Memory-mapped index construction.
//!
//! When the `mmap` feature is enabled and the row count exceeds
//! [`MMAP_THRESHOLD`], the permutation array is written to a temp file
//! and memory-mapped for access via the OS page cache.

use std::io::Write;

use crate::error::IndexError;
use crate::storage::PermutationStorage;

/// Row count above which mmap storage is used (when `mmap` feature is enabled).
///
/// 50M rows × 4 bytes = 200 MB.
pub(crate) const MMAP_THRESHOLD: u64 = 50_000_000;

/// Write row IDs to a temp file and memory-map it.
pub(crate) fn write_mmap(ids: &[u32]) -> Result<PermutationStorage, IndexError> {
    let mut file = tempfile::NamedTempFile::new().map_err(IndexError::MmapError)?;

    {
        let mut writer = std::io::BufWriter::new(file.as_file_mut());
        for &id in ids {
            writer
                .write_all(&id.to_le_bytes())
                .map_err(IndexError::MmapError)?;
        }
        writer.flush().map_err(IndexError::MmapError)?;
    }

    // SAFETY: The file descriptor is valid (we just wrote it), and the
    // NamedTempFile keeps it alive. We only read the mapping (Mmap, not MmapMut).
    #[allow(unsafe_code)]
    let map = unsafe {
        memmap2::MmapOptions::new()
            .map(file.as_file())
            .map_err(IndexError::MmapError)?
    };

    tracing::info!(
        rows = ids.len(),
        "created memory-mapped permutation storage"
    );

    Ok(PermutationStorage::Mmap {
        map,
        len: ids.len(),
        _file: file,
    })
}
