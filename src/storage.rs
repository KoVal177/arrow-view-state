//! Storage backends for permutation arrays.
//!
//! Variants:
//! - `InMemory`: backed by Arrow's `UInt32Array` (heap)
//! - `Mmap`: backed by a memory-mapped temp file (OS page cache)
//! - `MmapPersisted`: backed by a named file that survives process restarts

use arrow_array::UInt32Array;

/// Internal storage for the sorted permutation array.
///
/// Transparent to callers — [`super::PermutationIndex`] exposes only
/// `len()`, `read_range()`, `indices()`.
pub(crate) enum PermutationStorage {
    /// In-memory, backed by Arrow's buffer allocator.
    InMemory(UInt32Array),

    /// Memory-mapped temp file. The file is deleted when the `NamedTempFile`
    /// drops (on Unix: unlinked immediately, kept open via fd).
    #[cfg(feature = "mmap")]
    Mmap {
        /// Read-only memory map over the temp file.
        map: memmap2::Mmap,
        /// Number of `u32` elements stored.
        len: usize,
        /// Keeps the temp file alive; deleted on drop.
        _file: tempfile::NamedTempFile,
    },

    /// Memory-mapped named file. Persisted across process restarts.
    /// The file is NOT deleted on drop (unlike `Mmap` which uses `NamedTempFile`).
    /// The mmap includes a 16-byte header before the u32 array.
    #[cfg(feature = "persist")]
    MmapPersisted {
        /// Read-only memory map over the persisted file (includes header).
        map: memmap2::Mmap,
        /// Number of u32 elements.
        len: usize,
        /// File path for diagnostics.
        path: std::path::PathBuf,
    },
}

/// Header size in bytes for persisted permutation files.
#[cfg(feature = "persist")]
const PERSIST_HEADER: usize = 16;

impl PermutationStorage {
    /// Number of row IDs stored.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::InMemory(arr) => arr.len(),
            #[cfg(feature = "mmap")]
            Self::Mmap { len, .. } => *len,
            #[cfg(feature = "persist")]
            Self::MmapPersisted { len, .. } => *len,
        }
    }

    /// Read a contiguous range of physical row IDs.
    ///
    /// Clamped to `[0, len)`. Returns empty `Vec` if `start >= len`.
    pub(crate) fn read_range(&self, start: usize, end: usize) -> Vec<u32> {
        let end = end.min(self.len());
        if start >= end {
            return Vec::new();
        }
        match self {
            Self::InMemory(arr) => arr.values().as_ref()[start..end].to_vec(),
            #[cfg(feature = "mmap")]
            Self::Mmap { map, .. } => read_u32s_from_bytes(map, 0, start, end),
            #[cfg(feature = "persist")]
            Self::MmapPersisted { map, .. } => {
                read_u32s_from_bytes(map, PERSIST_HEADER, start, end)
            }
        }
    }

    /// Consume storage and return all row IDs as a `Vec<u32>`.
    pub(crate) fn into_vec(self) -> Vec<u32> {
        match self {
            Self::InMemory(arr) => arr.values().to_vec(),
            #[cfg(feature = "mmap")]
            Self::Mmap { map, len, .. } => read_u32s_from_bytes(&map, 0, 0, len),
            #[cfg(feature = "persist")]
            Self::MmapPersisted { map, len, .. } => {
                read_u32s_from_bytes(&map, PERSIST_HEADER, 0, len)
            }
        }
    }
}

/// Read a range of u32 LE values from a byte slice, offset by `header` bytes.
#[cfg(feature = "mmap")]
fn read_u32s_from_bytes(bytes: &[u8], header: usize, start: usize, end: usize) -> Vec<u32> {
    let byte_start = header + start * 4;
    let byte_end = header + end * 4;
    bytes[byte_start..byte_end]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
        .collect()
}

impl std::fmt::Debug for PermutationStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InMemory(arr) => write!(f, "InMemory(len={})", arr.len()),
            #[cfg(feature = "mmap")]
            Self::Mmap { len, .. } => write!(f, "Mmap(len={len})"),
            #[cfg(feature = "persist")]
            Self::MmapPersisted { len, path, .. } => {
                write!(f, "MmapPersisted(len={len}, path={})", path.display())
            }
        }
    }
}
