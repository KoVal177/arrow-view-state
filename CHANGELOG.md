# Changelog

All notable changes to `arrow-view-state` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — 2026-04-15

### Added

- `SortBuilder` — streaming multi-column sort builder using Arrow `RowConverter` and
  optional parallel argsort via Rayon.
- `PermutationIndex` — virtual-to-physical row mapping with `read_range` and
  `to_physical_selection` for late materialisation.
- `FilterIndex` — sparse filter engine backed by Roaring Bitmap; supports `and`, `or`,
  `not` set-algebra operations and `apply_to_permutation`.
- `PhysicalSelection` — type-safe physical row IDs suitable for Arrow `take` or Parquet
  row-selection pushdown.
- `EvaluateFilter` (`evaluate` feature) — SIMD predicate evaluation via `arrow-ord`
  producing `FilterIndex` from column expressions.
- `HashIndex` (`hash-index` feature) — O(1) equality lookup index with float hashing via
  `ordered-float`.
- `InvertedIndex` (`inverted-index` feature) — token-based text search index skeleton.
- `MmapBuilder` / `StorageRef` (`mmap` feature) — memory-mapped temporary storage for
  indices larger than available RAM.
- `PersistIndex` (`persist` feature) — save and reload `PermutationIndex` from disk via
  mmap.
- `ScalarValue` — column-type-agnostic scalar for predicate construction.
- `parallel` feature (default) — enables Rayon multi-thread argsort.
- `full` feature — convenience alias enabling all optional features.
