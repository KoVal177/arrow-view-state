# arrow-view-state

[![Crates.io version](https://img.shields.io/crates/v/arrow-view-state.svg)](https://crates.io/crates/arrow-view-state)
[![Docs.rs](https://img.shields.io/docsrs/arrow-view-state)](https://docs.rs/arrow-view-state)
[![CI](https://github.com/KoVal177/arrow-view-state/actions/workflows/ci.yml/badge.svg)](https://github.com/KoVal177/arrow-view-state/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![MSRV](https://img.shields.io/badge/rustc-1.94+-blue.svg)](https://blog.rust-lang.org/)

High-performance columnar permutation index and filter engine for Apache Arrow.

## Overview

`arrow-view-state` is a pure indexing engine — no I/O, no column names, no pagination, no
application state. It separates *view-state computation* (sorting, filtering, selection) from
data access so that UI layers can reorder and filter millions of Arrow rows without copying
or rewriting the underlying `RecordBatch`es.

## Features

- **`SortBuilder`** — streaming multi-column sort; feed `RecordBatch`es one at a time, finish into a `PermutationIndex`
- **`PermutationIndex`** — virtual-to-physical row mapping; windowed reads, late materialisation
- **`FilterIndex`** — sparse filter backed by Roaring Bitmap with set algebra (`and`, `or`, `not`)
- **`PhysicalSelection`** — physical row IDs ready for Arrow `take` or Parquet row-selection pushdown
- **Parallel argsort** via Rayon (optional, default-on)
- **Memory-mapped storage** for large indices that exceed RAM (`persist` feature)
- **Hash index** for O(1) equality lookups (`hash-index` feature)
- **Inverted index** for token-based text search (`inverted-index` feature)
- **SIMD predicate evaluation** via `arrow-ord` (`evaluate` feature)

## Installation

```toml
[dependencies]
arrow-view-state = "0.1"
```

For WASM or minimal builds, disable the default `parallel` feature:

```toml
arrow-view-state = { version = "0.1", default-features = false }
```

## Quick Start

```rust
use std::sync::Arc;
use arrow_array::{ArrayRef, Int64Array, StringArray, RecordBatch};
use arrow_schema::{Schema, Field, DataType, SortOptions};
use arrow_row::SortField;
use arrow_view_state::{SortBuilder, FilterIndex};

let schema = Arc::new(Schema::new(vec![
    Field::new("name",  DataType::Utf8,  false),
    Field::new("value", DataType::Int64, false),
]));
let batch = RecordBatch::try_new(schema.clone(), vec![
    Arc::new(StringArray::from(vec!["B", "A", "B", "A"])) as ArrayRef,
    Arc::new(Int64Array::from(vec![10, 20, 30, 40]))       as ArrayRef,
]).unwrap();

// Sort by name asc, then value desc.
let fields = vec![
    SortField::new(DataType::Utf8),
    SortField::new_with_options(
        DataType::Int64,
        SortOptions { descending: true, nulls_first: true },
    ),
];
let mut builder = SortBuilder::new(fields);
builder.push(&[batch.column(0).clone(), batch.column(1).clone()]).unwrap();
let sorted = builder.finish().unwrap();

let range     = sorted.read_range(0, 2);              // windowed read
let filter    = FilterIndex::from_ids([0, 2]);
let filtered  = filter.apply_to_permutation(&sorted); // sparse filter
let selection = sorted.to_physical_selection(0..2);   // late materialisation
```

## Examples

See [`EXAMPLES.md`](EXAMPLES.md) for annotated walkthroughs.

| Example | What it shows |
|---|---|
| [Sort & Window](EXAMPLES.md#sort--window) | Multi-column sort, windowed read, late materialisation |
| [Filter Algebra](EXAMPLES.md#filter-algebra) | Composing `FilterIndex` with `and` / `or` / `not` |
| [Persist to Disk](EXAMPLES.md#persist-to-disk) | Save and reload a `PermutationIndex` via the `persist` feature |

## Feature Flags

| Flag | Default | Description |
|---|---|---|
| `parallel` | ✓ | Parallel argsort via Rayon |
| `evaluate` | | SIMD predicate evaluation → `FilterIndex` via `arrow-ord` |
| `hash-index` | | O(1) equality lookup index |
| `inverted-index` | | Token-based text search index |
| `mmap` | | Memory-mapped temp storage for large sorts |
| `persist` | | Save/load indices to disk (implies `mmap`) |
| `full` | | All of the above |

## Further Reading

- [Index Types in Depth](docs/index-types.md)
- [Feature Flags Reference](docs/feature-flags.md)

## MSRV

Minimum supported Rust version: **1.94** (edition 2024).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
