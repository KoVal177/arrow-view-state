# arrow-view-state

High-performance columnar permutation index and filter engine for Apache Arrow.

Pure indexing engine — no I/O, no column names, no pagination, no application state.
Consumers own their own view/session state.

## Core types

- **`PermutationIndex`** — sorted permutation mapping virtual rows → physical row IDs.
  Apply to any Arrow array with `arrow_select::take`.
- **`FilterIndex`** — sparse filter backed by Roaring Bitmap, with set algebra
  (`intersection`, `union`, `negate`).

## Builder functions

- **`build`** — multi-column sort from `&[ArrayRef]` + `&[SortOptions]`.
  Uses `RowConverter` for row-encoding + parallel argsort via rayon.
- **`refine`** — stable tiebreaker refinement of an existing permutation.

## Usage

```rust
use std::sync::Arc;
use arrow_array::{ArrayRef, Int64Array, StringArray};
use arrow_schema::SortOptions;
use arrow_view_state::{build, refine, FilterIndex};

let names: ArrayRef = Arc::new(StringArray::from(vec!["B", "A", "B", "A"]));
let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40]));

// Sort by name ascending
let sorted = build(
    &[names],
    &[SortOptions { descending: false, nulls_first: false }],
).unwrap();

// Refine ties by value descending
let refined = refine(
    sorted,
    &[values],
    &[SortOptions { descending: true, nulls_first: true }],
).unwrap();

// Filter: keep only rows 0, 2
let filter = FilterIndex::from_ids([0, 2]);
let filtered = filter.apply_to_permutation(&refined);
```

## Features

- `parallel` (default) — enables rayon for parallel sort. Disable with `default-features = false`.

```toml
[dependencies]
arrow-view-state = { version = "0.2" }
```

## License

Licensed under either of Apache License 2.0 or MIT license at your option.
