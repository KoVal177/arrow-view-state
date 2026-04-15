# Examples

← [Back to README](README.md)

All examples below are self-contained code snippets that compile with `cargo test --all-features`.
`arrow-view-state` is a pure library crate with no interactive runner — the best way to explore
it is through its test suite and the snippets below.

---

## Sort & Window

Demonstrates multi-column sort with windowed reads and late materialisation.

### What it shows

- Building a `SortBuilder` with heterogeneous `SortField` types
- Streaming multiple `RecordBatch`es into a single sort
- Reading a viewport window via `read_range`
- Converting sorted positions to `PhysicalSelection` for Arrow `take`

### Code

```rust
use std::sync::Arc;
use arrow_array::{ArrayRef, Int64Array, StringArray, RecordBatch};
use arrow_schema::{Schema, Field, DataType, SortOptions};
use arrow_row::SortField;
use arrow_view_state::SortBuilder;

let schema = Arc::new(Schema::new(vec![
    Field::new("name",  DataType::Utf8,  false),
    Field::new("score", DataType::Int64, false),
]));

// Build two batches — simulating incremental ingestion.
let batch1 = RecordBatch::try_new(schema.clone(), vec![
    Arc::new(StringArray::from(vec!["Charlie", "Alice"])) as ArrayRef,
    Arc::new(Int64Array::from(vec![70, 95]))              as ArrayRef,
]).unwrap();
let batch2 = RecordBatch::try_new(schema.clone(), vec![
    Arc::new(StringArray::from(vec!["Bob", "Alice"])) as ArrayRef,
    Arc::new(Int64Array::from(vec![82, 61]))           as ArrayRef,
]).unwrap();

// Sort: name asc, score desc.
let fields = vec![
    SortField::new(DataType::Utf8),
    SortField::new_with_options(DataType::Int64, SortOptions { descending: true, nulls_first: false }),
];
let mut builder = SortBuilder::new(fields);
builder.push(&[batch1.column(0).clone(), batch1.column(1).clone()]).unwrap();
builder.push(&[batch2.column(0).clone(), batch2.column(1).clone()]).unwrap();
let sorted = builder.finish().unwrap();

// Read first page (rows 0..2 of the sorted view).
let _page = sorted.read_range(0, 2);

// Materialise into physical row IDs for Arrow take.
let selection = sorted.to_physical_selection(0..2);
// Pass `selection` to `arrow_select::take` or Parquet row-group pushdown.
```

### Key Concepts

- `SortBuilder::new(fields)` — one `SortField` per sort column (type + direction).
- `builder.push(columns)` — slice of `ArrayRef` aligned with the declared fields.
- `builder.finish()` — consumes the builder, returns a frozen `PermutationIndex`.
- `read_range(start, len)` — zero-copy windowed view at virtual row offsets.
- `to_physical_selection(range)` — produces row IDs in physical batch order.

### What to Try

- Add a third sort column and observe order stability.
- Call `read_range` with `start` beyond the last row — it returns an empty range, not a panic.
- Feed the `PhysicalSelection` into `arrow_select::take::take_record_batch` to get materialised rows.

---

## Filter Algebra

Demonstrates composing `FilterIndex` instances with set-algebra operations.

### What it shows

- Creating a `FilterIndex` from an explicit list of physical row IDs
- Combining filters with `and`, `or`, `not`
- Applying a filter to a `PermutationIndex` to produce a filtered, sorted view

### Code

```rust
use arrow_view_state::FilterIndex;

// Three independent filters (physical row IDs).
let active    = FilterIndex::from_ids([0, 1, 3, 5, 7]);
let high_value = FilterIndex::from_ids([1, 3, 6, 7]);
let excluded  = FilterIndex::from_ids([3]);

// active AND high_value AND NOT excluded → rows 1, 7
let result = active.and(&high_value).and(&excluded.not());
assert_eq!(result.to_ids(), vec![1, 7]);
```

### Key Concepts

- `FilterIndex::from_ids` — O(n) construction from any `IntoIterator<Item = u32>`.
- `.and` / `.or` / `.not` — produce new `FilterIndex` values; inputs are unchanged.
- `.to_ids()` — materialise the filtered set as a sorted `Vec<u32>`.
- `.apply_to_permutation(&perm)` — intersect a filter with a sorted `PermutationIndex`.

### What to Try

- Use `FilterIndex::full(n)` to create an "all rows" filter, then subtract with `.and(&exclusion.not())`.
- Chain three or more `.and` / `.or` calls — each returns a new owned value.

---

## Persist to Disk

Demonstrates saving a `PermutationIndex` to a memory-mapped file and reloading it.

Requires the `persist` (or `full`) feature.

### Code

```rust
#[cfg(feature = "persist")]
{
    use arrow_view_state::{SortBuilder, PersistIndex};
    use arrow_row::SortField;
    use arrow_schema::DataType;

    // ... build `sorted` as in "Sort & Window" above ...

    let tmp = tempfile::NamedTempFile::new().unwrap();
    sorted.save(tmp.path()).unwrap();

    let reloaded = PersistIndex::load(tmp.path()).unwrap();
    assert_eq!(sorted.len(), reloaded.len());
}
```

### Key Concepts

- `PermutationIndex::save(path)` — serialises the permutation array to an mmap file.
- `PersistIndex::load(path)` — deserialises without copying — backed by the mmap.
- Useful for caching sort results across process restarts.

### What to Try

- Measure the wall-clock difference between re-sorting from scratch vs loading from disk
  on a 1M-row dataset.
