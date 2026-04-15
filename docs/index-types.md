← [Back to README](../README.md) | [Examples](../EXAMPLES.md)

# Index Types in Depth

`arrow-view-state` exposes three complementary index types. Understanding when to use each
one avoids unnecessary computation.

---

## `PermutationIndex`

A sorted permutation that maps *virtual* row positions (as seen by the UI) to *physical*
row positions (as stored in `RecordBatch`).

**When to use:** You need a stable sorted view of one or more `RecordBatch`es. The batches
themselves are never reordered — only the index changes.

```rust
// After builder.finish():
let sorted: PermutationIndex = builder.finish().unwrap();

// Virtual row 0 → the row with the smallest sort key.
let physical_0 = sorted.get(0);          // → physical row index
let range       = sorted.read_range(0, 100); // → slice of physical IDs
```

**Cost:** O(n log n) construction (parallel with the `parallel` feature). O(1) reads.

---

## `FilterIndex`

A sparse bitset of *physical* row IDs backed by [Roaring Bitmap](https://roaringbitmap.org/).

**When to use:** You need to hide rows matching (or not matching) a predicate without
recomputing the sort order.

```rust
let filter: FilterIndex = FilterIndex::from_ids([2, 5, 8]);
let combined = filter.and(&other_filter); // set intersection
let inverted = filter.not();              // complement
```

**Combining with a permutation:**

```rust
let visible = filter.apply_to_permutation(&sorted);
// `visible` is a new PermutationIndex containing only the filtered rows
// in their original sort order.
```

**Cost:** O(n) construction from an iterator. Set operations are O(output) thanks to
Roaring Bitmap compression.

---

## `PhysicalSelection`

A thin wrapper around `Vec<u32>` of physical row IDs, ready to pass to:

- [`arrow_select::take::take_record_batch`](https://docs.rs/arrow-select)
- Parquet [`RowSelection`](https://docs.rs/parquet) for pushdown

**When to use:** You have the final sorted+filtered view and need to materialise rows.

```rust
let selection: PhysicalSelection = sorted.to_physical_selection(0..page_size);
// Use selection.as_indices() to pass to Arrow take.
```

---

## `HashIndex` (feature: `hash-index`)

An equality lookup index for a single column. Useful for fast row lookup by key.

```rust
let idx = HashIndex::build(batch.column(0)).unwrap();
let ids: Vec<u32> = idx.get("Alice").unwrap_or_default();
```

**Cost:** O(n) build, O(1) lookup. Supports `Utf8`, `Int64`, `Float64`, and other
scalar types via `ordered-float`.

---

## `InvertedIndex` (feature: `inverted-index`)

A token-based text search index. Tokenises string columns and maps tokens to row IDs.

Useful for "contains word" searches without full SIMD predicate evaluation.

```rust
let idx = InvertedIndex::build(batch.column(0)).unwrap();
let ids = idx.query("Alice"); // rows containing the token "Alice"
```

---

## Choosing the Right Index

| Need | Index |
|---|---|
| Sorted viewport, multi-column | `PermutationIndex` (via `SortBuilder`) |
| Hide/show rows by predicate | `FilterIndex` |
| Materialise rows for Arrow | `PhysicalSelection` |
| Fast equality lookup by key | `HashIndex` |
| Token-based text search | `InvertedIndex` |
| Predicate from column expression | `EvaluateFilter` (feature: `evaluate`) |
