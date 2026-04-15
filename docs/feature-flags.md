← [Back to README](../README.md) | [Examples](../EXAMPLES.md)

# Feature Flags Reference

All feature flags are additive. None of them change the public API surface for the
features that are already enabled.

---

## `parallel` (default: **on**)

Enables multi-threaded argsort via [Rayon](https://docs.rs/rayon).

Extra dependency: `rayon`

**When to disable:** WASM targets (Rayon does not compile to `wasm32-unknown-unknown`),
or when you need deterministic single-threaded behaviour.

```toml
arrow-view-state = { version = "0.1", default-features = false }
```

---

## `evaluate`

Enables SIMD-accelerated predicate evaluation via `arrow-ord`, producing a `FilterIndex`
directly from a column and a comparison expression.

Extra dependency: `arrow-ord`

```rust
#[cfg(feature = "evaluate")]
{
    use arrow_view_state::EvaluateFilter;
    let filter = EvaluateFilter::gt(batch.column(1), &ScalarValue::Int64(50)).unwrap();
}
```

---

## `hash-index`

Enables `HashIndex` — O(1) equality lookup by value.

Extra dependency: `ordered-float` (for `Float32` / `Float64` hashing)

---

## `inverted-index`

Enables `InvertedIndex` — token-based text search.

No extra dependencies.

---

## `mmap`

Enables `MmapBuilder` and `StorageRef` — memory-mapped temporary storage for large
intermediate sort buffers.

Extra dependencies: `memmap2`, `tempfile`

---

## `persist` (implies `mmap`)

Enables `PersistIndex` — save/load `PermutationIndex` to and from a file via mmap.

Automatically enables `mmap`.

---

## `full`

Convenience alias that enables all optional features:

```toml
arrow-view-state = { version = "0.1", features = ["full"] }
```

Equivalent to: `parallel + evaluate + hash-index + inverted-index + persist`
