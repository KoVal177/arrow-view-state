# Contributing to arrow-view-state

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/KoVal177/arrow-view-state
cd arrow-view-state
cargo build --all-features
```

Rust **1.94+** is required (edition 2024). Install via `rustup`.

## Running Tests

```bash
cargo test --all-features
```

For feature-specific testing:

```bash
cargo test                          # default features only
cargo test --features full          # all features
cargo test --no-default-features    # bare minimum
```

## Code Style

- Format: `cargo fmt --check` (enforced by CI)
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`
- No `unsafe` code — `unsafe_code = "deny"` is set workspace-wide.
- All public items must have at least a one-line `///` doc comment.

## Pull Request Process

1. Fork and create a feature branch from `main`.
2. Add tests for any new behaviour.
3. Ensure `cargo fmt`, `cargo clippy`, `cargo test --all-features`, and
   `cargo doc --no-deps --all-features` all pass.
4. Open a PR with a clear description of what changes and why.

## Bug Reports

Please open an issue on GitHub with a minimal reproducible example and the
output of `cargo --version` and `rustc --version`.
