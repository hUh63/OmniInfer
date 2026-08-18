# OmniInfer Rust Control Plane

This workspace is an incremental Rust rewrite of the OmniInfer control plane:
CLI parsing, local state/config handling, gateway orchestration, and the future
TUI. It intentionally does not rewrite inference runtimes such as llama.cpp,
vLLM, MLX, or MNN.

Current scope:

- `omniinfer-core`: shared local paths, config compatibility, state parsing, and
  minimal local HTTP helpers.
- `omniinfer-cli`: Rust control-plane binary with the target command surface,
  gateway orchestration, runtime management, and shell completion generation.

## Architecture

Dependencies flow in one direction: `omniinfer-cli` depends on
`omniinfer-core`; core never imports CLI code. Core is grouped by domain:

- `backend`: backend arguments, profiles, registry, platform templates, and
  compatibility detection.
- `model`: local artifacts, bundled catalogs, load requests, and public model
  manifests.
- `protocol`: OpenAI/Anthropic normalization and streamed chat parsing.
- `runtime`: launch planning, process lifecycle, and resource accounting.
- `state`: application configuration, selected-model state, and serve
  ownership state.
- `support`: authentication, HTTP, path, and version utilities.

The CLI keeps transport and presentation concerns outside core. Command
workflows, `gateway`, `serve`, installers, `advisor`, `benchmark`, and `tui`
are separate modules; large gateway/runtime and installer responsibilities are
split into focused child modules. Existing flat core module paths remain
re-exported from `lib.rs` for downstream compatibility, while new code may use
the grouped paths.

Keep tests next to small modules. Move a test suite into a module-local
`tests.rs` when it obscures the production implementation. Treat files near
700 lines as a review signal, not a CI limit: split by responsibility when a
stable boundary exists, but do not create pass-through modules solely to lower
line counts. A separate gateway crate should wait until its internal boundary
is stable enough to justify another public package and dependency boundary.

The production entrypoint is `./omniinfer`, which starts the Rust control
plane. Python control-plane fallback has been removed; unsupported commands
return explicit Rust errors.

Use the public `--state-root` and `--runtime-root` global options for isolated
application integration. `OMNIINFER_STATE_ROOT` and `OMNIINFER_RUNTIME_ROOT`
are their stable environment equivalents. The older
`OMNIINFER_RUST_STATE_ROOT` name remains accepted by tests and existing
automation for compatibility.

## Local Development

```bash
cargo test --workspace
cargo run -p omniinfer-cli -- --help
cargo run -p omniinfer-cli -- status
cargo run -p omniinfer-cli -- completion bash
```

Run `cargo fmt --all -- --check` and `cargo test --workspace` before each
Rust control-plane commit.

## Profiling

Capture Rust command profiles:

```bash
python3 scripts/profile_python_cli.py \
  --runs 7 \
  --binary target/debug/omniinfer \
  --scenario help \
  --scenario status \
  --skip-import-trace \
  --output-dir tmp/test_results/20260622-rust-control-plane-rust-profile
```
