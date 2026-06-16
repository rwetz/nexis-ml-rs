# Changelog

All notable changes to nexis-ml-rs (the Rust engine). Versions follow
[SemVer](https://semver.org/); pre-1.0, minor bumps may change the CLI.
This engine speaks the same protocol and writes the same run store as the
Python [`nexis-ml`](https://github.com/rwetz/nexis-ml) — see
[PLAN.md](PLAN.md) for the milestone-by-milestone story.

## [0.5.2] — 2026-06-15

### Added
- **Cross-platform release workflow** — CI builds the per-OS/arch single
  binary (`nexis-ml-windows-x64.exe`, `nexis-ml-macos-arm64`,
  `nexis-ml-macos-x64`, `nexis-ml-linux-x64`) that Nexis fetches from the
  GitHub "latest release" via the panel's setup card.

## [0.5.1] — 2026-06-15

### Fixed
- `new <template> [dir]` now takes the directory as a **positional**
  argument, matching the Python engine's argument order.

## [0.5.0] — 2026-06-15 — M5: ONNX export

### Added
- **`nexis-ml export --onnx [dir]`** — trains the tabular MLP from
  `train.toml` and writes `<dir>/model.onnx`. burn has no native ONNX export,
  so `src/onnx.rs` hand-encodes the protobuf (no extra dependency); the graph
  bakes in standardization (raw features → class logits) and is verified
  against onnxruntime for an exact prediction match. CNN export is a follow-up.

### Changed
- The backend RNG is seeded from `[train] seed`, so training and exports are
  reproducible.

## [0.4.0] — 2026-06-14 — M4: declarative model

### Added
- The model is **declared in `train.toml`**, chosen by what `[data] path`
  points at: a `.csv` (or unset → synthetic) trains a variable-depth MLP
  (`hidden = 16` or `[64, 32]`); a folder of class sub-folders trains a CNN
  over its images (`conv1`/`conv2`/`hidden`), decoding PNG/JPEG/BMP to
  grayscale via the `image` crate. The CNN emits a per-epoch confusion matrix
  and an `image-grid` PNG of sample predictions, matching the Python templates.

## [0.3.0] — 2026-06-14 — M3: GPU via wgpu

### Added
- `model.rs` is generic over the burn backend and picks CPU (ndarray) or
  **GPU via burn's `wgpu`** (Vulkan/DX12/Metal/OpenGL — no vendor toolchain)
  from `[train] device` (`auto`/`cpu`/`gpu`/`cuda`/`wgpu`; `auto` probes for
  an adapter and degrades to CPU). The resolved device rides on
  run.started/summary, GPU runs emit a per-epoch `mem/gpu_mb` footprint, and
  `nexis-ml env` reports `backend: "wgpu"` when a GPU is present.

## [0.2.1] — 2026-06-14 — run control

### Added
- The harness honors the same stdin **control commands** as the Python engine
  in protocol mode: `{"cmd":"cancel"}` (break cleanly, finish `cancelled`,
  keep the checkpoint) and `{"cmd":"pause"}` / `{"cmd":"resume"}`
  (block/release at the next epoch boundary) — so Nexis's Stop and Pause
  buttons work against this engine too.

## [0.2.0] — 2026-06-14 — M2: real burn MLP

### Added
- `train` runs a real **MLP** on [`burn`](https://github.com/tracel-ai/burn)'s
  ndarray/CPU backend with autodiff (Adam + cross-entropy, minibatches). Loads
  a CSV via `[data] path`/`target` (numeric features + a class column,
  train-split standardization) or falls back to synthetic data. Verified:
  trains to ~0 loss on the Python tabular `example.csv`, read back by
  `nexis-ml runs` unchanged.

## [0.1.0] — 2026-06-14 — M1: foundation

### Added
- First slice: a single Rust binary (named `nexis-ml` for drop-in detection)
  implementing **protocol v1 NDJSON** and the **exact Nexis run-store layout**
  (UTC run ids, atomic writes, `config.json` / `metrics.jsonl` /
  `summary.json` / `artifacts/`), with CLI `--version` / `env` / `new` /
  `train`. `train` drives a built-in linear classifier on synthetic data
  through the full `Run` lifecycle. Verified: a Rust-produced run is listed by
  the Python `nexis-ml runs` unchanged. `cargo test` / `clippy -D warnings` /
  `fmt` clean.
