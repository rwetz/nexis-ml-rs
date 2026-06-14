# nexis-ml-rs — plan

Phase 3 of the Nexis ML Suite: a downloadable, Python-free engine that
implements the same protocol as the Python `nexis-ml`. Build only as far
as real usage justifies (the spec gates Phase 3 on Phase 1/2 adoption).

## Milestones

### M1 — foundation ✅ (2026-06-14)
Protocol v1 NDJSON emitter, Nexis-compatible run store (UTC run ids,
atomic writes, `metrics.jsonl`/`config.json`/`summary.json`/`artifacts/`),
CLI (`--version`, `env`, `new`, `train`), and a built-in **linear
classifier** on synthetic two-blob data driving the full `Run` lifecycle
(run.started → metric/epoch → confusion-matrix artifact → run.finished).
Verified: a Rust-produced run is listed by the Python `nexis-ml runs`.
`cargo test`/`clippy -D warnings`/`fmt` clean.

### M2 — `burn` backend (real models, CPU) ✅ (2026-06-14)
`model.rs` now trains a real **MLP** (configurable hidden width) on
[`burn`](https://github.com/tracel-ai/burn)'s **ndarray/CPU** backend with
autodiff (Adam + cross-entropy, minibatches), behind the same `Run`
lifecycle. Loads a CSV via `[data] path`/`target` (numeric features + a
class column, train-split standardization) or falls back to synthetic
data. Verified: trains to ~0 loss on the Python tabular `example.csv`, and
the Python `nexis-ml runs` reads the burn-produced run unchanged. v0.2.0.

### M3 — GPU via `wgpu`
Add burn's `wgpu` backend so it runs on any modern GPU without a vendor
toolchain (the headline "GPU on any box" win over CUDA-only PyTorch).
`device = auto|cpu|gpu` in `train.toml`, mirroring the Python engine; emit
`device` on `run.started` and a `mem/gpu_mb` line where the backend exposes it.

### M4 — declarative model presets
`train.toml` describes the model (MLP/CNN presets: layer sizes, conv
channels) rather than code — the Rust engine can't run a user's `train.py`,
so configuration is the editable surface here. Cover the `tabular` and
`image` template shapes.

### M5 — `export --onnx`
Export a trained model to ONNX (door-opener for an `ort`-based inference
path), matching the stretch item in ML_SUITE.md.

### M6 — Nexis integration (download + detect)
Teach Nexis to download this binary to a managed dir and detect it
(`--version`) like an LSP server, then spawn it via the existing `ml_*`
commands. No Nexis protocol changes needed — only a download/locate path.

## Non-goals (carried from ML_SUITE.md)
No cloud, no distributed training, no model zoo. Small models, small data,
local-first.
