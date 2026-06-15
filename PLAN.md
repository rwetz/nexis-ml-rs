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

### Run control (cancel/pause/resume) ✅ (2026-06-14, v0.2.1)
The engine now honors the same stdin **control commands** as the Python
harness: a daemon-style watcher reads `{"cmd":"cancel"|"pause"|"resume"}`
in protocol mode and flips shared flags. `cancel` makes the training loop
break and finish as `cancelled` (keeping the last checkpoint); `pause` /
`resume` block/release at the next epoch boundary. Closes the gap where
Nexis's Stop/Pause buttons were no-ops against the Rust engine.

### M3 — GPU via `wgpu` ✅ (2026-06-14, v0.3.0)
`model.rs` is now generic over the burn backend and monomorphizes
`run_training` for CPU (`Autodiff<NdArray>`) or GPU
(`Autodiff<Wgpu>`) chosen from `[train] device`: `cpu`, `gpu`/`cuda`/`wgpu`
(warns + falls back if no adapter), or `auto`/empty (silently prefers GPU).
GPU availability is probed by forcing a one-element wgpu allocation inside
`catch_unwind` so a headless/driverless box degrades to CPU instead of
aborting. The resolved `device` rides on `run.started` + `summary.json`, and
a per-epoch `mem/gpu_mb` line reports the wgpu compute client's
`bytes_in_use` (the analogue of the Python engine's
`torch.cuda.memory_allocated`; CPU emits none). `nexis-ml env` reports
`backend: "wgpu"` when a GPU is present. Verified on an RTX 4070 SUPER:
`device=auto` trains on GPU to acc 1.0, the Python `nexis-ml runs` reads the
GPU-produced run unchanged, and `device=cpu` still works. The wgpu backend
needs no vendor toolchain — the headline "GPU on any box" win over
CUDA-only PyTorch.

### M4 — declarative model presets ✅ (2026-06-14, v0.4.0)
The model is described in `train.toml`, not code (the Rust engine can't run a
user's `train.py`, so config is the editable surface). The model kind is
chosen from `[data] path`: a folder of class sub-folders → **CNN** over its
images; a CSV (or no path → synthetic) → **MLP** over tabular rows.
- **Tabular MLP**: `[model] hidden` is a single width (`16`) or an explicit
  list of widths (`[64, 32]`) → an arbitrary-depth `Vec<Linear>` (empty list
  = bare linear classifier).
- **Image CNN**: folder-per-class images decoded to grayscale via the
  [`image`](https://crates.io/crates/image) crate (resized to the first
  image's size), trained by a `conv1`/`conv2`/`hidden` CNN (3×3 Same convs +
  2×2 pools, flat size found by a dry run). Emits the per-epoch
  confusion-matrix and an `image-grid` PNG (green/red correctness borders),
  matching the Python `image` template.

Both run the same `Run` lifecycle (cancel/pause, device selection,
`mem/gpu_mb`). Verified on the RTX 4070 SUPER: `hidden=[32,16]` trains and
records the layer list; a 4-class image folder trains to acc 1.0 on CPU and
GPU with both artifacts, and the Python `nexis-ml runs` reads either run.

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
