# nexis-ml-rs

A **Python-free, single-binary** ML engine for the [Nexis](https://github.com/rwetz/Nexis)
terminal — Phase 3 of the ML Suite (see `ML_SUITE.md` in the Nexis repo).

It speaks the **same NDJSON protocol** and writes the **same run store** as
the Python [`nexis-ml`](https://github.com/rwetz/nexis-ml) engine, so Nexis
renders runs from either with zero changes. The goal: an LSP-style
downloadable engine for machines without a Python/PyTorch toolchain.

> **Status: real `burn` MLP + CNN, CPU *and* GPU.** Protocol, run store, and
> CLI are complete and verified end-to-end (a Rust-produced run is read by the
> Python `nexis-ml runs`). `train` runs on [`burn`](https://github.com/tracel-ai/burn) —
> pick the backend with `[train] device` (`auto`/`cpu`/`gpu`): GPU runs on
> burn's **wgpu** backend (Vulkan/DX12/Metal, no vendor toolchain), CPU on
> ndarray; both with autodiff. The model is **declared in `train.toml`**: a
> CSV `[data] path` (or none → synthetic) trains a variable-depth MLP; a
> folder of class sub-folders trains a CNN over its images. Next: `export
> --onnx`, then Nexis download/detect — see [PLAN.md](PLAN.md).

## Build & run

```sh
cargo build --release           # produces target/release/nexis-ml(.exe)

nexis-ml --version              # → "nexis-ml 0.5.2" (Nexis-detectable)
nexis-ml env                    # one-line JSON capability report (backend: cpu|wgpu)
nexis-ml new tabular my-run     # scaffold a project (templates: tabular | image)
nexis-ml train my-run           # train; writes .nexis-ml/runs/<id>/
nexis-ml --nexis-protocol train my-run   # stream NDJSON protocol on stdout
nexis-ml export --onnx my-run   # train the MLP and write my-run/model.onnx
```

### CPU or GPU

Set `[train] device` in `train.toml`:

| value | backend |
|---|---|
| `auto` (default) | GPU via wgpu when an adapter is present, else CPU |
| `gpu` / `cuda` / `wgpu` | GPU via wgpu (warns and falls back to CPU if none) |
| `cpu` | ndarray CPU backend |

The GPU path uses burn's **wgpu** backend (Vulkan/DX12/Metal/OpenGL) — no
CUDA or vendor toolchain required, so the same binary uses the GPU on any
modern machine. GPU runs emit a per-epoch `mem/gpu_mb` footprint metric.

### The model (declarative)

The model is described in `train.toml`, not code — chosen by what `[data]
path` points at:

| `[data] path` | model | `[model]` keys |
|---|---|---|
| a `.csv` (or unset → synthetic) | tabular **MLP** | `hidden = 16` or `hidden = [64, 32]` |
| a folder of class sub-folders | image **CNN** | `conv1`, `conv2`, `hidden` |

`hidden` as a list sets the MLP's hidden-layer widths (any depth; an empty
list is a bare linear classifier). The CNN decodes folder-per-class images to
grayscale (PNG/JPEG/BMP, resized to the first image's size) and emits a
per-epoch confusion matrix plus an `image-grid` PNG of sample predictions.

### ONNX export

`nexis-ml export --onnx [dir]` trains the tabular MLP from `train.toml` and
writes `<dir>/model.onnx` — a portable model for `ort`/onnxruntime inference
without Python. Standardization is baked into the graph, so it takes **raw
features** (input name `input`) and returns class logits (`output`). burn has
no native ONNX export, so the protobuf is hand-encoded (no extra dependency);
the export is verified against onnxruntime for an exact prediction match.
(CNN/image ONNX export is a follow-up.)

The binary is named `nexis-ml` (not `nexis-ml-rs`) so Nexis detects and
spawns it exactly like the Python engine — the two are never on `PATH`
together; this one is downloaded to its own directory.

## Compatibility

A run produced here is indistinguishable to Nexis from a Python-engine run:

```sh
nexis-ml train demo                 # Rust engine writes the run
python -m nexis_ml.cli runs demo    # Python engine lists it ✓
```

Same `protocol` version (1), same `metric`/`epoch`/`artifact`/
`run.finished` events, same `config.json` / `metrics.jsonl` /
`summary.json` / `artifacts/` layout.

In protocol mode the engine also honors the same stdin **control commands**
as the Python harness — `{"cmd":"cancel"}` (stop gracefully and finish as
`cancelled`, keeping the last checkpoint), `{"cmd":"pause"}` /
`{"cmd":"resume"}` (block/release at the next epoch boundary) — so Nexis's
Stop and Pause buttons work against either engine.

## Layout

| File | Role |
|---|---|
| `src/protocol.rs` | NDJSON emitter (protocol v1) |
| `src/run_store.rs` | run directory + atomic file writes (UTC stamp, no deps) |
| `src/harness.rs` | `Run` lifecycle + stdin control watcher (cancel/pause/resume) |
| `src/model.rs` | the `train` command's `burn` MLP + CNN (CPU/GPU), device selection, CSV/synthetic + image data |
| `src/onnx.rs` | dependency-free ONNX writer for the tabular MLP (`export --onnx`) |
| `src/main.rs` | CLI (`--version` / `env` / `new` / `train` / `export`) |

## License

Apache-2.0, same as Nexis.
