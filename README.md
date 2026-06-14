# nexis-ml-rs

A **Python-free, single-binary** ML engine for the [Nexis](https://github.com/rwetz/Nexis)
terminal — Phase 3 of the ML Suite (see `ML_SUITE.md` in the Nexis repo).

It speaks the **same NDJSON protocol** and writes the **same run store** as
the Python [`nexis-ml`](https://github.com/rwetz/nexis-ml) engine, so Nexis
renders runs from either with zero changes. The goal: an LSP-style
downloadable engine for machines without a Python/PyTorch toolchain.

> **Status: real `burn` MLP, CPU *and* GPU.** Protocol, run store, and CLI
> are complete and verified end-to-end (a Rust-produced run is read by the
> Python `nexis-ml runs`). `train` runs a true MLP classifier on
> [`burn`](https://github.com/tracel-ai/burn) — pick the backend with
> `[train] device` (`auto`/`cpu`/`gpu`): GPU runs on burn's **wgpu** backend
> (Vulkan/DX12/Metal, no vendor toolchain), CPU on ndarray; both with
> autodiff. Load a CSV via `[data] path` (the "my spreadsheet" case) or fall
> back to built-in synthetic data. Next: declarative MLP/CNN presets — see
> [PLAN.md](PLAN.md).

## Build & run

```sh
cargo build --release           # produces target/release/nexis-ml(.exe)

nexis-ml --version              # → "nexis-ml 0.3.0" (Nexis-detectable)
nexis-ml env                    # one-line JSON capability report (backend: cpu|wgpu)
nexis-ml new my-run             # scaffold a train.toml (device = "auto")
nexis-ml train my-run           # train; writes .nexis-ml/runs/<id>/
nexis-ml --nexis-protocol train my-run   # stream NDJSON protocol on stdout
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
| `src/model.rs` | the `train` command's `burn` MLP (CPU/GPU) + device selection + CSV/synthetic data |
| `src/main.rs` | CLI (`--version` / `env` / `new` / `train`) |

## License

Apache-2.0, same as Nexis.
