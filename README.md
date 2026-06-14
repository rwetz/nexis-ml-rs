# nexis-ml-rs

A **Python-free, single-binary** ML engine for the [Nexis](https://github.com/rwetz/Nexis)
terminal — Phase 3 of the ML Suite (see `ML_SUITE.md` in the Nexis repo).

It speaks the **same NDJSON protocol** and writes the **same run store** as
the Python [`nexis-ml`](https://github.com/rwetz/nexis-ml) engine, so Nexis
renders runs from either with zero changes. The goal: an LSP-style
downloadable engine for machines without a Python/PyTorch toolchain.

> **Status: foundation slice.** Protocol, run store, and CLI are complete
> and verified end-to-end (a Rust-produced run is read by the Python
> `nexis-ml runs`). The `train` command currently uses a small built-in
> linear classifier on synthetic data to prove the pipeline. The real
> model backend is [`burn`](https://github.com/tracel-ai/burn) — see
> [PLAN.md](PLAN.md).

## Build & run

```sh
cargo build --release           # produces target/release/nexis-ml(.exe)

nexis-ml --version              # → "nexis-ml 0.1.0" (Nexis-detectable)
nexis-ml env                    # one-line JSON capability report
nexis-ml new my-run             # scaffold a train.toml
nexis-ml train my-run           # train; writes .nexis-ml/runs/<id>/
nexis-ml --nexis-protocol train my-run   # stream NDJSON protocol on stdout
```

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

## Layout

| File | Role |
|---|---|
| `src/protocol.rs` | NDJSON emitter (protocol v1) |
| `src/run_store.rs` | run directory + atomic file writes (UTC stamp, no deps) |
| `src/harness.rs` | `Run` lifecycle — ties emitter to the run store |
| `src/model.rs` | the `train` command's classifier (to be `burn`-backed) |
| `src/main.rs` | CLI (`--version` / `env` / `new` / `train`) |

## License

Apache-2.0, same as Nexis.
