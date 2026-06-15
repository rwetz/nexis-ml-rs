// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Nexis ML (Rust)                     ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! nexis-ml (Rust engine, Phase 3) — a Python-free, single-binary engine
//! speaking the same NDJSON protocol and writing the same run store as the
//! Python `nexis-ml`, so Nexis consumes either with no changes.
//!
//! Commands: --version | env | new <template> [dir] | train [dir] | export --onnx [dir]
//! Exit codes: 0 ok, 1 error, 2 usage.

mod harness;
mod model;
mod onnx;
mod protocol;
mod run_store;

use std::path::Path;

use protocol::{Emitter, ENV_FLAG};

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let proto_flag = args.iter().any(|a| a == "--nexis-protocol");
    if proto_flag {
        // Mirror the Python engine: the flag rides in the env so it's
        // inherited consistently.
        std::env::set_var(ENV_FLAG, "1");
    }
    if args.iter().any(|a| a == "--version") {
        // Format matches the Python engine so Nexis's detector parses it.
        println!("nexis-ml {}", env!("CARGO_PKG_VERSION"));
        return 0;
    }

    let cmd = args.iter().find(|a| !a.starts_with("--")).cloned();
    let emitter = Emitter::from_env_or_flag(proto_flag);
    match cmd.as_deref() {
        Some("env") => cmd_env(),
        Some("new") => cmd_new(&positionals(&args, "new")),
        Some("train") => cmd_train(&positionals(&args, "train"), &emitter),
        Some("export") => cmd_export(&args, &positionals(&args, "export"), &emitter),
        Some(other) => {
            eprintln!("error: unknown command: {other}");
            2
        }
        None => {
            eprintln!("usage: nexis-ml [--version] <env|new|train|export> [dir]");
            2
        }
    }
}

/// Non-flag args after the command token (the command itself dropped once).
fn positionals(args: &[String], cmd: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut dropped = false;
    for a in args {
        if a.starts_with("--") {
            continue;
        }
        if !dropped && a == cmd {
            dropped = true;
            continue;
        }
        out.push(a.clone());
    }
    out
}

/// Machine-readable capability report — one JSON line. Keeps the
/// python/torch/cudaAvailable/gpuName keys so Nexis's env probe doesn't
/// need a special case, and adds engine/backend fields.
fn cmd_env() -> i32 {
    // wgpu GPU is reported via `backend` (cudaAvailable/gpuName stay
    // CUDA-specific — they drive the Python engine's torch-install UI).
    let backend = if model::gpu_available() {
        "wgpu"
    } else {
        "cpu"
    };
    let info = serde_json::json!({
        "engine": "nexis-ml-rs",
        "nexisMl": env!("CARGO_PKG_VERSION"),
        "python": serde_json::Value::Null,
        "torch": serde_json::Value::Null,
        "cudaAvailable": false,
        "gpuName": serde_json::Value::Null,
        "backend": backend,
    });
    println!("{}", serde_json::to_string(&info).unwrap_or_default());
    0
}

const TABULAR_TOML: &str = "\
# nexis-ml-rs (Phase 3 engine) — an MLP over tabular data.
# Point [data] at a CSV, or leave it for built-in synthetic data, then
# run `nexis-ml train`.

[train]
epochs = 20
batch_size = 16
lr = 0.2
val_split = 0.2
seed = 42
samples = 240        # synthetic two-blob points (when no [data] path)
device = \"auto\"       # auto | cpu | gpu  (auto uses the GPU via wgpu when present)

[model]
hidden = [16]        # MLP hidden-layer widths (a single int also works)
";

const IMAGE_TOML: &str = "\
# nexis-ml-rs (Phase 3 engine) — a small CNN over folder-per-class images.
# Put images in data/<class>/*.png (one folder per class), then run
# `nexis-ml train`.

[train]
epochs = 12
batch_size = 16
lr = 0.01
val_split = 0.2
seed = 42
device = \"auto\"       # auto | cpu | gpu

[data]
path = \"data\"         # a folder of class sub-folders (one per class)

[model]
conv1 = 16
conv2 = 32
hidden = 64
";

/// `new <template> [dir]` — matches the Python engine's argument order
/// (template first; dir defaults to `./<template>`). Scaffolds the
/// template's `train.toml`. The Rust engine supports `tabular` and `image`;
/// `textgen` is Python-only.
fn cmd_new(pos: &[String]) -> i32 {
    let template = pos.first().map(String::as_str).unwrap_or("tabular");
    let dir = pos.get(1).map(String::as_str).unwrap_or(template);
    let toml = match template {
        "tabular" => TABULAR_TOML,
        "image" => IMAGE_TOML,
        other => {
            eprintln!("error: unknown template '{other}' (this engine supports: tabular, image)");
            return 2;
        }
    };
    let path = Path::new(dir);
    if let Err(e) = std::fs::create_dir_all(path) {
        eprintln!("error: {e}");
        return 1;
    }
    if let Err(e) = std::fs::write(path.join("train.toml"), toml) {
        eprintln!("error: {e}");
        return 1;
    }
    // Protocol mode keeps stdout clean (humans read stderr).
    if std::env::var(ENV_FLAG).as_deref() == Ok("1") {
        eprintln!("created {template} project at {dir}");
        return 0;
    }
    println!("created {template} project at {dir}\nnext:\n  cd {dir}\n  nexis-ml train");
    0
}

fn cmd_train(pos: &[String], emitter: &Emitter) -> i32 {
    let dir = pos.first().map(String::as_str).unwrap_or(".");
    match model::train(Path::new(dir), emitter) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

/// `export --onnx [dir]` — train the tabular MLP from train.toml and write it
/// as `<dir>/model.onnx`. `--onnx` is required (the only supported format);
/// requiring it keeps room for other formats later.
fn cmd_export(args: &[String], pos: &[String], emitter: &Emitter) -> i32 {
    if !args.iter().any(|a| a == "--onnx") {
        eprintln!("error: export requires --onnx (the only supported format)");
        return 2;
    }
    let dir = pos.first().map(String::as_str).unwrap_or(".");
    let out = Path::new(dir).join("model.onnx");
    match model::export_onnx(Path::new(dir), &out) {
        Ok(()) => {
            emitter.console(&format!("wrote {}", out.display()));
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}
