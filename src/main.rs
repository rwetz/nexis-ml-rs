// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Nexis ML (Rust)                     ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! nexis-ml (Rust engine, Phase 3) — a Python-free, single-binary engine
//! speaking the same NDJSON protocol and writing the same run store as the
//! Python `nexis-ml`, so Nexis consumes either with no changes.
//!
//! Commands: --version | env | new [dir] | train [dir]
//! Exit codes: 0 ok, 1 error, 2 usage.

mod harness;
mod model;
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
        Some(other) => {
            eprintln!("error: unknown command: {other}");
            2
        }
        None => {
            eprintln!("usage: nexis-ml [--version] <env|new|train> [dir]");
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
    let info = serde_json::json!({
        "engine": "nexis-ml-rs",
        "nexisMl": env!("CARGO_PKG_VERSION"),
        "python": serde_json::Value::Null,
        "torch": serde_json::Value::Null,
        "cudaAvailable": false,
        "gpuName": serde_json::Value::Null,
        "backend": "cpu",
    });
    println!("{}", serde_json::to_string(&info).unwrap_or_default());
    0
}

const DEFAULT_TOML: &str = "\
# nexis-ml-rs (Phase 3 engine) — a linear classifier on synthetic data.
# Edit and re-run `nexis-ml train`.

[train]
epochs = 20
batch_size = 16
lr = 0.2
val_split = 0.2
seed = 42
samples = 240        # synthetic two-blob points to generate
device = \"cpu\"        # cpu only for now; GPU arrives with the burn backend
";

fn cmd_new(pos: &[String]) -> i32 {
    let dir = pos
        .first()
        .map(String::as_str)
        .unwrap_or("nexis-ml-project");
    let path = Path::new(dir);
    if let Err(e) = std::fs::create_dir_all(path) {
        eprintln!("error: {e}");
        return 1;
    }
    let toml_path = path.join("train.toml");
    if let Err(e) = std::fs::write(&toml_path, DEFAULT_TOML) {
        eprintln!("error: {e}");
        return 1;
    }
    let out = if std::env::var(ENV_FLAG).as_deref() == Ok("1") {
        eprintln!("created project at {dir}");
        return 0;
    } else {
        format!("created project at {dir}\nnext:\n  cd {dir}\n  nexis-ml train")
    };
    println!("{out}");
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
