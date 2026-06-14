// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Nexis ML (Rust)                     ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! The `train` command's model — a multinomial logistic-regression
//! classifier (linear + softmax + cross-entropy, minibatch SGD) on a
//! synthetic two-blob dataset. Pure Rust, no ML framework: this slice
//! exists to prove the protocol + run-store + CLI are Nexis-compatible
//! end to end. The real model backend is `burn` (see PLAN.md) — it will
//! replace the math here behind the same `Run` lifecycle.

use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::json;

use crate::harness::Run;
use crate::protocol::Emitter;
use crate::run_store;

const FEATURES: usize = 3; // x1, x2, and a noise column the model should ignore
const CLASSES: usize = 2;

#[derive(Deserialize)]
#[serde(default)]
pub struct TrainCfg {
    pub epochs: u32,
    pub batch_size: usize,
    pub lr: f64,
    pub val_split: f64,
    pub seed: u64,
    pub samples: usize,
    pub device: String,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 16,
            lr: 0.2,
            val_split: 0.2,
            seed: 42,
            samples: 240,
            device: "cpu".into(),
        }
    }
}

#[derive(Deserialize, Default)]
struct Root {
    #[serde(default)]
    train: TrainCfg,
}

pub fn load_cfg(project: &Path) -> TrainCfg {
    let text = fs::read_to_string(project.join("train.toml")).unwrap_or_default();
    toml::from_str::<Root>(&text)
        .map(|r| r.train)
        .unwrap_or_default()
}

/// Minimal deterministic RNG (SplitMix64) — keeps the engine dependency-free.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        // Box–Muller
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn make_data(cfg: &TrainCfg) -> (Vec<[f64; FEATURES]>, Vec<usize>) {
    let mut rng = Rng(cfg.seed);
    let mut xs = Vec::with_capacity(cfg.samples);
    let mut ys = Vec::with_capacity(cfg.samples);
    for i in 0..cfg.samples {
        let label = i % 2;
        let cx = if label == 0 { -1.5 } else { 1.5 };
        xs.push([
            cx + rng.gauss() * 0.7,
            rng.gauss() * 0.9,
            rng.gauss(), // pure noise
        ]);
        ys.push(label);
    }
    (xs, ys)
}

fn softmax(logits: [f64; CLASSES]) -> [f64; CLASSES] {
    let m = logits.iter().cloned().fold(f64::MIN, f64::max);
    let exps: [f64; CLASSES] = [(logits[0] - m).exp(), (logits[1] - m).exp()];
    let sum = exps[0] + exps[1];
    [exps[0] / sum, exps[1] / sum]
}

fn forward(
    w: &[[f64; FEATURES]; CLASSES],
    b: &[f64; CLASSES],
    x: &[f64; FEATURES],
) -> [f64; CLASSES] {
    let mut logits = *b;
    for (k, row) in w.iter().enumerate() {
        for d in 0..FEATURES {
            logits[k] += row[d] * x[d];
        }
    }
    logits
}

/// Runs the whole training loop, driving the protocol/run store via `Run`.
pub fn train(project: &Path, emitter: &Emitter) -> std::io::Result<i32> {
    let cfg = load_cfg(project);
    let (mut xs, mut ys) = make_data(&cfg);

    // Shuffle, split, standardize on train stats (mirrors the tabular template).
    let mut rng = Rng(cfg.seed ^ 0xABCD);
    for i in (1..xs.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        xs.swap(i, j);
        ys.swap(i, j);
    }
    let n_val = ((xs.len() as f64 * cfg.val_split) as usize).max(1);
    let (val_x, train_x) = xs.split_at(n_val);
    let (val_y, train_y) = ys.split_at(n_val);

    let mut mean = [0.0; FEATURES];
    let mut std = [0.0; FEATURES];
    for row in train_x {
        for d in 0..FEATURES {
            mean[d] += row[d];
        }
    }
    for m in &mut mean {
        *m /= train_x.len() as f64;
    }
    for row in train_x {
        for d in 0..FEATURES {
            std[d] += (row[d] - mean[d]).powi(2);
        }
    }
    for s in &mut std {
        *s = (*s / train_x.len() as f64).sqrt().max(1e-8);
    }
    let norm = |row: &[f64; FEATURES]| {
        let mut o = [0.0; FEATURES];
        for d in 0..FEATURES {
            o[d] = (row[d] - mean[d]) / std[d];
        }
        o
    };
    let train_xn: Vec<[f64; FEATURES]> = train_x.iter().map(norm).collect();
    let val_xn: Vec<[f64; FEATURES]> = val_x.iter().map(norm).collect();

    let device = if cfg.device == "cpu" {
        "cpu"
    } else {
        emitter.console(&format!(
            "note: device \"{}\" not supported by the Rust engine yet — using CPU",
            cfg.device
        ));
        "cpu"
    };

    let dir = run_store::new_run_dir(project, "tabular")?;
    let config_json = json!({
        "train": {
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "val_split": cfg.val_split, "seed": cfg.seed, "samples": cfg.samples,
            "device": cfg.device,
        },
        "engine": "nexis-ml-rs",
        "derived": {"classes": ["0", "1"], "task": "classification",
                    "features": ["x1", "x2", "noise"]},
    });
    let mut run = Run::start(emitter, dir, "tabular", config_json, cfg.epochs, device);
    run.info(&format!(
        "linear classifier (Rust engine): {} train / {} val rows, {FEATURES} features",
        train_xn.len(),
        val_xn.len()
    ));

    let mut w = [[0.0; FEATURES]; CLASSES];
    let mut b = [0.0; CLASSES];
    let mut order: Vec<usize> = (0..train_xn.len()).collect();
    let mut best_val = f64::INFINITY;

    for epoch in 1..=cfg.epochs {
        for i in (1..order.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            order.swap(i, j);
        }
        for chunk in order.chunks(cfg.batch_size.max(1)) {
            let mut gw = [[0.0; FEATURES]; CLASSES];
            let mut gb = [0.0; CLASSES];
            let mut loss = 0.0;
            for &idx in chunk {
                let x = &train_xn[idx];
                let y = train_y[idx];
                let p = softmax(forward(&w, &b, x));
                loss += -(p[y].max(1e-12)).ln();
                for k in 0..CLASSES {
                    let d_logit = p[k] - if k == y { 1.0 } else { 0.0 };
                    gb[k] += d_logit;
                    for d in 0..FEATURES {
                        gw[k][d] += d_logit * x[d];
                    }
                }
            }
            let m = chunk.len() as f64;
            for k in 0..CLASSES {
                b[k] -= cfg.lr * gb[k] / m;
                for d in 0..FEATURES {
                    w[k][d] -= cfg.lr * gw[k][d] / m;
                }
            }
            run.log(&[("loss/train", loss / m)], epoch);
        }

        // Validation: loss, accuracy, confusion matrix.
        let mut vloss = 0.0;
        let mut correct = 0;
        let mut cm = [[0u32; CLASSES]; CLASSES];
        for (x, &y) in val_xn.iter().zip(val_y) {
            let p = softmax(forward(&w, &b, x));
            vloss += -(p[y].max(1e-12)).ln();
            let pred = if p[1] >= p[0] { 1 } else { 0 };
            if pred == y {
                correct += 1;
            }
            cm[y][pred] += 1;
        }
        let vloss = vloss / val_xn.len() as f64;
        let acc = correct as f64 / val_xn.len() as f64;
        run.log(&[("loss/val", vloss), ("acc/val", acc)], epoch);

        let cm_path = run.artifacts_dir().join(format!("cm-epoch{epoch}.json"));
        let cm_json = json!({
            "labels": ["0", "1"],
            "matrix": [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
        });
        let _ = fs::write(
            &cm_path,
            serde_json::to_string(&cm_json).unwrap_or_default(),
        );
        run.artifact("confusion-matrix", &cm_path);

        if vloss < best_val {
            best_val = vloss;
            let ckpt = json!({"w": w, "b": b, "mean": mean, "std": std, "classes": ["0", "1"]});
            let _ = fs::write(
                run.checkpoints_dir().join("best.json"),
                serde_json::to_string_pretty(&ckpt).unwrap_or_default(),
            );
        }

        run.epoch(epoch);
    }

    run.info(&format!("best val loss: {best_val:.4}"));
    run.finish("ok");
    Ok(0)
}
