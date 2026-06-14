// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Nexis ML (Rust)                     ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! The `train` command's model — a real MLP classifier built on
//! [`burn`](https://github.com/tracel-ai/burn), generic over the backend:
//! `[train] device` picks CPU (ndarray) or GPU (wgpu), both with autodiff.
//! Loads a CSV when `[data] path` points at one (the "my spreadsheet, what
//! predicts what" case), otherwise trains on a built-in synthetic two-blob
//! dataset so `train` works out of the box. Drives the same `Run` lifecycle
//! as the Python engine, so Nexis renders it identically.

use std::fs;
use std::path::Path;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::{
    backend::AutodiffBackend, backend::Backend, ElementConversion, Int, Tensor, TensorData,
};
use serde::Deserialize;
use serde_json::json;

use crate::harness::Run;
use crate::protocol::Emitter;
use crate::run_store;

// ── config (train.toml) ───────────────────────────────────────────────

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
            epochs: 30,
            batch_size: 16,
            lr: 0.05,
            val_split: 0.2,
            seed: 42,
            samples: 240,
            device: "cpu".into(),
        }
    }
}

#[derive(Deserialize, Default)]
struct DataCfg {
    path: Option<String>,
    target: Option<String>,
}

#[derive(Deserialize)]
#[serde(default)]
struct ModelCfg {
    hidden: usize,
}
impl Default for ModelCfg {
    fn default() -> Self {
        Self { hidden: 16 }
    }
}

#[derive(Deserialize, Default)]
struct Root {
    #[serde(default)]
    train: TrainCfg,
    #[serde(default)]
    data: DataCfg,
    #[serde(default)]
    model: ModelCfg,
}

fn load_root(project: &Path) -> Root {
    let text = fs::read_to_string(project.join("train.toml")).unwrap_or_default();
    toml::from_str(&text).unwrap_or_default()
}

// ── data ──────────────────────────────────────────────────────────────

struct Dataset {
    x: Vec<Vec<f32>>,
    y: Vec<usize>,
    feature_names: Vec<String>,
    classes: Vec<String>,
}

/// Minimal deterministic RNG (SplitMix64) — keeps the synthetic-data path
/// dependency-free and reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn gauss(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

fn synthetic(cfg: &TrainCfg) -> Dataset {
    let mut rng = Rng(cfg.seed);
    let mut x = Vec::with_capacity(cfg.samples);
    let mut y = Vec::with_capacity(cfg.samples);
    for i in 0..cfg.samples {
        let label = i % 2;
        let cx = if label == 0 { -1.5 } else { 1.5 };
        x.push(vec![
            cx + rng.gauss() * 0.7,
            rng.gauss() * 0.9,
            rng.gauss(), // a pure-noise column the model should learn to ignore
        ]);
        y.push(label);
    }
    Dataset {
        x,
        y,
        feature_names: vec!["x1".into(), "x2".into(), "noise".into()],
        classes: vec!["0".into(), "1".into()],
    }
}

fn load_csv(path: &Path, target: &str) -> std::io::Result<Dataset> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .ok_or_else(|| err(format!("{} is empty", path.display())))?
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let target_idx = header
        .iter()
        .position(|h| h == target)
        .ok_or_else(|| err(format!("target column '{target}' not in {header:?}")))?;

    let rows: Vec<Vec<String>> = lines
        .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
        .collect();
    if rows.is_empty() {
        return Err(err(format!("no data rows in {}", path.display())));
    }

    // Feature columns = every non-target column whose first value is numeric.
    let feature_cols: Vec<usize> = (0..header.len())
        .filter(|&i| {
            i != target_idx && rows[0].get(i).and_then(|v| v.parse::<f32>().ok()).is_some()
        })
        .collect();
    if feature_cols.is_empty() {
        return Err(err("no numeric feature columns found".into()));
    }
    let feature_names: Vec<String> = feature_cols.iter().map(|&i| header[i].clone()).collect();

    let mut classes: Vec<String> = rows
        .iter()
        .filter_map(|r| r.get(target_idx).cloned())
        .collect();
    classes.sort();
    classes.dedup();
    if classes.len() < 2 {
        return Err(err("need at least 2 distinct target classes".into()));
    }
    let class_index = |v: &str| classes.iter().position(|c| c == v).unwrap_or(0);

    let mut x = Vec::with_capacity(rows.len());
    let mut y = Vec::with_capacity(rows.len());
    for r in &rows {
        let feats: Vec<f32> = feature_cols
            .iter()
            .map(|&i| r.get(i).and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0))
            .collect();
        let label = r.get(target_idx).map(|v| class_index(v)).unwrap_or(0);
        x.push(feats);
        y.push(label);
    }
    Ok(Dataset {
        x,
        y,
        feature_names,
        classes,
    })
}

fn err(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

// ── model (MLP via burn) ──────────────────────────────────────────────

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn new(in_dim: usize, hidden: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(in_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, out_dim).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.fc2.forward(relu(self.fc1.forward(x)))
    }
}

// ── training ──────────────────────────────────────────────────────────

fn apply_standardization(rows: &mut [Vec<f32>], mean: &[f32], std: &[f32]) {
    for row in rows.iter_mut() {
        for (d, v) in row.iter_mut().enumerate() {
            *v = (*v - mean[d]) / std[d];
        }
    }
}

/// Compute train-split mean/std, standardize `x` in place, and return the
/// stats so the val split can be transformed with the same numbers.
fn standardize(x: &mut [Vec<f32>], n_features: usize) -> (Vec<f32>, Vec<f32>) {
    let n = x.len().max(1) as f32;
    let mut mean = vec![0.0f32; n_features];
    let mut std = vec![0.0f32; n_features];
    for row in x.iter() {
        for d in 0..n_features {
            mean[d] += row[d];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    for row in x.iter() {
        for d in 0..n_features {
            std[d] += (row[d] - mean[d]).powi(2);
        }
    }
    for s in &mut std {
        *s = (*s / n).sqrt().max(1e-8);
    }
    apply_standardization(x, &mean, &std);
    (mean, std)
}

fn to_tensor<B: Backend>(rows: &[Vec<f32>], n_features: usize, device: &B::Device) -> Tensor<B, 2> {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    // from_data resolves the backend's float dtype from the device and
    // converts, so the same f32 source works for ndarray and wgpu.
    Tensor::<B, 2>::from_data(TensorData::new(flat, [rows.len(), n_features]), device)
}

fn to_targets<B: Backend>(labels: &[usize], device: &B::Device) -> Tensor<B, 1, Int> {
    let ints: Vec<i64> = labels.iter().map(|&v| v as i64).collect();
    Tensor::<B, 1, Int>::from_data(TensorData::new(ints, [labels.len()]), device)
}

// ── device selection ──────────────────────────────────────────────────

/// Probe for a usable wgpu adapter without aborting the process. The wgpu
/// runtime initializes lazily and panics when no adapter is found, so we
/// force a one-element allocation inside `catch_unwind` (panic hook
/// silenced) and report success. `auto` falls back to CPU silently; `gpu`
/// warns first.
pub(crate) fn gpu_available() -> bool {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // keep the probe's panic off stderr
    let ok = std::panic::catch_unwind(|| {
        let device = WgpuDevice::default();
        let _ = Tensor::<Wgpu, 1>::zeros([1], &device).into_data();
    })
    .is_ok();
    std::panic::set_hook(prev);
    ok
}

/// Resolve the `[train] device` preference to a backend choice. `cpu` (and
/// unknown values) use CPU; `gpu`/`cuda`/`wgpu` use the GPU when present
/// (warning on fallback); `auto` (or empty) silently prefers the GPU. Keeps
/// the same vocabulary as the Python engine's `device`.
fn want_gpu(pref: &str, emitter: &Emitter) -> bool {
    match pref.trim().to_ascii_lowercase().as_str() {
        "cpu" => false,
        "gpu" | "cuda" | "wgpu" => {
            if gpu_available() {
                true
            } else {
                emitter.console(
                    "note: device \"gpu\" requested but no compatible GPU adapter found — using CPU",
                );
                false
            }
        }
        "auto" | "" => gpu_available(),
        other => {
            emitter.console(&format!("note: unknown device \"{other}\" — using CPU"));
            false
        }
    }
}

/// Current wgpu memory footprint in MiB (best-effort; `None` if unavailable).
/// Queries the same cubecl compute client the backend uses, keyed by device,
/// so it reflects this run's allocations — the wgpu analogue of the Python
/// engine's `torch.cuda.memory_allocated`.
fn wgpu_mem_mb(device: &WgpuDevice) -> Option<f64> {
    use burn::cubecl::Runtime;
    let client = burn::backend::wgpu::WgpuRuntime::client(device);
    let usage = client.memory_usage().ok()?;
    Some(usage.bytes_in_use as f64 / (1024.0 * 1024.0))
}

// ── training ──────────────────────────────────────────────────────────

/// Pick the backend from the configured device and run training on it. The
/// backend is a compile-time type, so we monomorphize `run_training` for
/// each path rather than dispatching dynamically.
pub fn train(project: &Path, emitter: &Emitter) -> std::io::Result<i32> {
    let root = load_root(project);
    if want_gpu(&root.train.device, emitter) {
        let device = WgpuDevice::default();
        let mem_dev = device.clone();
        run_training::<Autodiff<Wgpu>>(project, emitter, &root, device, "gpu", move || {
            wgpu_mem_mb(&mem_dev)
        })
    } else {
        run_training::<Autodiff<NdArray>>(
            project,
            emitter,
            &root,
            NdArrayDevice::default(),
            "cpu",
            || None,
        )
    }
}

// `slice([a..b])` is burn's ranges-array API, not an accidental 1-element
// range vec — clippy's lint is a false positive here.
#[allow(clippy::single_range_in_vec_init)]
fn run_training<B: AutodiffBackend>(
    project: &Path,
    emitter: &Emitter,
    root: &Root,
    device: B::Device,
    device_label: &str,
    mem_probe: impl Fn() -> Option<f64>,
) -> std::io::Result<i32> {
    let cfg = &root.train;

    // Load CSV if configured, else synthetic.
    let mut data = match &root.data.path {
        Some(p) if project.join(p).is_file() => {
            let target = root.data.target.as_deref().unwrap_or("label");
            load_csv(&project.join(p), target)?
        }
        _ => synthetic(cfg),
    };
    let n_features = data.feature_names.len();
    let n_classes = data.classes.len();

    // Shuffle + split.
    let mut rng = Rng(cfg.seed ^ 0xABCD);
    for i in (1..data.x.len()).rev() {
        let j = rng.below(i + 1);
        data.x.swap(i, j);
        data.y.swap(i, j);
    }
    let n_val = ((data.x.len() as f64 * cfg.val_split) as usize).max(1);
    let mut train_x = data.x.split_off(n_val);
    let train_y = data.y.split_off(n_val);
    let mut val_x = data.x; // first n_val
    let val_y = data.y;

    // Standardize on train stats, apply the same transform to the val split.
    let (mean, std) = standardize(&mut train_x, n_features);
    apply_standardization(&mut val_x, &mean, &std);

    let x_train = to_tensor::<B>(&train_x, n_features, &device);
    let y_train = to_targets::<B>(&train_y, &device);
    let x_val = to_tensor::<B>(&val_x, n_features, &device);

    let mut model = Mlp::<B>::new(n_features, root.model.hidden, n_classes, &device);
    let mut opt = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    let dir = run_store::new_run_dir(project, "tabular")?;
    let config_json = json!({
        "train": {
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "val_split": cfg.val_split, "seed": cfg.seed, "device": cfg.device,
        },
        "model": {"hidden": root.model.hidden},
        "engine": "nexis-ml-rs",
        "derived": {"classes": data.classes, "task": "classification",
                    "features": data.feature_names},
    });
    let mut run = Run::start(
        emitter,
        dir,
        "tabular",
        config_json,
        cfg.epochs,
        device_label,
    );
    run.info(&format!(
        "burn MLP (Rust engine, {device_label}): {} train / {} val rows, {n_features} features, {n_classes} classes, hidden={}",
        train_x.len(),
        val_x.len(),
        root.model.hidden
    ));

    let n_train = train_x.len();
    let bs = cfg.batch_size.max(1);
    let mut best_val = f64::INFINITY;

    for epoch in 1..=cfg.epochs {
        // Contiguous minibatches over the (once-shuffled) training tensor.
        let mut start = 0;
        while start < n_train {
            let end = (start + bs).min(n_train);
            let xb = x_train.clone().slice([start..end]);
            let yb = y_train.clone().slice([start..end]);
            let logits = model.forward(xb);
            let loss = loss_fn.forward(logits, yb);
            let loss_val = loss.clone().into_scalar().elem::<f64>();
            let grads = loss.backward();
            let gp = GradientsParams::from_grads(grads, &model);
            model = opt.step(cfg.lr, model, gp);
            run.log(&[("loss/train", loss_val)], epoch);
            start = end;
        }

        // Stop promptly on a cancel command — the last best.json is kept.
        if run.cancelled() {
            break;
        }

        // Validation.
        let logits = model.forward(x_val.clone());
        let vloss = loss_fn
            .forward(logits.clone(), to_targets::<B>(&val_y, &device))
            .into_scalar()
            .elem::<f64>();
        // iter() converts the backend's int dtype (i64 on ndarray, i32 on
        // wgpu); to_vec() would reject the mismatch and silently drop preds.
        let preds: Vec<i64> = logits.argmax(1).into_data().iter::<i64>().collect();
        let mut correct = 0usize;
        let mut cm = vec![vec![0u32; n_classes]; n_classes];
        for (p, &t) in preds.iter().zip(val_y.iter()) {
            let pred = *p as usize;
            if pred == t {
                correct += 1;
            }
            if t < n_classes && pred < n_classes {
                cm[t][pred] += 1;
            }
        }
        let acc = correct as f64 / val_y.len().max(1) as f64;
        run.log(&[("loss/val", vloss), ("acc/val", acc)], epoch);
        // GPU memory footprint, when the backend reports it (CPU → None).
        if let Some(mb) = mem_probe() {
            run.log(&[("mem/gpu_mb", mb)], epoch);
        }

        let cm_path = run.artifacts_dir().join(format!("cm-epoch{epoch}.json"));
        let cm_json = json!({"labels": data.classes, "matrix": cm});
        let _ = fs::write(
            &cm_path,
            serde_json::to_string(&cm_json).unwrap_or_default(),
        );
        run.artifact("confusion-matrix", &cm_path);

        if vloss < best_val {
            best_val = vloss;
            // Checkpoint metadata (burn weight serialization arrives with
            // the inference milestone); enough to satisfy the contract.
            let ckpt = json!({
                "classes": data.classes, "features": data.feature_names,
                "hidden": root.model.hidden, "mean": mean, "std": std,
            });
            let _ = fs::write(
                run.checkpoints_dir().join("best.json"),
                serde_json::to_string_pretty(&ckpt).unwrap_or_default(),
            );
        }

        run.epoch(epoch); // honors a pause request at the boundary
        if run.cancelled() {
            break;
        }
    }

    run.info(&format!("best val loss: {best_val:.4}"));
    let status = if run.cancelled() { "cancelled" } else { "ok" };
    run.finish(status);
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_csv_parses_features_and_classes() {
        let dir = std::env::temp_dir().join(format!("nexisml-csv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("d.csv");
        std::fs::write(&csv, "x1,x2,label\n0.1,0.2,a\n0.3,0.4,b\n0.5,0.6,a\n").unwrap();
        let ds = load_csv(&csv, "label").unwrap();
        assert_eq!(ds.feature_names, vec!["x1", "x2"]);
        assert_eq!(ds.classes, vec!["a", "b"]);
        assert_eq!(ds.y, vec![0, 1, 0]);
        assert_eq!(ds.x.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
