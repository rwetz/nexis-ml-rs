// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Nexis ML (Rust)                     ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! The run handle: ties the protocol emitter to the on-disk run store and
//! tracks per-metric stats for the summary. Mirrors the Python
//! `nexis_ml.track()` lifecycle (run.started → metric/epoch/artifact →
//! run.finished) so the event stream and files are interchangeable.

use std::collections::BTreeMap;

use serde_json::json;

use crate::protocol::{Emitter, PROTOCOL_VERSION};
use crate::run_store::{now_iso, RunDir};

struct Stat {
    last: f64,
    min: f64,
    max: f64,
    count: u64,
}

pub struct Run<'a> {
    name: String,
    emitter: &'a Emitter,
    dir: RunDir,
    total_epochs: u32,
    device: String,
    step: u64,
    epoch: u32,
    started_at: String,
    stats: BTreeMap<String, Stat>,
    last_values: BTreeMap<String, f64>,
    artifacts: Vec<(String, String)>,
}

impl<'a> Run<'a> {
    pub fn start(
        emitter: &'a Emitter,
        mut dir: RunDir,
        name: &str,
        config: serde_json::Value,
        total_epochs: u32,
        device: &str,
    ) -> Self {
        dir.write_config(&config);
        let started_at = now_iso();
        // absolute() (not canonicalize) so we match the Python engine's
        // clean abspaths — canonicalize adds Windows \\?\ verbatim prefixes.
        let abs = std::path::absolute(&dir.path).unwrap_or_else(|_| dir.path.clone());
        let event = json!({
            "ev": "run.started",
            "run": dir.run_id(),
            "name": name,
            "dir": abs.to_string_lossy(),
            "config": config,
            "totalEpochs": total_epochs,
            "device": device,
            "protocol": PROTOCOL_VERSION,
            "startedAt": started_at,
        });
        dir.append_event(&event);
        emitter.emit(event);
        emitter.console(&format!("run {} started", dir.run_id()));
        Self {
            name: name.to_string(),
            emitter,
            dir,
            total_epochs,
            device: device.to_string(),
            step: 0,
            epoch: 0,
            started_at,
            stats: BTreeMap::new(),
            last_values: BTreeMap::new(),
            artifacts: Vec::new(),
        }
    }

    pub fn checkpoints_dir(&self) -> std::path::PathBuf {
        self.dir.checkpoints_dir()
    }

    pub fn artifacts_dir(&self) -> std::path::PathBuf {
        self.dir.artifacts_dir()
    }

    /// Log one or more scalar metrics at the current step (auto-incremented
    /// once per call, like the Python harness).
    pub fn log(&mut self, metrics: &[(&str, f64)], epoch: u32) {
        self.step += 1;
        for &(name, value) in metrics {
            self.track_stat(name, value);
            let event = json!({
                "ev": "metric",
                "run": self.dir.run_id(),
                "step": self.step,
                "epoch": epoch,
                "name": name,
                "value": value,
            });
            self.dir.append_event(&event);
            self.emitter.emit(event);
        }
    }

    pub fn epoch(&mut self, i: u32) {
        self.epoch = i;
        let event = json!({
            "ev": "epoch",
            "run": self.dir.run_id(),
            "epoch": i,
            "of": self.total_epochs,
        });
        self.dir.append_event(&event);
        self.emitter.emit(event);
        let latest: Vec<String> = self
            .last_values
            .iter()
            .map(|(k, v)| format!("{k}={v:.4}"))
            .collect();
        self.emitter.console(&format!(
            "epoch {i}/{}  {}",
            self.total_epochs,
            latest.join("  ")
        ));
    }

    pub fn artifact(&mut self, kind: &str, path: &std::path::Path) {
        let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        let p = abs.to_string_lossy().to_string();
        self.artifacts.push((kind.to_string(), p.clone()));
        let event = json!({
            "ev": "artifact",
            "run": self.dir.run_id(),
            "kind": kind,
            "path": p,
        });
        self.dir.append_event(&event);
        self.emitter.emit(event);
    }

    pub fn info(&mut self, msg: &str) {
        let event = json!({
            "ev": "log", "run": self.dir.run_id(), "level": "info", "msg": msg,
        });
        self.dir.append_event(&event);
        self.emitter.emit(event);
        self.emitter.console(msg);
    }

    pub fn finish(mut self, status: &str) {
        let metrics: serde_json::Map<String, serde_json::Value> = self
            .stats
            .iter()
            .map(|(k, s)| {
                (
                    k.clone(),
                    json!({"last": s.last, "min": s.min, "max": s.max, "count": s.count}),
                )
            })
            .collect();
        let artifacts: Vec<serde_json::Value> = self
            .artifacts
            .iter()
            .map(|(k, p)| json!({"kind": k, "path": p}))
            .collect();
        let summary = json!({
            "status": status,
            "name": self.name,
            "startedAt": self.started_at,
            "finishedAt": now_iso(),
            "totalEpochs": self.total_epochs,
            "lastEpoch": self.epoch,
            "device": self.device,
            "metrics": metrics,
            "artifacts": artifacts,
        });
        self.dir.write_summary(&summary);
        let event = json!({
            "ev": "run.finished",
            "run": self.dir.run_id(),
            "status": status,
            "summary": summary,
        });
        self.dir.append_event(&event);
        self.emitter.emit(event);
        self.emitter
            .console(&format!("run {} finished: {status}", self.dir.run_id()));
    }

    fn track_stat(&mut self, name: &str, value: f64) {
        self.last_values.insert(name.to_string(), value);
        self.stats
            .entry(name.to_string())
            .and_modify(|s| {
                s.last = value;
                s.min = s.min.min(value);
                s.max = s.max.max(value);
                s.count += 1;
            })
            .or_insert(Stat {
                last: value,
                min: value,
                max: value,
                count: 1,
            });
    }
}
