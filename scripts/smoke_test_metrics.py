#!/usr/bin/env python3
"""Smoke tests for metric label schema handling on tiny synthetic data."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("report") / "scratch" / "smoke_metrics"


def make_embeddings(path: Path, n: int = 60, d: int = 6) -> np.ndarray:
    rng = np.random.RandomState(7)
    centers = np.array(
        [
            [-4, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    labels = np.repeat(np.arange(3), n // 3)
    X = centers[labels] + rng.normal(0, 0.35, size=(n, d)).astype(np.float32)
    np.save(path, X)
    return X


def write_labels(path: Path, labels, row_index=None):
    data = {"label": labels}
    if row_index is not None:
        data = {"row_index": row_index, "label": labels}
    pd.DataFrame(data).to_csv(path, index=False)


def run_metrics(emb: Path, labels_dir: Path, outdir: Path, overwrite: bool = False):
    cmd = [
        sys.executable,
        "scripts/compute_metrics_from_labels.py",
        "--emb",
        str(emb),
        "--labels-dir",
        str(labels_dir),
        "--outdir",
        str(outdir),
        "--metric-sample-size",
        "20",
        "--sample",
        "30",
        "--seed",
        "123",
    ]
    if overwrite:
        cmd.append("--overwrite")
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_summary(outdir: Path) -> pd.DataFrame:
    return pd.read_csv(outdir / "clustering_results.csv")


def expect(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run_case(name: str, emb: Path, labels_fn, validate_fn, expect_success: bool = True):
    case_dir = RUN_DIR / name
    labels_dir = case_dir / "labels"
    outdir = case_dir / "out"
    labels_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    labels_fn(labels_dir)
    proc = run_metrics(emb, labels_dir, outdir)
    if expect_success:
        expect(proc.returncode == 0, f"expected success, got {proc.returncode}: {proc.stderr or proc.stdout}")
        validate_fn(outdir, proc)
    else:
        expect(proc.returncode != 0, "expected failure but command succeeded")
        validate_fn(outdir, proc)
    return {"case": name, "status": "PASS", "returncode": proc.returncode}


def main():
    global RUN_DIR
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    RUN_DIR = ROOT / run_id
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    emb = RUN_DIR / "embeddings.npy"
    make_embeddings(emb)
    base_labels = np.repeat(np.arange(3), 20)

    results = []

    def full_row_index(labels_dir):
        write_labels(labels_dir / "cluster_labels_full_row_index.csv", base_labels, np.arange(60))

    def validate_full(outdir, _proc):
        df = read_summary(outdir)
        row = df.iloc[0]
        expect(int(row["n_points_labeled"]) == 60, "full row_index n_points_labeled mismatch")
        expect(abs(float(row["coverage"]) - 1.0) < 1e-9, "full row_index coverage mismatch")
        expect(int(row["n_clusters"]) == 3, "full row_index n_clusters mismatch")

    results.append(run_case("full_row_index", emb, full_row_index, validate_full))

    def legacy_full(labels_dir):
        pd.DataFrame({"label": base_labels}).to_csv(labels_dir / "cluster_labels_legacy_full.csv", index=False)

    results.append(run_case("legacy_full_no_row_index", emb, legacy_full, validate_full))

    def sample_row_index(labels_dir):
        rows = np.array([0, 1, 2, 20, 21, 22, 40, 41, 42])
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        write_labels(labels_dir / "cluster_labels_sample.csv", labels, rows)

    def validate_sample(outdir, _proc):
        row = read_summary(outdir).iloc[0]
        expect(int(row["n_points_labeled"]) == 9, "sample n_points_labeled mismatch")
        expect(int(row["n_embedding_points"]) == 60, "sample n_embedding_points mismatch")
        expect(int(row["n_clusters"]) == 3, "sample n_clusters mismatch")

    results.append(run_case("sample_row_index", emb, sample_row_index, validate_sample))

    def noise_labels(labels_dir):
        labels = base_labels.copy()
        labels[:6] = -1
        write_labels(labels_dir / "cluster_labels_noise.csv", labels, np.arange(60))

    def validate_noise(outdir, _proc):
        row = read_summary(outdir).iloc[0]
        expect(int(row["n_assigned"]) == 54, "noise n_assigned mismatch")
        expect(abs(float(row["coverage"]) - 0.9) < 1e-9, "noise coverage mismatch")
        expect(int(row["n_clusters"]) == 3, "noise n_clusters should exclude -1")

    results.append(run_case("noise_minus_one", emb, noise_labels, validate_noise))

    def one_cluster(labels_dir):
        write_labels(labels_dir / "cluster_labels_one_cluster.csv", np.zeros(60, dtype=int), np.arange(60))

    def validate_one_cluster(outdir, _proc):
        row = read_summary(outdir).iloc[0]
        expect(int(row["n_clusters"]) == 1, "one-cluster n_clusters mismatch")
        expect(pd.isna(row["silhouette"]), "one-cluster silhouette should be NaN")
        expect(pd.isna(row["dbi"]), "one-cluster DBI should be NaN")
        expect("less_than_2" in str(row["notes"]), "one-cluster notes missing skip reason")

    results.append(run_case("one_cluster", emb, one_cluster, validate_one_cluster))

    def invalid_mismatch(labels_dir):
        pd.DataFrame({"label": [0, 1, 0, 1, 0]}).to_csv(labels_dir / "cluster_labels_bad.csv", index=False)

    def validate_invalid(_outdir, proc):
        combined = proc.stdout + proc.stderr
        expect("sample labels must include a row_index column" in combined, "invalid mismatch error text missing")

    results.append(run_case("invalid_mismatch_no_row_index", emb, invalid_mismatch, validate_invalid, expect_success=False))

    def safe_write(labels_dir):
        write_labels(labels_dir / "cluster_labels_safe_write.csv", base_labels, np.arange(60))

    safe_case = RUN_DIR / "safe_write"
    labels_dir = safe_case / "labels"
    outdir = safe_case / "out"
    labels_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_write(labels_dir)
    first = run_metrics(emb, labels_dir, outdir)
    expect(first.returncode == 0, f"safe-write first run failed: {first.stderr or first.stdout}")
    second = run_metrics(emb, labels_dir, outdir)
    expect(second.returncode != 0, "safe-write second run should fail without --overwrite")
    expect("Output already exists" in (second.stdout + second.stderr), "safe-write failure text missing")
    results.append({"case": "safe_write_no_overwrite", "status": "PASS", "returncode": second.returncode})

    print("Smoke metric/schema tests")
    print("Output:", RUN_DIR)
    for item in results:
        print(f"PASS {item['case']}")
    (RUN_DIR / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
