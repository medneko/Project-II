#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_samples
from sklearn.preprocessing import normalize

try:
    from scripts.run_multifeature_100k import dataframe_to_markdown
except ModuleNotFoundError:
    from run_multifeature_100k import dataframe_to_markdown


N_ROWS = 100_000
EMB_DIM = 768
OUT_ROOT = Path("report") / "runs" / "100k" / "_hdbscan_tuning_compact"
FEATURE_SPACES = {"text_768_original", "text_768_l2", "text_pca64_only", "text_pca64_lexical"}
TEXT_PCA64_LEXICAL = Path("report") / "runs" / "100k" / "_multifeature" / "bounded_ablation" / "features" / "text_pca64_lexical.npy"
BASELINE_TEXT_HDBSCAN = Path("report") / "runs" / "100k" / "hdbscan_minsize50" / "metrics" / "results_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compact HDBSCAN tuning for 100k dense/event-like detection.")
    p.add_argument("--emb", default="data/embeddings_100k.npy")
    p.add_argument("--text-pca", default="report/runs/100k/_multifeature/artifacts/X_text_pca64.npy")
    p.add_argument("--meta", default="report/runs/100k/_multifeature/artifacts/multifeature_meta.csv")
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--feature-spaces", default="text_768_original,text_768_l2,text_pca64_only,text_pca64_lexical")
    p.add_argument("--min-cluster-sizes", default="30,50,100")
    p.add_argument("--min-samples-list", default="5,10,20")
    p.add_argument("--cluster-selection-methods", default="eom,leaf")
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--metric-sample-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--core-dist-n-jobs", type=int, default=-1)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--refresh-docs-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def validate_out_root(path: Path) -> None:
    try:
        path.resolve().relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"--out-root must stay under {OUT_ROOT}: {path}") from exc


def check_write_path(path: Path, overwrite: bool) -> Path:
    try:
        path.resolve().relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside {OUT_ROOT}: {path}") from exc
    if path.exists() and path.is_file() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_ints(raw: str, name: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} cannot be empty")
    return vals


def parse_csv(raw: str, valid: set[str], name: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(vals) - valid)
    if unknown:
        raise ValueError(f"Unknown {name}: {unknown}. Valid: {sorted(valid)}")
    if not vals:
        raise ValueError(f"{name} cannot be empty")
    return vals


def order_feature_spaces(spaces: list[str]) -> list[str]:
    priority = {
        "text_pca64_only": 0,
        "text_pca64_lexical": 1,
        "text_768_l2": 2,
        "text_768_original": 3,
    }
    return sorted(spaces, key=lambda x: priority.get(x, 99))


def sample_indices(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=min(size, n), replace=False))


def load_feature_space(space: str, args, rows: np.ndarray) -> np.ndarray:
    if space in {"text_768_original", "text_768_l2"}:
        emb = np.load(args.emb, mmap_mode="r")
        if emb.shape != (N_ROWS, EMB_DIM):
            raise ValueError(f"{args.emb} must have shape ({N_ROWS}, {EMB_DIM}), got {emb.shape}")
        X = np.asarray(emb[rows], dtype=np.float32)
        return normalize(X, norm="l2", axis=1).astype(np.float32) if space == "text_768_l2" else X
    if space == "text_pca64_only":
        X = np.load(args.text_pca, mmap_mode="r")
        if X.shape[0] != N_ROWS:
            raise ValueError(f"{args.text_pca} has invalid row count: {X.shape}")
        return np.asarray(X[rows], dtype=np.float32)
    if space == "text_pca64_lexical":
        if not TEXT_PCA64_LEXICAL.exists():
            raise FileNotFoundError(f"Missing reusable text_pca64_lexical feature matrix: {TEXT_PCA64_LEXICAL}")
        X = np.load(TEXT_PCA64_LEXICAL, mmap_mode="r")
        if X.shape[0] != N_ROWS:
            raise ValueError(f"{TEXT_PCA64_LEXICAL} has invalid row count: {X.shape}")
        return np.asarray(X[rows], dtype=np.float32)
    raise ValueError(f"Unknown feature space: {space}")


def cluster_size_metrics(labels: np.ndarray, assigned: np.ndarray) -> dict:
    counts = pd.Series(labels[assigned]).value_counts().sort_index()
    if counts.empty:
        return {
            "min_assigned_cluster_size": 0,
            "median_assigned_cluster_size": 0.0,
            "max_assigned_cluster_size": 0,
            "largest_cluster_pct": 0.0,
        }
    sizes = counts.to_numpy(dtype=np.float64)
    return {
        "min_assigned_cluster_size": int(sizes.min()),
        "median_assigned_cluster_size": float(np.median(sizes)),
        "max_assigned_cluster_size": int(sizes.max()),
        "largest_cluster_pct": float(sizes.max() / sizes.sum() * 100.0),
    }


def metric_subset_indices(assigned_idx: np.ndarray, metric_sample_size: int, seed: int) -> np.ndarray:
    if len(assigned_idx) <= metric_sample_size:
        return assigned_idx
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(assigned_idx, size=metric_sample_size, replace=False))


def evaluate(labels: np.ndarray, X: np.ndarray, args) -> dict:
    assigned = labels != -1
    n_assigned = int(assigned.sum())
    coverage_pct = float(n_assigned / len(labels) * 100.0)
    noise_pct = 100.0 - coverage_pct
    cluster_labels = sorted(set(labels[assigned]))
    n_clusters = len(cluster_labels)
    metrics = {
        "coverage_pct": coverage_pct,
        "noise_pct": noise_pct,
        "n_clusters": n_clusters,
        "n_assigned": n_assigned,
        "silhouette_cosine": math.nan,
        "dbi": math.nan,
        "negative_silhouette_pct": math.nan,
        **cluster_size_metrics(labels, assigned),
    }
    if n_clusters >= 2 and n_assigned >= 3:
        assigned_idx = np.flatnonzero(assigned)
        metric_idx = metric_subset_indices(assigned_idx, args.metric_sample_size, args.seed)
        Xs = X[metric_idx]
        ys = labels[metric_idx]
        if len(set(ys)) >= 2:
            sil = silhouette_samples(Xs, ys, metric="cosine")
            metrics["silhouette_cosine"] = float(np.mean(sil))
            metrics["negative_silhouette_pct"] = float((sil < 0).mean() * 100.0)
            metrics["dbi"] = float(davies_bouldin_score(Xs, ys))
    metrics["collapse_warning"] = bool(n_clusters <= 3 and coverage_pct > 50.0)
    metrics["too_sparse_warning"] = bool(coverage_pct < 5.0)
    return metrics


def run_one(space: str, X: np.ndarray, min_cluster_size: int, min_samples: int, method: str, args) -> dict:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
        metric="euclidean",
        core_dist_n_jobs=args.core_dist_n_jobs,
    )
    labels = clusterer.fit_predict(X)
    metric = evaluate(labels, X, args)
    return {
        "feature_space": space,
        "sample_size": int(len(labels)),
        "metric": "euclidean",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": method,
        **metric,
    }


def score_rows(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    valid = ~(scored["collapse_warning"].astype(bool) | scored["too_sparse_warning"].astype(bool))
    scored["event_score"] = -np.inf
    if not valid.any():
        return scored.sort_values(["coverage_pct", "n_clusters"], ascending=[False, False])
    sub = scored.loc[valid].copy()
    specs = [
        ("silhouette_cosine", True, 0.35),
        ("dbi", False, 0.25),
        ("negative_silhouette_pct", False, 0.15),
        ("coverage_pct", True, 0.15),
        ("n_clusters", True, 0.10),
    ]
    sub["event_score"] = 0.0
    for col, higher, weight in specs:
        vals = pd.to_numeric(sub[col], errors="coerce").astype(float)
        rng = vals.max() - vals.min()
        if np.isfinite(rng) and rng > 0:
            norm = (vals - vals.min()) / rng if higher else (vals.max() - vals) / rng
        else:
            norm = pd.Series(0.5, index=sub.index)
        sub["event_score"] += weight * norm.fillna(0.0)
    scored.loc[sub.index, "event_score"] = sub["event_score"]
    return scored.sort_values(["event_score", "silhouette_cosine"], ascending=[False, False])


def baseline_text_hdbscan() -> dict | None:
    if not BASELINE_TEXT_HDBSCAN.exists():
        return None
    row = pd.read_csv(BASELINE_TEXT_HDBSCAN).iloc[0].to_dict()
    return {
        "coverage_pct": float(row.get("coverage_pct", math.nan)),
        "n_clusters": int(row.get("n_clusters", 0)),
        "silhouette_cosine": float(row.get("silhouette", math.nan)),
        "dbi": float(row.get("dbi", math.nan)),
        "source": str(BASELINE_TEXT_HDBSCAN),
    }


def compare_to_baseline(best: pd.Series, baseline: dict | None) -> bool:
    if baseline is None:
        return False
    return (
        float(best["coverage_pct"]) >= 0.8 * float(baseline["coverage_pct"])
        and int(best["n_clusters"]) >= 0.5 * int(baseline["n_clusters"])
        and float(best["silhouette_cosine"]) >= float(baseline["silhouette_cosine"])
        and float(best["dbi"]) <= float(baseline["dbi"])
        and not bool(best["collapse_warning"])
        and not bool(best["too_sparse_warning"])
    )


def config_key(row: dict | pd.Series) -> tuple:
    return (
        str(row["feature_space"]),
        int(row["min_cluster_size"]),
        int(row["min_samples"]),
        str(row["cluster_selection_method"]),
    )


def write_outputs(summary_rows: list[dict], args) -> pd.DataFrame:
    summary = pd.DataFrame(summary_rows)
    scored = score_rows(summary)
    metrics_dir = Path(args.out_root) / "metrics"
    scored.to_csv(check_write_path(metrics_dir / "hdbscan_tuning_summary.csv", args.overwrite), index=False)
    write_doc(scored, args)
    return scored


def write_doc(scored: pd.DataFrame, args) -> None:
    docs = Path(args.out_root) / "docs"
    baseline = baseline_text_hdbscan()
    best = scored.iloc[0]
    beats = compare_to_baseline(best, baseline)
    collapse = scored[scored["collapse_warning"].astype(bool)]
    sparse = scored[scored["too_sparse_warning"].astype(bool)]
    expected_total = len(getattr(args, "expected_configs", []))
    completed_total = len(scored)
    completed_spaces = sorted(scored["feature_space"].dropna().astype(str).unique().tolist())
    expected_spaces = getattr(args, "expected_spaces", completed_spaces)
    missing_spaces = sorted(set(expected_spaces) - set(completed_spaces))
    top_cols = [
        "feature_space",
        "min_cluster_size",
        "min_samples",
        "cluster_selection_method",
        "event_score",
        "coverage_pct",
        "noise_pct",
        "n_clusters",
        "silhouette_cosine",
        "dbi",
        "negative_silhouette_pct",
        "largest_cluster_pct",
        "collapse_warning",
        "too_sparse_warning",
    ]
    lines = [
        "# HDBSCAN Event Model Recommendation",
        "",
        "## Scope",
        "",
        "- HDBSCAN is evaluated only as a dense/event-like detector, not as a full-coverage clustering model.",
        f"- Fit sample size: {args.sample_size}; metric sample size: {args.metric_sample_size}; seed: {args.seed}.",
        "- Publisher/stock metadata was not used in fitting.",
        f"- Completed configurations in current summary: {completed_total}" + (f" of {expected_total} requested." if expected_total else "."),
        f"- Completed feature spaces: {', '.join(completed_spaces) if completed_spaces else 'none'}.",
        "",
        "## Baseline",
        "",
    ]
    if missing_spaces:
        lines.insert(9, f"- Missing feature spaces in current summary: {', '.join(missing_spaces)}.")
    if baseline is None:
        lines.append("- Text-only `hdbscan_minsize50` baseline metrics were not found.")
    else:
        lines.append(
            f"- Text-only `hdbscan_minsize50`: coverage={baseline['coverage_pct']:.3f}%, "
            f"n_clusters={baseline['n_clusters']}, silhouette={baseline['silhouette_cosine']:.6f}, DBI={baseline['dbi']:.6f}."
        )
    lines.extend(
        [
            "",
            "## Top Configurations",
            "",
            dataframe_to_markdown(scored.head(10)[top_cols]),
            "",
            "## Answers",
            "",
            f"1. Any configuration clearly exceeds text-only `hdbscan_minsize50`: {'yes' if beats else 'no'}.",
            f"2. Collapse configurations: {len(collapse)}.",
            f"3. Too-sparse configurations: {len(sparse)}.",
            f"4. Recommended compact tuning candidate: `{best['feature_space']}`, min_cluster_size={int(best['min_cluster_size'])}, min_samples={int(best['min_samples'])}, method={best['cluster_selection_method']}.",
            f"5. Event model decision: {'switch to the compact tuned candidate' if beats else 'keep text-only HDBSCAN minsize50'}.",
        ]
    )
    if not beats:
        lines.append("- Reason to keep baseline: no compact candidate beats it cleanly on silhouette, DBI, coverage, cluster count, and warning checks together.")
    check_write_path(docs / "hdbscan_event_model_recommendation.md", args.overwrite).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out_root)
    validate_out_root(out_root)
    spaces = order_feature_spaces(parse_csv(args.feature_spaces, FEATURE_SPACES, "--feature-spaces"))
    min_cluster_sizes = parse_ints(args.min_cluster_sizes, "--min-cluster-sizes")
    min_samples_list = parse_ints(args.min_samples_list, "--min-samples-list")
    methods = parse_csv(args.cluster_selection_methods, {"eom", "leaf"}, "--cluster-selection-methods")
    args.expected_spaces = spaces
    args.expected_configs = [
        (space, min_cluster_size, min_samples, method)
        for space in spaces
        for min_cluster_size in min_cluster_sizes
        for min_samples in min_samples_list
        for method in methods
    ]
    _ = pd.read_csv(args.meta, nrows=1)
    rows = sample_indices(N_ROWS, args.sample_size, args.seed)
    metrics_path = out_root / "metrics" / "hdbscan_tuning_summary.csv"
    summary_rows = []
    completed = set()
    if (args.skip_existing or args.refresh_docs_only) and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        summary_rows = existing.drop(columns=[c for c in ["event_score"] if c in existing.columns]).to_dict("records")
        completed = {config_key(row) for row in summary_rows}
    if args.refresh_docs_only:
        if not summary_rows:
            raise FileNotFoundError(f"No existing summary to refresh: {metrics_path}")
        write_outputs(summary_rows, args)
        print(f"Refreshed compact HDBSCAN docs from {metrics_path}")
        return 0
    for space in spaces:
        print(f"Loading feature space {space}", flush=True)
        X = load_feature_space(space, args, rows)
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_list:
                for method in methods:
                    key = (space, min_cluster_size, min_samples, method)
                    if key in completed:
                        print(f"Skipping existing {space} min_cluster_size={min_cluster_size} min_samples={min_samples} method={method}", flush=True)
                        continue
                    print(f"Running {space} min_cluster_size={min_cluster_size} min_samples={min_samples} method={method}", flush=True)
                    try:
                        row = run_one(space, X, min_cluster_size, min_samples, method, args)
                    except Exception as exc:
                        row = {
                            "feature_space": space,
                            "sample_size": int(args.sample_size),
                            "metric": "euclidean",
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            "cluster_selection_method": method,
                            "coverage_pct": math.nan,
                            "noise_pct": math.nan,
                            "n_clusters": 0,
                            "n_assigned": 0,
                            "silhouette_cosine": math.nan,
                            "dbi": math.nan,
                            "negative_silhouette_pct": math.nan,
                            "min_assigned_cluster_size": 0,
                            "median_assigned_cluster_size": 0.0,
                            "max_assigned_cluster_size": 0,
                            "largest_cluster_pct": 0.0,
                            "collapse_warning": False,
                            "too_sparse_warning": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    summary_rows.append(row)
                    completed.add(key)
                    write_outputs(summary_rows, args)
    write_outputs(summary_rows, args)
    print(f"Wrote compact HDBSCAN tuning to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
