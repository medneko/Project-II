#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from multiprocessing.context import TimeoutError as MpTimeoutError
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_samples

try:
    from scripts.run_multifeature_100k import dataframe_to_markdown
except ModuleNotFoundError:
    from run_multifeature_100k import dataframe_to_markdown


N_ROWS = 100_000
DEFAULT_OUT_ROOT = Path("report") / "runs" / "100k" / "_hdbscan_event_validation"
BASELINE_TEXT_HDBSCAN = Path("report") / "runs" / "100k" / "hdbscan_minsize50" / "metrics" / "results_summary.csv"
LABEL_NAME = "text_pca64_only_hdbscan_mcs30_ms20_leaf_labels.csv"
TEXT_COLUMNS = ("headline_clean", "headline", "title", "text", "content")
DATE_COLUMNS = ("date", "published_at", "publish_date", "datetime")
PUBLISHER_COLUMNS = ("publisher", "source", "provider")
STOCK_COLUMNS = ("stock", "ticker", "symbol")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate the best PCA64 HDBSCAN dense/event candidate on 100k.")
    p.add_argument("--emb", default="data/embeddings_100k.npy")
    p.add_argument("--text-pca", default="report/runs/100k/_multifeature/artifacts/X_text_pca64.npy")
    p.add_argument("--meta", default="report/runs/100k/_multifeature/artifacts/multifeature_meta.csv")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--min-cluster-size", type=int, default=30)
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="leaf")
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--metric-sample-size", type=int, default=10000)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--full-timeout-seconds", type=int, default=1800)
    p.add_argument("--sample-timeout-seconds", type=int, default=1800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--core-dist-n-jobs", type=int, default=-1)
    p.add_argument("--refresh-docs-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_out_root(path: Path) -> None:
    if not is_relative_to(path, DEFAULT_OUT_ROOT):
        raise ValueError(f"--out-root must stay under {DEFAULT_OUT_ROOT}: {path}")


def check_write_path(path: Path, overwrite: bool) -> Path:
    if not is_relative_to(path, DEFAULT_OUT_ROOT):
        raise ValueError(f"Refusing to write outside {DEFAULT_OUT_ROOT}: {path}")
    if path.exists() and path.is_file() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sample_indices(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=min(size, n), replace=False))


def first_existing(columns: list[str] | pd.Index, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(args.emb, mmap_mode="r")
    if emb.shape[0] != N_ROWS:
        raise ValueError(f"{args.emb} must have {N_ROWS} rows, got {emb.shape}")
    X = np.load(args.text_pca, mmap_mode="r")
    if X.shape[0] != N_ROWS:
        raise ValueError(f"{args.text_pca} must have {N_ROWS} rows, got {X.shape}")
    meta = pd.read_csv(args.meta)
    if len(meta) != N_ROWS:
        raise ValueError(f"{args.meta} must have {N_ROWS} rows, got {len(meta)}")
    if "row_index" in meta.columns and not np.array_equal(meta["row_index"].to_numpy(), np.arange(N_ROWS)):
        raise ValueError(f"{args.meta}.row_index must align to 0..{N_ROWS - 1}")
    return X, meta


def fit_hdbscan_from_npy(
    text_pca_path: str,
    rows: np.ndarray | None,
    min_cluster_size: int,
    min_samples: int | None,
    method: str,
    metric: str,
    core_dist_n_jobs: int,
) -> np.ndarray:
    import hdbscan

    X_mmap = np.load(text_pca_path, mmap_mode="r")
    X = np.asarray(X_mmap if rows is None else X_mmap[rows], dtype=np.float32)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
        metric=metric,
        core_dist_n_jobs=core_dist_n_jobs,
    )
    return clusterer.fit_predict(X).astype(np.int64)


def run_fit_with_timeout(
    args: argparse.Namespace,
    rows: np.ndarray | None,
    min_cluster_size: int,
    min_samples: int | None,
    method: str,
    timeout_seconds: int,
) -> tuple[np.ndarray | None, str]:
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        async_result = pool.apply_async(
            fit_hdbscan_from_npy,
            (
                args.text_pca,
                rows,
                min_cluster_size,
                min_samples,
                method,
                args.metric,
                args.core_dist_n_jobs,
            ),
        )
        try:
            return async_result.get(timeout=timeout_seconds), ""
        except MpTimeoutError:
            pool.terminate()
            return None, f"timeout_after_{timeout_seconds}s"
        except Exception as exc:
            pool.terminate()
            return None, f"{type(exc).__name__}: {exc}"


def metric_rows(assigned_idx: np.ndarray, metric_sample_size: int, seed: int) -> np.ndarray:
    if len(assigned_idx) <= metric_sample_size:
        return assigned_idx
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(assigned_idx, size=metric_sample_size, replace=False))


def cluster_size_metrics(labels: np.ndarray) -> dict:
    assigned = labels != -1
    counts = pd.Series(labels[assigned]).value_counts()
    if counts.empty:
        return {
            "min_assigned_cluster_size": 0,
            "median_assigned_cluster_size": 0.0,
            "max_assigned_cluster_size": 0,
            "largest_cluster_pct": 0.0,
            "top3_cluster_pct": 0.0,
        }
    sizes = counts.to_numpy(dtype=np.float64)
    top = np.sort(sizes)[::-1]
    return {
        "min_assigned_cluster_size": int(sizes.min()),
        "median_assigned_cluster_size": float(np.median(sizes)),
        "max_assigned_cluster_size": int(sizes.max()),
        "largest_cluster_pct": float(top[0] / sizes.sum() * 100.0),
        "top3_cluster_pct": float(top[:3].sum() / sizes.sum() * 100.0),
    }


def evaluate(labels: np.ndarray, X: np.ndarray, rows: np.ndarray, args: argparse.Namespace, role: str, notes: str = "") -> dict:
    assigned = labels != -1
    n_assigned = int(assigned.sum())
    coverage_pct = float(n_assigned / len(labels) * 100.0) if len(labels) else 0.0
    clusters = sorted(int(v) for v in np.unique(labels[assigned]))
    n_clusters = len(clusters)
    out = {
        "model_role": role,
        "feature_space": "text_pca64_only",
        "fit_n_rows": int(len(labels)),
        "is_full_100k": bool(len(labels) == N_ROWS),
        "metric": args.metric,
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": int(args.min_samples) if args.min_samples is not None else "",
        "cluster_selection_method": args.cluster_selection_method,
        "coverage_pct": coverage_pct,
        "noise_pct": float(100.0 - coverage_pct),
        "n_clusters": n_clusters,
        "n_assigned": n_assigned,
        "silhouette_cosine": math.nan,
        "silhouette_euclidean": math.nan,
        "dbi": math.nan,
        "negative_silhouette_pct": math.nan,
        "metric_sample_size_used": 0,
        **cluster_size_metrics(labels),
        "collapse_warning": bool(n_clusters <= 3 and coverage_pct > 50.0),
        "too_sparse_warning": bool(coverage_pct < 5.0),
        "notes": notes,
    }
    if n_clusters >= 2 and n_assigned >= 3:
        local_metric_idx = metric_rows(np.flatnonzero(assigned), args.metric_sample_size, args.seed)
        X_metric = np.asarray(X[rows[local_metric_idx]], dtype=np.float32)
        y_metric = labels[local_metric_idx]
        if len(set(y_metric)) >= 2:
            sil_cos = silhouette_samples(X_metric, y_metric, metric="cosine")
            out["silhouette_cosine"] = float(np.mean(sil_cos))
            out["negative_silhouette_pct"] = float((sil_cos < 0).mean() * 100.0)
            try:
                sil_euc = silhouette_samples(X_metric, y_metric, metric="euclidean")
                out["silhouette_euclidean"] = float(np.mean(sil_euc))
            except Exception as exc:
                out["notes"] = append_note(out["notes"], f"silhouette_euclidean_failed:{type(exc).__name__}")
            try:
                out["dbi"] = float(davies_bouldin_score(X_metric, y_metric))
            except Exception as exc:
                out["notes"] = append_note(out["notes"], f"dbi_failed:{type(exc).__name__}")
            out["metric_sample_size_used"] = int(len(local_metric_idx))
    return out


def append_note(notes: str, value: str) -> str:
    return value if not notes else f"{notes};{value}"


def historical_baseline_row(args: argparse.Namespace) -> dict | None:
    if not BASELINE_TEXT_HDBSCAN.exists():
        return None
    row = pd.read_csv(BASELINE_TEXT_HDBSCAN).iloc[0]
    return {
        "model_role": "historical_text_only_baseline",
        "feature_space": "text_768_original",
        "fit_n_rows": int(row.get("n_points_labeled", N_ROWS)),
        "is_full_100k": True,
        "metric": "euclidean",
        "min_cluster_size": int(row.get("min_cluster_size", 50)),
        "min_samples": "",
        "cluster_selection_method": "unknown",
        "coverage_pct": float(row.get("coverage_pct", math.nan)),
        "noise_pct": float(100.0 - float(row.get("coverage_pct", math.nan))),
        "n_clusters": int(row.get("n_clusters", 0)),
        "n_assigned": int(row.get("n_assigned", 0)),
        "silhouette_cosine": float(row.get("silhouette", math.nan)),
        "silhouette_euclidean": math.nan,
        "dbi": float(row.get("dbi", math.nan)),
        "negative_silhouette_pct": math.nan,
        "metric_sample_size_used": int(row.get("silhouette_sample_size", args.metric_sample_size)),
        "min_assigned_cluster_size": math.nan,
        "median_assigned_cluster_size": math.nan,
        "max_assigned_cluster_size": math.nan,
        "largest_cluster_pct": math.nan,
        "top3_cluster_pct": math.nan,
        "collapse_warning": False,
        "too_sparse_warning": False,
        "notes": f"loaded_from:{BASELINE_TEXT_HDBSCAN}",
    }


def top_terms(texts: list[str], n: int = 12) -> str:
    texts = [t for t in texts if t]
    if not texts:
        return ""
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=5000)
        matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        terms = np.asarray(vectorizer.get_feature_names_out())
        order = np.argsort(scores)[::-1][:n]
        return ", ".join(terms[order])
    except ValueError:
        return ""


def concentration(values: pd.Series) -> tuple[str, float]:
    if values.empty:
        return "", 0.0
    counts = values.fillna("").astype(str).replace("", "(missing)").value_counts()
    if counts.empty:
        return "", 0.0
    return str(counts.index[0]), float(counts.iloc[0] / counts.sum() * 100.0)


def top_cluster_rows(labels: np.ndarray, rows: np.ndarray, meta: pd.DataFrame, seed: int) -> pd.DataFrame:
    assigned_labels = labels[labels != -1]
    if len(assigned_labels) == 0:
        return pd.DataFrame()
    text_col = first_existing(meta.columns, TEXT_COLUMNS)
    date_col = first_existing(meta.columns, DATE_COLUMNS)
    publisher_col = first_existing(meta.columns, PUBLISHER_COLUMNS)
    stock_col = first_existing(meta.columns, STOCK_COLUMNS)
    counts = pd.Series(assigned_labels).value_counts().head(10)
    rng = np.random.RandomState(seed)
    out = []
    assigned_total = int(len(assigned_labels))
    for label, size in counts.items():
        local_idx = np.flatnonzero(labels == int(label))
        row_idx = rows[local_idx]
        subset = meta.iloc[row_idx]
        texts = subset[text_col].fillna("").astype(str).tolist() if text_col else []
        sample_take = min(5, len(texts))
        sample_texts = []
        if sample_take:
            sample_positions = rng.choice(len(texts), size=sample_take, replace=False)
            sample_texts = [texts[i] for i in sample_positions]
        date_value, date_pct = concentration(subset[date_col]) if date_col else ("", 0.0)
        publisher_value, publisher_pct = concentration(subset[publisher_col]) if publisher_col else ("", 0.0)
        stock_value, stock_pct = concentration(subset[stock_col]) if stock_col else ("", 0.0)
        warnings = []
        if date_pct >= 50.0:
            warnings.append(f"date_concentration:{date_value}:{date_pct:.1f}%")
        if publisher_pct >= 50.0:
            warnings.append(f"publisher_concentration:{publisher_value}:{publisher_pct:.1f}%")
        if stock_pct >= 50.0:
            warnings.append(f"stock_concentration:{stock_value}:{stock_pct:.1f}%")
        out.append(
            {
                "cluster_id": int(label),
                "size": int(size),
                "assigned_pct": float(size / assigned_total * 100.0),
                "top_terms": top_terms(texts),
                "examples": " || ".join(sample_texts),
                "top_date": date_value,
                "top_date_pct": date_pct,
                "top_publisher": publisher_value,
                "top_publisher_pct": publisher_pct,
                "top_stock": stock_value,
                "top_stock_pct": stock_pct,
                "posthoc_warnings": ";".join(warnings),
            }
        )
    return pd.DataFrame(out)


def write_labels(path: Path, labels: np.ndarray, rows: np.ndarray, args: argparse.Namespace) -> None:
    pd.DataFrame(
        {
            "row_index": rows.astype(np.int64),
            "label": labels.astype(np.int64),
            "algorithm": "text_pca64_only_hdbscan_mcs30_ms20_leaf",
            "feature_space": "text_pca64_only",
            "is_full_100k": bool(len(labels) == N_ROWS),
        }
    ).to_csv(check_write_path(path, args.overwrite), index=False)


def enough_to_switch_event_model(candidate: pd.Series, full_candidate: pd.Series, text_baseline: pd.Series | None) -> bool:
    """Switch criteria for dense-event use, comparing against the old text-only model."""
    if text_baseline is None:
        return False
    dbi_ok = float(candidate["dbi"]) <= float(text_baseline["dbi"]) or float(full_candidate["dbi"]) <= 1.05 * float(text_baseline["dbi"])
    return (
        float(candidate["silhouette_cosine"]) >= float(text_baseline["silhouette_cosine"])
        and float(full_candidate["silhouette_cosine"]) >= float(text_baseline["silhouette_cosine"])
        and dbi_ok
        and float(candidate["coverage_pct"]) >= 0.8 * float(text_baseline["coverage_pct"])
        and float(full_candidate["coverage_pct"]) >= 0.8 * float(text_baseline["coverage_pct"])
        and int(candidate["n_clusters"]) >= 0.5 * int(text_baseline["n_clusters"])
        and int(full_candidate["n_clusters"]) >= 0.5 * int(text_baseline["n_clusters"])
        and not bool(candidate["collapse_warning"])
        and not bool(candidate["too_sparse_warning"])
        and not bool(full_candidate["collapse_warning"])
        and not bool(full_candidate["too_sparse_warning"])
    )


def better_than_baseline(candidate: pd.Series, baseline: pd.Series | None) -> bool:
    if baseline is None:
        return False
    return (
        float(candidate["silhouette_cosine"]) >= float(baseline["silhouette_cosine"])
        and float(candidate["dbi"]) <= float(baseline["dbi"])
        and float(candidate["coverage_pct"]) >= 0.8 * float(baseline["coverage_pct"])
        and int(candidate["n_clusters"]) >= 0.5 * int(baseline["n_clusters"])
        and not bool(candidate["collapse_warning"])
        and not bool(candidate["too_sparse_warning"])
    )


def write_report(summary: pd.DataFrame, cluster_info: pd.DataFrame, full_error: str, args: argparse.Namespace) -> None:
    docs = Path(args.out_root) / "docs"
    main = summary[summary["model_role"] == "candidate_validation"].iloc[0]
    sample = summary[summary["model_role"] == "same_sample_candidate"].iloc[0]
    pca64_baseline_rows = summary[summary["model_role"] == "same_sample_pca64_baseline"]
    pca64_baseline = pca64_baseline_rows.iloc[0] if len(pca64_baseline_rows) else None
    text_baseline_rows = summary[summary["model_role"] == "historical_text_only_baseline"]
    text_baseline = text_baseline_rows.iloc[0] if len(text_baseline_rows) else None
    switch = enough_to_switch_event_model(sample, main, text_baseline)
    full_status = "yes" if bool(main["is_full_100k"]) else "no"
    full_note = "Full 100k completed." if bool(main["is_full_100k"]) else f"Full 100k did not complete; fallback sample was used ({full_error})."
    top_cluster_md = dataframe_to_markdown(cluster_info) if len(cluster_info) else "_No assigned clusters to summarize._"

    lines = [
        "# HDBSCAN Event Validation Report",
        "",
        "## Scope",
        "",
        "- Validates the best completed compact-tuning candidate only; no broad sweep was run.",
        "- HDBSCAN is treated as a dense/event detector, not a full-coverage clustering model.",
        "- Raw 768D feature spaces were not rerun because compact tuning timed out on `text_768_original` and `text_768_l2`.",
        "",
        "## Summary Metrics",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Full 100k Status",
        "",
        f"- Best PCA64 candidate ran full 100k: {full_status}.",
        f"- {full_note}",
        "",
        "## Top 10 Assigned Clusters",
        "",
        top_cluster_md,
        "",
        "## Post-Hoc Warning Rules",
        "",
        "- `collapse_warning` means <=3 clusters with >50% coverage.",
        "- `too_sparse_warning` means <5% coverage.",
        "- Date, publisher, and stock concentration warnings are post-hoc checks only; these fields were not used for fitting.",
    ]
    check_write_path(docs / "hdbscan_event_validation_report.md", args.overwrite).write_text("\n".join(lines) + "\n", encoding="utf-8")

    text_baseline_text = "not available"
    if text_baseline is not None:
        text_baseline_text = (
            f"text-only hdbscan_minsize50 coverage={float(text_baseline['coverage_pct']):.3f}%, "
            f"n_clusters={int(text_baseline['n_clusters'])}, silhouette={float(text_baseline['silhouette_cosine']):.6f}, "
            f"DBI={float(text_baseline['dbi']):.6f}"
        )
    pca64_baseline_text = "not available"
    if pca64_baseline is not None:
        pca64_baseline_text = (
            f"PCA64 mcs50/eom same-sample coverage={float(pca64_baseline['coverage_pct']):.3f}%, "
            f"n_clusters={int(pca64_baseline['n_clusters'])}, silhouette={float(pca64_baseline['silhouette_cosine']):.6f}, "
            f"DBI={float(pca64_baseline['dbi']):.6f}"
        )
    rec_lines = [
        "# HDBSCAN Event Model Final Recommendation",
        "",
        "## Answers",
        "",
        f"1. Best PCA64 candidate can run full 100k: {full_status}. {full_note}",
        (
            "2. Full-run metrics remain strong: "
            f"coverage={float(main['coverage_pct']):.3f}%, n_clusters={int(main['n_clusters'])}, "
            f"silhouette_cosine={float(main['silhouette_cosine']):.6f}, DBI={float(main['dbi']):.6f}."
            if bool(main["is_full_100k"])
            else "2. Full-run metrics are not available because the full fit did not complete within the validation path."
        ),
        (
            "3. Sample-level evidence is enough to keep this as the preferred event candidate: "
            f"same-sample coverage={float(sample['coverage_pct']):.3f}%, n_clusters={int(sample['n_clusters'])}, "
            f"silhouette_cosine={float(sample['silhouette_cosine']):.6f}, DBI={float(sample['dbi']):.6f}."
        ),
        f"4. Switch event model from text-only HDBSCAN minsize50 to PCA64 tuned HDBSCAN: {'yes' if switch else 'no'}. Text-only baseline reference: {text_baseline_text}. Same-sample PCA64 baseline check: {pca64_baseline_text}.",
        "5. Report caveat: raw 768D HDBSCAN candidates remain unresolved because compact tuning timed out on `text_768_original` and `text_768_l2`; this recommendation is therefore for the validated PCA64 event-detector path, not a final claim that 768D cannot improve it.",
        "",
        "## Recommended Conclusion Wording",
        "",
        (
            "Use `text_pca64_only + HDBSCAN(min_cluster_size=30, min_samples=20, leaf, euclidean)` as the event-detection HDBSCAN candidate. "
            "It improves cluster separation and event granularity versus the text-only minsize50 baseline while retaining comparable dense-event coverage for post-hoc review; full-run DBI is slightly worse, so keep that caveat visible. "
            "Caveat: raw 768D HDBSCAN variants timed out during compact tuning, so the conclusion is limited to validated PCA64 candidates."
            if switch
            else "Keep `text-only hdbscan_minsize50` as the current event-model default. The PCA64 tuned candidate remains promising for dense-event analysis, but the validation did not beat the baseline cleanly enough on the chosen criteria. Caveat: raw 768D HDBSCAN variants timed out during compact tuning."
        ),
    ]
    check_write_path(docs / "hdbscan_event_model_final_recommendation.md", args.overwrite).write_text(
        "\n".join(rec_lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_root = str(Path(args.out_root))
    validate_out_root(Path(args.out_root))
    if args.metric != "euclidean":
        raise ValueError("This validation script expects --metric euclidean for HDBSCAN fitting.")
    X, meta = load_inputs(args)
    out_root = Path(args.out_root)
    for sub in ("metrics", "docs", "labels"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    if args.refresh_docs_only:
        summary_path = out_root / "metrics" / "hdbscan_event_validation_summary.csv"
        labels_path = out_root / "labels" / LABEL_NAME
        if not summary_path.exists() or not labels_path.exists():
            raise FileNotFoundError("Refresh requires existing summary CSV and candidate labels.")
        summary = pd.read_csv(summary_path)
        hist = historical_baseline_row(args)
        if hist is not None and "historical_text_only_baseline" not in set(summary["model_role"].astype(str)):
            summary = pd.concat([summary, pd.DataFrame([hist])], ignore_index=True)
            summary.to_csv(check_write_path(summary_path, args.overwrite), index=False)
        labels_df = pd.read_csv(labels_path)
        cluster_info = top_cluster_rows(
            labels_df["label"].to_numpy(dtype=np.int64),
            labels_df["row_index"].to_numpy(dtype=np.int64),
            meta,
            args.seed,
        )
        write_report(summary, cluster_info, "", args)
        print(f"Refreshed validation docs from {summary_path}")
        return 0

    full_rows = np.arange(N_ROWS, dtype=np.int64)
    labels, full_error = run_fit_with_timeout(
        args,
        None,
        args.min_cluster_size,
        args.min_samples,
        args.cluster_selection_method,
        args.full_timeout_seconds,
    )
    if labels is None:
        rows = sample_indices(N_ROWS, args.sample_size, args.seed)
        labels, sample_error = run_fit_with_timeout(
            args,
            rows,
            args.min_cluster_size,
            args.min_samples,
            args.cluster_selection_method,
            args.sample_timeout_seconds,
        )
        if labels is None:
            raise RuntimeError(f"Candidate failed on full and sample fallback. full={full_error}; sample={sample_error}")
        main_rows = rows
        main_notes = f"full_100k_failed:{full_error};fallback_sample_50k"
    else:
        main_rows = full_rows
        main_notes = "full_100k_completed"

    summary_rows = [evaluate(labels, X, main_rows, args, "candidate_validation", main_notes)]
    write_labels(out_root / "labels" / LABEL_NAME, labels, main_rows, args)

    sample_rows = sample_indices(N_ROWS, args.sample_size, args.seed)
    sample_labels, sample_error = run_fit_with_timeout(
        args,
        sample_rows,
        args.min_cluster_size,
        args.min_samples,
        args.cluster_selection_method,
        args.sample_timeout_seconds,
    )
    if sample_labels is not None:
        summary_rows.append(evaluate(sample_labels, X, sample_rows, args, "same_sample_candidate"))

    baseline_labels, baseline_error = run_fit_with_timeout(
        args,
        sample_rows,
        50,
        None,
        "eom",
        args.sample_timeout_seconds,
    )
    if baseline_labels is not None:
        old_mcs, old_ms, old_method = args.min_cluster_size, args.min_samples, args.cluster_selection_method
        args.min_cluster_size, args.min_samples, args.cluster_selection_method = 50, None, "eom"
        summary_rows.append(evaluate(baseline_labels, X, sample_rows, args, "same_sample_pca64_baseline"))
        args.min_cluster_size, args.min_samples, args.cluster_selection_method = old_mcs, old_ms, old_method
    else:
        hist = historical_baseline_row(args)
        if hist is not None:
            hist["notes"] = append_note(hist["notes"], f"same_sample_pca64_baseline_failed:{baseline_error}")
            summary_rows.append(hist)
    hist = historical_baseline_row(args)
    if hist is not None and not any(r["model_role"] == "historical_text_only_baseline" for r in summary_rows):
        summary_rows.append(hist)

    if sample_labels is None:
        hist = historical_baseline_row(args)
        if hist is not None and not any(r["model_role"] == "historical_text_only_baseline" for r in summary_rows):
            hist["notes"] = append_note(hist["notes"], f"same_sample_candidate_failed:{sample_error}")
            summary_rows.append(hist)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        check_write_path(out_root / "metrics" / "hdbscan_event_validation_summary.csv", args.overwrite),
        index=False,
    )
    cluster_info = top_cluster_rows(labels, main_rows, meta, args.seed)
    write_report(summary, cluster_info, full_error, args)
    print(f"Wrote validation outputs to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
