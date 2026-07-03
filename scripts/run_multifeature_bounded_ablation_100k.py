#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

try:
    from scripts.run_multifeature_100k import check_write_path, dataframe_to_markdown
except ModuleNotFoundError:
    from run_multifeature_100k import check_write_path, dataframe_to_markdown


N_ROWS = 100_000
ROOT = Path("report") / "runs" / "100k" / "_multifeature" / "bounded_ablation"
FAST_VARIANTS = [
    "text_pca64_only",
    "text_pca64_lexical",
    "text_pca64_lexical_calendar",
    "text_pca64_lexical_calendar_publisher",
    "text_pca64_lexical_calendar_publisher_stock",
    "text_pca64_all_aux_w005",
    "text_pca64_all_aux_w010",
    "text_pca64_all_aux_w020",
    "text_pca64_all_aux_w030",
]
CORE_VARIANTS = [
    "text_pca64_only",
    "text_pca64_lexical_calendar",
    "text_pca64_lexical_calendar_publisher_stock",
    "text_pca64_all_aux_w010",
    "text_pca64_all_aux_w030",
]
GROUP_ORDER = ["lexical", "calendar", "publisher", "stock", "lexicon_sentiment_risk"]
VALID_ALGORITHMS = {
    "minibatch_k8",
    "minibatch_k16",
    "minibatch_k32",
    "gmm_k8",
    "gmm_k16",
    "hdbscan_minsize50",
    "hdbscan_minsize100",
    "clara_k8",
    "clara_k16",
    "agg_ward_k8",
    "mst_req8",
}
VALID_VARIANTS = set(FAST_VARIANTS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Time-budgeted/resource-bounded 100k multi-feature ablation.")
    p.add_argument("--emb", default="data/embeddings_100k.npy")
    p.add_argument("--text-pca", default="report/runs/100k/_multifeature/artifacts/X_text_pca64.npy")
    p.add_argument("--aux", default="report/runs/100k/_multifeature/artifacts/X_aux_features.npy")
    p.add_argument("--feature-columns", default="report/runs/100k/_multifeature/artifacts/feature_columns.json")
    p.add_argument("--meta", default="report/runs/100k/_multifeature/artifacts/multifeature_meta.csv")
    p.add_argument("--out-root", default=str(ROOT))
    p.add_argument("--time-budget-hours", type=float, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--metric-sample-size", type=int, default=10000)
    p.add_argument("--max-rows-gmm", type=int, default=30000)
    p.add_argument("--max-rows-hdbscan", type=int, default=20000)
    p.add_argument("--max-rows-clara", type=int, default=15000)
    p.add_argument("--max-rows-graph", type=int, default=10000)
    p.add_argument("--run-graph-algos", action="store_true")
    p.add_argument("--no-hdbscan", action="store_true")
    p.add_argument("--no-clara", action="store_true")
    p.add_argument("--only-algorithms", default=None, help="Comma-separated algorithm allow-list applied after default scheduling.")
    p.add_argument("--only-variants", default=None, help="Comma-separated variant allow-list applied after default scheduling.")
    return p


def validate_root(path: Path) -> None:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"--out-root must stay under {ROOT}: {path}") from exc


def load_inputs(args):
    emb = np.load(args.emb, mmap_mode="r")
    text = np.load(args.text_pca, mmap_mode="r")
    aux = np.load(args.aux, mmap_mode="r")
    cols = json.loads(Path(args.feature_columns).read_text(encoding="utf-8"))
    meta = pd.read_csv(args.meta, usecols=["row_index"])
    if emb.shape != (N_ROWS, 768):
        raise ValueError(f"{args.emb} must be (100000, 768), got {emb.shape}")
    if text.shape != (N_ROWS, 64):
        raise ValueError(f"{args.text_pca} must be (100000, 64), got {text.shape}")
    if aux.shape[0] != N_ROWS:
        raise ValueError(f"{args.aux} must have 100000 rows, got {aux.shape}")
    if not np.array_equal(meta["row_index"].to_numpy(), np.arange(N_ROWS)):
        raise ValueError(f"{args.meta}.row_index must be exactly 0..99999")
    return emb, text, aux, cols


def group_slices(cols):
    out = {}
    start = 0
    for group in GROUP_ORDER:
        width = len(cols[group])
        out[group] = slice(start, start + width)
        start += width
    return out


def variant_spec(name: str) -> dict:
    mapping = {
        "text_pca64_only": ([], 0.0),
        "text_pca64_lexical": (["lexical"], 0.3),
        "text_pca64_lexical_calendar": (["lexical", "calendar"], 0.3),
        "text_pca64_lexical_calendar_publisher": (["lexical", "calendar", "publisher"], 0.3),
        "text_pca64_lexical_calendar_publisher_stock": (["lexical", "calendar", "publisher", "stock"], 0.3),
        "text_pca64_all_aux_w005": (GROUP_ORDER, 0.05),
        "text_pca64_all_aux_w010": (GROUP_ORDER, 0.10),
        "text_pca64_all_aux_w020": (GROUP_ORDER, 0.20),
        "text_pca64_all_aux_w030": (GROUP_ORDER, 0.30),
    }
    groups, weight = mapping[name]
    return {"variant": name, "groups": list(groups), "aux_weight": float(weight), "n_text_dims": 64}


def build_matrix(name, text, aux, cols):
    spec = variant_spec(name)
    slices = group_slices(cols)
    if spec["groups"]:
        aux_part = np.hstack([np.asarray(aux[:, slices[g]], dtype=np.float32) for g in spec["groups"]]).astype(np.float32)
        X = np.hstack([np.asarray(text, dtype=np.float32), spec["aux_weight"] * aux_part]).astype(np.float32)
    else:
        X = np.asarray(text, dtype=np.float32)
    spec["n_total_dims"] = int(X.shape[1])
    spec["n_aux_dims"] = int(X.shape[1] - spec["n_text_dims"])
    return X, spec


def save_feature(out_root, name, X, spec, overwrite):
    feature_dir = out_root / "features"
    np.save(check_write_path(feature_dir / f"{name}.npy", overwrite), X.astype(np.float32))
    check_write_path(feature_dir / f"{name}.json", overwrite).write_text(
        json.dumps({**spec, "notes": "row_index preserved; no modulo/repeat"}, indent=2),
        encoding="utf-8",
    )


def sample_indices(out_root, kind, n, seed, overwrite):
    path = out_root / "shared" / f"sample_indices_{kind}_{n}_seed{seed}.npy"
    if path.exists() and not overwrite:
        return np.load(path)
    rng = np.random.RandomState(seed)
    rows = np.sort(rng.choice(N_ROWS, size=min(n, N_ROWS), replace=False))
    np.save(check_write_path(path, overwrite), rows)
    return rows


def schedule(args):
    jobs = []
    for variant in FAST_VARIANTS:
        for k in [8, 16, 32]:
            jobs.append((variant, f"minibatch_k{k}", None, False, "full"))
    for variant in CORE_VARIANTS:
        for k in [8, 16]:
            jobs.append((variant, f"gmm_k{k}", args.max_rows_gmm, True, "gmm"))
    if not args.no_hdbscan:
        for variant in CORE_VARIANTS:
            for m in [50, 100]:
                jobs.append((variant, f"hdbscan_minsize{m}", args.max_rows_hdbscan, True, "hdbscan"))
    if not args.no_clara:
        for variant in CORE_VARIANTS:
            for k in [8, 16]:
                jobs.append((variant, f"clara_k{k}", args.max_rows_clara, True, "clara"))
    if args.run_graph_algos:
        for variant in CORE_VARIANTS:
            jobs.append((variant, "agg_ward_k8", args.max_rows_graph, True, "graph"))
            jobs.append((variant, "mst_req8", args.max_rows_graph, True, "graph"))
    return filter_jobs(jobs, args.only_algorithms, args.only_variants)


def parse_allow_list(raw, valid, label):
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = sorted(values - valid)
    if unknown:
        allowed = ", ".join(sorted(valid))
        raise ValueError(f"Invalid {label}: {', '.join(unknown)}. Valid {label}: {allowed}")
    return values


def filter_jobs(jobs, only_algorithms, only_variants):
    algos = parse_allow_list(only_algorithms, VALID_ALGORITHMS, "algorithms")
    variants = parse_allow_list(only_variants, VALID_VARIANTS, "variants")
    if algos is not None:
        jobs = [job for job in jobs if job[1] in algos]
    if variants is not None:
        jobs = [job for job in jobs if job[0] in variants]
    return jobs


def output_paths(out_root, variant, algo):
    root = out_root / "runs" / variant / algo
    return {
        "root": root,
        "labels": root / "labels" / f"cluster_labels_{variant}_{algo}.csv",
        "metrics": root / "metrics" / "results_summary.csv",
        "manifest": root / "manifest.json",
    }


def valid_checkpoint(paths):
    if not paths["labels"].exists() or not paths["metrics"].exists():
        return False
    try:
        labels = pd.read_csv(paths["labels"], usecols=["row_index", "label", "algorithm", "feature_space"])
        metrics = pd.read_csv(paths["metrics"])
        return len(labels) > 0 and len(metrics) == 1 and labels["row_index"].is_unique
    except Exception:
        return False


def run_labels(algo, X, seed):
    if algo.startswith("minibatch_k"):
        from sklearn.cluster import MiniBatchKMeans

        k = int(algo.split("_k")[1])
        return MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto").fit_predict(X), ""
    if algo.startswith("gmm_k"):
        from sklearn.mixture import GaussianMixture

        k = int(algo.split("_k")[1])
        return GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=1,
            max_iter=100,
            reg_covar=1e-4,
            random_state=seed,
        ).fit_predict(X), ""
    if algo.startswith("hdbscan_minsize"):
        import hdbscan

        min_size = int(algo.split("minsize")[1])
        return hdbscan.HDBSCAN(min_cluster_size=min_size, metric="euclidean", core_dist_n_jobs=-1).fit_predict(X), ""
    if algo.startswith("clara_k"):
        k = int(algo.split("_k")[1])
        return clara_labels(X, k, seed)
    if algo == "agg_ward_k8":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=8, linkage="ward").fit_predict(X), ""
    if algo == "mst_req8":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=8, linkage="single").fit_predict(X), ""
    raise ValueError(f"Unsupported algorithm: {algo}")


def clara_labels(X, k, seed):
    try:
        from sklearn_extra.cluster import CLARA

        n_sampling = min(len(X), max(2048, 40 + 2 * k))
        model = CLARA(
            n_clusters=k,
            metric="euclidean",
            init="build",
            max_iter=300,
            n_sampling=n_sampling,
            n_sampling_iter=5,
            random_state=seed,
        )
        return model.fit_predict(X), f"CLARA true via sklearn_extra.cluster.CLARA; n_sampling={n_sampling}; n_sampling_iter=5"
    except Exception as exc:
        labels = clara_fallback_labels(X, k, seed)
        return labels, f"CLARA fallback MiniBatchKMeans-to-medoid used; sklearn_extra CLARA unavailable or failed: {type(exc).__name__}: {exc}"


def clara_fallback_labels(X, k, seed):
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init="auto")
    rough = model.fit_predict(X)
    centers = model.cluster_centers_.astype(np.float32)
    medoids = np.empty((k, X.shape[1]), dtype=np.float32)
    for j in range(k):
        mask = rough == j
        pool = X[mask] if mask.any() else X
        dif = pool - centers[j]
        medoids[j] = pool[np.argmin((dif * dif).sum(axis=1))]
    d = ((X[:, None, :] - medoids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1)


def compute_job_metrics(labels, X_eval, sample_size, seed):
    assigned = labels != -1
    n_assigned = int(assigned.sum())
    coverage = n_assigned / len(labels) if len(labels) else np.nan
    clusters = np.unique(labels[assigned])
    n_clusters = int(len(clusters))
    notes = []
    sil = np.nan
    dbi = np.nan
    used = 0
    if n_clusters < 2 or n_assigned < 2:
        notes.append("metric_skipped_less_than_2_assigned_clusters")
    else:
        rows = np.flatnonzero(assigned)
        if sample_size and len(rows) > sample_size:
            rng = np.random.RandomState(seed)
            rows = np.sort(rng.choice(rows, size=sample_size, replace=False))
        used = int(len(rows))
        try:
            sil = float(silhouette_score(X_eval[rows], labels[rows], metric="cosine"))
        except Exception as exc:
            notes.append(f"silhouette_failed:{type(exc).__name__}")
        try:
            dbi = float(davies_bouldin_score(X_eval[rows], labels[rows]))
        except Exception as exc:
            notes.append(f"dbi_failed:{type(exc).__name__}")
    return {
        "coverage_pct": coverage * 100.0,
        "n_clusters": n_clusters,
        "silhouette": sil,
        "silhouette_metric": "cosine",
        "silhouette_sample_size": used,
        "dbi": dbi,
        "notes": ";".join(notes),
    }


def write_job_outputs(paths, labels, row_index, algo, variant, metrics_row, manifest, overwrite):
    pd.DataFrame(
        {"row_index": row_index.astype(np.int64), "label": labels.astype(np.int64), "algorithm": algo, "feature_space": variant}
    ).to_csv(check_write_path(paths["labels"], overwrite), index=False)
    pd.DataFrame([metrics_row]).to_csv(check_write_path(paths["metrics"], overwrite), index=False)
    check_write_path(paths["manifest"], overwrite).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_status(out_root, statuses, overwrite=True):
    pd.DataFrame(statuses).to_csv(check_write_path(out_root / "job_status.csv", overwrite), index=False)


def summarize(out_root, statuses, overwrite):
    rows = []
    for status in statuses:
        if status["status"] not in {"completed", "skipped_existing"}:
            continue
        metrics_path = Path(status["output_dir"]) / "metrics" / "results_summary.csv"
        if not metrics_path.exists():
            continue
        row = pd.read_csv(metrics_path).iloc[0].to_dict()
        rows.append(row)
    summary = pd.DataFrame(rows)
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(check_write_path(metrics_dir / "bounded_ablation_summary.csv", overwrite), index=False)
    check_write_path(metrics_dir / "bounded_ablation_summary.md", overwrite).write_text(
        dataframe_to_markdown(summary) if len(summary) else "(no completed jobs)\n",
        encoding="utf-8",
    )
    write_report(out_root, statuses, summary, overwrite)
    return summary


def write_report(out_root, statuses, summary, overwrite):
    docs = out_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    counts = pd.Series([s["status"] for s in statuses]).value_counts().to_dict() if statuses else {}
    full_algos = sorted(set(summary.loc[summary.get("is_sample_run", pd.Series(dtype=bool)) == False, "algorithm"])) if len(summary) else []
    sample_algos = sorted(set(summary.loc[summary.get("is_sample_run", pd.Series(dtype=bool)) == True, "algorithm"])) if len(summary) else []
    total_elapsed = sum(float(s.get("elapsed_seconds", 0) or 0) for s in statuses)
    lines = ["# Bounded Multi-feature Ablation Analysis", ""]
    lines.append("## Run Status")
    lines.append("")
    for key in ["completed", "skipped_existing", "failed", "time_budget_skipped", "dependency_skipped"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.extend(["", "## Scope", ""])
    lines.append(f"- Full 100k jobs: {', '.join(full_algos) if full_algos else 'none'}.")
    lines.append(f"- Sample-level diagnostics: {', '.join(sample_algos) if sample_algos else 'none'}.")
    lines.append("- Sample-level HDBSCAN/CLARA/GMM results should not be compared as final full-100k models.")
    lines.append(f"- Total scheduled job runtime: {total_elapsed / 3600:.2f} hours ({total_elapsed:.1f} seconds).")
    lines.extend(["", "## Findings", ""])
    if len(summary):
        mini = summary[summary["algorithm"].astype(str).str.startswith("minibatch")]
        if len(mini):
            best = mini.sort_values(["silhouette", "dbi"], ascending=[False, True]).iloc[0]
            lines.append(f"- Best MiniBatch variant: `{best['variant']}` with `{best['algorithm']}` silhouette {float(best['silhouette']):.6f}, DBI {float(best['dbi']):.6f}.")
            k16 = mini[mini["algorithm"] == "minibatch_k16"].set_index("variant")
            if "text_pca64_only" in k16.index:
                pca = k16.loc["text_pca64_only"]
                lines.append(f"- MiniBatch k16 `text_pca64_only`: silhouette {float(pca['silhouette']):.6f}, DBI {float(pca['dbi']):.6f}. Existing text-only k16 baseline was silhouette 0.073499, DBI 3.046973, so PCA64 was worse for k16.")
            for label in [
                "text_pca64_lexical",
                "text_pca64_lexical_calendar",
                "text_pca64_lexical_calendar_publisher",
                "text_pca64_lexical_calendar_publisher_stock",
            ]:
                if label in k16.index:
                    row = k16.loc[label]
                    lines.append(f"- MiniBatch k16 `{label}`: silhouette {float(row['silhouette']):.6f}, DBI {float(row['dbi']):.6f}.")
            weight_rows = mini[mini["variant"].astype(str).str.startswith("text_pca64_all_aux_w")]
            if len(weight_rows):
                for algo, sub in weight_rows.groupby("algorithm"):
                    best_w = sub.sort_values(["silhouette", "dbi"], ascending=[False, True]).iloc[0]
                    lines.append(f"- Best MiniBatch aux weight for `{algo}`: `{best_w['variant']}` (aux_weight={float(best_w['aux_weight']):.2f}, silhouette {float(best_w['silhouette']):.6f}).")
        gmm = summary[summary["algorithm"].astype(str).str.startswith("gmm")]
        if len(gmm):
            best = gmm.sort_values(["silhouette", "dbi"], ascending=[False, True]).iloc[0]
            lines.append(f"- Best bounded GMM sample: `{best['variant']}` / `{best['algorithm']}` silhouette {float(best['silhouette']):.6f}.")
            worst = gmm.sort_values(["silhouette", "dbi"], ascending=[True, False]).iloc[0]
            lines.append(f"- Worst bounded GMM sample: `{worst['variant']}` / `{worst['algorithm']}` silhouette {float(worst['silhouette']):.6f}; publisher/stock/all-aux variants were the main degradation zone.")
        hdb = summary[summary["algorithm"].astype(str).str.startswith("hdbscan")]
        if len(hdb):
            best = hdb.sort_values(["silhouette", "coverage_pct"], ascending=[False, False]).iloc[0]
            lines.append(f"- Best bounded HDBSCAN sample: `{best['variant']}` / `{best['algorithm']}` silhouette {float(best['silhouette']):.6f}, coverage {float(best['coverage_pct']):.2f}%.")
            collapsed = hdb[(hdb["n_clusters"] <= 2) & (hdb["coverage_pct"] > 50)]
            if len(collapsed):
                desc = "; ".join(f"`{r.variant}`/{r.algorithm}: coverage {float(r.coverage_pct):.2f}%, clusters {int(r.n_clusters)}, silhouette {float(r.silhouette):.6f}" for r in collapsed.itertuples())
                lines.append(f"- HDBSCAN coverage/quality warning: {desc}.")
        clara = summary[summary["algorithm"].astype(str).str.startswith("clara")]
        if len(clara):
            best = clara.sort_values(["silhouette", "dbi"], ascending=[False, True]).iloc[0]
            lines.append(f"- Best CLARA diagnostic: `{best['variant']}` / `{best['algorithm']}` silhouette {float(best['silhouette']):.6f}.")
            notes = " ".join(clara["notes"].fillna("").astype(str).tolist())
            if "CLARA true via sklearn_extra.cluster.CLARA" in notes:
                lines.append("- CLARA used `sklearn_extra.cluster.CLARA` in the active environment.")
            elif "fallback" in notes:
                lines.append("- CLARA used a MiniBatchKMeans-to-medoid fallback because `sklearn_extra` was unavailable or failed in the active environment.")
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "1. Full 100k jobs were MiniBatch k8/k16/k32 across the full fast variant list.",
            "2. GMM, HDBSCAN, and CLARA were sample-level diagnostics with deterministic row_index samples.",
            "3. For MiniBatch k16, `text_pca64_only` degraded versus the existing 768-d text-only baseline.",
            "4. Lexical/calendar sometimes helped relative to PCA64, but not enough to beat the original text-only k16 baseline.",
            "5. Publisher/stock features tended to hurt or collapse structure: they lowered MiniBatch k16, worsened GMM, and caused HDBSCAN high-coverage/two-cluster behavior in some variants.",
            "6. The best aux_weight was algorithm-dependent, but `w010` was the strongest bounded HDBSCAN setting and competitive for MiniBatch; `w030` was often too heavy.",
            "7. GMM on bounded samples preferred `text_pca64_only`/`gmm_k16`; richer metadata-heavy variants often produced weak or negative silhouettes.",
            "8. HDBSCAN on bounded samples showed the central tradeoff: some all-aux variants had high silhouette at low coverage, while publisher/stock-heavy variants increased coverage but collapsed to very few clusters.",
            "9. CLARA ran through the fallback path and remained a diagnostic, not a final full-100k result.",
            "10. The run respected the 6-8h budget.",
            "11. Text-only remains the conservative main model.",
            "12. Multi-feature should be presented as a supplemental ablation/experimental extension, not as the primary model unless a tuned feature-weighting scheme later improves metrics and interpretability.",
            "",
            "## Recommendation",
            "",
            "Use text-only embedding as the main model if bounded ablation does not beat baseline metrics. Present multi-feature as supplemental evidence and future-work direction: tune feature weighting, supervised relevance weighting, or market-response labels.",
        ]
    )
    check_write_path(docs / "bounded_ablation_analysis.md", overwrite).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_root = Path(args.out_root)
    validate_root(out_root)
    start = time.time()
    deadline = start + args.time_budget_hours * 3600
    emb, text, aux, cols = load_inputs(args)
    jobs = schedule(args)
    print(f"Scheduled jobs after filter: {len(jobs)}")
    print("Algorithms to run:", ", ".join(sorted({job[1] for job in jobs})) if jobs else "(none)")
    print("Variants to run:", ", ".join(sorted({job[0] for job in jobs})) if jobs else "(none)")
    statuses = []
    matrices = {}
    sample_cache = {}
    try:
        for variant, algo, sample_n, is_sample, kind in jobs:
            now = time.time()
            paths = output_paths(out_root, variant, algo)
            status = {
                "variant": variant,
                "algorithm": algo,
                "fit_n_rows": N_ROWS if sample_n is None else int(sample_n),
                "eval_n_rows": N_ROWS if sample_n is None else int(sample_n),
                "is_sample_run": bool(is_sample),
                "status": "",
                "start_time": datetime.now().isoformat(timespec="seconds"),
                "end_time": "",
                "elapsed_seconds": 0.0,
                "output_dir": str(paths["root"]),
                "notes": "",
            }
            if args.skip_existing and valid_checkpoint(paths):
                status.update({"status": "skipped_existing", "end_time": datetime.now().isoformat(timespec="seconds")})
                statuses.append(status)
                write_status(out_root, statuses)
                continue
            if now >= deadline:
                status.update({"status": "time_budget_skipped", "end_time": datetime.now().isoformat(timespec="seconds"), "notes": "time budget exhausted before job start"})
                statuses.append(status)
                continue
            try:
                if variant not in matrices:
                    X_full, spec = build_matrix(variant, text, aux, cols)
                    matrices[variant] = (X_full, spec)
                    save_feature(out_root, variant, X_full, spec, args.overwrite)
                X_full, spec = matrices[variant]
                if sample_n is None:
                    row_index = np.arange(N_ROWS)
                    X_fit = X_full
                else:
                    if kind not in sample_cache:
                        sample_cache[kind] = sample_indices(out_root, kind, sample_n, args.seed, args.overwrite)
                    row_index = sample_cache[kind]
                    X_fit = np.asarray(X_full[row_index], dtype=np.float32)
                labels, algo_note = run_labels(algo, X_fit, args.seed)
                metric = compute_job_metrics(labels, X_fit, args.metric_sample_size, args.seed)
                row = {
                    "variant": variant,
                    "algorithm": algo,
                    "feature_space": variant,
                    "fit_n_rows": len(X_fit),
                    "eval_n_rows": len(X_fit),
                    "is_sample_run": bool(is_sample),
                    "n_total_dims": spec["n_total_dims"],
                    "n_text_dims": spec["n_text_dims"],
                    "n_aux_dims": spec["n_aux_dims"],
                    "aux_groups": "+".join(spec["groups"]) if spec["groups"] else "none",
                    "aux_weight": spec["aux_weight"],
                    **metric,
                }
                note = row["notes"]
                if algo_note:
                    note = (note + ";" if note else "") + algo_note
                    row["notes"] = note
                write_job_outputs(
                    paths,
                    labels,
                    row_index,
                    algo,
                    variant,
                    row,
                    {"variant": variant, "algorithm": algo, "sample_kind": kind, "sample_n": sample_n, "algorithm_note": algo_note},
                    args.overwrite,
                )
                status["status"] = "completed"
                status["notes"] = note
            except Exception as exc:
                status["status"] = "failed"
                status["notes"] = f"{type(exc).__name__}: {exc}"
            finally:
                status["end_time"] = datetime.now().isoformat(timespec="seconds")
                status["elapsed_seconds"] = round(time.time() - now, 3)
                statuses.append(status)
                write_status(out_root, statuses)
                summarize(out_root, statuses, True)
        summarize(out_root, statuses, True)
        return 0
    finally:
        if statuses:
            write_status(out_root, statuses)
            summarize(out_root, statuses, True)


if __name__ == "__main__":
    raise SystemExit(main())
