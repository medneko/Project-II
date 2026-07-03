#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

try:
    from scripts.run_multifeature_100k import dataframe_to_markdown
except ModuleNotFoundError:
    from run_multifeature_100k import dataframe_to_markdown


N_ROWS = 100_000
OUT_ROOT = Path("report") / "runs" / "100k" / "_k_metric_exploration"
FEATURE_SPACES = {"text_768_original", "text_pca64_only", "text_pca64_lexical"}


def build_parser():
    p = argparse.ArgumentParser(description="Explore k and metrics around MiniBatchKMeans 100k models.")
    p.add_argument("--emb", default="data/embeddings_100k.npy")
    p.add_argument("--text-pca", default="report/runs/100k/_multifeature/artifacts/X_text_pca64.npy")
    p.add_argument("--aux", default="report/runs/100k/_multifeature/artifacts/X_aux_features.npy")
    p.add_argument("--feature-columns", default="report/runs/100k/_multifeature/artifacts/feature_columns.json")
    p.add_argument("--meta", default="report/runs/100k/_multifeature/artifacts/multifeature_meta.csv")
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--k-values", default="8,12,16,24,32,40,48,64,96")
    p.add_argument("--feature-spaces", default="text_768_original,text_pca64_only,text_pca64_lexical")
    p.add_argument("--metric-sample-size", type=int, default=10000)
    p.add_argument("--stability-seeds", default="7,13,21,42,100")
    p.add_argument("--stability-sample-size", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p


def check_write_path(path: Path, overwrite: bool) -> Path:
    try:
        path.resolve().relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside {OUT_ROOT}: {path}") from exc
    if path.exists() and path.is_file() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_out_root(path: Path):
    try:
        path.resolve().relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"--out-root must stay under {OUT_ROOT}: {path}") from exc


def parse_ints(raw: str, name: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} cannot be empty")
    return vals


def parse_feature_spaces(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(vals) - FEATURE_SPACES)
    if unknown:
        raise ValueError(f"Unknown feature spaces: {unknown}. Valid: {sorted(FEATURE_SPACES)}")
    return vals


def load_feature_matrix(space: str, args, feature_columns: dict) -> np.ndarray:
    if space == "text_768_original":
        return np.load(args.emb, mmap_mode="r")
    if space == "text_pca64_only":
        return np.load(args.text_pca, mmap_mode="r")
    existing = Path("report/runs/100k/_multifeature/bounded_ablation/features/text_pca64_lexical.npy")
    if existing.exists():
        return np.load(existing, mmap_mode="r")
    text = np.load(args.text_pca, mmap_mode="r")
    aux = np.load(args.aux, mmap_mode="r")
    lexical_width = len(feature_columns["lexical"])
    return np.hstack([np.asarray(text, dtype=np.float32), 0.3 * np.asarray(aux[:, :lexical_width], dtype=np.float32)]).astype(np.float32)


def sample_indices(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=min(size, n), replace=False))


def balance_metrics(labels: np.ndarray) -> dict:
    counts = pd.Series(labels).value_counts().sort_index()
    sizes = counts.to_numpy(dtype=np.float64)
    total = sizes.sum()
    sorted_sizes = np.sort(sizes)[::-1]
    ent = entropy(sizes)
    return {
        "largest_cluster_pct": float(sorted_sizes[:1].sum() / total * 100.0),
        "top3_cluster_pct": float(sorted_sizes[:3].sum() / total * 100.0),
        "top5_cluster_pct": float(sorted_sizes[:5].sum() / total * 100.0),
        "min_cluster_size": int(sizes.min()),
        "median_cluster_size": float(np.median(sizes)),
        "max_cluster_size": int(sizes.max()),
        "normalized_entropy": float(ent / math.log(len(sizes))) if len(sizes) > 1 else 0.0,
        "gini": gini(sizes),
        "n_clusters_lt_100": int((sizes < 100).sum()),
        "n_clusters_lt_500": int((sizes < 500).sum()),
    }


def entropy(values: np.ndarray) -> float:
    p = values[values > 0] / values.sum()
    return float(-(p * np.log(p)).sum())


def gini(values: np.ndarray) -> float:
    x = np.sort(values.astype(np.float64))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def token_counts(texts: pd.Series) -> Counter:
    c = Counter()
    for text in texts.fillna("").astype(str):
        toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower()) if t not in ENGLISH_STOP_WORDS]
        c.update(toks)
    return c


def interpretability_notes(meta: pd.DataFrame, labels: np.ndarray, max_clusters: int = 10) -> tuple[int, int, int]:
    generic = {"stock", "stocks", "market", "company", "shares", "earnings", "news", "report", "update"}
    low_interp = 0
    pub_warn = 0
    stock_warn = 0
    for lab in pd.unique(labels):
        sub = meta[labels == lab]
        top_terms = token_counts(sub["headline_clean"]).most_common(10)
        if not top_terms or sum(term in generic for term, _ in top_terms[:5]) >= 4:
            low_interp += 1
        pub_share = sub["publisher"].fillna("").astype(str).value_counts(normalize=True).head(1)
        stock_share = sub["stock"].fillna("").astype(str).value_counts(normalize=True).head(1)
        if len(pub_share) and float(pub_share.iloc[0]) > 0.70:
            pub_warn += 1
        if len(stock_share) and float(stock_share.iloc[0]) > 0.50:
            stock_warn += 1
    return low_interp, pub_warn, stock_warn


def model_paths(out_root: Path, space: str, k: int) -> dict[str, Path]:
    root = out_root / "runs" / space / f"minibatch_k{k}"
    return {
        "root": root,
        "labels": root / "labels" / f"cluster_labels_{space}_minibatch_k{k}.csv",
        "metrics": root / "metrics" / "results_summary.csv",
    }


def fit_or_load(space: str, k: int, X, args, meta: pd.DataFrame) -> tuple[np.ndarray, dict]:
    paths = model_paths(Path(args.out_root), space, k)
    if paths["labels"].exists() and paths["metrics"].exists() and not args.overwrite:
        labels = pd.read_csv(paths["labels"])["label"].to_numpy()
        metric = pd.read_csv(paths["metrics"]).iloc[0].to_dict()
        return labels, metric

    model = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=4096, n_init=3)
    labels = model.fit_predict(np.asarray(X, dtype=np.float32))
    rows = sample_indices(len(labels), args.metric_sample_size, args.seed)
    Xs = np.asarray(X[rows], dtype=np.float32)
    ys = labels[rows]
    notes = []
    try:
        sil_cos = float(silhouette_score(Xs, ys, metric="cosine"))
    except Exception as exc:
        sil_cos = np.nan
        notes.append(f"silhouette_cosine_failed:{type(exc).__name__}")
    try:
        sil_euc = float(silhouette_score(Xs, ys, metric="euclidean"))
    except Exception as exc:
        sil_euc = np.nan
        notes.append(f"silhouette_euclidean_failed:{type(exc).__name__}")
    try:
        dbi = float(davies_bouldin_score(Xs, ys))
    except Exception as exc:
        dbi = np.nan
        notes.append(f"dbi_failed:{type(exc).__name__}")
    try:
        ch = float(calinski_harabasz_score(Xs, ys))
    except Exception as exc:
        ch = np.nan
        notes.append(f"calinski_harabasz_failed:{type(exc).__name__}")
    low_interp, pub_warn, stock_warn = interpretability_notes(meta, labels)
    bal = balance_metrics(labels)
    metric = {
        "feature_space": space,
        "algorithm": "MiniBatchKMeans",
        "k": k,
        "fit_n_rows": int(len(labels)),
        "eval_n_rows": int(len(rows)),
        "coverage_pct": 100.0,
        "n_clusters": int(len(np.unique(labels))),
        "silhouette_cosine": sil_cos,
        "silhouette_euclidean": sil_euc,
        "dbi": dbi,
        "calinski_harabasz": ch,
        "inertia": float(model.inertia_),
        **bal,
        "notes": ";".join(notes + [f"low_interpretability_clusters={low_interp}", f"publisher_warnings={pub_warn}", f"stock_warnings={stock_warn}"]),
    }
    pd.DataFrame({"row_index": np.arange(len(labels)), "label": labels, "algorithm": f"minibatch_k{k}", "feature_space": space}).to_csv(check_write_path(paths["labels"], args.overwrite), index=False)
    pd.DataFrame([metric]).to_csv(check_write_path(paths["metrics"], args.overwrite), index=False)
    return labels, metric


def stability_for(space: str, k: int, X, seeds: list[int], sample_rows: np.ndarray, args) -> list[dict]:
    out_dir = Path(args.out_root) / "stability" / space / f"minibatch_k{k}"
    labels_by_seed = {}
    Xs = np.asarray(X[sample_rows], dtype=np.float32)
    for seed in seeds:
        path = out_dir / f"labels_seed{seed}.csv"
        if path.exists() and not args.overwrite:
            labels = pd.read_csv(path)["label"].to_numpy()
        else:
            labels = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=3).fit_predict(Xs)
            pd.DataFrame({"row_index": sample_rows, "label": labels, "seed": seed}).to_csv(check_write_path(path, args.overwrite), index=False)
        labels_by_seed[seed] = labels
    ref = labels_by_seed[args.seed]
    rows = []
    for seed in seeds:
        if seed == args.seed:
            continue
        rows.append(
            {
                "feature_space": space,
                "k": k,
                "reference_seed": args.seed,
                "compared_seed": seed,
                "ARI": adjusted_rand_score(ref, labels_by_seed[seed]),
                "NMI": normalized_mutual_info_score(ref, labels_by_seed[seed]),
                "is_stability_sample": True,
                "sample_n": int(len(sample_rows)),
                "notes": "deterministic stability sample",
            }
        )
    return rows


def plot_feature_space(space: str, summary: pd.DataFrame, stability: pd.DataFrame, out_root: Path, overwrite: bool):
    import matplotlib.pyplot as plt

    charts = out_root / "charts" / space
    sub = summary[summary["feature_space"] == space].sort_values("k")
    stab = stability[stability["feature_space"] == space].groupby("k")[["ARI", "NMI"]].mean().reset_index()
    for y, name, ylabel in [
        ("silhouette_cosine", "silhouette_vs_k.png", "Silhouette cosine"),
        ("dbi", "dbi_vs_k.png", "DBI"),
        ("largest_cluster_pct", "cluster_size_balance_vs_k.png", "Largest cluster %"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sub["k"], sub[y], marker="o")
        ax.set_title(f"{space}: {ylabel}")
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(check_write_path(charts / name, overwrite), dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(stab):
        ax.plot(stab["k"], stab["ARI"], marker="o", label="ARI")
        ax.plot(stab["k"], stab["NMI"], marker="o", label="NMI")
    ax.set_title(f"{space}: stability vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(check_write_path(charts / "stability_vs_k.png", overwrite), dpi=180)
    plt.close(fig)


def write_docs(summary: pd.DataFrame, stability: pd.DataFrame, args):
    docs = Path(args.out_root) / "docs"
    lines = ["# K Selection Analysis", ""]
    recs = ["# K Selection Recommendation", ""]
    for space in ["text_768_original", "text_pca64_only", "text_pca64_lexical"]:
        sub = summary[summary["feature_space"] == space]
        if sub.empty:
            continue
        best_sil = sub.sort_values(["silhouette_cosine", "dbi"], ascending=[False, True]).iloc[0]
        best_dbi = sub.sort_values(["dbi", "silhouette_cosine"], ascending=[True, False]).iloc[0]
        stab = stability[stability["feature_space"] == space].groupby("k")[["ARI", "NMI"]].mean().reset_index()
        best_stab = stab.sort_values(["ARI", "NMI"], ascending=[False, False]).iloc[0] if len(stab) else None
        lines.extend(
            [
                f"## {space}",
                "",
                f"- Best k by silhouette: k={int(best_sil['k'])}, silhouette={float(best_sil['silhouette_cosine']):.6f}, DBI={float(best_sil['dbi']):.6f}.",
                f"- Best k by DBI: k={int(best_dbi['k'])}, DBI={float(best_dbi['dbi']):.6f}, silhouette={float(best_dbi['silhouette_cosine']):.6f}.",
            ]
        )
        if best_stab is not None:
            lines.append(f"- Best k by sample stability: k={int(best_stab['k'])}, mean ARI={float(best_stab['ARI']):.4f}, mean NMI={float(best_stab['NMI']):.4f}.")
        for k in [32, 48, 64]:
            row = sub[sub["k"] == k]
            if not row.empty:
                r = row.iloc[0]
                lines.append(f"- k={k}: silhouette={float(r['silhouette_cosine']):.6f}, DBI={float(r['dbi']):.6f}, largest={float(r['largest_cluster_pct']):.2f}%, min_size={int(r['min_cluster_size'])}.")
        lines.append("")
    lex = summary[summary["feature_space"] == "text_pca64_lexical"]
    chosen = choose_recommended(lex, stability[stability["feature_space"] == "text_pca64_lexical"])
    lines.extend(
        [
            "## Cross-cutting Answers",
            "",
            "- Raising k can improve silhouette, but the decision should be checked against DBI, cluster balance, stability, and interpretability notes.",
            "- Very high k values are not automatically better; they can fragment clusters and reduce stability.",
            f"- Recommended experimental k for `text_pca64_lexical`: k={chosen}.",
            f"- `text_pca64_lexical + minibatch_k32` {'remains a strong choice' if chosen == 32 else 'is no longer the single best recommendation'} under this sweep.",
            "- Conservative baseline should remain text-only MiniBatch around k=16 unless the report emphasizes finer-grained exploratory clusters.",
        ]
    )
    recs.extend(
        [
            f"- Conservative baseline: keep `text_768_original + minibatch_k16`.",
            f"- Best experimental model from this sweep: `text_pca64_lexical + minibatch_k{chosen}`.",
            "- Main report should present silhouette/DBI together with cluster balance and stability, not silhouette alone.",
            "- Publisher/stock are not part of this fitting loop; metadata remains post-hoc profiling only.",
        ]
    )
    check_write_path(docs / "k_selection_analysis.md", args.overwrite).write_text("\n".join(lines) + "\n", encoding="utf-8")
    check_write_path(docs / "k_selection_recommendation.md", args.overwrite).write_text("\n".join(recs) + "\n", encoding="utf-8")


def choose_recommended(sub: pd.DataFrame, stab: pd.DataFrame) -> int:
    merged = sub.copy()
    if len(stab):
        st = stab.groupby("k")[["ARI", "NMI"]].mean().reset_index()
        merged = merged.merge(st, on="k", how="left")
    else:
        merged["ARI"] = np.nan
        merged["NMI"] = np.nan
    for col in ["silhouette_cosine", "normalized_entropy", "ARI", "NMI"]:
        vals = merged[col].astype(float)
        rng = vals.max() - vals.min()
        merged[col + "_score"] = (vals - vals.min()) / rng if np.isfinite(rng) and rng > 0 else 0.5
    vals = merged["dbi"].astype(float)
    rng = vals.max() - vals.min()
    merged["dbi_score"] = (vals.max() - vals) / rng if np.isfinite(rng) and rng > 0 else 0.5
    merged["small_penalty"] = (merged["n_clusters_lt_500"] > 0).astype(float) * 0.2
    merged["score"] = (
        0.35 * merged["silhouette_cosine_score"]
        + 0.20 * merged["dbi_score"]
        + 0.15 * merged["normalized_entropy_score"]
        + 0.15 * merged["ARI_score"].fillna(0.5)
        + 0.15 * merged["NMI_score"].fillna(0.5)
        - merged["small_penalty"]
    )
    return int(merged.sort_values(["score", "silhouette_cosine"], ascending=[False, False]).iloc[0]["k"])


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_root = Path(args.out_root)
    validate_out_root(out_root)
    k_values = parse_ints(args.k_values, "--k-values")
    seeds = parse_ints(args.stability_seeds, "--stability-seeds")
    if args.seed not in seeds:
        seeds.append(args.seed)
    spaces = parse_feature_spaces(args.feature_spaces)
    feature_columns = json.loads(Path(args.feature_columns).read_text(encoding="utf-8"))
    meta = pd.read_csv(args.meta)
    summary_rows = []
    stability_rows = []
    stability_idx = sample_indices(N_ROWS, args.stability_sample_size, args.seed)
    for space in spaces:
        print(f"Loading feature space {space}")
        X = load_feature_matrix(space, args, feature_columns)
        if X.shape[0] != N_ROWS:
            raise ValueError(f"{space} has invalid row count: {X.shape}")
        for k in k_values:
            print(f"Running {space} k={k}")
            _labels, metric = fit_or_load(space, k, X, args, meta)
            summary_rows.append(metric)
            stability_rows.extend(stability_for(space, k, X, seeds, stability_idx, args))
    summary = pd.DataFrame(summary_rows)
    stability = pd.DataFrame(stability_rows)
    metrics_dir = out_root / "metrics"
    summary.to_csv(check_write_path(metrics_dir / "k_sweep_summary.csv", args.overwrite), index=False)
    check_write_path(metrics_dir / "k_sweep_summary.md", args.overwrite).write_text(dataframe_to_markdown(summary), encoding="utf-8")
    stability.to_csv(check_write_path(metrics_dir / "k_sweep_stability.csv", args.overwrite), index=False)
    for space in spaces:
        plot_feature_space(space, summary, stability, out_root, args.overwrite)
    write_docs(summary, stability, args)
    print("Wrote k exploration to", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
