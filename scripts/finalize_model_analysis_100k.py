#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

try:
    from scripts.run_multifeature_100k import dataframe_to_markdown
except ModuleNotFoundError:
    from run_multifeature_100k import dataframe_to_markdown


N_ROWS = 100_000
FINAL_ROOT = Path("report") / "runs" / "100k" / "_final_analysis"
POSITIVE = {"gain", "gains", "rise", "rises", "rising", "surge", "surges", "beat", "beats", "growth", "bullish", "upgrade", "upgraded", "outperform", "profit", "profits", "record", "strong"}
NEGATIVE = {"fall", "falls", "falling", "drop", "drops", "plunge", "plunges", "miss", "misses", "loss", "losses", "bearish", "downgrade", "downgraded", "underperform", "weak", "warning", "cut", "cuts"}
RISK = {"risk", "risks", "uncertain", "uncertainty", "volatile", "volatility", "lawsuit", "probe", "investigation", "debt", "default", "inflation", "recession", "crisis"}


def check_write_path(path: Path, overwrite: bool) -> Path:
    try:
        path.resolve().relative_to(FINAL_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside {FINAL_ROOT}: {path}") from exc
    if path.exists() and path.is_file() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_parser():
    p = argparse.ArgumentParser(description="Finalize 100k model comparison and deep analysis.")
    p.add_argument("--runs-root", default="report/runs/100k")
    p.add_argument("--bounded-root", default="report/runs/100k/_multifeature/bounded_ablation")
    p.add_argument("--clara-true-root", default="report/runs/100k/_multifeature/bounded_ablation/clara_true")
    p.add_argument("--multifeature-artifacts", default="report/runs/100k/_multifeature/artifacts")
    p.add_argument("--outdir", default=str(FINAL_ROOT))
    p.add_argument("--best-variant", default="text_pca64_lexical")
    p.add_argument("--best-algorithm", default="minibatch_k32")
    p.add_argument("--sample-size-metrics", type=int, default=10000)
    p.add_argument("--sample-size-pairwise", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p


def validate_outdir(path: Path):
    try:
        path.resolve().relative_to(FINAL_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"--outdir must stay under {FINAL_ROOT}: {path}") from exc


def read_one_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def metric_row_from_text(runs_root: Path, algo: str, role: str, selected: str, reason: str) -> dict | None:
    path = runs_root / algo / "metrics" / "results_summary.csv"
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    return {
        "model_role": role,
        "feature_space": "text_only",
        "variant": "text_768_original",
        "algorithm": algo,
        "fit_n_rows": int(row.get("n_embedding_points", N_ROWS)),
        "eval_n_rows": int(row.get("n_points_labeled", N_ROWS)),
        "is_sample_run": False,
        "coverage_pct": row.get("coverage_pct"),
        "n_clusters": row.get("n_clusters"),
        "silhouette": row.get("silhouette"),
        "silhouette_metric": row.get("silhouette_metric", "cosine"),
        "dbi": row.get("dbi"),
        "notes": row.get("notes", ""),
        "selected_for_final": selected,
        "reason": reason,
    }


def metric_row_from_summary(summary: pd.DataFrame, variant: str, algo: str, role: str, selected: str, reason: str) -> dict | None:
    hit = summary[(summary["variant"] == variant) & (summary["algorithm"] == algo)]
    if hit.empty:
        return None
    row = hit.iloc[0].to_dict()
    return {
        "model_role": role,
        "feature_space": "multifeature",
        "variant": variant,
        "algorithm": algo,
        "fit_n_rows": row.get("fit_n_rows"),
        "eval_n_rows": row.get("eval_n_rows"),
        "is_sample_run": row.get("is_sample_run"),
        "coverage_pct": row.get("coverage_pct"),
        "n_clusters": row.get("n_clusters"),
        "silhouette": row.get("silhouette"),
        "silhouette_metric": row.get("silhouette_metric", "cosine"),
        "dbi": row.get("dbi"),
        "notes": row.get("notes", ""),
        "selected_for_final": selected,
        "reason": reason,
    }


def write_final_comparison(args, outdir: Path, bounded: pd.DataFrame, clara: pd.DataFrame):
    rows = []
    runs_root = Path(args.runs_root)
    candidates = [
        metric_row_from_text(runs_root, "minibatch_k16", "conservative main baseline", "yes", "Stable full-coverage text-only baseline."),
        metric_row_from_text(runs_root, "minibatch_k32", "text-only k32 reference", "no", "Useful k32 reference for experimental model."),
        metric_row_from_summary(bounded, "text_pca64_only", "minibatch_k32", "PCA64 diagnostic", "no", "Tests PCA compression without auxiliary features."),
        metric_row_from_summary(bounded, args.best_variant, args.best_algorithm, "best full-100k experimental model", "yes", "Best full-100k MiniBatch ablation result."),
        metric_row_from_text(runs_root, "gmm_k8", "compact probabilistic baseline", "yes", "Full-coverage probabilistic text-only baseline."),
        metric_row_from_text(runs_root, "hdbscan_minsize50", "dense/event detection model", "yes", "High-quality dense cluster detector with partial coverage."),
        metric_row_from_summary(bounded, "text_pca64_all_aux_w010", "hdbscan_minsize50", "bounded HDBSCAN diagnostic", "no", "Sample-level diagnostic; not a full-100k final model."),
    ]
    if len(clara):
        best_clara = clara.sort_values(["silhouette", "dbi"], ascending=[False, True]).iloc[0]
        candidates.append(
            metric_row_from_summary(clara, str(best_clara["variant"]), str(best_clara["algorithm"]), "CLARA true diagnostic baseline", "no", "Rerun with sklearn_extra.cluster.CLARA; still weaker than MiniBatch.")
        )
    candidates.append(
        metric_row_from_summary(bounded, "text_pca64_all_aux_w030", "minibatch_k16", "previous all-aux w030 full multi-feature", "no", "Represents heavier all-aux fusion at weight 0.30.")
    )
    rows = [row for row in candidates if row is not None]
    df = pd.DataFrame(rows)
    metrics_dir = outdir / "metrics"
    df.to_csv(check_write_path(metrics_dir / "final_model_comparison.csv", args.overwrite), index=False)
    check_write_path(metrics_dir / "final_model_comparison.md", args.overwrite).write_text(dataframe_to_markdown(df), encoding="utf-8")
    return df


def load_best_labels(bounded_root: Path, variant: str, algo: str) -> pd.DataFrame:
    run_dir = bounded_root / "runs" / variant / algo
    files = sorted((run_dir / "labels").glob("cluster_labels_*.csv"))
    if not files:
        raise FileNotFoundError(f"No label CSV found under {run_dir / 'labels'}")
    df = pd.read_csv(files[0])
    if not np.array_equal(df["row_index"].to_numpy(), np.arange(len(df))):
        raise ValueError("Best model labels are not full row_index 0..n-1")
    return df


def build_best_matrix(artifacts: Path, feature_columns: dict) -> np.ndarray:
    text = np.load(artifacts / "X_text_pca64.npy", mmap_mode="r")
    aux = np.load(artifacts / "X_aux_features.npy", mmap_mode="r")
    lexical_width = len(feature_columns["lexical"])
    return np.hstack([np.asarray(text, dtype=np.float32), 0.3 * np.asarray(aux[:, :lexical_width], dtype=np.float32)]).astype(np.float32)


def entropy(values):
    p = np.asarray(values, dtype=np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def gini(values):
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def write_cluster_size_and_balance(labels, out_base: Path, overwrite: bool):
    counts = pd.Series(labels).value_counts().sort_index()
    total = int(counts.sum())
    sizes = counts.to_numpy()
    profile = pd.DataFrame(
        {
            "cluster_id": counts.index.astype(int),
            "size": counts.values.astype(int),
        }
    )
    profile["size_pct"] = profile["size"] / total * 100.0
    profile = profile.sort_values("size", ascending=False).reset_index(drop=True)
    profile["cumulative_pct"] = profile["size_pct"].cumsum()
    profile["min_size"] = int(sizes.min())
    profile["median_size"] = float(np.median(sizes))
    profile["max_size"] = int(sizes.max())
    profile["largest_cluster_pct"] = float(sizes.max() / total * 100.0)
    profile["n_clusters_size_lt_100"] = int((sizes < 100).sum())
    profile["n_clusters_size_lt_500"] = int((sizes < 500).sum())
    profile.to_csv(check_write_path(out_base / "cluster_size_profile.csv", overwrite), index=False)
    sorted_sizes = np.sort(sizes)[::-1]
    bal = {
        "n_clusters": int(len(sizes)),
        "entropy": entropy(sizes),
        "normalized_entropy": entropy(sizes) / math.log(len(sizes)) if len(sizes) > 1 else 0.0,
        "gini": gini(sizes),
        "top_1_cluster_pct": float(sorted_sizes[:1].sum() / total * 100.0),
        "top_3_cluster_pct": float(sorted_sizes[:3].sum() / total * 100.0),
        "top_5_cluster_pct": float(sorted_sizes[:5].sum() / total * 100.0),
        "min_size": int(sizes.min()),
        "median_size": float(np.median(sizes)),
        "max_size": int(sizes.max()),
        "largest_cluster_pct": float(sorted_sizes[0] / total * 100.0),
        "n_clusters_size_lt_100": int((sizes < 100).sum()),
        "n_clusters_size_lt_500": int((sizes < 500).sum()),
    }
    check_write_path(out_base / "cluster_balance_metrics.json", overwrite).write_text(json.dumps(bal, indent=2), encoding="utf-8")
    return profile, bal


def sample_rows(n, size, seed):
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=min(size, n), replace=False))


def silhouette_profile(X, labels, rows, out_base, overwrite):
    vals = silhouette_samples(X[rows], labels[rows], metric="cosine")
    df = pd.DataFrame({"row_index": rows, "label": labels[rows], "silhouette": vals})
    rows_out = []
    for lab, sub in df.groupby("label"):
        rows_out.append(
            {
                "cluster_id": int(lab),
                "sample_size": int(len(sub)),
                "silhouette_mean": float(sub["silhouette"].mean()),
                "silhouette_median": float(sub["silhouette"].median()),
                "silhouette_p10": float(sub["silhouette"].quantile(0.10)),
                "silhouette_p25": float(sub["silhouette"].quantile(0.25)),
                "silhouette_p75": float(sub["silhouette"].quantile(0.75)),
                "silhouette_p90": float(sub["silhouette"].quantile(0.90)),
                "negative_silhouette_fraction": float((sub["silhouette"] < 0).mean()),
            }
        )
    prof = pd.DataFrame(rows_out).sort_values("silhouette_mean")
    prof.to_csv(check_write_path(out_base / "cluster_silhouette_profile.csv", overwrite), index=False)
    return prof, float(vals.mean())


def centroid_diagnostics(X, labels, rows, out_base, overwrite):
    labs = np.array(sorted(pd.unique(labels)))
    centers = np.vstack([X[labels == lab].mean(axis=0) for lab in labs]).astype(np.float32)
    d_centers = euclidean_distances(centers)
    np.fill_diagonal(d_centers, np.inf)
    rows_out = []
    for i, lab in enumerate(labs):
        mask = labels[rows] == lab
        Xs = X[rows][mask]
        own = np.linalg.norm(Xs - centers[i], axis=1) if len(Xs) else np.array([np.nan])
        nearest_idx = int(np.argmin(d_centers[i]))
        rows_out.append(
            {
                "cluster_id": int(lab),
                "centroid_norm": float(np.linalg.norm(centers[i])),
                "sample_size": int(mask.sum()),
                "avg_distance_to_own_centroid": float(np.nanmean(own)),
                "nearest_cluster": int(labs[nearest_idx]),
                "nearest_centroid_distance": float(d_centers[i, nearest_idx]),
                "centroid_margin": float(d_centers[i, nearest_idx] - np.nanmean(own)),
                "scatter_estimate": float(np.nanmean(own)),
            }
        )
    diag = pd.DataFrame(rows_out)
    pairs = []
    for i, j in combinations(range(len(labs)), 2):
        dist = float(d_centers[i, j])
        s_i = float(diag.loc[diag["cluster_id"] == labs[i], "scatter_estimate"].iloc[0])
        s_j = float(diag.loc[diag["cluster_id"] == labs[j], "scatter_estimate"].iloc[0])
        pairs.append(
            {
                "cluster_a": int(labs[i]),
                "cluster_b": int(labs[j]),
                "centroid_distance": dist,
                "dbi_like_ratio": float((s_i + s_j) / dist) if dist > 0 else np.inf,
            }
        )
    pair_df = pd.DataFrame(pairs).sort_values("centroid_distance")
    diag.to_csv(check_write_path(out_base / "cluster_cohesion_separation.csv", overwrite), index=False)
    pair_df.head(50).to_csv(check_write_path(out_base / "worst_dbi_like_cluster_pairs.csv", overwrite), index=False)
    return diag, pair_df


def lexicon_scores(text):
    toks = [t.lower().strip(".:-") for t in str(text).split()]
    pos = sum(t in POSITIVE for t in toks)
    neg = sum(t in NEGATIVE for t in toks)
    risk = sum(t in RISK for t in toks)
    return pos - neg, risk, risk / max(len(toks), 1)


def metadata_profile(meta, labels, out_base, overwrite):
    df = meta.copy()
    df["label"] = labels
    scores = df["headline_clean"].fillna("").apply(lexicon_scores)
    df["sentiment_score"] = [x[0] for x in scores]
    df["risk_count"] = [x[1] for x in scores]
    df["risk_ratio"] = [x[2] for x in scores]
    df["headline_length"] = df["headline_clean"].fillna("").astype(str).str.len()
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m")
    rows = []
    for lab, sub in df.groupby("label"):
        pub = sub["publisher"].fillna("").astype(str).value_counts().head(5)
        stock = sub["stock"].fillna("").astype(str).value_counts().head(5)
        top_pub_share = float(pub.iloc[0] / len(sub) * 100.0) if len(pub) else 0
        top_stock_share = float(stock.iloc[0] / len(sub) * 100.0) if len(stock) else 0
        rows.append(
            {
                "cluster_id": int(lab),
                "size": int(len(sub)),
                "top_publishers": " | ".join(f"{k}:{v}" for k, v in pub.items()),
                "top_stocks": " | ".join(f"{k}:{v}" for k, v in stock.items()),
                "date_min": str(sub["date"].min()),
                "date_max": str(sub["date"].max()),
                "top_months": " | ".join(f"{k}:{v}" for k, v in sub["month"].value_counts().head(5).items()),
                "avg_headline_length": float(sub["headline_length"].mean()),
                "avg_sentiment_score": float(sub["sentiment_score"].mean()),
                "avg_risk_count": float(sub["risk_count"].mean()),
                "avg_risk_ratio": float(sub["risk_ratio"].mean()),
                "top_publisher_share_pct": top_pub_share,
                "top_stock_share_pct": top_stock_share,
                "publisher_dominance_warning": bool(top_pub_share > 70.0),
                "stock_dominance_warning": bool(top_stock_share > 50.0),
            }
        )
    out = pd.DataFrame(rows).sort_values("cluster_id")
    out.to_csv(check_write_path(out_base / "cluster_metadata_profile.csv", overwrite), index=False)
    return out


def tokenize(text):
    return [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", str(text).lower()) if t not in ENGLISH_STOP_WORDS]


def write_terms_examples(meta, labels, out_base, overwrite):
    lines = ["# Cluster Terms and Example Headlines", ""]
    generic_flags = []
    for lab in sorted(pd.unique(labels)):
        sub = meta[labels == lab]
        counter = Counter()
        for headline in sub["headline_clean"].fillna("").astype(str):
            counter.update(tokenize(headline))
        top_terms = counter.most_common(12)
        generic = len(top_terms) < 5
        if generic:
            generic_flags.append(int(lab))
        examples = sub["headline_clean"].fillna("").astype(str).head(5).tolist()
        lines.extend(
            [
                f"## Cluster {int(lab)}",
                "",
                f"- Size: {len(sub):,}",
                f"- Top terms: {', '.join(f'{term} ({count})' for term, count in top_terms)}",
                f"- Low-interpretability flag: {'yes' if generic else 'no'}",
                "- Examples:",
            ]
        )
        lines.extend([f"  - {ex}" for ex in examples])
        lines.append("")
    check_write_path(out_base / "cluster_terms_examples.md", overwrite).write_text("\n".join(lines), encoding="utf-8")
    return generic_flags


def stability(args, X, base_labels, outdir, overwrite):
    stab_dir = outdir / "stability" / "minibatch_k32_text_pca64_lexical"
    seeds = [7, 13, 21, 42, 100]
    rows = []
    label_by_seed = {}
    for seed in seeds:
        label_path = stab_dir / f"labels_seed{seed}.csv"
        if label_path.exists():
            labels = pd.read_csv(label_path)["label"].to_numpy()
        elif seed == 42:
            labels = base_labels
            pd.DataFrame({"row_index": np.arange(len(labels)), "label": labels}).to_csv(check_write_path(label_path, False), index=False)
        else:
            labels = MiniBatchKMeans(n_clusters=32, random_state=seed, batch_size=4096, n_init="auto").fit_predict(X)
            pd.DataFrame({"row_index": np.arange(len(labels)), "label": labels}).to_csv(check_write_path(label_path, False), index=False)
        label_by_seed[seed] = labels
    for seed in seeds:
        if seed == 42:
            continue
        rows.append(
            {
                "seed": seed,
                "reference_seed": 42,
                "n_rows": len(base_labels),
                "is_sample_run": False,
                "ARI_vs_seed42": adjusted_rand_score(base_labels, label_by_seed[seed]),
                "NMI_vs_seed42": normalized_mutual_info_score(base_labels, label_by_seed[seed]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(check_write_path(stab_dir / "stability_ari_nmi.csv", overwrite), index=False)
    return df


def plot_outputs(X, labels, rows, sil_prof, size_prof, out_base, overwrite):
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    from sklearn.decomposition import PCA

    img_dir = out_base / "images"
    Xs = X[rows]
    ys = labels[rows]
    pca_coords = PCA(n_components=2, random_state=42).fit_transform(Xs).astype(np.float32)
    np.save(check_write_path(out_base / "pca_2d_sample20000_seed42.npy", overwrite), pca_coords)
    plot_scatter(pca_coords, ys, "pca", img_dir, overwrite)
    try:
        import umap

        umap_path = out_base / "umap_2d_sample20000_seed42.npy"
        if umap_path.exists() and not overwrite:
            umap_coords = np.load(umap_path)
        else:
            umap_coords = umap.UMAP(n_components=2, random_state=42).fit_transform(Xs).astype(np.float32)
            np.save(check_write_path(umap_path, overwrite), umap_coords)
        plot_scatter(umap_coords, ys, "umap", img_dir, overwrite)
    except Exception as exc:
        check_write_path(out_base / "umap_skipped.txt", overwrite).write_text(str(exc), encoding="utf-8")

    size_plot = size_prof.sort_values("cluster_id")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(size_plot["cluster_id"].astype(str), size_plot["size"])
    ax.set_title("Cluster size profile")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Size")
    fig.tight_layout()
    fig.savefig(check_write_path(img_dir / "cluster_size_bar.png", overwrite), dpi=180)
    plt.close(fig)

    sil_plot = sil_prof.sort_values("cluster_id")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(sil_plot["cluster_id"].astype(str), sil_plot["silhouette_mean"])
    ax.set_title("Mean silhouette per cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean silhouette")
    fig.tight_layout()
    fig.savefig(check_write_path(img_dir / "silhouette_per_cluster.png", overwrite), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = size_prof["size_pct"].sort_values(ascending=False).to_numpy()
    ax.plot(np.arange(1, len(vals) + 1), np.cumsum(vals), marker="o")
    ax.set_title("Cluster balance cumulative share")
    ax.set_xlabel("Top clusters")
    ax.set_ylabel("Cumulative %")
    fig.tight_layout()
    fig.savefig(check_write_path(img_dir / "cluster_balance.png", overwrite), dpi=180)
    plt.close(fig)


def plot_scatter(coords, labels, method, img_dir, overwrite):
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    palette = qualitative.Plotly
    labs = sorted(pd.unique(labels))
    color = {lab: palette[i % len(palette)] for i, lab in enumerate(labs)}
    fig, ax = plt.subplots(figsize=(9, 7))
    for lab in labs:
        mask = labels == lab
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.65, c=color[lab], label=f"cluster {int(lab)}")
    for lab in labs:
        mask = labels == lab
        c = coords[mask].mean(axis=0)
        ax.scatter(c[0], c[1], s=90, marker="o", c=color[lab], edgecolors="black", linewidths=1.2)
    ax.set_title(f"text_pca64_lexical + minibatch_k32 {method.upper()} 2D")
    fig.tight_layout()
    fig.savefig(check_write_path(img_dir / f"best_model_{method}_2d.png", overwrite), dpi=200)
    plt.close(fig)

    pfig = go.Figure()
    for lab in labs:
        mask = labels == lab
        pfig.add_trace(go.Scatter(x=coords[mask, 0], y=coords[mask, 1], mode="markers", name=f"cluster {int(lab)}", meta={"role": "points"}, marker=dict(size=4, opacity=0.72, color=color[lab])))
    for lab in labs:
        mask = labels == lab
        c = coords[mask].mean(axis=0)
        pfig.add_trace(go.Scatter(x=[c[0]], y=[c[1]], mode="markers", name=f"centroid c{int(lab)}", meta={"role": "centroids"}, marker=dict(size=14, symbol="circle", color=color[lab], line=dict(color="black", width=2)), visible=False))
    roles = [tr.meta or {} for tr in pfig.data]

    def vis(points, centroids):
        return [(r.get("role") == "points" and points) or (r.get("role") == "centroids" and centroids) for r in roles]

    pfig.update_layout(
        title=f"text_pca64_lexical + minibatch_k32 {method.upper()} 2D",
        xaxis=dict(autorange=False, range=axis_range(coords[:, 0])),
        yaxis=dict(autorange=False, range=axis_range(coords[:, 1])),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=-0.08,
                buttons=[
                    dict(label="Show all", method="update", args=[{"visible": vis(True, True)}]),
                    dict(label="Points only", method="update", args=[{"visible": vis(True, False)}]),
                    dict(label="Centroids only", method="update", args=[{"visible": vis(False, True)}]),
                ],
            )
        ],
        margin=dict(b=90),
    )
    pfig.write_html(check_write_path(img_dir / f"best_model_{method}_2d.html", overwrite))


def axis_range(values):
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    return [lo - pad, hi + pad]


def write_docs(args, outdir, final_df, bounded, clara, size_profile, balance, sil_prof, cohesion, metadata, stability_df, global_sil):
    docs = outdir / "docs"
    best = f"{args.best_variant} + {args.best_algorithm}"
    clara_best = clara.sort_values(["silhouette", "dbi"], ascending=[False, True]).iloc[0] if len(clara) else None
    warnings = metadata[(metadata["publisher_dominance_warning"]) | (metadata["stock_dominance_warning"])]
    worst = sil_prof.sort_values("silhouette_mean").head(5)
    best_sil = sil_prof.sort_values("silhouette_mean", ascending=False).head(5)
    common = {
        "best": best,
        "global_sil": global_sil,
        "largest": balance["largest_cluster_pct"],
        "entropy": balance["normalized_entropy"],
        "gini": balance["gini"],
        "warnings": len(warnings),
    }
    write_text(docs / "clara_true_update.md", f"""# CLARA True Update

Old bounded-ablation CLARA fallback results were diagnostic only. CLARA was rerun under `.venv311` with `sklearn_extra.cluster.CLARA`, and every `clara_true` row records `CLARA true via sklearn_extra.cluster.CLARA`.

Best CLARA true result: `{clara_best['variant']} + {clara_best['algorithm']}` with silhouette {float(clara_best['silhouette']):.6f} and DBI {float(clara_best['dbi']):.6f}.

CLARA true still does not beat MiniBatch. It remains a diagnostic baseline rather than the selected final model.
""", args.overwrite)
    write_text(docs / "final_model_selection.md", f"""# Final Model Selection

- Conservative main model: `text-only minibatch_k16`.
- Best full-100k experimental model: `{best}`.
- Dense/event detection model: `text-only hdbscan_minsize50`.
- Compact probabilistic baseline: `text-only gmm_k8`.
- Diagnostic baselines: CLARA true and bounded multi-feature variants.

Do not summarize the experiment as "multi-feature is always worse." Lightweight lexical features improved MiniBatch at k=32 in the bounded ablation, while metadata-heavy publisher/stock/all-aux variants could hurt GMM/HDBSCAN or collapse clusters. All-aux weight 0.30 is often too heavy.
""", args.overwrite)
    write_text(docs / "multifeature_ablation_conclusion.md", f"""# Multi-feature Ablation Conclusion

The best full-100k experimental model is `{best}`. Its global silhouette on the diagnostic sample is {global_sil:.6f}; the bounded summary reports silhouette 0.109388 and DBI 3.381518.

Lexical-only augmentation is the cleanest positive signal. Publisher/stock-heavy features are useful for context but can dominate or blur semantic clustering, especially for GMM and HDBSCAN.

CLARA true was tested and remains below the best MiniBatch result.
""", args.overwrite)
    write_text(docs / "final_experiment_summary.md", f"""# Final Experiment Summary

This final pass integrates text-only, bounded multi-feature ablation, and CLARA true rerun results.

The final comparison table is in `metrics/final_model_comparison.csv`. The deep dive for `{best}` is in `best_model/text_pca64_lexical_minibatch_k32/`.

Key facts:

- Largest cluster share: {common['largest']:.2f}%.
- Normalized cluster entropy: {common['entropy']:.4f}.
- Cluster size Gini: {common['gini']:.4f}.
- Metadata dominance warnings: {common['warnings']} clusters.
""", args.overwrite)
    write_text(docs / "report_ready_conclusion.md", f"""# Kết luận sẵn sàng đưa vào báo cáo

Kết quả thực nghiệm cho thấy biểu diễn text-only embedding là lựa chọn ổn định và thận trọng nhất để làm baseline chính. Mô hình `text-only minibatch_k16` giữ coverage 100%, metric ổn định và dễ so sánh với các thuật toán còn lại.

Nhánh multi-feature không nên bị diễn giải là luôn kém hơn. Ablation cho thấy phần mở rộng nhẹ bằng đặc trưng lexical có thể cải thiện MiniBatch ở cấu hình k=32. Cụ thể, `{best}` là mô hình thực nghiệm full-coverage tốt nhất trong nhánh bounded ablation.

Ngược lại, fusion metadata nặng với publisher/stock hoặc all-aux weight lớn có thể làm giảm chất lượng GMM/HDBSCAN hoặc làm cấu trúc cụm bị collapse. Vì vậy, metadata nên được dùng cẩn trọng như tín hiệu bổ trợ và công cụ giải thích hậu nghiệm.

CLARA true đã được kiểm thử bằng `sklearn_extra.cluster.CLARA`, không còn là fallback. Tuy nhiên, CLARA true vẫn yếu hơn MiniBatch và chỉ nên xem là diagnostic baseline.

HDBSCAN phù hợp hơn cho phát hiện cụm dày đặc hoặc cụm kiểu sự kiện, không phải mô hình full-coverage chính.

Khuyến nghị cuối cùng:

- Dùng `text-only minibatch_k16` làm conservative baseline.
- Dùng `{best}` làm best experimental full-coverage model nếu phần kiểm tra diễn giải cụm đạt yêu cầu.
- Dùng `text-only hdbscan_minsize50` cho dense/event detection.
""", args.overwrite)
    deep = outdir / "best_model" / "text_pca64_lexical_minibatch_k32" / "best_model_metric_deep_dive.md"
    write_text(deep, f"""# Best Model Metric Deep Dive

Model: `{best}`.

## Cluster Size and Balance

- Number of clusters: {balance['n_clusters']}.
- Largest cluster: {balance['largest_cluster_pct']:.2f}%.
- Top 3 clusters: {balance['top_3_cluster_pct']:.2f}%.
- Top 5 clusters: {balance['top_5_cluster_pct']:.2f}%.
- Normalized entropy: {balance['normalized_entropy']:.4f}.
- Gini: {balance['gini']:.4f}.
- Clusters with size < 100: {balance['n_clusters_size_lt_100']}.
- Clusters with size < 500: {balance['n_clusters_size_lt_500']}.

## Silhouette

- Global sampled silhouette: {global_sil:.6f}.
- Worst clusters by mean silhouette: {', '.join(str(int(x)) for x in worst['cluster_id'].tolist())}.
- Best clusters by mean silhouette: {', '.join(str(int(x)) for x in best_sil['cluster_id'].tolist())}.

## Cohesion and Separation

The closest centroid pairs and DBI-like pairs are saved in `cluster_cohesion_separation.csv` and `worst_dbi_like_cluster_pairs.csv`.

## Metadata Dominance

Clusters with publisher/stock dominance warnings: {len(warnings)}.

## Stability

Stability rerun used MiniBatch k32 seeds 7, 13, 21, 42, 100 on full 100k rows. Mean ARI vs seed 42: {stability_df['ARI_vs_seed42'].mean():.4f}; mean NMI vs seed 42: {stability_df['NMI_vs_seed42'].mean():.4f}.
""", args.overwrite)


def write_text(path, text, overwrite):
    check_write_path(path, overwrite).write_text(text, encoding="utf-8")


def main(argv=None):
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    validate_outdir(outdir)
    docs = outdir / "docs"
    metrics = outdir / "metrics"
    best_base = outdir / "best_model" / "text_pca64_lexical_minibatch_k32"
    docs.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    best_base.mkdir(parents=True, exist_ok=True)

    bounded = read_one_csv(Path(args.bounded_root) / "metrics" / "bounded_ablation_summary.csv")
    clara = read_one_csv(Path(args.clara_true_root) / "metrics" / "bounded_ablation_summary.csv")
    final_df = write_final_comparison(args, outdir, bounded, clara)

    feature_columns = json.loads((Path(args.multifeature_artifacts) / "feature_columns.json").read_text(encoding="utf-8"))
    X = build_best_matrix(Path(args.multifeature_artifacts), feature_columns)
    labels_df = load_best_labels(Path(args.bounded_root), args.best_variant, args.best_algorithm)
    labels = labels_df["label"].to_numpy(dtype=np.int64)
    meta = pd.read_csv(Path(args.multifeature_artifacts) / "multifeature_meta.csv")

    size_profile, balance = write_cluster_size_and_balance(labels, best_base, args.overwrite)
    metric_rows = sample_rows(len(labels), args.sample_size_metrics, args.seed)
    sil_prof, global_sil = silhouette_profile(X, labels, metric_rows, best_base, args.overwrite)
    cohesion, pair_df = centroid_diagnostics(X, labels, metric_rows, best_base, args.overwrite)
    meta_profile = metadata_profile(meta, labels, best_base, args.overwrite)
    low_interp = write_terms_examples(meta, labels, best_base, args.overwrite)
    stability_df = stability(args, X, labels, outdir, args.overwrite)
    viz_rows = sample_rows(len(labels), 20000, args.seed)
    plot_outputs(X, labels, viz_rows, sil_prof, size_profile, best_base, args.overwrite)
    write_docs(args, outdir, final_df, bounded, clara, size_profile, balance, sil_prof, cohesion, meta_profile, stability_df, global_sil)
    print("Wrote final analysis to", outdir)
    print("Best model global sampled silhouette:", f"{global_sil:.6f}")
    print("Metadata dominance warnings:", int(((meta_profile["publisher_dominance_warning"]) | (meta_profile["stock_dominance_warning"])).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
