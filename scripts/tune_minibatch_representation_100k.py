#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler, normalize

try:
    from scripts.run_multifeature_100k import dataframe_to_markdown
except ModuleNotFoundError:
    from run_multifeature_100k import dataframe_to_markdown


N_ROWS = 100_000
EMB_DIM = 768
OUT_ROOT = Path("report") / "runs" / "100k" / "_representation_tuning"
BASELINE_SUMMARY = Path("report") / "runs" / "100k" / "_k_metric_exploration" / "metrics" / "k_sweep_summary.csv"
BASELINE_STABILITY = Path("report") / "runs" / "100k" / "_k_metric_exploration" / "metrics" / "k_sweep_stability.csv"

LEXICAL_COLUMNS = [
    "char_length",
    "token_count",
    "avg_token_length",
    "digit_count",
    "digit_ratio",
    "uppercase_count",
    "uppercase_ratio",
    "punctuation_count",
    "question_mark_count",
    "exclamation_mark_count",
    "comma_period_colon_count",
    "ticker_like_token_count",
    "finance_keyword_count",
]

FINANCE_WORDS = {
    "acquire",
    "acquires",
    "acquisition",
    "analyst",
    "bank",
    "bond",
    "buyback",
    "capital",
    "cash",
    "ceo",
    "company",
    "credit",
    "deal",
    "debt",
    "dividend",
    "earnings",
    "eps",
    "equity",
    "forecast",
    "guidance",
    "ipo",
    "market",
    "merger",
    "nasdaq",
    "nyse",
    "outlook",
    "price",
    "profit",
    "rating",
    "revenue",
    "sales",
    "share",
    "shares",
    "stock",
    "target",
    "trading",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune MiniBatchKMeans representations around the best 100k experimental model.")
    p.add_argument("--emb", default="data/embeddings_100k.npy")
    p.add_argument("--meta", default="report/runs/100k/_multifeature/artifacts/multifeature_meta.csv")
    p.add_argument("--feature-columns", default="report/runs/100k/_multifeature/artifacts/feature_columns.json")
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--pca-dims", default="128,256")
    p.add_argument("--lexical-weights", default="0.03,0.05,0.10")
    p.add_argument("--k-values", default="32,40,48,64")
    p.add_argument("--metric-sample-size", type=int, default=10000)
    p.add_argument("--stability-seeds", default="7,13,21,42,100")
    p.add_argument("--top-n-stability", type=int, default=5)
    p.add_argument("--stability-sample-size", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-init", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--reuse-existing-runs", action="store_true", help="Load existing per-run metrics/labels while refreshing summaries/docs.")
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


def parse_floats(raw: str, name: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} cannot be empty")
    return vals


def sample_indices(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=min(size, n), replace=False))


def entropy(values: np.ndarray) -> float:
    p = values[values > 0] / values.sum()
    return float(-(p * np.log(p)).sum())


def gini(values: np.ndarray) -> float:
    x = np.sort(values.astype(np.float64))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9&.\-]*|\d+(?:\.\d+)?", text)


def build_lexical_from_meta(meta: pd.DataFrame) -> np.ndarray:
    headlines = meta["headline_clean"].fillna("").astype(str)
    lexical = np.zeros((len(headlines), len(LEXICAL_COLUMNS)), dtype=np.float32)
    punctuation = set(string.punctuation)
    for i, text in enumerate(headlines):
        tokens = tokenize(text)
        token_count = len(tokens)
        char_length = len(text)
        token_lengths = [len(tok) for tok in tokens]
        digit_count = sum(ch.isdigit() for ch in text)
        uppercase_count = sum(ch.isupper() for ch in text)
        punct_count = sum(ch in punctuation for ch in text)
        comma_period_colon_count = text.count(",") + text.count(".") + text.count(":")
        ticker_like_count = sum(1 for tok in tokens if 1 <= len(tok) <= 5 and tok.isupper() and any(ch.isalpha() for ch in tok))
        lower_tokens = [tok.lower().strip(".:-") for tok in tokens]
        finance_count = sum(1 for tok in lower_tokens if tok in FINANCE_WORDS)
        lexical[i] = [
            char_length,
            token_count,
            float(np.mean(token_lengths)) if token_lengths else 0.0,
            digit_count,
            digit_count / max(char_length, 1),
            uppercase_count,
            uppercase_count / max(char_length, 1),
            punct_count,
            text.count("?"),
            text.count("!"),
            comma_period_colon_count,
            ticker_like_count,
            finance_count,
        ]
    return StandardScaler().fit_transform(lexical).astype(np.float32)


def load_or_build_lexical(args, feature_columns: dict, meta: pd.DataFrame) -> tuple[np.ndarray, str]:
    aux_path = Path(args.feature_columns).with_name("X_aux_features.npy")
    lexical_width = len(feature_columns.get("lexical", LEXICAL_COLUMNS))
    if aux_path.exists():
        aux = np.load(aux_path, mmap_mode="r")
        if aux.shape[0] == N_ROWS and aux.shape[1] >= lexical_width:
            return np.asarray(aux[:, :lexical_width], dtype=np.float32), f"reused {aux_path}"
    return build_lexical_from_meta(meta), "reconstructed from multifeature_meta.headline_clean"


def build_text_pca(emb: np.ndarray, dim: int, chunk: int = 20_000) -> np.ndarray:
    pca = IncrementalPCA(n_components=dim)
    for start in range(0, N_ROWS, chunk):
        pca.partial_fit(np.asarray(emb[start : start + chunk], dtype=np.float32))
    reduced = np.empty((N_ROWS, dim), dtype=np.float32)
    for start in range(0, N_ROWS, chunk):
        batch = np.asarray(emb[start : start + chunk], dtype=np.float32)
        transformed = pca.transform(batch).astype(np.float32)
        reduced[start : start + transformed.shape[0]] = transformed
    return StandardScaler().fit_transform(reduced).astype(np.float32)


def load_or_build_pca(emb: np.ndarray, dim: int, args) -> np.ndarray:
    path = Path(args.out_root) / "artifacts" / f"X_text_pca{dim}.npy"
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape == (N_ROWS, dim):
            print(f"Reusing cached PCA{dim}: {path}")
            return cached
    print(f"Building PCA{dim}")
    reduced = build_text_pca(emb, dim)
    np.save(check_write_path(path, args.overwrite), reduced)
    return np.load(path, mmap_mode="r")


def maybe_l2(X: np.ndarray, enabled: bool) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if not enabled:
        return arr
    return normalize(arr, norm="l2", axis=1).astype(np.float32)


def feature_spaces_from_args(pca_dims: list[int], weights: list[float]) -> list[str]:
    spaces = []
    for dim in pca_dims:
        spaces.append(f"text_pca{dim}_only")
    for dim in pca_dims:
        for w in weights:
            spaces.append(f"text_pca{dim}_lexical_w{int(round(w * 100)):03d}")
    spaces.extend(["text_768_l2", "text_pca128_lexical_w005_l2", "text_pca256_lexical_w005_l2"])
    return spaces


def load_feature_space(space: str, emb, pcas: dict[int, np.ndarray], lexical: np.ndarray) -> np.ndarray:
    if space == "text_768_l2":
        return maybe_l2(emb, True)
    m = re.fullmatch(r"text_pca(\d+)_(only|lexical_w(\d{3})(?:_l2)?)", space)
    if not m:
        raise ValueError(f"Unknown feature space: {space}")
    dim = int(m.group(1))
    mode = m.group(2)
    l2 = mode.endswith("_l2")
    if mode == "only":
        return np.asarray(pcas[dim], dtype=np.float32)
    weight = int(m.group(3)) / 100.0
    X = np.hstack([np.asarray(pcas[dim], dtype=np.float32), weight * lexical]).astype(np.float32)
    return maybe_l2(X, l2)


def token_counts(texts: pd.Series) -> Counter:
    c = Counter()
    for text in texts.fillna("").astype(str):
        toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower()) if t not in ENGLISH_STOP_WORDS]
        c.update(toks)
    return c


def posthoc_warnings(meta: pd.DataFrame, labels: np.ndarray) -> tuple[int, int, int]:
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
        "labels": root / "labels" / f"cluster_labels_{space}_minibatch_k{k}.csv",
        "metrics": root / "metrics" / "results_summary.csv",
    }


def fit_or_load(space: str, k: int, X, args, meta: pd.DataFrame) -> tuple[np.ndarray, dict]:
    paths = model_paths(Path(args.out_root), space, k)
    if paths["labels"].exists() and paths["metrics"].exists() and (not args.overwrite or args.reuse_existing_runs):
        labels = pd.read_csv(paths["labels"])["label"].to_numpy()
        metric = pd.read_csv(paths["metrics"]).iloc[0].to_dict()
        return labels, metric

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=args.seed,
        batch_size=args.batch_size,
        n_init=args.n_init,
        max_iter=args.max_iter,
        init="k-means++",
    )
    labels = model.fit_predict(np.asarray(X, dtype=np.float32))
    rows = sample_indices(len(labels), args.metric_sample_size, args.seed)
    Xs = np.asarray(X[rows], dtype=np.float32)
    ys = labels[rows]
    notes = []
    try:
        sil_values = silhouette_samples(Xs, ys, metric="cosine")
        sil_cos = float(np.mean(sil_values))
        negative_pct = float((sil_values < 0).mean() * 100.0)
        sil_quantiles = np.percentile(sil_values, [10, 25, 50, 75, 90])
    except Exception as exc:
        sil_cos = np.nan
        negative_pct = np.nan
        sil_quantiles = [np.nan] * 5
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
    low_interp, pub_warn, stock_warn = posthoc_warnings(meta, labels)
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
        "negative_silhouette_pct": negative_pct,
        "silhouette_p10": float(sil_quantiles[0]),
        "silhouette_p25": float(sil_quantiles[1]),
        "silhouette_median": float(sil_quantiles[2]),
        "silhouette_p75": float(sil_quantiles[3]),
        "silhouette_p90": float(sil_quantiles[4]),
        **balance_metrics(labels),
        "low_interpretability_clusters": low_interp,
        "publisher_warning_count": pub_warn,
        "stock_warning_count": stock_warn,
        "notes": ";".join(notes),
    }
    pd.DataFrame({"row_index": np.arange(len(labels)), "label": labels, "algorithm": f"minibatch_k{k}", "feature_space": space}).to_csv(check_write_path(paths["labels"], args.overwrite), index=False)
    pd.DataFrame([metric]).to_csv(check_write_path(paths["metrics"], args.overwrite), index=False)
    return labels, metric


def score_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    spec = [
        ("silhouette_cosine", True, 0.28),
        ("dbi", False, 0.18),
        ("negative_silhouette_pct", False, 0.14),
        ("normalized_entropy", True, 0.14),
        ("largest_cluster_pct", False, 0.08),
        ("n_clusters_lt_500", False, 0.08),
        ("publisher_warning_count", False, 0.06),
        ("stock_warning_count", False, 0.04),
    ]
    scored["base_score"] = 0.0
    for col, higher_is_better, weight in spec:
        vals = pd.to_numeric(scored[col], errors="coerce").astype(float)
        rng = vals.max() - vals.min()
        if np.isfinite(rng) and rng > 0:
            norm = (vals - vals.min()) / rng if higher_is_better else (vals.max() - vals) / rng
        else:
            norm = pd.Series(0.5, index=scored.index)
        scored[f"{col}_score"] = norm
        scored["base_score"] += weight * norm.fillna(0.0)
    return scored.sort_values(["base_score", "silhouette_cosine"], ascending=[False, False])


def stability_for(space: str, k: int, X, seeds: list[int], sample_rows: np.ndarray, args) -> list[dict]:
    out_dir = Path(args.out_root) / "stability" / space / f"minibatch_k{k}"
    labels_by_seed = {}
    Xs = np.asarray(X[sample_rows], dtype=np.float32)
    for seed in seeds:
        path = out_dir / f"labels_seed{seed}.csv"
        if path.exists() and (not args.overwrite or args.reuse_existing_runs):
            labels = pd.read_csv(path)["label"].to_numpy()
        else:
            labels = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed,
                batch_size=args.batch_size,
                n_init=args.n_init,
                max_iter=args.max_iter,
                init="k-means++",
            ).fit_predict(Xs)
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
                "notes": "top candidate deterministic stability sample",
            }
        )
    return rows


def add_stability_scores(scored: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    if stability.empty:
        out["mean_ARI"] = np.nan
        out["mean_NMI"] = np.nan
        out["final_score"] = out["base_score"]
        return out
    st = stability.groupby(["feature_space", "k"])[["ARI", "NMI"]].mean().reset_index().rename(columns={"ARI": "mean_ARI", "NMI": "mean_NMI"})
    out = out.merge(st, on=["feature_space", "k"], how="left")
    for col in ["mean_ARI", "mean_NMI"]:
        vals = out[col].astype(float)
        rng = vals.max() - vals.min()
        out[col + "_score"] = (vals - vals.min()) / rng if np.isfinite(rng) and rng > 0 else np.where(vals.notna(), 0.5, np.nan)
    out["final_score"] = out["base_score"] + 0.10 * out["mean_ARI_score"].fillna(0.0) + 0.10 * out["mean_NMI_score"].fillna(0.0)
    return out.sort_values(["final_score", "base_score"], ascending=[False, False])


def plot_outputs(summary: pd.DataFrame, stability: pd.DataFrame, args) -> None:
    import matplotlib.pyplot as plt

    out = Path(args.out_root) / "charts"
    for y, name, ylabel in [
        ("silhouette_cosine", "silhouette_vs_k.png", "Silhouette cosine"),
        ("dbi", "dbi_vs_k.png", "DBI"),
        ("negative_silhouette_pct", "negative_silhouette_vs_k.png", "Negative silhouette %"),
        ("largest_cluster_pct", "cluster_balance_vs_k.png", "Largest cluster %"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 7))
        for space, sub in summary.sort_values("k").groupby("feature_space"):
            ax.plot(sub["k"], sub[y], marker="o", linewidth=1.4, label=space)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs k")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(check_write_path(out / name, args.overwrite), dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    if not stability.empty:
        st = stability.groupby(["feature_space", "k"])[["ARI", "NMI"]].mean().reset_index()
        for space, sub in st.sort_values("k").groupby("feature_space"):
            ax.plot(sub["k"], sub["ARI"], marker="o", label=f"{space} ARI")
            ax.plot(sub["k"], sub["NMI"], marker="x", linestyle="--", label=f"{space} NMI")
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title("Top-candidate stability vs k")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(check_write_path(out / "stability_vs_k.png", args.overwrite), dpi=180)
    plt.close(fig)


def get_baseline_row() -> dict | None:
    if not BASELINE_SUMMARY.exists():
        return None
    base = pd.read_csv(BASELINE_SUMMARY)
    row = base[(base["feature_space"] == "text_pca64_lexical") & (base["k"] == 40)]
    if row.empty:
        return None
    result = row.iloc[0].to_dict()
    if BASELINE_STABILITY.exists():
        st = pd.read_csv(BASELINE_STABILITY)
        st = st[(st["feature_space"] == "text_pca64_lexical") & (st["k"] == 40)]
        if not st.empty:
            result["mean_ARI"] = float(st["ARI"].mean())
            result["mean_NMI"] = float(st["NMI"].mean())
    return result


def best_weight(scored: pd.DataFrame) -> str:
    sub = scored[scored["feature_space"].str.contains("lexical_w") & ~scored["feature_space"].str.endswith("_l2")]
    if sub.empty:
        return "n/a"
    row = sub.sort_values(["base_score", "silhouette_cosine"], ascending=[False, False]).iloc[0]
    m = re.search(r"w(\d{3})", str(row["feature_space"]))
    return f"{int(m.group(1)) / 100:.2f}" if m else "n/a"


def write_docs(summary: pd.DataFrame, scored: pd.DataFrame, stability: pd.DataFrame, args, lexical_source: str) -> None:
    docs = Path(args.out_root) / "docs"
    baseline = get_baseline_row()
    best = scored.iloc[0]
    top = scored.head(10)[
        [
            "feature_space",
            "k",
            "base_score",
            "final_score",
            "silhouette_cosine",
            "dbi",
            "negative_silhouette_pct",
            "min_cluster_size",
            "largest_cluster_pct",
            "publisher_warning_count",
            "stock_warning_count",
            "mean_ARI",
            "mean_NMI",
        ]
    ]
    pca128_best = scored[scored["feature_space"].str.contains("pca128")].sort_values(["base_score", "silhouette_cosine"], ascending=[False, False]).iloc[0]
    pca256_best = scored[scored["feature_space"].str.contains("pca256")].sort_values(["base_score", "silhouette_cosine"], ascending=[False, False]).iloc[0]
    l2_mask = scored["feature_space"].str.endswith("_l2") | (scored["feature_space"] == "text_768_l2")
    l2_best = scored[l2_mask].sort_values(["base_score", "silhouette_cosine"], ascending=[False, False]).iloc[0] if l2_mask.any() else None
    non_l2_best = scored[~l2_mask].sort_values(["base_score", "silhouette_cosine"], ascending=[False, False]).iloc[0] if (~l2_mask).any() else None
    l2_helps = l2_best is not None and non_l2_best is not None and float(l2_best["base_score"]) > float(non_l2_best["base_score"])
    w_best = best_weight(scored)
    beats_baseline = False
    baseline_line = "- Baseline `text_pca64_lexical + minibatch_k40` was not found for direct comparison."
    if baseline is not None:
        beats_baseline = (
            float(best["silhouette_cosine"]) > float(baseline["silhouette_cosine"])
            and float(best["dbi"]) < float(baseline["dbi"])
            and int(best["n_clusters_lt_500"]) == 0
            and int(best["publisher_warning_count"]) <= int(str(baseline.get("notes", "")).split("publisher_warnings=")[-1].split(";")[0] if "publisher_warnings=" in str(baseline.get("notes", "")) else 999)
        )
        baseline_line = (
            f"- Baseline `text_pca64_lexical + minibatch_k40`: silhouette={float(baseline['silhouette_cosine']):.6f}, "
            f"DBI={float(baseline['dbi']):.6f}, largest={float(baseline['largest_cluster_pct']):.2f}%, "
            f"min_size={int(baseline['min_cluster_size'])}."
        )

    analysis = [
        "# Representation Tuning Analysis",
        "",
        "## Run Scope",
        "",
        f"- Feature source for lexical columns: {lexical_source}.",
        f"- MiniBatchKMeans params: n_init={args.n_init}, batch_size={args.batch_size}, max_iter={args.max_iter}, init=k-means++.",
        "- Publisher and stock were excluded from fitting and used only for post-hoc warnings.",
        baseline_line,
        "",
        "## Top Candidates",
        "",
        dataframe_to_markdown(top),
        "",
        "## Answers",
        "",
        f"1. PCA128/PCA256: best PCA128 candidate is `{pca128_best['feature_space']} + k{int(pca128_best['k'])}`; best PCA256 candidate is `{pca256_best['feature_space']} + k{int(pca256_best['k'])}`. The final ranking decides by balanced score, not PCA dimension alone.",
        f"2. Best lexical weight among non-L2 lexical runs: `{w_best}`.",
        f"3. L2-normalize {'helps the balanced ranking in this sweep' if l2_helps else 'does not clearly help'} under the seed-42 pre-stability score.",
        f"4. Best k by balanced final score is `k{int(best['k'])}` for `{best['feature_space']}`.",
        f"5. Clear improvement over `text_pca64_lexical + minibatch_k40`: {'yes' if beats_baseline else 'no'} under the conservative comparison rule.",
        f"6. Recommended final experimental model: `{best['feature_space']} + minibatch_k{int(best['k'])}` if accepting this tuning score; otherwise keep PCA64 lexical k40 as the simpler prior model.",
        "7. If the final recommendation does not change, the reason is that higher-dimensional or L2 variants did not produce a clean improvement across silhouette, DBI, balance, and metadata warnings together.",
        "",
        "## Stability",
        "",
        "- Stability was run only for the top candidates selected after the seed-42 full 100k sweep.",
    ]
    rec_change = "change" if beats_baseline else "do not change"
    recommendation = [
        "# Representation Tuning Recommendation",
        "",
        f"- Best tuned candidate: `{best['feature_space']} + minibatch_k{int(best['k'])}`.",
        f"- Best lexical weight: `{w_best}`.",
        f"- L2-normalize: {'useful for this tuned sweep, but not enough by itself to replace the prior experimental model' if l2_helps else 'not recommended as default'} based on this tuning run.",
        f"- Final experimental model decision: {rec_change} from `text_pca64_lexical + minibatch_k40`.",
        "- Use the recommendation only with the paired metric table; do not select by silhouette alone.",
    ]
    check_write_path(docs / "representation_tuning_analysis.md", args.overwrite).write_text("\n".join(analysis) + "\n", encoding="utf-8")
    check_write_path(docs / "representation_tuning_recommendation.md", args.overwrite).write_text("\n".join(recommendation) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out_root)
    validate_out_root(out_root)
    pca_dims = sorted(set(parse_ints(args.pca_dims, "--pca-dims")))
    weights = parse_floats(args.lexical_weights, "--lexical-weights")
    k_values = parse_ints(args.k_values, "--k-values")
    seeds = parse_ints(args.stability_seeds, "--stability-seeds")
    if args.seed not in seeds:
        seeds.append(args.seed)
    emb = np.load(args.emb, mmap_mode="r")
    if emb.shape != (N_ROWS, EMB_DIM):
        raise ValueError(f"{args.emb} must have shape ({N_ROWS}, {EMB_DIM}), got {emb.shape}")
    meta = pd.read_csv(args.meta)
    if len(meta) != N_ROWS:
        raise ValueError(f"{args.meta} must have {N_ROWS} rows, got {len(meta)}")
    feature_columns = json.loads(Path(args.feature_columns).read_text(encoding="utf-8"))
    lexical, lexical_source = load_or_build_lexical(args, feature_columns, meta)
    pcas = {dim: load_or_build_pca(emb, dim, args) for dim in pca_dims}
    spaces = feature_spaces_from_args(pca_dims, weights)
    summary_rows = []
    feature_cache = {}
    for space in spaces:
        print(f"Loading feature space {space}")
        X = load_feature_space(space, emb, pcas, lexical)
        if X.shape[0] != N_ROWS:
            raise ValueError(f"{space} has invalid row count: {X.shape}")
        feature_cache[space] = X
        for k in k_values:
            print(f"Running {space} k={k}")
            _labels, metric = fit_or_load(space, k, X, args, meta)
            summary_rows.append(metric)
    summary = pd.DataFrame(summary_rows)
    scored_initial = score_candidates(summary)
    top_candidates = scored_initial.head(args.top_n_stability)[["feature_space", "k"]]
    stability_rows = []
    stability_idx = sample_indices(N_ROWS, args.stability_sample_size, args.seed)
    for row in top_candidates.itertuples(index=False):
        space = str(row.feature_space)
        k = int(row.k)
        print(f"Running stability for {space} k={k}")
        stability_rows.extend(stability_for(space, k, feature_cache[space], seeds, stability_idx, args))
    stability = pd.DataFrame(stability_rows)
    scored = add_stability_scores(scored_initial, stability)
    metrics_dir = out_root / "metrics"
    summary_out = scored.merge(
        summary.drop(columns=[c for c in ["base_score", "final_score", "mean_ARI", "mean_NMI"] if c in summary.columns]),
        how="right",
        on=list(summary.columns),
    ) if False else scored
    summary_out.to_csv(check_write_path(metrics_dir / "representation_tuning_summary.csv", args.overwrite), index=False)
    check_write_path(metrics_dir / "representation_tuning_summary.md", args.overwrite).write_text(dataframe_to_markdown(summary_out), encoding="utf-8")
    stability.to_csv(check_write_path(metrics_dir / "representation_tuning_stability.csv", args.overwrite), index=False)
    plot_outputs(summary, stability, args)
    write_docs(summary, scored, stability, args, lexical_source)
    print(f"Wrote representation tuning to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
