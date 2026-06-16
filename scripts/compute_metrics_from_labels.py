#!/usr/bin/env python3
"""Compute clustering metrics safely from label CSV files.

Label CSVs should contain ``row_index,label``. Legacy ``id,label`` files are
accepted as full-label files only when they cover the full embedding matrix.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.utils.io import check_output_path, ensure_dir, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, ensure_dir, legacy_default_output


def load_label_file(path: Path, n_embedding: int):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"{path} is missing required column 'label'")

    labels = df["label"].to_numpy()
    if "row_index" in df.columns:
        row_index = df["row_index"].to_numpy()
    elif "id" in df.columns and len(df) == n_embedding:
        row_index = df["id"].to_numpy()
    elif len(df) == n_embedding:
        row_index = np.arange(n_embedding)
    else:
        raise ValueError(
            f"{path} has {len(df)} labels but embedding has {n_embedding} rows; "
            "sample labels must include a row_index column"
        )

    row_index = np.asarray(row_index)
    if row_index.ndim != 1 or len(row_index) != len(labels):
        raise ValueError(f"{path} has invalid row_index shape")
    if not np.issubdtype(row_index.dtype, np.integer):
        if not np.all(np.equal(row_index, row_index.astype(np.int64))):
            raise ValueError(f"{path} row_index must contain integer values")
        row_index = row_index.astype(np.int64)
    else:
        row_index = row_index.astype(np.int64)
    if len(row_index) and (row_index.min() < 0 or row_index.max() >= n_embedding):
        raise ValueError(f"{path} row_index values are outside embedding bounds")
    if len(np.unique(row_index)) != len(row_index):
        raise ValueError(f"{path} row_index contains duplicate values")
    if len(row_index) == n_embedding and not np.array_equal(np.sort(row_index), np.arange(n_embedding)):
        raise ValueError(f"{path} full labels must cover row_index 0..n-1 exactly")

    return row_index, labels


def sample_assigned(row_index, labels, assigned_mask, max_size, seed):
    assigned_rows = row_index[assigned_mask]
    assigned_labels = labels[assigned_mask]
    if max_size and len(assigned_rows) > max_size:
        rng = np.random.RandomState(seed)
        pick = np.sort(rng.choice(len(assigned_rows), size=max_size, replace=False))
        return assigned_rows[pick], assigned_labels[pick], int(max_size)
    return assigned_rows, assigned_labels, int(len(assigned_rows))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--out", default=None, help="Output directory (legacy alias for --outdir)")
    p.add_argument("--outdir", default=None)
    p.add_argument("--labels-dir", default=None)
    p.add_argument("--sample", type=int, default=100000, help="Rows used for pca_sample.csv only")
    p.add_argument("--silhouette-sample", type=int, default=None, help="Legacy alias for --metric-sample-size")
    p.add_argument("--metric-sample-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(args.outdir or args.out) if (args.outdir or args.out) else legacy_default_output("report/scratch/metrics")
    labels_dir = Path(args.labels_dir) if args.labels_dir else out_dir
    ensure_dir(out_dir)
    pca_out = check_output_path(out_dir / "pca_sample.csv", overwrite=args.overwrite)
    clustering_out = check_output_path(out_dir / "clustering_results.csv", overwrite=args.overwrite)
    consensus_out = out_dir / "consensus_pairwise.csv"
    metric_sample_size = args.silhouette_sample if args.silhouette_sample is not None else args.metric_sample_size

    X = np.load(args.emb, mmap_mode="r")
    n_embedding = int(X.shape[0])

    try:
        from sklearn.decomposition import PCA

        pca_n = min(args.sample, n_embedding) if args.sample else n_embedding
        Xsmall = np.array(X[:pca_n], dtype=np.float32)
        vis = PCA(n_components=2).fit_transform(Xsmall)
        pd.DataFrame(vis, columns=["x", "y"]).to_csv(pca_out, index=False)
    except Exception as e:
        print("PCA failed:", e)

    files = sorted(glob.glob(str(labels_dir / "cluster_labels_*.csv")))
    if not files:
        print("No cluster label files found in", labels_dir)
        return

    from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score

    results = []
    label_sets = {}

    for f in files:
        path = Path(f)
        name = path.stem
        algorithm = name.replace("cluster_labels_", "")
        notes = []
        row_index, labels = load_label_file(path, n_embedding)
        n_points_labeled = int(len(labels))
        assigned_mask = labels != -1
        n_assigned = int(assigned_mask.sum())
        coverage = float(n_assigned / n_points_labeled) if n_points_labeled else float("nan")
        assigned_labels_all = labels[assigned_mask]
        unique = np.unique(assigned_labels_all) if n_assigned else np.array([])
        n_clusters = int(len(unique))

        sil = float("nan")
        dbi = float("nan")
        sil_sample_used = 0
        if n_clusters < 2:
            notes.append("metric_skipped_less_than_2_assigned_clusters")
        elif n_assigned < 2:
            notes.append("metric_skipped_less_than_2_assigned_points")
        else:
            metric_rows, metric_labels, sil_sample_used = sample_assigned(
                row_index,
                labels,
                assigned_mask,
                metric_sample_size,
                args.seed,
            )
            X_metric = np.asarray(X[metric_rows], dtype=np.float32)
            try:
                sil = float(
                    silhouette_score(
                        X_metric,
                        metric_labels,
                        metric="cosine",
                        sample_size=None,
                    )
                )
            except Exception as exc:
                notes.append(f"silhouette_failed:{type(exc).__name__}")
            try:
                dbi = float(davies_bouldin_score(X_metric, metric_labels))
            except Exception as exc:
                notes.append(f"dbi_failed:{type(exc).__name__}")

        results.append(
            {
                "algorithm": algorithm,
                "algo": algorithm,
                "label_file": path.name,
                "n_points_labeled": n_points_labeled,
                "n_embedding_points": n_embedding,
                "n_assigned": n_assigned,
                "coverage": coverage,
                "coverage_pct": coverage * 100.0 if np.isfinite(coverage) else coverage,
                "n_clusters": n_clusters,
                "silhouette": sil,
                "silhouette_metric": "cosine",
                "silhouette_sample_size": sil_sample_used,
                "dbi": dbi,
                "notes": ";".join(notes),
            }
        )
        label_sets[algorithm] = (row_index, labels)

    pd.DataFrame(results).to_csv(clustering_out, index=False)

    pairs = []
    keys = list(label_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            rows_a, labels_a = label_sets[a]
            rows_b, labels_b = label_sets[b]
            map_a = pd.Series(labels_a, index=rows_a)
            map_b = pd.Series(labels_b, index=rows_b)
            common = map_a.index.intersection(map_b.index)
            if len(common) == 0:
                continue
            l1 = map_a.loc[common].to_numpy()
            l2 = map_b.loc[common].to_numpy()
            mask = (l1 != -1) & (l2 != -1)
            if mask.sum() == 0:
                continue
            pairs.append(
                {
                    "algo_a": a,
                    "algo_b": b,
                    "ARI": adjusted_rand_score(l1[mask], l2[mask]),
                    "NMI": normalized_mutual_info_score(l1[mask], l2[mask]),
                    "overlap_count": int(mask.sum()),
                }
            )

    if pairs:
        consensus_out = check_output_path(consensus_out, overwrite=args.overwrite)
        pd.DataFrame(pairs).to_csv(consensus_out, index=False)
    print("Wrote clustering_results.csv and consensus_pairwise.csv to", out_dir)


if __name__ == "__main__":
    main()
