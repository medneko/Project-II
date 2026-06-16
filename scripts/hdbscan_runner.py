#!/usr/bin/env python3
"""Run HDBSCAN on memmapped embeddings using chunked PCA reduction.

Usage: python scripts/hdbscan_runner.py --emb data/embeddings_clean.npy --out report/results/cluster_labels_hdbscan_kmin50.csv
"""
from __future__ import annotations
import argparse
import numpy as np

try:
    from scripts.utils.io import check_output_path, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, legacy_default_output


def incremental_pca(emb_path, n_components=64, chunk=20000):
    from sklearn.decomposition import IncrementalPCA

    X = np.load(emb_path, mmap_mode="r")
    n, d = X.shape
    ipca = IncrementalPCA(n_components=n_components)
    for i in range(0, n, chunk):
        xb = np.array(X[i : i + chunk], dtype=np.float32)
        ipca.partial_fit(xb)

    reduced = np.empty((n, n_components), dtype=np.float32)
    for i in range(0, n, chunk):
        xb = np.array(X[i : i + chunk], dtype=np.float32)
        reduced[i : i + xb.shape[0]] = ipca.transform(xb).astype(np.float32)
    return reduced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--pca", type=int, default=64)
    p.add_argument("--use-umap", action="store_true")
    p.add_argument("--umap-dim", type=int, default=5)
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--outdir", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.out is None:
        file_name = f"cluster_labels_hdbscan_minsize{args.min_cluster_size}.csv"
        if args.outdir:
            args.out = f"{args.outdir}/{file_name}"
        else:
            args.out = legacy_default_output(f"report/results/{file_name}")
    out_path = check_output_path(args.out, overwrite=args.overwrite)

    print("Computing PCA...")
    Xred = incremental_pca(args.emb, n_components=args.pca, chunk=args.chunk)

    try:
        import hdbscan
    except Exception as exc:
        raise RuntimeError("Please install hdbscan in your environment (conda-forge recommended).") from exc

    if args.use_umap:
        import umap

        print("Running UMAP to", args.umap_dim)
        reducer = umap.UMAP(n_components=args.umap_dim)
        Xcluster = reducer.fit_transform(Xred)
    else:
        Xcluster = Xred

    print("Running HDBSCAN(min_cluster_size=%d)" % args.min_cluster_size)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
    labels = clusterer.fit_predict(Xcluster)

    import csv

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, lab in enumerate(labels):
            w.writerow([i, int(lab)])

    print("Wrote labels ->", out_path)


if __name__ == "__main__":
    main()
