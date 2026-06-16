#!/usr/bin/env python3
"""Agglomerative clustering using sparse connectivity from k-NN (avoids full distance matrix).

Pipeline:
  - Optionally compute IncrementalPCA to reduce dims (memmap)
  - Build sparse connectivity from knn NPZ
  - Run AgglomerativeClustering(connectivity=connectivity)
"""
from __future__ import annotations
import argparse
import numpy as np
import os
from pathlib import Path

from scipy.sparse import coo_matrix

try:
    from scripts.utils.io import check_output_path, ensure_dir, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, ensure_dir, legacy_default_output


def incremental_pca_memmap(emb_path: str, out_path: str, n_components: int = 64, chunk: int = 20000):
    from sklearn.decomposition import IncrementalPCA

    X = np.load(emb_path, mmap_mode="r")
    n, d = X.shape
    if os.path.exists(out_path):
        print("PCA memmap exists ->", out_path)
        return out_path

    ipca = IncrementalPCA(n_components=n_components)
    for i in range(0, n, chunk):
        xb = np.array(X[i : i + chunk], dtype=np.float32)
        ipca.partial_fit(xb)

    # create a valid .npy memmap so it can be loaded later with np.load(..., mmap_mode="r")
    from numpy.lib.format import open_memmap

    reduced = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, n_components))
    for i in range(0, n, chunk):
        xb = np.array(X[i : i + chunk], dtype=np.float32)
        tr = ipca.transform(xb)
        reduced[i : i + tr.shape[0]] = tr.astype(np.float32)
    del reduced
    return out_path


def build_connectivity_from_knn(knn_npz: str):
    arr = np.load(knn_npz)
    indices = arr["indices"]
    n, k = indices.shape
    rows_all = np.repeat(np.arange(n, dtype=np.int32), k)
    cols_all = indices.ravel()
    mask = cols_all >= 0
    rows = rows_all[mask]
    cols = cols_all[mask]
    # make symmetric
    rows2 = np.concatenate([rows, cols]).astype(np.int32)
    cols2 = np.concatenate([cols, rows]).astype(np.int32)
    data = np.ones(rows2.shape[0], dtype=np.uint8)
    return coo_matrix((data, (rows2, cols2)), shape=(n, n)).tocsr()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--knn", required=True)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--out", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.out is None:
        file_name = f"cluster_labels_agg_ward_k{args.n_clusters}.csv"
        if args.outdir:
            args.out = f"{args.outdir}/{file_name}"
        else:
            args.out = legacy_default_output(f"report/results/{file_name}")
    out_path = check_output_path(args.out, overwrite=args.overwrite)
    out_dir = ensure_dir(out_path.parent)
    pca_path = out_dir / f"pca_n{args.n_components}.npy"
    if pca_path.exists() and not args.overwrite:
        print("PCA memmap exists ->", pca_path)
    pca_path = incremental_pca_memmap(args.emb, pca_path, n_components=args.n_components, chunk=args.chunk)

    print("Building connectivity from:", args.knn)
    connectivity = build_connectivity_from_knn(args.knn)

    print("Loading reduced embeddings (memmap):", pca_path)
    Xred = np.load(pca_path, allow_pickle=True)

    from sklearn.cluster import AgglomerativeClustering

    print("Running AgglomerativeClustering (ward) with connectivity")
    model = AgglomerativeClustering(n_clusters=args.n_clusters, linkage="ward", connectivity=connectivity)
    labels = model.fit_predict(Xred)

    import csv

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, lab in enumerate(labels):
            w.writerow([i, int(lab)])

    print("Wrote labels ->", out_path)


if __name__ == "__main__":
    main()
