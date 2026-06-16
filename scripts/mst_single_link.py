#!/usr/bin/env python3
"""Single-linkage via MST built from a sparse k-NN graph.

Input: a k-NN `.npz` produced by `build_knn.py` containing `indices` and `dists`.
Output: cluster labels by cutting the MST at the (n_clusters-1) largest edges.
"""
from __future__ import annotations
import argparse
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

try:
    from scripts.utils.io import check_output_path, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, legacy_default_output


def load_knn(npz_path):
    arr = np.load(npz_path)
    indices = arr["indices"]
    dists = arr["dists"]
    return indices, dists


def build_symmetric_coo_from_knn(indices, dists):
    n, k = indices.shape
    rows_all = np.repeat(np.arange(n, dtype=np.int32), k)
    cols_all = indices.ravel()
    data_all = dists.ravel()
    mask = cols_all >= 0
    rows = rows_all[mask]
    cols = cols_all[mask]
    data = data_all[mask]

    # make symmetric by adding reverse edges (duplicates OK)
    rows2 = np.concatenate([rows, cols]).astype(np.int32)
    cols2 = np.concatenate([cols, rows]).astype(np.int32)
    data2 = np.concatenate([data, data]).astype(np.float32)

    return coo_matrix((data2, (rows2, cols2)), shape=(n, n))


def mst_cut_clusters(A_sparse_coo, n_clusters):
    n = A_sparse_coo.shape[0]
    A_csr = A_sparse_coo.tocsr()
    mst = minimum_spanning_tree(A_csr)
    mst = mst.tocoo()

    m = mst.data.size
    # undirected representation (duplicate edges)
    rows_u = np.concatenate([mst.row, mst.col]).astype(np.int32)
    cols_u = np.concatenate([mst.col, mst.row]).astype(np.int32)
    data_u = np.concatenate([mst.data, mst.data]).astype(np.float32)

    # unique undirected edges are those with row < col (pick one per undirected edge)
    unique_pos = np.where(rows_u < cols_u)[0]
    unique_data = data_u[unique_pos]

    # choose largest (n_clusters-1) edges to remove
    to_remove = np.argsort(unique_data)[-max(0, n_clusters - 1) :]
    # map back to positions in rows_u (and their symmetric counterparts)
    rem_pos = unique_pos[to_remove]
    rem_pos_pair = np.array([p + m if p < m else p - m for p in rem_pos], dtype=np.int64)
    remove_positions = np.concatenate([rem_pos, rem_pos_pair]) if rem_pos.size else np.array([], dtype=np.int64)

    keep_mask = np.ones(rows_u.shape[0], dtype=bool)
    if remove_positions.size:
        keep_mask[remove_positions] = False

    rows_keep = rows_u[keep_mask]
    cols_keep = cols_u[keep_mask]
    data_keep = data_u[keep_mask]

    A_cut = coo_matrix((data_keep, (rows_keep, cols_keep)), shape=(n, n)).tocsr()
    n_comp, labels = connected_components(A_cut, directed=False)
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--knn", required=True, help="NPZ with indices/dists from build_knn.py")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.out is None:
        file_name = f"cluster_labels_mst_k{args.n_clusters}.csv"
        if args.outdir:
            args.out = f"{args.outdir}/{file_name}"
        else:
            args.out = legacy_default_output(f"report/results/{file_name}")
    out_path = check_output_path(args.out, overwrite=args.overwrite)

    indices, dists = load_knn(args.knn)
    n = indices.shape[0]
    A = build_symmetric_coo_from_knn(indices, dists)
    labels = mst_cut_clusters(A, args.n_clusters)

    # write CSV: id,label
    import csv

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, lab in enumerate(labels):
            w.writerow([i, int(lab)])

    print(f"Wrote labels -> {out_path} (n={n}, clusters={args.n_clusters})")


if __name__ == "__main__":
    main()
