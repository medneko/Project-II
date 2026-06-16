#!/usr/bin/env python3
"""Run MiniBatchKMeans on memmapped embeddings using chunked partial_fit.

Produces a CSV of labels and saves the trained model with joblib if desired.
"""
from __future__ import annotations
import argparse
import numpy as np

try:
    from scripts.utils.io import check_output_path, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, legacy_default_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--batch", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.out is None:
        if args.outdir:
            args.out = f"{args.outdir}/cluster_labels_minibatch_k{args.k}.csv"
        else:
            args.out = legacy_default_output(
                f"report/results/cluster_labels_minibatch_k{args.k}.csv"
            )
    out_path = check_output_path(args.out, overwrite=args.overwrite)

    X = np.load(args.emb, mmap_mode="r")
    n, d = X.shape
    from sklearn.cluster import MiniBatchKMeans

    mbk = MiniBatchKMeans(n_clusters=args.k)

    for ep in range(args.epochs):
        print("epoch", ep)
        for i in range(0, n, args.batch):
            xb = np.array(X[i : i + args.batch], dtype=np.float32)
            mbk.partial_fit(xb)

    import csv

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i in range(0, n, args.batch):
            xb = np.array(X[i : i + args.batch], dtype=np.float32)
            labs = mbk.predict(xb)
            for off, lab in enumerate(labs):
                w.writerow([i + off, int(lab)])

    print("Wrote labels ->", out_path)


if __name__ == "__main__":
    main()
