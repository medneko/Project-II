#!/usr/bin/env python3
"""CLARA wrapper for k-medoids on large memmapped datasets.

Approach:
 - Repeat t times: sample m points, run KMedoids (or KMeans->medoid fallback) on sample
 - Evaluate medoids on full dataset by chunked distance computation
 - Keep best medoids, then assign full labels by nearest medoid (chunked)
"""
from __future__ import annotations
import argparse
import numpy as np

try:
    from scripts.utils.io import check_output_path, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, legacy_default_output


def load_memmap(path):
    return np.load(path, mmap_mode="r")


def medoid_cost_full(X, medoids, chunk=20000):
    # medoids: (K, d)
    K = medoids.shape[0]
    med_norm = (medoids * medoids).sum(axis=1).astype(np.float64)
    total = 0.0
    n = X.shape[0]
    for i in range(0, n, chunk):
        xb = np.array(X[i : i + chunk], dtype=np.float32)
        xb_norm = (xb * xb).sum(axis=1).astype(np.float64)
        D = xb_norm[:, None] + med_norm[None, :] - 2.0 * (xb @ medoids.T)
        total += D.min(axis=1).sum()
    return float(total)


def assign_labels_chunked(X, medoids, out_csv, chunk=20000):
    import csv

    n = X.shape[0]
    K = medoids.shape[0]
    med_norm = (medoids * medoids).sum(axis=1).astype(np.float64)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i in range(0, n, chunk):
            xb = np.array(X[i : i + chunk], dtype=np.float32)
            xb_norm = (xb * xb).sum(axis=1).astype(np.float64)
            D = xb_norm[:, None] + med_norm[None, :] - 2.0 * (xb @ medoids.T)
            labs = np.argmin(D, axis=1)
            for off, lab in enumerate(labs):
                w.writerow([i + off, int(lab)])


def run_clara(X_path, n_clusters=8, m_sample=10000, t=5, chunk=20000, seed=0):
    X = load_memmap(X_path)
    n, d = X.shape
    rng = np.random.RandomState(seed)
    best_cost = float("inf")
    best_meds = None

    for it in range(t):
        sample_idx = rng.choice(n, size=min(m_sample, n), replace=False)
        sample = np.array(X[sample_idx], dtype=np.float32)

        # try sklearn_extra KMedoids
        try:
            from sklearn_extra.cluster import KMedoids

            model = KMedoids(n_clusters=n_clusters, metric="euclidean", random_state=seed)
            model.fit(sample)
            # try to extract medoids as coordinates
            if hasattr(model, "cluster_centers_"):
                medoids = np.array(model.cluster_centers_, dtype=np.float32)
            elif hasattr(model, "medoid_indices_"):
                medoids = sample[model.medoid_indices_]
            else:
                medoids = sample[model.labels_ == 0][:n_clusters]
        except Exception:
            # fallback: MiniBatchKMeans -> choose nearest sample point per centroid
            from sklearn.cluster import MiniBatchKMeans

            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
            km.fit(sample)
            cents = km.cluster_centers_
            # find nearest sample point in `sample` to each centroid
            medoids = np.empty((n_clusters, d), dtype=np.float32)
            for j in range(n_clusters):
                dif = sample - cents[j]
                dd = (dif * dif).sum(axis=1)
                medoids[j] = sample[np.argmin(dd)]

        cost = medoid_cost_full(X, medoids, chunk=chunk)
        print(f"iter={it} cost={cost:.3f}")
        if cost < best_cost:
            best_cost = cost
            best_meds = medoids.copy()

    return best_meds, best_cost


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--m", type=int, default=10000)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--out", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.out is None:
        file_name = f"cluster_labels_clara_k{args.k}_m{args.m}_t{args.t}.csv"
        if args.outdir:
            args.out = f"{args.outdir}/{file_name}"
        else:
            args.out = legacy_default_output(f"report/results/{file_name}")
    out_path = check_output_path(args.out, overwrite=args.overwrite)

    medoids, cost = run_clara(args.emb, n_clusters=args.k, m_sample=args.m, t=args.t, chunk=args.chunk, seed=args.seed)
    print("Best cost:", cost)
    if medoids is None:
        raise RuntimeError("CLARA failed to produce medoids")

    assign_labels_chunked(np.load(args.emb, mmap_mode="r"), medoids, out_path, chunk=args.chunk)
    print("Wrote labels ->", out_path)


if __name__ == "__main__":
    main()
