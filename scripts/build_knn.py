#!/usr/bin/env python3
"""Chunked k-NN builder for large memmapped embeddings.

Usage:
  python scripts/build_knn.py --emb data/embeddings_clean.npy --out report/runs/10k/run_example/knn_k50.npz --k 50

Tries FAISS (HNSW) if available; otherwise falls back to an exact chunked computation
that never loads the full matrix into memory.
"""
from __future__ import annotations
import argparse
import numpy as np

try:
    from scripts.utils.io import check_output_path, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, legacy_default_output


def load_memmap(path):
    X = np.load(path, mmap_mode="r")
    if X.dtype != np.float32:
        # don't modify on disk — cast when reading
        X = X.astype(np.float32)
    return X


def chunked_topk(X, k=50, q_batch=512, chunk_c=20000, metric="l2"):
    n, d = X.shape
    indices = np.full((n, k), -1, dtype=np.int32)
    dists = np.full((n, k), np.inf, dtype=np.float32)

    if metric == "l2":
        norms = np.empty(n, dtype=np.float32)
        for j in range(0, n, chunk_c):
            chunk = X[j : j + chunk_c]
            norms[j : j + len(chunk)] = (chunk * chunk).sum(axis=1).astype(np.float32)
    else:
        norms = None

    for q0 in range(0, n, q_batch):
        q = np.array(X[q0 : q0 + q_batch], dtype=np.float32)
        b = q.shape[0]
        if metric == "l2":
            qn = (q * q).sum(axis=1).astype(np.float32)
        else:
            qn = None

        best_idx = np.full((b, k), -1, dtype=np.int32)
        best_d = np.full((b, k), np.inf, dtype=np.float32)

        for j in range(0, n, chunk_c):
            chunk = np.array(X[j : j + chunk_c], dtype=np.float32)
            c = chunk.shape[0]
            if metric == "l2":
                D = q @ chunk.T
                d_chunk = qn[:, None] + norms[j : j + c][None, :] - 2.0 * D
                d_chunk = d_chunk.astype(np.float32)
            else:
                qnrm = q / np.linalg.norm(q, axis=1)[:, None]
                cnrm = chunk / np.linalg.norm(chunk, axis=1)[:, None]
                S = qnrm @ cnrm.T
                d_chunk = (1.0 - S).astype(np.float32)

            # mask self-distances if overlap
            overlap_start = max(q0, j)
            overlap_end = min(q0 + b, j + c)
            if overlap_end > overlap_start:
                for gi in range(overlap_start, overlap_end):
                    r = gi - q0
                    cc = gi - j
                    d_chunk[r, cc] = np.inf

            # per-row top-k from this chunk
            idx_chunk = np.argpartition(d_chunk, kth=k - 1, axis=1)[:, :k]
            d_top = np.take_along_axis(d_chunk, idx_chunk, axis=1)
            idx_global = idx_chunk + j

            # merge with best so far
            concat_d = np.concatenate([best_d, d_top], axis=1)
            concat_idx = np.concatenate([best_idx, idx_global], axis=1)
            pick = np.argpartition(concat_d, kth=k - 1, axis=1)[:, :k]
            best_d = np.take_along_axis(concat_d, pick, axis=1)
            best_idx = np.take_along_axis(concat_idx, pick, axis=1)

        order = np.argsort(best_d, axis=1)
        indices[q0 : q0 + b] = np.take_along_axis(best_idx, order, axis=1)
        dists[q0 : q0 + b] = np.take_along_axis(best_d, order, axis=1)

    return indices, dists


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--metric", choices=("l2", "cosine"), default="l2")
    p.add_argument("--q-batch", type=int, default=512)
    p.add_argument("--chunk-c", type=int, default=20000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.out is None:
        args.out = legacy_default_output("report/scratch/knn_k50.npz")
    out_path = check_output_path(args.out, overwrite=args.overwrite)

    X = load_memmap(args.emb)
    n, d = X.shape

    # try faiss if installed
    try:
        import faiss  # type: ignore

        print("FAISS available; building HNSW index (may require RAM for index).")
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 200
        for i in range(0, n, args.chunk_c):
            xb = np.array(X[i : i + args.chunk_c], dtype=np.float32)
            index.add(xb)

        indices = np.full((n, args.k), -1, dtype=np.int32)
        dists = np.full((n, args.k), np.inf, dtype=np.float32)
        for q0 in range(0, n, args.q_batch):
            q = np.array(X[q0 : q0 + args.q_batch], dtype=np.float32)
            D, I = index.search(q, args.k + 1)
            for r in range(D.shape[0]):
                rid = q0 + r
                row_idx = I[r]
                row_d = D[r]
                mask = row_idx != rid
                sel_idx = row_idx[mask][: args.k]
                sel_d = row_d[mask][: args.k]
                if sel_idx.size < args.k:
                    sel_idx = np.pad(sel_idx, (0, args.k - sel_idx.size), constant_values=-1)
                    sel_d = np.pad(sel_d, (0, args.k - sel_d.size), constant_values=np.inf)
                indices[rid] = sel_idx
                dists[rid] = sel_d
        np.savez_compressed(out_path, indices=indices, dists=dists, n=n, k=args.k)
        print("Saved knn ->", out_path)
        return
    except Exception:
        print("FAISS not available or failed; using exact chunked top-k (slower).")

    indices, dists = chunked_topk(X, k=args.k, q_batch=args.q_batch, chunk_c=args.chunk_c, metric=args.metric)
    np.savez_compressed(out_path, indices=indices, dists=dists, n=n, k=args.k)
    print("Saved knn ->", out_path)


if __name__ == "__main__":
    main()
