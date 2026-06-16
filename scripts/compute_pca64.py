#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA

try:
    from scripts.utils.io import check_output_path, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, legacy_default_output


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="data/embeddings_100k.npy")
    p.add_argument("--out", default=None)
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    if args.out is None:
        args.out = legacy_default_output(f"report/scratch/pca_n{args.n_components}.npy")
    out_path = check_output_path(Path(args.out), overwrite=args.overwrite)

    X = np.load(args.emb, mmap_mode="r")
    n, _ = X.shape
    if args.sample_size is not None and args.sample_size < n:
        rng = np.random.RandomState(args.seed)
        rows = np.sort(rng.choice(n, size=args.sample_size, replace=False))
        X_source = X[rows]
        n = args.sample_size
    else:
        X_source = X

    ip = IncrementalPCA(n_components=args.n_components)
    for i in range(0, n, args.chunk):
        xb = np.array(X_source[i : i + args.chunk], dtype=np.float32)
        ip.partial_fit(xb)

    reduced = np.empty((n, args.n_components), dtype=np.float32)
    for i in range(0, n, args.chunk):
        xb = np.array(X_source[i : i + args.chunk], dtype=np.float32)
        tr = ip.transform(xb).astype(np.float32)
        reduced[i : i + tr.shape[0]] = tr

    np.save(out_path, reduced)
    print("saved", out_path, reduced.shape)


if __name__ == "__main__":
    main()
