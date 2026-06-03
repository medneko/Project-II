#!/usr/bin/env python3
"""Create a small sample file from a large numpy embeddings memmap.

Usage:
  python scripts/create_embeddings_sample.py --src data/embeddings_clean.npy --out data/embeddings_10k.npy --n 10000
"""
import argparse
import numpy as np
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='data/embeddings_clean.npy')
    p.add_argument('--out', default='data/embeddings_10k.npy')
    p.add_argument('--n', type=int, default=10000)
    args = p.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit('Source embeddings not found: ' + args.src)
    X = np.load(args.src, mmap_mode='r')
    n = min(args.n, X.shape[0])
    Y = np.array(X[:n], dtype=np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, Y)
    print('saved', args.out, 'shape=', Y.shape, 'dtype=', Y.dtype, 'nbytes=', Y.nbytes)


if __name__ == '__main__':
    main()
