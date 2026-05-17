#!/usr/bin/env python3
"""Run helper: load embeddings and metadata (project-relative paths).

Usage examples:
  python run.py                 # load embeddings (full or sample) and print summary
  python run.py --emb data/embeddings_sample.npy --meta data/embeddings_sample_meta.csv --nn --nn-index 0
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def find_existing(paths, base):
    for p in paths:
        path = p if os.path.isabs(p) else os.path.join(base, p)
        if os.path.exists(path):
            return path
    return None


def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb', help='Path to embeddings .npy (relative to project root or absolute)')
    parser.add_argument('--meta', help='Path to embeddings meta CSV')
    parser.add_argument('--nn', action='store_true', help='Compute nearest neighbors for one index (may be slow)')
    parser.add_argument('--nn-index', type=int, default=0, help='Index to query for nearest neighbors')
    parser.add_argument('--n-neigh', type=int, default=6, help='Number of neighbors to return')
    parser.add_argument('--status', action='store_true', help='Print status and sanity checks for embeddings and meta')
    parser.add_argument('--report', action='store_true', help='Generate simple report assets (PCA sample PNG and date counts CSV)')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    default_embs = ['data/embeddings.npy', 'data/embeddings_sample.npy']
    default_metas = ['data/embeddings_meta.csv', 'data/embeddings_sample_meta.csv']

    emb_path = args.emb or find_existing(default_embs, base)
    if emb_path and not os.path.isabs(emb_path):
        emb_path = os.path.join(base, emb_path) if not os.path.exists(emb_path) else emb_path

    if emb_path is None or not os.path.exists(emb_path):
        raise SystemExit(f"No embeddings file found. Provide --emb or place one of: {default_embs}")

    # determine meta path early to allow fallback when embeddings file is a raw memmap
    meta_path = args.meta or find_existing(default_metas, base)
    meta = None
    if meta_path and os.path.exists(meta_path):
        print('Loading meta from', meta_path)
        meta = pd.read_csv(meta_path)
        print('Meta: rows=', len(meta), 'columns=', list(meta.columns)[:6])
        print(meta.head(3).to_string(index=False))
    else:
        print('No meta CSV found (proceeding without).')

    # choose mmap for safety (doesn't copy whole file into RAM)
    print('Loading embeddings from', emb_path)
    try:
        emb = np.load(emb_path, mmap_mode='r')
        try:
            n, d = emb.shape
        except Exception:
            arr = np.asarray(emb)
            if arr.ndim == 1:
                n, d = arr.shape[0], 1
            else:
                n, d = arr.shape
    except ValueError as e:
        # Could be a raw memmap file without .npy header; try opening as raw memmap using meta rows
        print('np.load failed:', e)
        if meta is not None:
            try:
                n = len(meta)
                d = 768
                mm = np.memmap(emb_path, dtype=np.float32, mode='r', shape=(n, d))
                emb = mm
                print('Opened embeddings as raw memmap with shape', mm.shape)
            except Exception as e2:
                raise SystemExit(f'Failed to open embeddings as raw memmap: {e2}')
        else:
            raise SystemExit('Embeddings file is not a .npy file and no meta available to infer shape.')

    size_bytes = os.path.getsize(emb_path)
    print(f'Embeddings: shape={n}x{d}, dtype={emb.dtype}, size={sizeof_fmt(size_bytes)}')

    if args.nn:
        # compute nearest neighbors for a single index
        from sklearn.neighbors import NearestNeighbors

        idx = int(args.nn_index)
        if idx < 0 or idx >= n:
            raise SystemExit('nn-index out of range')

        print(f'Computing {args.n_neigh} nearest neighbors for index {idx} (this may take time)...')
        # load as memmap-backed array (still works with sklearn but may be slow)
        emb_array = np.array(emb) if getattr(emb, 'mode', None) else emb
        nn = NearestNeighbors(n_neighbors=args.n_neigh, metric='cosine', n_jobs=1)
        nn.fit(emb_array)
        dists, idxs = nn.kneighbors(emb_array[idx].reshape(1, -1))
        print('Neighbors indices:', idxs[0].tolist())
        print('Distances:', dists[0].tolist())
        if meta is not None:
            print('\nNeighbor meta rows:')
            print(meta.iloc[idxs[0]].to_string(index=False))

    print('\nDone.')

    if args.status:
        # progress JSON
        prog_path = os.path.join(base, 'data', 'emb_progress.json')
        if os.path.exists(prog_path):
            try:
                print('Progress:', json.load(open(prog_path, 'r', encoding='utf-8')))
            except Exception as e:
                print('Could not read progress file:', e)

        # sanity checks (NaN, all-zero, sample norms)
        try:
            mm = np.memmap(emb_path, dtype=emb.dtype, mode='r', shape=(n, d))
            nan_rows = int(np.isnan(mm).any(axis=1).sum())
            zero_rows = int((mm == 0).all(axis=1).sum())
            import numpy.linalg as la
            sample_n = min(1000, n)
            norms = la.norm(mm[:sample_n], axis=1)
            print('Sanity: nan_rows=', nan_rows, 'all-zero rows=', zero_rows, f'sample_norm_mean={float(norms.mean()):.3f}')
        except Exception as e:
            print('Sanity checks failed:', e)

        # fused clusters info
        fused = os.path.join(base, 'data', 'fused_clusters.csv')
        if os.path.exists(fused):
            try:
                fc = pd.read_csv(fused)
                if 'cluster' in fc.columns:
                    print('Fused clusters: rows=', len(fc), 'unique clusters=', fc['cluster'].nunique())
                else:
                    print('Fused_clusters exists but no cluster column')
            except Exception as e:
                print('Could not read fused_clusters.csv:', e)

    if args.report:
        # create report dir
        report_dir = Path(base) / 'report'
        report_dir.mkdir(parents=True, exist_ok=True)
        # date counts
        try:
            if meta is not None and 'date' in meta.columns:
                m = meta.copy()
                m['date'] = pd.to_datetime(m['date'], errors='coerce').dt.date
                m.groupby('date').size().to_csv(report_dir / 'date_counts.csv', header=['count'])
                print('Wrote', report_dir / 'date_counts.csv')
            else:
                print('Meta missing or has no date column; skipping date_counts')
        except Exception as e:
            print('Failed date_counts:', e)

        # PCA sample scatter
        try:
            import matplotlib
            matplotlib.use('Agg')
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            cnt = min(50000, n)
            idx = np.random.choice(n, cnt, replace=False)
            mm = np.memmap(emb_path, dtype=emb.dtype, mode='r', shape=(n, d))
            X = mm[idx]
            pca = PCA(n_components=2).fit_transform(X)
            plt.figure(figsize=(8, 6))
            plt.scatter(pca[:, 0], pca[:, 1], s=2, alpha=0.6)
            plt.title('PCA (sample)')
            out_png = report_dir / 'pca_sample.png'
            plt.savefig(out_png, dpi=150)
            plt.close()
            print('Saved', out_png)
        except Exception as e:
            print('PCA/report generation failed:', e)


if __name__ == '__main__':
    main()
