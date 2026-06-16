#!/usr/bin/env python3
"""Compute clustering metrics (Silhouette, DBI), consensus (ARI/NMI) and PCA sample CSV.

Reads cluster label CSVs from an explicit labels/output directory and writes
metrics artifacts to the output directory.
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scripts.utils.io import check_output_path, ensure_dir, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, ensure_dir, legacy_default_output


def load_embeddings(path, nmax=None):
    X = np.load(path, mmap_mode='r')
    if nmax is not None:
        return X[:nmax]
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb', required=True)
    p.add_argument('--out', default=None, help='Output directory (legacy alias for --outdir)')
    p.add_argument('--outdir', default=None)
    p.add_argument('--labels-dir', default=None)
    p.add_argument('--sample', type=int, default=100000)
    p.add_argument('--silhouette-sample', type=int, default=10000)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    out_dir = Path(args.outdir or args.out) if (args.outdir or args.out) else legacy_default_output('report/scratch/metrics')
    labels_dir = Path(args.labels_dir) if args.labels_dir else out_dir
    ensure_dir(out_dir)
    pca_out = check_output_path(out_dir / 'pca_sample.csv', overwrite=args.overwrite)
    clustering_out = check_output_path(out_dir / 'clustering_results.csv', overwrite=args.overwrite)
    consensus_out = out_dir / 'consensus_pairwise.csv'

    # load sample embeddings (memmap)
    X = load_embeddings(args.emb, nmax=args.sample)
    n = X.shape[0]

    # compute PCA 2 for visualization
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        Xsmall = np.array(X, dtype=np.float32)
        vis = pca.fit_transform(Xsmall)
        pd.DataFrame(vis, columns=['x','y']).to_csv(pca_out, index=False)
    except Exception as e:
        print('PCA failed:', e)

    # gather label files
    files = sorted(glob.glob(str(labels_dir / 'cluster_labels_*.csv')))
    results = []
    label_sets = {}

    if not files:
        print('No cluster label files found in', labels_dir)
        return

    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    for f in files:
        name = os.path.basename(f).replace('.csv','')
        df = pd.read_csv(f)
        labels = df['label'].to_numpy()
        if labels.shape[0] != n:
            print('Skipping', f, 'length mismatch', labels.shape[0], 'vs', n)
            continue
        mask = labels != -1
        coverage = float(mask.sum()) / n * 100.0
        unique = np.unique(labels[mask]) if mask.sum() > 0 else np.array([])
        n_clusters = len(unique)
        if n_clusters > 1:
            try:
                metric_n = int(mask.sum())
                sil_sample = min(args.silhouette_sample, metric_n) if args.silhouette_sample else None
                sil = float(
                    silhouette_score(
                        X[mask],
                        labels[mask],
                        metric='cosine',
                        sample_size=sil_sample,
                        random_state=42,
                    )
                )
            except Exception:
                sil = float('nan')
            try:
                dbi = float(davies_bouldin_score(X[mask], labels[mask]))
            except Exception:
                dbi = float('nan')
        else:
            sil = float('nan')
            dbi = float('nan')
        algo = name.replace('cluster_labels_','')
        results.append({'algo': algo, 'param': '', 'silhouette': sil, 'dbi': dbi, 'coverage_pct': coverage, 'n_clusters': n_clusters, 'silhouette_sample': args.silhouette_sample})
        label_sets[algo] = labels

    res_df = pd.DataFrame(results)
    res_df.to_csv(clustering_out, index=False)

    # pairwise consensus
    pairs = []
    keys = list(label_sets.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a = keys[i]
            b = keys[j]
            l1 = label_sets[a]
            l2 = label_sets[b]
            mask = (l1 != -1) & (l2 != -1)
            if mask.sum() == 0:
                continue
            ari = adjusted_rand_score(l1[mask], l2[mask])
            nmi = normalized_mutual_info_score(l1[mask], l2[mask])
            pairs.append({'algo_a': a, 'algo_b': b, 'ARI': ari, 'NMI': nmi, 'overlap_count': int(mask.sum())})

    if pairs:
        consensus_out = check_output_path(consensus_out, overwrite=args.overwrite)
        pd.DataFrame(pairs).to_csv(consensus_out, index=False)
    print('Wrote clustering_results.csv and consensus_pairwise.csv to', out_dir)


if __name__ == '__main__':
    main()
