"""Run a simple clustering pipeline for multiple algorithms and compute metrics.

Usage:
    python scripts/run_pipeline.py --emb data/embeddings_clean.npy --meta data/features_aggregated.csv --out report/
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import umap


def load_embeddings(path, max_rows=None):
    emb = np.load(path, mmap_mode='r') if os.path.exists(path) else None
    if max_rows and emb is not None:
        return emb[:max_rows]
    return emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb', required=True)
    p.add_argument('--meta', required=False)
    p.add_argument('--out', default='report')
    p.add_argument('--sample', type=int, default=10000)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    emb = load_embeddings(args.emb, max_rows=args.sample)
    if emb is None:
        raise SystemExit('Embeddings file not found')

    # sample if necessary
    if emb.shape[0] > args.sample:
        idx = np.random.RandomState(42).choice(emb.shape[0], size=args.sample, replace=False)
        sample = emb[idx]
    else:
        sample = emb

    # PCA for visualization
    pca = PCA(n_components=2)
    vis = pca.fit_transform(sample)
    pd.DataFrame(vis, columns=['x','y']).to_csv(os.path.join(args.out, 'pca_sample.csv'), index=False)

    results = []

    # KMeans
    for k in [8, 16, 32]:
        km = MiniBatchKMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(sample)
        sil = silhouette_score(sample, labels, metric='cosine')
        dbi = davies_bouldin_score(sample, labels)
        results.append({'algo':'MiniBatchKMeans','param':k,'silhouette':sil,'dbi':dbi})
        pd.DataFrame({'label':labels}).to_csv(os.path.join(args.out, f'cluster_labels_kmeans_k{k}.csv'), index=False)

    # GMM
    for k in [8,16]:
        gm = GaussianMixture(n_components=k, random_state=42)
        labels = gm.fit_predict(sample)
        sil = silhouette_score(sample, labels, metric='cosine')
        dbi = davies_bouldin_score(sample, labels)
        results.append({'algo':'GMM','param':k,'silhouette':sil,'dbi':dbi})
        pd.DataFrame({'label':labels}).to_csv(os.path.join(args.out, f'cluster_labels_gmm_k{k}.csv'), index=False)

    # UMAP + HDBSCAN (optional dependency)
    try:
        import hdbscan
        reducer = umap.UMAP(n_components=5, random_state=42)
        red = reducer.fit_transform(sample)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
        labels = clusterer.fit_predict(red)
        sil = silhouette_score(sample, labels[labels!=-1], metric='cosine') if (labels!=-1).sum()>1 else float('nan')
        dbi = davies_bouldin_score(sample[labels!=-1], labels[labels!=-1]) if (labels!=-1).sum()>1 else float('nan')
        results.append({'algo':'HDBSCAN','param':'min_cluster_size=50','silhouette':sil,'dbi':dbi})
        pd.DataFrame({'label':labels}).to_csv(os.path.join(args.out, f'cluster_labels_hdbscan.csv'), index=False)
    except Exception:
        print('HDBSCAN not installed; skipping HDBSCAN step')

    pd.DataFrame(results).to_csv(os.path.join(args.out, 'clustering_results.csv'), index=False)
    print('Done. Results written to', args.out)


if __name__ == '__main__':
    main()
