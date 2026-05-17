import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb', default='data/embeddings.npy')
    parser.add_argument('--meta', default='data/embeddings_meta.csv')
    parser.add_argument('--features', default='data/features_aggregated.csv')
    parser.add_argument('--out', default='data/fused_clusters.csv')
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    emb = np.load(args.emb)
    meta = pd.read_csv(args.meta)
    feat = pd.read_csv(args.features)

    # Try to align by index mapping saved in meta
    meta = meta.rename(columns={'index': 'row_index'})
    # If features have ticker/date, we expect merging upstream; here we'll join by position if possible
    # For simplicity, if lengths match, concatenate by order.
    if len(meta) == len(feat):
        fused = np.hstack([emb, feat.values])
    else:
        # fallback: use embeddings only
        fused = emb

    kmeans = KMeans(n_clusters=args.k, random_state=42).fit(fused)
    labels = kmeans.labels_

    outdf = pd.DataFrame({'label': labels})
    outdf.to_csv(args.out, index=False)
    print('Saved clusters to', args.out)


if __name__ == '__main__':
    main()
