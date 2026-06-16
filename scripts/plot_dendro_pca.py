"""Plot dendrogram (for a subsample) and PCA scatter of clustered samples.

Usage:
    python scripts/plot_dendro_pca.py --emb data/embeddings_clean.npy --labels report/runs/10k/run_example/cluster_labels_agg_ward_k8.csv --out report/runs/10k/run_example

The script will:
 - Load labels (CSV with a `label` column) produced by `run_pipeline.py`.
 - Recreate the same sample indices used by `run_pipeline.py` (RandomState(42)), so labels align with embeddings.
 - For PCA scatter: use `pca_sample.csv` beside the labels if available, otherwise compute PCA(2) from the sampled embeddings.
 - For dendrogram: build a linkage on a smaller subsample (default 500) after reducing dimensions with PCA, then plot dendrogram with leaves colored by cluster label.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

try:
    from scripts.utils.io import check_output_path, ensure_dir, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, ensure_dir, legacy_default_output

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
except Exception:
    linkage = None
    dendrogram = None


def load_labels(path):
    df = pd.read_csv(path)
    if 'label' in df.columns:
        return df['label'].to_numpy()
    # fallback to first column
    return df.iloc[:, 0].to_numpy()


def make_color_map(unique_vals, palette='tab20'):
    n = len(unique_vals)
    if n <= 20:
        colors = sns.color_palette(palette, n_colors=n)
    else:
        # use HSV continuous colormap for many categories
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i / float(n)) for i in range(n)]
    return {val: colors[i] for i, val in enumerate(unique_vals)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb', default='data/embeddings_clean.npy')
    p.add_argument('--labels', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--n-dendro', type=int, default=500, help='number of leaves to use for dendrogram (subsample)')
    p.add_argument('--random-state', type=int, default=0, help='random state for subsampling dendrogram leaves')
    p.add_argument('--method', default='ward', help='linkage method for dendrogram (ward/average/single/complete)')
    p.add_argument('--leaf-font-size', type=float, default=6.0)
    p.add_argument('--leaf-rotation', type=float, default=90.0)
    p.add_argument('--fig-scale', type=float, default=1.5)
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--use-pca-from-emb', action='store_true', help='compute PCA(2) from embeddings instead of reading pca_sample.csv')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if args.out is None:
        args.out = legacy_default_output('report/scratch/plots')
    out_dir = ensure_dir(args.out)
    label_stem = os.path.splitext(os.path.basename(args.labels))[0]
    out_scatter = check_output_path(out_dir / f'clusters_pca_{label_stem}.png', overwrite=args.overwrite)
    out_dendro = check_output_path(out_dir / f'clusters_dendrogram_{label_stem}.png', overwrite=args.overwrite)

    if linkage is None:
        raise SystemExit('scipy is required for dendrogram plotting. Install scipy in the environment.')

    if not os.path.exists(args.labels):
        raise SystemExit(f'Labels file not found: {args.labels}')

    labels_all = load_labels(args.labels)
    n_labels = len(labels_all)
    print('Loaded labels:', args.labels, 'n=', n_labels)

    # load embeddings memmap
    if not os.path.exists(args.emb):
        raise SystemExit(f'Embeddings file not found: {args.emb}')
    emb = np.load(args.emb, mmap_mode='r')
    total_n = emb.shape[0]
    print('Embeddings shape:', emb.shape)

    # recreate the same sample indices used by run_pipeline (if sample < total)
    if n_labels < total_n:
        rng_idx = np.random.RandomState(42)
        sample_idx = rng_idx.choice(total_n, size=n_labels, replace=False)
        sample_emb = emb[sample_idx]
    else:
        sample_idx = np.arange(total_n)
        sample_emb = emb

    # PCA scatter coordinates: prefer existing pca_sample.csv
    pca_csv = os.path.join(os.path.dirname(args.labels), 'pca_sample.csv')
    if os.path.exists(pca_csv) and not args.use_pca_from_emb:
        pca_df = pd.read_csv(pca_csv)
        if len(pca_df) == n_labels:
            xs = pca_df['x'].to_numpy()
            ys = pca_df['y'].to_numpy()
            print('Using existing', pca_csv, 'for PCA scatter')
        else:
            print('pca_sample.csv exists but length mismatch; computing PCA from embeddings')
            pca = PCA(n_components=2)
            xy = pca.fit_transform(sample_emb)
            xs, ys = xy[:, 0], xy[:, 1]
    else:
        pca = PCA(n_components=2)
        xy = pca.fit_transform(sample_emb)
        xs, ys = xy[:, 0], xy[:, 1]

    # color map for clusters
    unique = np.unique(labels_all)
    cmap = make_color_map(unique)

    # PCA scatter plot
    fig_w = max(6, n_labels ** 0.5 * args.fig_scale)
    fig_h = max(4, n_labels ** 0.5 * args.fig_scale)
    plt.figure(figsize=(fig_w, fig_h))
    # map colors for each point
    colors = [cmap[l] if l in cmap else (0.5, 0.5, 0.5) for l in labels_all]
    plt.scatter(xs, ys, c=colors, s=8, alpha=0.8)
    plt.title('PCA scatter colored by cluster')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=args.dpi)
    plt.close()
    print('Saved PCA scatter to', out_scatter)

    # Dendrogram on a manageable subsample
    n_dendro = min(args.n_dendro, n_labels)
    if n_dendro < 2:
        print('Not enough points for dendrogram; n_dendro < 2')
        return

    rng = np.random.RandomState(args.random_state)
    subs_in_sample = rng.choice(n_labels, size=n_dendro, replace=False)
    global_idx = sample_idx[subs_in_sample]
    X_dendro = emb[global_idx]
    labels_sub = labels_all[subs_in_sample]

    # Reduce dimensionality before linkage to speed up
    n_comp = min(20, X_dendro.shape[1], X_dendro.shape[0] - 1)
    if n_comp < 2:
        # not enough dims; use raw
        reduced = X_dendro
    else:
        pca_d = PCA(n_components=n_comp)
        reduced = pca_d.fit_transform(X_dendro)

    print('Computing linkage on', reduced.shape)
    Z = linkage(reduced, method=args.method)

    # Plot dendrogram
    fig_w = max(10, n_dendro * 0.02 * args.fig_scale)
    fig_h = max(6, n_dendro * 0.02 * args.fig_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # labels for leaves are cluster ids
    leaf_labels = [str(int(l)) for l in labels_sub]
    den = dendrogram(Z, labels=leaf_labels, leaf_rotation=args.leaf_rotation, leaf_font_size=args.leaf_font_size, ax=ax)

    # color leaf tick labels according to cluster
    ivl = den.get('ivl', [])
    # mapping for colors
    unique_sub = np.unique(labels_sub)
    cmap_sub = make_color_map(unique_sub)
    ticks = ax.get_xmajorticklabels()
    for tick, label_text in zip(ticks, ivl):
        try:
            lab_val = int(label_text)
        except Exception:
            lab_val = label_text
        color = cmap_sub.get(lab_val, (0.2, 0.2, 0.2))
        tick.set_color(color)

    plt.title(f'Dendrogram ({args.method}) on {n_dendro} samples')
    plt.tight_layout()
    plt.savefig(out_dendro, dpi=args.dpi)
    plt.close()
    print('Saved dendrogram to', out_dendro)


if __name__ == '__main__':
    main()
