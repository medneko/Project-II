"""Generate a clearer PCA scatter plot from `pca_sample.csv` or embeddings + labels.
Removes noise points (labeled as -1 or "-1") from the visualization.

Usage:
    python scripts/plot_pca_scatter.py --labels report/runs/10k/run_example/cluster_labels_agg_ward_k8.csv --out report/runs/10k/run_example --marker-size 30 --alpha 0.9
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import sys
import io

try:
    from scripts.utils.io import check_output_path, ensure_dir, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, ensure_dir, legacy_default_output


def make_color_map(unique_vals, cmap_name='tab20'):
    n = len(unique_vals)
    if n <= 20:
        pal = sns.color_palette(cmap_name, n_colors=n)
    else:
        cmap = plt.get_cmap('hsv')
        pal = [cmap(i / float(n)) for i in range(n)]
    return {val: pal[i] for i, val in enumerate(unique_vals)}


def try_load_pca(pca_csv, n_expected=None):
    if os.path.exists(pca_csv):
        df = pd.read_csv(pca_csv)
        if n_expected is None or len(df) == n_expected:
            return df['x'].to_numpy(), df['y'].to_numpy()
    return None


def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--labels', required=True)
    p.add_argument('--pca', default=None)
    p.add_argument('--emb', default='data/embeddings_clean.npy')
    p.add_argument('--out', default=None)
    p.add_argument('--marker-size', type=float, default=30.0)
    p.add_argument('--alpha', type=float, default=0.9)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--cmap', default='tab20')
    p.add_argument('--legend', action='store_true')
    p.add_argument('--rasterize', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if args.out is None:
        args.out = legacy_default_output('report/scratch/plots')
    if args.pca is None:
        args.pca = str(Path(args.out) / 'pca_sample.csv')
    out_dir = ensure_dir(args.out)

    if not os.path.exists(args.labels):
        raise SystemExit('Labels file not found: ' + args.labels)

    labels_df = pd.read_csv(args.labels)
    if 'label' in labels_df.columns:
        labels = labels_df['label'].to_numpy()
    else:
        labels = labels_df.iloc[:, 0].to_numpy()
    n = len(labels)
    print('Loaded', n, 'labels from', args.labels)

    # try load PCA coordinates
    pca_coords = try_load_pca(args.pca, n_expected=n)
    if pca_coords is not None:
        xs, ys = pca_coords
        print('Using', args.pca)
    else:
        # fallback: compute PCA from embeddings for the sampled indices
        if not os.path.exists(args.emb):
            raise SystemExit('No PCA and embeddings not found: ' + args.emb)
        emb = np.load(args.emb, mmap_mode='r')
        total_n = emb.shape[0]
        if n < total_n:
            rng = np.random.RandomState(42)
            idx = rng.choice(total_n, size=n, replace=False)
            sample = emb[idx]
        else:
            sample = emb
        pca = PCA(n_components=2)
        xy = pca.fit_transform(sample)
        xs, ys = xy[:, 0], xy[:, 1]
        print('Computed PCA from embeddings (fallback)')

    # ==========================================================================
    # BƯỚC SỬA ĐỔI: LOẠI BỎ ĐIỂM NHIỄU (-1 HOẶC "-1") TRƯỚC KHI VẼ
    # ==========================================================================
    # Tạo mặt nạ boolean: True cho các điểm KHÔNG PHẢI là nhiễu
    non_noise_mask = (labels != -1) & (labels != "-1")
    n_noise = np.sum(~non_noise_mask)
    
    if n_noise > 0:
        print(f"--> Phát hiện và loại bỏ {n_noise} điểm nhiễu (-1) khỏi đồ thị trực quan.")
        # Lọc lại toàn bộ tọa độ và nhãn dữ liệu
        xs = xs[non_noise_mask]
        ys = ys[non_noise_mask]
        labels = labels[non_noise_mask]
    else:
        print("--> Không phát hiện điểm nhiễu (Thuật toán phủ toàn bộ 100%).")

    # Tạo colormap và danh sách màu dựa trên tập nhãn đã lọc sạch
    unique = np.unique(labels)
    cmap = make_color_map(unique, cmap_name=args.cmap)
    colors = [cmap.get(l, (0.5, 0.5, 0.5)) for l in labels]

    # plot
    fig_w, fig_h = 14, 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.scatter(xs, ys, c=colors, s=args.marker_size, alpha=args.alpha, edgecolors='none', rasterized=args.rasterize)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    base_title = os.path.basename(args.labels)
    if n_noise > 0:
        ax.set_title(f'PCA scatter ({base_title}) - Noise Removed ({n_noise} points)')
    else:
        ax.set_title(f'PCA scatter ({base_title})')
        
    ax.grid(False)
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(args.labels))[0]
    out_png = check_output_path(out_dir / f'{base}_pca_big.png', overwrite=args.overwrite)
    out_pdf = check_output_path(out_dir / f'{base}_pca_big.pdf', overwrite=args.overwrite)
    plt.savefig(out_png, dpi=args.dpi)
    # save vector PDF
    plt.savefig(out_pdf)
    plt.close()
    print('Saved', out_png, 'and', out_pdf)


if __name__ == '__main__':
    main()
