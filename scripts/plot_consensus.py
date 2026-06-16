"""Plot pairwise consensus heatmaps (ARI, NMI, overlap)

Usage:
    python scripts/plot_consensus.py --consensus report/runs/10k/run_example/consensus_pairwise.csv --out report/runs/10k/run_example --suffix 10k
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scripts.utils.io import check_output_path, ensure_dir, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, ensure_dir, legacy_default_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--consensus', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--suffix', default='')
    p.add_argument('--annot-size', type=float, default=8.0, help='font size for cell annotations')
    p.add_argument('--tick-size', type=float, default=8.0, help='font size for tick labels')
    p.add_argument('--title-size', type=float, default=12.0, help='font size for title')
    p.add_argument('--fig-scale', type=float, default=1.0, help='scale factor for figure size (1.0 default)')
    p.add_argument('--dpi', type=int, default=150, help='output image dpi')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if args.out is None:
        args.out = legacy_default_output('report/scratch/plots')
    if args.consensus is None:
        args.consensus = str(Path(args.out) / 'consensus_pairwise.csv')

    if not os.path.exists(args.consensus):
        raise SystemExit(f'Consensus file not found: {args.consensus}')

    df = pd.read_csv(args.consensus)

    # collect unique algorithm keys
    algos = sorted(set(df['algo_a']).union(set(df['algo_b'])))
    n = len(algos)
    idx = {a: i for i, a in enumerate(algos)}

    ari = np.full((n, n), np.nan)
    nmi = np.full((n, n), np.nan)
    overlap = np.zeros((n, n), dtype=int)

    # fill diagonal
    for i in range(n):
        ari[i, i] = 1.0
        nmi[i, i] = 1.0

    for _, row in df.iterrows():
        a = row['algo_a']
        b = row['algo_b']
        i = idx[a]
        j = idx[b]
        val_ari = float(row.get('ARI', np.nan))
        val_nmi = float(row.get('NMI', np.nan))
        cnt = int(row.get('overlap_count', 0))
        ari[i, j] = val_ari
        ari[j, i] = val_ari
        nmi[i, j] = val_nmi
        nmi[j, i] = val_nmi
        overlap[i, j] = cnt
        overlap[j, i] = cnt

    mask_ari = np.isnan(ari)
    mask_nmi = np.isnan(nmi)

    suf = f"_{args.suffix}" if args.suffix else ""
    out_dir = ensure_dir(args.out)

    # helper to plot with adjustable font sizes
    def save_heatmap(mat, mask, labels, title, out_path, fmt='.2f', cmap='viridis', cbar_label='', annot_size=8.0, tick_size=8.0, title_size=12.0, fig_scale=1.0, dpi=150):
        fig_w = max(6, n * 0.5 * fig_scale)
        fig_h = max(4, n * 0.35 * fig_scale)
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(mat, xticklabels=labels, yticklabels=labels, annot=True, fmt=fmt, cmap=cmap, mask=mask, annot_kws={'fontsize': annot_size}, cbar_kws={'label': cbar_label})
        plt.title(title, fontsize=title_size)
        plt.xticks(rotation=45, ha='right', fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    out_ari = check_output_path(out_dir / f'consensus_ARI_heatmap{suf}.png', overwrite=args.overwrite)
    save_heatmap(ari, mask_ari, algos, f'Pairwise ARI{suf}', out_ari, fmt='.2f', cmap='viridis', cbar_label='ARI', annot_size=args.annot_size, tick_size=args.tick_size, title_size=args.title_size, fig_scale=args.fig_scale, dpi=args.dpi)

    out_nmi = check_output_path(out_dir / f'consensus_NMI_heatmap{suf}.png', overwrite=args.overwrite)
    save_heatmap(nmi, mask_nmi, algos, f'Pairwise NMI{suf}', out_nmi, fmt='.2f', cmap='viridis', cbar_label='NMI', annot_size=args.annot_size, tick_size=args.tick_size, title_size=args.title_size, fig_scale=args.fig_scale, dpi=args.dpi)

    out_overlap = check_output_path(out_dir / f'consensus_overlap_heatmap{suf}.png', overwrite=args.overwrite)
    # overlap matrix has zeros for missing entries; no mask
    save_heatmap(overlap, None, algos, f'Pairwise Overlap Count{suf}', out_overlap, fmt='d', cmap='magma', cbar_label='overlap_count', annot_size=args.annot_size, tick_size=args.tick_size, title_size=args.title_size, fig_scale=args.fig_scale, dpi=args.dpi)

    print('Saved:', out_ari, out_nmi, out_overlap)


if __name__ == '__main__':
    main()
