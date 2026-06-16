"""Plot clustering results from a run/report directory and save summaries.

Generates:
 - results_summary.csv
 - results_summary.png (silhouette & DBI)
 - results_cluster_sizes.png
 - results.txt (plain text summary)

Usage:
    python scripts/plot_results.py --report report/runs/10k/run_example
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.utils.io import check_output_path, ensure_dir, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, ensure_dir, legacy_default_output


def load_results(path):
    f = os.path.join(path, 'clustering_results.csv')
    if not os.path.exists(f):
        raise SystemExit('clustering_results.csv not found in ' + path)
    return pd.read_csv(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', default=None)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if args.report is None:
        args.report = legacy_default_output('report/scratch/metrics')
    report_dir = ensure_dir(args.report)
    df = load_results(str(report_dir))
    summary_csv = check_output_path(report_dir / 'results_summary.csv', overwrite=args.overwrite)
    summary_png = check_output_path(report_dir / 'results_summary.png', overwrite=args.overwrite)
    sizes_png = check_output_path(report_dir / 'results_cluster_sizes.png', overwrite=args.overwrite)
    results_txt = check_output_path(report_dir / 'results.txt', overwrite=args.overwrite)

    # save a copy
    df.to_csv(summary_csv, index=False)

    # Plot silhouette and DBI
    fig, ax1 = plt.subplots(figsize=(10, 5)) # Tăng nhẹ kích thước để text không bị đè nhau
    x = np.arange(len(df))
    ax1.plot(x, df['silhouette'], marker='o', color='C0', label='Silhouette')
    ax1.set_ylabel('Silhouette', color='C0')
    
    ax2 = ax1.twinx()
    ax2.plot(x, df['dbi'], marker='s', color='C1', label='DBI')
    ax2.set_ylabel('Davies-Bouldin Index', color='C1')
    
    # Xử lý nhãn trục X an toàn, loại bỏ triệt để hiện tượng dính "nan" do cộng chuỗi khuyết định dạng
    xtick_labels = []
    for _, row in df.iterrows():
        algo_name = str(row['algo']) if pd.notna(row['algo']) else "Unknown"
        if 'param' in df.columns and pd.notna(row['param']) and str(row['param']).strip().lower() != 'nan':
            xtick_labels.append(f"{algo_name}\n({row['param']})")
        else:
            xtick_labels.append(algo_name)
            
    ax1.set_xticks(x)
    ax1.set_xticklabels(xtick_labels, rotation=45, ha='right')
    
    fig.tight_layout()
    fig.savefig(summary_png, dpi=200) # Thêm DPI cho sắc nét
    plt.close(fig)

    # Cluster sizes for each label file present
    cluster_plots = []
    for col in os.listdir(report_dir):
        if col.startswith('cluster_labels_') and col.endswith('.csv'):
            labels = pd.read_csv(report_dir / col)['label']
            counts = labels.value_counts().sort_index()
            cluster_plots.append((col, counts))

    if cluster_plots:
        fig, axs = plt.subplots(len(cluster_plots), 1, figsize=(6, 3*len(cluster_plots)))
        if len(cluster_plots) == 1:
            axs = [axs]
        for ax, (fname, counts) in zip(axs, cluster_plots):
            counts.plot(kind='bar', ax=ax)
            ax.set_title(fname)
            ax.set_xlabel('Cluster label')
            ax.set_ylabel('Count')
        fig.tight_layout()
        fig.savefig(sizes_png)
        plt.close(fig)

    # write text summary
    with open(results_txt, 'w', encoding='utf8') as fh:
        fh.write('Clustering results summary\n')
        fh.write(df.to_string(index=False))
        fh.write('\n\n')
        for fname, counts in cluster_plots:
            fh.write(f'File: {fname}\n')
            fh.write(counts.to_string())
            fh.write('\n\n')

    print('Wrote results to', report_dir)


if __name__ == '__main__':
    main()
