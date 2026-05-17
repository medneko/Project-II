"""Headless EDA script for Project 2
Generates report/eda_summary.txt, CSV summaries and PNG plots.
"""
import os
import sys
import argparse
import json
import math
import warnings
from collections import Counter
import re

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("ERROR: pandas and numpy are required. Install with: pip install pandas numpy")
    raise

# optional plotting / ML packages
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

try:
    import umap
    UMAP = umap.UMAP
except Exception:
    UMAP = None

try:
    import hdbscan
except Exception:
    hdbscan = None


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def load_csv(path, nrows=None):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, nrows=nrows)


def detect_text_col(df):
    candidates = ['headline', 'title', 'text', 'content', 'body']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first string-like column (object or pandas string dtype)
    for c in df.columns:
        try:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                return c
        except Exception:
            continue
    return None


def top_tokens_from_series(s, sample=500, n=30):
    s = s.dropna().astype(str)
    if len(s) == 0:
        return []
    sample_n = min(sample, len(s))
    tokens = Counter()
    for t in s.sample(sample_n, random_state=1):
        tokens.update(re.findall(r"\w+", t.lower()))
    return tokens.most_common(n)


def save_fig(fig, path):
    try:
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print("Could not save figure:", path, e)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--emb-path', default=None, help='Override embeddings file path')
    parser.add_argument('--out', default='report')
    parser.add_argument('--sample-rows', type=int, default=2000)
    parser.add_argument('--emb-sample', type=int, default=5000)
    args = parser.parse_args(argv)

    DATA_DIR = args.data_dir
    OUT = args.out
    SAMPLE = args.sample_rows
    EMB_SAMPLE = args.emb_sample

    ensure_dir(OUT)

    summary = []

    # Paths
    news_path = os.path.join(DATA_DIR, 'news_clean.csv')
    features_path = os.path.join(DATA_DIR, 'features_aggregated.csv')
    # embeddings path: allow override, prefer cleaned file if present
    if args.emb_path:
        emb_path = args.emb_path
    else:
        candidate_clean = os.path.join(DATA_DIR, 'embeddings_clean.npy')
        default_emb = os.path.join(DATA_DIR, 'embeddings.npy')
        emb_path = candidate_clean if os.path.exists(candidate_clean) else default_emb
    emb_meta_path = os.path.join(DATA_DIR, 'embeddings_meta.csv')

    print('Loading sample data...')
    df_news = load_csv(news_path, nrows=SAMPLE)
    df_features = load_csv(features_path, nrows=SAMPLE)
    emb_exists = os.path.exists(emb_path)

    # Schema & missing
    def describe_df(name, df):
        if df is None:
            summary.append(f'{name}: NOT FOUND')
            return
        summary.append(f'{name}: shape={df.shape}')
        summary.append(str(df.dtypes))
        miss = df.isna().mean().sort_values(ascending=False).head(50)
        summary.append(f'{name} missing rates:\n{miss.to_string()}')
        dup = int(df.duplicated().sum())
        summary.append(f'{name} duplicates: {dup}')

    describe_df('news', df_news)
    describe_df('features', df_features)

    # Timestamps
    if df_news is not None and 'timestamp' in df_news.columns:
        try:
            df_news['ts'] = pd.to_datetime(df_news['timestamp'], errors='coerce')
            tmin = df_news['ts'].min()
            tmax = df_news['ts'].max()
            summary.append(f'news.time_range: {tmin} - {tmax}')
            counts = df_news.set_index('ts').resample('D').size()
            counts_path = os.path.join(OUT, 'date_counts.csv')
            counts.to_csv(counts_path, header=['count'])
            summary.append(f'Wrote time series counts to {counts_path}')
            if plt is not None:
                fig, ax = plt.subplots(figsize=(10,4))
                counts.plot(ax=ax, title='Daily counts (sample)')
                ax.set_xlabel('date')
                ax.set_ylabel('count')
                fig_path = os.path.join(OUT, 'daily_counts.png')
                save_fig(fig, fig_path)
                summary.append(f'Wrote plot {fig_path}')
        except Exception as e:
            summary.append('Timestamp processing failed: '+str(e))
    else:
        summary.append('No timestamp column detected in news sample')

    # Text EDA
    if df_news is not None:
        text_col = detect_text_col(df_news)
        if text_col:
            s = df_news[text_col].dropna().astype(str)
            summary.append(f'detected text column: {text_col}')
            summary.append('text length stats: ' + str(s.str.len().describe()))
            top_tokens = top_tokens_from_series(s, sample=500, n=50)
            tok_path = os.path.join(OUT, 'top_tokens.csv')
            pd.DataFrame(top_tokens, columns=['token','count']).to_csv(tok_path, index=False)
            summary.append(f'Wrote top tokens to {tok_path}')
            if plt is not None:
                try:
                    fig, ax = plt.subplots(figsize=(6,3))
                    if sns is not None:
                        sns.histplot(s.str.len(), bins=50, ax=ax)
                    else:
                        ax.hist(s.str.len().dropna(), bins=50)
                    ax.set_title('Text length distribution')
                    fig_path = os.path.join(OUT, 'text_length.png')
                    save_fig(fig, fig_path)
                    summary.append(f'Wrote plot {fig_path}')
                except Exception as e:
                    summary.append('Text plot failed: '+str(e))
        else:
            summary.append('No text-like column found in news sample')

    # Numeric features
    if df_features is not None:
        num = df_features.select_dtypes(include=['number'])
        desc_path = os.path.join(OUT, 'features_describe.csv')
        if num.shape[0] > 0 and num.shape[1] > 0:
            num.describe().T.to_csv(desc_path)
            summary.append(f'Wrote numeric features describe to {desc_path}')
            if num.shape[1] > 1 and plt is not None:
                try:
                    corr = num.corr()
                    fig, ax = plt.subplots(figsize=(8,6))
                    if sns is not None:
                        sns.heatmap(corr, vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
                    else:
                        ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
                        ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
                        ax.set_xticklabels(corr.columns, rotation=90)
                        ax.set_yticklabels(corr.columns)
                    ax.set_title('Numeric feature correlations')
                    fig_path = os.path.join(OUT, 'features_corr.png')
                    save_fig(fig, fig_path)
                    summary.append(f'Wrote correlation plot {fig_path}')
                except Exception as e:
                    summary.append('Correlation plot failed: '+str(e))
        else:
            summary.append('No numeric features found in features dataset')
    else:
        summary.append('No features dataset loaded')

    # Embeddings diagnostics
    if emb_exists:
        try:
            # Try memory-mapped load first for large numeric arrays
            try:
                emb = np.load(emb_path, mmap_mode='r')
            except Exception as e_load:
                summary.append('embeddings load (mmap) failed: '+str(e_load))
                # fallback to allow_pickle in case file contains object arrays
                try:
                    emb = np.load(emb_path, allow_pickle=True)
                    summary.append('Loaded embeddings with allow_pickle=True (unsafe).')
                except Exception as e2:
                    raise e2

            # If embeddings are object-dtype (e.g., array of vectors), try to convert
            if getattr(emb, 'dtype', None) == object:
                try:
                    emb = np.vstack([np.asarray(x) for x in emb])
                    summary.append('Converted object-dtype embeddings to numeric 2D array.')
                except Exception as e_conv:
                    summary.append('Could not convert object-dtype embeddings: '+str(e_conv))
                    raise

            summary.append(f'embeddings shape: {emb.shape}')
            norms = np.sqrt((emb**2).sum(axis=1))
            summary.append(f'emb norms: min={float(norms.min())}, mean={float(norms.mean())}, max={float(norms.max())}')
            # PCA visualization on sample
            sample_n = min(EMB_SAMPLE, emb.shape[0])
            try:
                pca = PCA(2, random_state=42)
                X2 = pca.fit_transform(emb[:sample_n])
                if plt is not None:
                    fig, ax = plt.subplots(figsize=(6,5))
                    ax.scatter(X2[:,0], X2[:,1], s=5, alpha=0.6)
                    ax.set_title('PCA(2) of embeddings (sample)')
                    fig_path = os.path.join(OUT, 'emb_pca.png')
                    save_fig(fig, fig_path)
                    summary.append(f'Wrote embedding PCA plot {fig_path}')
            except Exception as e:
                summary.append('Embed PCA failed: '+str(e))

            # Baseline clustering
            try:
                # reduce to 2d with UMAP if available
                if UMAP is not None:
                    reducer = UMAP(n_components=2, random_state=42)
                    X2 = reducer.fit_transform(emb[:sample_n])
                else:
                    pca = PCA(2, random_state=42)
                    X2 = pca.fit_transform(emb[:sample_n])
                labels = None
                if hdbscan is not None:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
                    labels = clusterer.fit_predict(X2)
                    summary.append('Used HDBSCAN for clustering')
                else:
                    km = KMeans(n_clusters=10, random_state=42)
                    labels = km.fit_predict(X2)
                    summary.append('Used KMeans for clustering')
                # label stats
                import pandas as _pd
                lab_ser = _pd.Series(labels)
                summary.append('Cluster label counts: '+str(lab_ser.value_counts().head(20).to_dict()))
                valid = labels != -1
                if valid.sum() > 10:
                    try:
                        sc = silhouette_score(X2[valid], labels[valid]) if len(set(labels[valid]))>1 else float('nan')
                        summary.append(f'Silhouette (excluding noise or -1): {sc}')
                    except Exception as e:
                        summary.append('Silhouette failed: '+str(e))
                # save labels CSV (for sample ids)
                out_labels = os.path.join(OUT, 'cluster_labels_sample.csv')
                _pd.DataFrame({'idx': list(range(sample_n)), 'label': labels}).to_csv(out_labels, index=False)
                summary.append(f'Wrote cluster labels to {out_labels}')
            except Exception as e:
                summary.append('Clustering failed: '+str(e))

        except Exception as e:
            summary.append('Could not load embeddings: '+str(e))
    else:
        summary.append('No embeddings file found')

    # write summary
    out = os.path.join(OUT, 'eda_summary.txt')
    with open(out, 'w', encoding='utf8') as f:
        f.write('\n'.join(summary))
    print('Wrote', out)
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
