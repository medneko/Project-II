"""Orchestrator to run clustering for the 10k dataset.

Default output goes to a fresh run directory under ``report/runs/10k``.
The historical ``report/results_10k_approved`` directory is only targeted when
``--approved-output`` is passed explicitly.
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from scripts.utils.io import check_output_path, ensure_dir
except ModuleNotFoundError:
    from utils.io import check_output_path, ensure_dir


def default_run_dir() -> Path:
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return Path("report") / "runs" / "10k" / run_id


def run(cmd, desc, log_path, env_extra=None):
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    if env_extra:
        env.update(env_extra)
    with open(log_path, "a", encoding="utf8") as fh:
        fh.write("\n\n" + "=" * 80 + "\n")
        fh.write(f"RUN: {desc}\n")
        fh.write("CMD: " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        try:
            p = subprocess.run(
                list(map(str, cmd)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.getcwd(),
            )
            out = p.stdout.decode("utf8", errors="replace")
            fh.write(out)
            fh.write(f"EXIT CODE: {p.returncode}\n")
            fh.flush()
            return p.returncode
        except Exception as e:
            fh.write(f"EXCEPTION running {desc}: {e}\n")
            fh.flush()
            return 2


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="data/embeddings_10k.npy")
    p.add_argument("--features", default="data/features_aggregated.csv")
    p.add_argument("--outdir", default=None)
    p.add_argument("--approved-output", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Run scripts/rebuild_features.py before fusion. This can modify data/features_aggregated.csv.",
    )
    return p


def output_paths(out_dir: Path):
    paths = {
        "log": out_dir / "run_full_10k.log",
        "multimodal": out_dir / "embeddings_multimodal_10k.npy",
        "knn": out_dir / "knn_k50.npz",
        "mst": out_dir / "cluster_labels_mst_k8.csv",
        "agg": out_dir / "cluster_labels_agg_ward_k8.csv",
        "hdbscan": out_dir / "cluster_labels_hdbscan_minsize50.csv",
        "minibatch": out_dir / "cluster_labels_minibatch_k8.csv",
        "clara": out_dir / "cluster_labels_clara_k8_m10000_t5.csv",
        "gmm": out_dir / "cluster_labels_gmm_k8.csv",
        "clustering_results": out_dir / "clustering_results.csv",
        "consensus": out_dir / "consensus_pairwise.csv",
        "pca_sample": out_dir / "pca_sample.csv",
        "summary_csv": out_dir / "results_summary.csv",
        "summary_png": out_dir / "results_summary.png",
        "sizes_png": out_dir / "results_cluster_sizes.png",
        "results_txt": out_dir / "results.txt",
        "consensus_ari": out_dir / "consensus_ARI_heatmap_10k.png",
        "consensus_nmi": out_dir / "consensus_NMI_heatmap_10k.png",
        "consensus_overlap": out_dir / "consensus_overlap_heatmap_10k.png",
    }
    label_names = [
        "cluster_labels_clara_k8_m10000_t5",
        "cluster_labels_minibatch_k8",
        "cluster_labels_gmm_k8",
        "cluster_labels_agg_ward_k8",
        "cluster_labels_hdbscan_minsize50",
        "cluster_labels_mst_k8",
    ]
    for name in label_names:
        paths[f"{name}_pca_png"] = out_dir / f"{name}_pca_big.png"
        paths[f"{name}_pca_pdf"] = out_dir / f"{name}_pca_big.pdf"
    paths["agg_dendro"] = out_dir / "clusters_dendrogram_cluster_labels_agg_ward_k8.png"
    return paths


def preflight(paths, overwrite):
    for path in paths.values():
        check_output_path(path, overwrite=overwrite)


def main(argv=None):
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    args = build_parser().parse_args(argv)
    py = sys.executable
    out_dir = Path(args.outdir) if args.outdir else default_run_dir()
    if args.approved_output and args.outdir is None:
        out_dir = Path("report") / "results_10k_approved"
    out_dir = ensure_dir(out_dir)
    paths = output_paths(out_dir)
    preflight(paths, args.overwrite)
    log_path = paths["log"]

    print(f"--> Pipeline 10k output: {out_dir}")
    print(f"--> Log: {log_path}")

    if args.rebuild_features:
        run([py, "scripts/rebuild_features.py"], "rebuild_financial_features", log_path)

    emb = args.emb
    try:
        text_emb = np.load(args.emb)
        num_df = pd.read_csv(args.features)
        num_data = num_df[["sentiment_mean", "sentiment_count", "news_density"]].values
        if len(num_data) != len(text_emb):
            print(f"WARNING: feature rows ({len(num_data)}) != embedding rows ({len(text_emb)}); keeping existing modulo fusion behavior for this routing-only change.")
            indices = np.arange(len(text_emb)) % len(num_data)
            num_data = num_data[indices]
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_data)
        multimodal_emb = np.hstack((text_emb, num_scaled * 10.0))
        np.save(paths["multimodal"], multimodal_emb)
        print(f"Saved multimodal embeddings -> {paths['multimodal']}")
        emb = str(paths["multimodal"])
    except Exception as e:
        print(f"WARNING: multimodal fusion failed; using text embeddings. Reason: {e}")

    overwrite_flag = ["--overwrite"] if args.overwrite else []

    run([py, "scripts/build_knn.py", "--emb", emb, "--out", paths["knn"], "--k", "50", *overwrite_flag], "build_knn", log_path)
    run([py, "scripts/mst_single_link.py", "--knn", paths["knn"], "--n-clusters", "8", "--out", paths["mst"], *overwrite_flag], "single_linkage_mst", log_path)
    run([py, "scripts/agg_with_connectivity.py", "--emb", emb, "--knn", paths["knn"], "--n-clusters", "8", "--out", paths["agg"], *overwrite_flag], "ward_connectivity", log_path)
    run([py, "scripts/hdbscan_runner.py", "--emb", emb, "--min-cluster-size", "50", "--out", paths["hdbscan"], *overwrite_flag], "hdbscan", log_path)
    run([py, "scripts/minibatch_kmeans_runner.py", "--emb", emb, "--k", "8", "--out", paths["minibatch"], *overwrite_flag], "minibatch_kmeans", log_path)
    run([py, "scripts/clara_kmedoids.py", "--emb", emb, "--k", "8", "--out", paths["clara"], *overwrite_flag], "clara_kmedoids", log_path)
    run([py, "scripts/gmm_runner.py", "--emb", emb, "--k", "8", "--seed", str(args.seed), "--out", paths["gmm"], *overwrite_flag], "gmm", log_path)

    run([py, "scripts/compute_metrics_from_labels.py", "--emb", emb, "--out", out_dir, "--sample", "10000", "--silhouette-sample", "1000", *overwrite_flag], "compute_metrics", log_path)
    run([py, "scripts/plot_results.py", "--report", out_dir, *overwrite_flag], "plot_results", log_path)
    run([py, "scripts/plot_consensus.py", "--consensus", paths["consensus"], "--out", out_dir, "--suffix", "10k", "--annot-size", "9", "--tick-size", "9", "--title-size", "14", "--fig-scale", "2.0", "--dpi", "200", *overwrite_flag], "plot_consensus", log_path)

    target_label_files = [paths["clara"], paths["minibatch"], paths["gmm"], paths["agg"], paths["hdbscan"], paths["mst"]]
    for label_path in target_label_files:
        if label_path.exists():
            algo_id = label_path.stem.replace("cluster_labels_", "")
            run([py, "scripts/plot_pca_scatter.py", "--labels", label_path, "--emb", emb, "--pca", paths["pca_sample"], "--out", out_dir, "--marker-size", "40", "--alpha", "0.6", "--dpi", "300", "--rasterize", *overwrite_flag], f"plot_pca_{algo_id}", log_path)

    if paths["agg"].exists():
        run([py, "scripts/plot_dendro_pca.py", "--emb", emb, "--labels", paths["agg"], "--out", out_dir, "--n-dendro", "150", "--method", "ward", "--fig-scale", "1.0", "--dpi", "150", *overwrite_flag], "plot_dendro_agg_ward_k8", log_path)

    text_eda_out = out_dir / "text_eda"
    run([py, "scripts/text_eda.py", "--data-dir", "data", "--out", text_eda_out, "--sample-rows", "10000", "--num-topics", "10"], "text_eda_LDA", log_path)

    with open(log_path, "a", encoding="utf8") as fh:
        fh.write("\nALL STEPS COMPLETE. Check outputs in " + str(out_dir) + "\n")
    print("--> Pipeline finished. Output:", out_dir)


if __name__ == "__main__":
    main()
