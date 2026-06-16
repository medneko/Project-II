"""Orchestrator to run clustering with the approved algorithm runners.

Default output goes to a fresh run directory under ``report/runs/100k``.
Use ``--approved-output`` only when intentionally targeting the historical
approved directory, and pair it with ``--overwrite`` only after reviewing the
existing artifacts.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from scripts.utils.io import check_output_path, ensure_dir
except ModuleNotFoundError:
    from utils.io import check_output_path, ensure_dir


def default_run_dir() -> Path:
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return Path("report") / "runs" / "100k" / run_id


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
            if p.returncode != 0:
                fh.write(f"WARNING: command {desc} exited with code {p.returncode}\n")
            return p.returncode
        except Exception as e:
            fh.write(f"EXCEPTION running {desc}: {e}\n")
            fh.flush()
            return 2


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="data/embeddings_100k.npy")
    p.add_argument("--outdir", default=None)
    p.add_argument("--approved-output", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--algorithms",
        default="mst,agg,hdbscan,minibatch,clara,gmm",
        help="Comma-separated algorithms to run.",
    )
    return p


def output_paths(out_dir: Path):
    return {
        "log": out_dir / "run_full_100k.log",
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
        "consensus_ari": out_dir / "consensus_ARI_heatmap_100k.png",
        "consensus_nmi": out_dir / "consensus_NMI_heatmap_100k.png",
        "consensus_overlap": out_dir / "consensus_overlap_heatmap_100k.png",
        "minibatch_pca_png": out_dir / "cluster_labels_minibatch_k8_pca_big.png",
        "minibatch_pca_pdf": out_dir / "cluster_labels_minibatch_k8_pca_big.pdf",
        "agg_pca_png": out_dir / "cluster_labels_agg_ward_k8_pca_big.png",
        "agg_pca_pdf": out_dir / "cluster_labels_agg_ward_k8_pca_big.pdf",
        "agg_dendro": out_dir / "clusters_dendrogram_cluster_labels_agg_ward_k8.png",
    }


def preflight(paths, algorithms, overwrite):
    keys = [
        "log",
        "knn",
        "clustering_results",
        "consensus",
        "pca_sample",
        "summary_csv",
        "summary_png",
        "sizes_png",
        "results_txt",
        "consensus_ari",
        "consensus_nmi",
        "consensus_overlap",
        "minibatch_pca_png",
        "minibatch_pca_pdf",
        "agg_pca_png",
        "agg_pca_pdf",
        "agg_dendro",
    ]
    keys.extend(a for a in algorithms if a in paths)
    for key in keys:
        check_output_path(paths[key], overwrite=overwrite)


def main(argv=None):
    args = build_parser().parse_args(argv)
    py = sys.executable
    out_dir = Path(args.outdir) if args.outdir else default_run_dir()
    if args.approved_output and args.outdir is None:
        out_dir = Path("report") / "results_100k_approved"
    out_dir = ensure_dir(out_dir)
    paths = output_paths(out_dir)
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]

    preflight(paths, algorithms, args.overwrite)
    log_path = paths["log"]

    print("Orchestration output:", out_dir)
    print("Log:", log_path)

    overwrite_flag = ["--overwrite"] if args.overwrite else []

    run([py, "scripts/build_knn.py", "--emb", args.emb, "--out", paths["knn"], "--k", "50", *overwrite_flag], "build_knn", log_path)
    if "mst" in algorithms:
        run([py, "scripts/mst_single_link.py", "--knn", paths["knn"], "--n-clusters", "8", "--out", paths["mst"], *overwrite_flag], "single_linkage_mst", log_path)
    if "agg" in algorithms:
        run([py, "scripts/agg_with_connectivity.py", "--emb", args.emb, "--knn", paths["knn"], "--n-clusters", "8", "--out", paths["agg"], *overwrite_flag], "ward_connectivity", log_path)
    if "hdbscan" in algorithms:
        run([py, "scripts/hdbscan_runner.py", "--emb", args.emb, "--min-cluster-size", "50", "--out", paths["hdbscan"], *overwrite_flag], "hdbscan", log_path)
    if "minibatch" in algorithms:
        run([py, "scripts/minibatch_kmeans_runner.py", "--emb", args.emb, "--k", "8", "--out", paths["minibatch"], *overwrite_flag], "minibatch_kmeans", log_path)
    if "clara" in algorithms:
        run([py, "scripts/clara_kmedoids.py", "--emb", args.emb, "--k", "8", "--out", paths["clara"], *overwrite_flag], "clara_kmedoids", log_path)
    if "gmm" in algorithms:
        run([py, "scripts/gmm_runner.py", "--emb", args.emb, "--k", "8", "--seed", str(args.seed), "--out", paths["gmm"], *overwrite_flag], "gmm", log_path)

    run([py, "scripts/compute_metrics_from_labels.py", "--emb", args.emb, "--out", out_dir, "--sample", "100000", "--silhouette-sample", "10000", *overwrite_flag], "compute_metrics", log_path)
    run([py, "scripts/plot_results.py", "--report", out_dir, *overwrite_flag], "plot_results", log_path)
    run([py, "scripts/plot_consensus.py", "--consensus", paths["consensus"], "--out", out_dir, "--suffix", "100k", "--annot-size", "9", "--tick-size", "9", "--title-size", "14", "--fig-scale", "2.0", "--dpi", "200", *overwrite_flag], "plot_consensus", log_path)

    if paths["minibatch"].exists():
        run([py, "scripts/plot_pca_scatter.py", "--labels", paths["minibatch"], "--pca", paths["pca_sample"], "--out", out_dir, "--marker-size", "40", "--alpha", "0.9", "--dpi", "300", "--rasterize", *overwrite_flag], "plot_pca_kmeans_k8", log_path)
    if paths["agg"].exists():
        run([py, "scripts/plot_pca_scatter.py", "--labels", paths["agg"], "--pca", paths["pca_sample"], "--out", out_dir, "--marker-size", "40", "--alpha", "0.9", "--dpi", "300", "--rasterize", *overwrite_flag], "plot_pca_agg_ward_k8", log_path)
        run([py, "scripts/plot_dendro_pca.py", "--emb", args.emb, "--labels", paths["agg"], "--out", out_dir, "--n-dendro", "500", "--method", "ward", "--fig-scale", "1.5", "--dpi", "200", *overwrite_flag], "plot_dendro_agg_ward_k8", log_path)

    text_eda_out = out_dir / "text_eda"
    run([py, "scripts/text_eda.py", "--data-dir", "data", "--out", text_eda_out, "--sample-rows", "100000", "--num-topics", "10"], "text_eda_LDA", log_path)

    with open(log_path, "a", encoding="utf8") as fh:
        fh.write("\nALL STEPS COMPLETE. Check outputs in " + str(out_dir) + "\n")
    print("Orchestration finished (logs at", log_path, ")")


if __name__ == "__main__":
    main()
