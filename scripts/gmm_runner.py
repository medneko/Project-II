"""Run Gaussian Mixture Model clustering on memmapped embeddings."""
import argparse

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

try:
    from scripts.utils.io import check_output_path, legacy_default_output
except ModuleNotFoundError:
    from utils.io import check_output_path, legacy_default_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--sample", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--predict-sample-only", action="store_true")
    parser.add_argument("--chunk", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out is None:
        suffix = "_sample" if args.predict_sample_only else ""
        if args.outdir:
            args.out = f"{args.outdir}/cluster_labels_gmm_k{args.k}{suffix}.csv"
        else:
            args.out = legacy_default_output(f"report/results/cluster_labels_gmm_k{args.k}{suffix}.csv")
    out_path = check_output_path(args.out, overwrite=args.overwrite)

    X = np.load(args.emb, mmap_mode="r")
    n = X.shape[0]
    sample_n = min(args.sample, n)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(n, size=sample_n, replace=False) if sample_n < n else np.arange(n)
    sample = np.asarray(X[idx], dtype=np.float32)

    model = GaussianMixture(n_components=args.k, random_state=args.seed)
    sample_labels = model.fit_predict(sample)

    if args.predict_sample_only:
        pd.DataFrame(
            {"row_index": idx, "label": sample_labels, "algorithm": f"gmm_k{args.k}"}
        ).to_csv(out_path, index=False)
        print(f"Wrote sample labels -> {out_path} (n={sample_n}, k={args.k})")
        return

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("row_index,label,algorithm\n")
        for start in range(0, n, args.chunk):
            xb = np.asarray(X[start : start + args.chunk], dtype=np.float32)
            labels = model.predict(xb)
            for off, lab in enumerate(labels):
                fh.write(f"{start + off},{int(lab)},gmm_k{args.k}\n")
    print(f"Wrote full labels -> {out_path} (n={n}, fit_sample={sample_n}, k={args.k})")


if __name__ == "__main__":
    main()
