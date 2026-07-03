# Final Event Model Update

## Decision

`text_pca64_only + HDBSCAN(min_cluster_size=30, min_samples=20, cluster_selection_method=leaf)` is now the official event/dense detection model.

## Baseline vs Updated Event Model

- Previous baseline: `text-only hdbscan_minsize50`, coverage `20.638%`, clusters `105`, silhouette cosine `0.565234`, DBI `0.98608`.
- Updated model: `text_pca64_only + HDBSCAN(mcs=30, ms=20, leaf)`, coverage `20.119%`, noise `79.881%`, clusters `219`, silhouette cosine `0.60858`, silhouette euclidean `0.452374`, DBI `1.02601`, negative silhouette `0.69%`.
- The updated HDBSCAN model is not a full-coverage clustering model. It is selected specifically for dense/event detection.
- The new model improves silhouette and negative-silhouette behavior while retaining coverage close to the text-only HDBSCAN baseline.
- Caveat: full-run DBI is slightly worse than the text-only baseline (`1.026005` vs `0.986080`).
- Caveat: the raw 768D HDBSCAN grid timed out, so this conclusion is limited to the validated PCA64 path and does not prove that 768D cannot improve it.
