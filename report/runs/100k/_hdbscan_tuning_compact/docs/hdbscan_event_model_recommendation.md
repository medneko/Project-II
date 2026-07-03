# HDBSCAN Event Model Recommendation

## Scope

- HDBSCAN is evaluated only as a dense/event-like detector, not as a full-coverage clustering model.
- Fit sample size: 50000; metric sample size: 10000; seed: 42.
- Publisher/stock metadata was not used in fitting.
- Completed configurations in current summary: 36 of 72 requested.
- Completed feature spaces: text_pca64_lexical, text_pca64_only.
- Missing feature spaces in current summary: text_768_l2, text_768_original.

## Baseline

- Text-only `hdbscan_minsize50`: coverage=20.638%, n_clusters=105, silhouette=0.565234, DBI=0.986080.

## Top Configurations

| feature_space      | min_cluster_size | min_samples | cluster_selection_method | event_score | coverage_pct | noise_pct | n_clusters | silhouette_cosine | dbi      | negative_silhouette_pct | largest_cluster_pct | collapse_warning | too_sparse_warning |
| ------------------ | ---------------- | ----------- | ------------------------ | ----------- | ------------ | --------- | ---------- | ----------------- | -------- | ----------------------- | ------------------- | ---------------- | ------------------ |
| text_pca64_only    | 30               | 20          | leaf                     | 0.815595    | 17.23        | 82.77     | 113        | 0.619671          | 0.980566 | 0.835752                | 8.98433             | False            | False              |
| text_pca64_lexical | 30               | 20          | leaf                     | 0.801188    | 17.988       | 82.012    | 114        | 0.612034          | 0.994398 | 0.856126                | 8.66133             | False            | False              |
| text_pca64_only    | 30               | 20          | eom                      | 0.748527    | 21.026       | 78.974    | 105        | 0.570673          | 0.980264 | 0.87                    | 13.5261             | False            | False              |
| text_pca64_lexical | 30               | 20          | eom                      | 0.632906    | 21.746       | 78.254    | 106        | 0.559446          | 1.10948  | 2.1                     | 13.7772             | False            | False              |
| text_pca64_only    | 30               | 10          | leaf                     | 0.611947    | 21.192       | 78.808    | 132        | 0.5481            | 1.13514  | 1.97                    | 7.48396             | False            | False              |
| text_pca64_lexical | 30               | 10          | leaf                     | 0.59724     | 21.832       | 78.168    | 135        | 0.543651          | 1.15614  | 2.13                    | 7.27373             | False            | False              |
| text_pca64_only    | 50               | 20          | leaf                     | 0.58831     | 18.482       | 81.518    | 62         | 0.587294          | 1.22697  | 0.768315                | 8.37572             | False            | False              |
| text_pca64_lexical | 50               | 20          | leaf                     | 0.566288    | 19.368       | 80.632    | 65         | 0.576745          | 1.24195  | 0.939694                | 8.0442              | False            | False              |
| text_pca64_only    | 50               | 20          | eom                      | 0.539547    | 20.814       | 79.186    | 59         | 0.554386          | 1.22994  | 0.84                    | 13.6639             | False            | False              |
| text_pca64_lexical | 50               | 20          | eom                      | 0.531568    | 21.65        | 78.35     | 63         | 0.548457          | 1.24321  | 0.9                     | 13.8383             | False            | False              |


## Answers

1. Any configuration clearly exceeds text-only `hdbscan_minsize50`: yes.
2. Collapse configurations: 0.
3. Too-sparse configurations: 0.
4. Recommended compact tuning candidate: `text_pca64_only`, min_cluster_size=30, min_samples=20, method=leaf.
5. Event model decision: switch to the compact tuned candidate.
