# Representation Tuning Analysis

## Run Scope

- Feature source for lexical columns: reused report\runs\100k\_multifeature\artifacts\X_aux_features.npy.
- MiniBatchKMeans params: n_init=5, batch_size=8192, max_iter=200, init=k-means++.
- Publisher and stock were excluded from fitting and used only for post-hoc warnings.
- Baseline `text_pca64_lexical + minibatch_k40`: silhouette=0.106481, DBI=3.346290, largest=5.17%, min_size=599.

## Top Candidates

| feature_space               | k  | base_score | final_score | silhouette_cosine | dbi     | negative_silhouette_pct | min_cluster_size | largest_cluster_pct | publisher_warning_count | stock_warning_count | mean_ARI | mean_NMI |
| --------------------------- | -- | ---------- | ----------- | ----------------- | ------- | ----------------------- | ---------------- | ------------------- | ----------------------- | ------------------- | -------- | -------- |
| text_768_l2                 | 40 | 0.923166   | 1.09846     | 0.0803915         | 3.25287 | 40.93                   | 271              | 4.309               | 11                      | 0                   | 0.425816 | 0.662565 |
| text_768_l2                 | 64 | 0.913861   | 1.09611     | 0.0817524         | 3.21536 | 41.44                   | 266              | 2.642               | 25                      | 0                   | 0.408265 | 0.68217  |
| text_768_l2                 | 32 | 0.925362   | 1.07876     | 0.0826371         | 3.18153 | 39.39                   | 507              | 7.19                | 12                      | 0                   | 0.412648 | 0.655757 |
| text_768_l2                 | 48 | 0.912528   | 1.07781     | 0.0773665         | 3.20115 | 42.46                   | 532              | 3.952               | 16                      | 0                   | 0.40761  | 0.669232 |
| text_pca128_lexical_w005_l2 | 64 | 0.916456   | 0.916456    | 0.0983992         | 3.77525 | 33.72                   | 485              | 3.057               | 30                      | 0                   | 0.326964 | 0.602812 |
| text_pca128_only            | 64 | 0.87053    | 0.87053     | 0.0901653         | 3.99104 | 35.22                   | 273              | 3.248               | 25                      | 1                   |          |          |
| text_pca128_lexical_w003    | 64 | 0.860981   | 0.860981    | 0.0828267         | 3.96532 | 34.52                   | 6                | 3.215               | 24                      | 0                   |          |          |
| text_pca128_only            | 48 | 0.845305   | 0.845305    | 0.0872942         | 4.23273 | 37.12                   | 577              | 5.405               | 24                      | 1                   |          |          |
| text_pca128_lexical_w003    | 48 | 0.84441    | 0.84441     | 0.087373          | 4.23873 | 37.38                   | 577              | 5.413               | 24                      | 1                   |          |          |
| text_pca128_lexical_w005_l2 | 48 | 0.843293   | 0.843293    | 0.0831234         | 4.17112 | 37.72                   | 528              | 5.259               | 25                      | 0                   |          |          |


## Answers

1. PCA128/PCA256: best PCA128 candidate is `text_pca128_lexical_w005_l2 + k64`; best PCA256 candidate is `text_pca256_lexical_w005_l2 + k64`. The final ranking decides by balanced score, not PCA dimension alone.
2. Best lexical weight among non-L2 lexical runs: `0.03`.
3. L2-normalize helps the balanced ranking in this sweep under the seed-42 pre-stability score.
4. Best k by balanced final score is `k40` for `text_768_l2`.
5. Clear improvement over `text_pca64_lexical + minibatch_k40`: no under the conservative comparison rule.
6. Recommended final experimental model: `text_768_l2 + minibatch_k40` if accepting this tuning score; otherwise keep PCA64 lexical k40 as the simpler prior model.
7. If the final recommendation does not change, the reason is that higher-dimensional or L2 variants did not produce a clean improvement across silhouette, DBI, balance, and metadata warnings together.

## Stability

- Stability was run only for the top candidates selected after the seed-42 full 100k sweep.
