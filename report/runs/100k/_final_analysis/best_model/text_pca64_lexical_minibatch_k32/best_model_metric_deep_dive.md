# Best Model Metric Deep Dive

Model: `text_pca64_lexical + minibatch_k32`.

## Cluster Size and Balance

- Number of clusters: 32.
- Largest cluster: 6.03%.
- Top 3 clusters: 15.77%.
- Top 5 clusters: 24.53%.
- Normalized entropy: 0.9774.
- Gini: 0.2129.
- Clusters with size < 100: 0.
- Clusters with size < 500: 0.

## Silhouette

- Global sampled silhouette: 0.109388.
- Worst clusters by mean silhouette: 23, 26, 24, 31, 19.
- Best clusters by mean silhouette: 6, 13, 15, 7, 2.

## Cohesion and Separation

The closest centroid pairs and DBI-like pairs are saved in `cluster_cohesion_separation.csv` and `worst_dbi_like_cluster_pairs.csv`.

## Metadata Dominance

Clusters with publisher/stock dominance warnings: 10.

## Stability

Stability rerun used MiniBatch k32 seeds 7, 13, 21, 42, 100 on full 100k rows. Mean ARI vs seed 42: 0.3684; mean NMI vs seed 42: 0.5524.
