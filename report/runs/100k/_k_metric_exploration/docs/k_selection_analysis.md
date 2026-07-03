# K Selection Analysis

## text_768_original

- Best k by silhouette: k=32, silhouette=0.083570, DBI=3.228854.
- Best k by DBI: k=16, DBI=3.085353, silhouette=0.063232.
- Best k by sample stability: k=8, mean ARI=0.4960, mean NMI=0.6152.
- k=32: silhouette=0.083570, DBI=3.228854, largest=6.51%, min_size=633.
- k=48: silhouette=0.076036, DBI=3.220168, largest=4.21%, min_size=269.
- k=64: silhouette=0.083209, DBI=3.204224, largest=3.01%, min_size=210.

## text_pca64_only

- Best k by silhouette: k=96, silhouette=0.116079, DBI=2.996845.
- Best k by DBI: k=96, DBI=2.996845, silhouette=0.116079.
- Best k by sample stability: k=96, mean ARI=0.4069, mean NMI=0.6468.
- k=32: silhouette=0.096369, DBI=3.472506, largest=5.79%, min_size=503.
- k=48: silhouette=0.109194, DBI=3.134515, largest=3.81%, min_size=256.
- k=64: silhouette=0.106974, DBI=3.096599, largest=3.03%, min_size=69.

## text_pca64_lexical

- Best k by silhouette: k=96, silhouette=0.112344, DBI=2.990959.
- Best k by DBI: k=96, DBI=2.990959, silhouette=0.112344.
- Best k by sample stability: k=96, mean ARI=0.4204, mean NMI=0.6567.
- k=32: silhouette=0.109067, DBI=3.396135, largest=6.09%, min_size=914.
- k=48: silhouette=0.110851, DBI=3.134524, largest=3.95%, min_size=165.
- k=64: silhouette=0.103531, DBI=3.047795, largest=4.18%, min_size=63.

## Cross-cutting Answers

- Raising k can improve silhouette, but the decision should be checked against DBI, cluster balance, stability, and interpretability notes.
- Very high k values are not automatically better; they can fragment clusters and reduce stability.
- Recommended experimental k for `text_pca64_lexical`: k=40.
- `text_pca64_lexical + minibatch_k32` is no longer the single best recommendation under this sweep.
- Conservative baseline should remain text-only MiniBatch around k=16 unless the report emphasizes finer-grained exploratory clusters.
