# K Selection Recommendation

- Conservative baseline: keep `text_768_original + minibatch_k16`.
- Best experimental model from this sweep: `text_pca64_lexical + minibatch_k40`.
- Main report should present silhouette/DBI together with cluster balance and stability, not silhouette alone.
- Publisher/stock are not part of this fitting loop; metadata remains post-hoc profiling only.
