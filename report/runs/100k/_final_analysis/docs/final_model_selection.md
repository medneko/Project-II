# Final Model Selection

- Conservative main model: `text-only minibatch_k16`.
- Best full-100k experimental model: `text_pca64_lexical + minibatch_k32`.
- Dense/event detection model: `text-only hdbscan_minsize50`.
- Compact probabilistic baseline: `text-only gmm_k8`.
- Diagnostic baselines: CLARA true and bounded multi-feature variants.

Do not summarize the experiment as "multi-feature is always worse." Lightweight lexical features improved MiniBatch at k=32 in the bounded ablation, while metadata-heavy publisher/stock/all-aux variants could hurt GMM/HDBSCAN or collapse clusters. All-aux weight 0.30 is often too heavy.
