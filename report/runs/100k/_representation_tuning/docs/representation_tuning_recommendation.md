# Representation Tuning Recommendation

- Best tuned candidate: `text_768_l2 + minibatch_k40`.
- Best lexical weight: `0.03`.
- L2-normalize: useful for this tuned sweep, but not enough by itself to replace the prior experimental model based on this tuning run.
- Final experimental model decision: do not change from `text_pca64_lexical + minibatch_k40`.
- Use the recommendation only with the paired metric table; do not select by silhouette alone.
