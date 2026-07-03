# Final Model Selection Updated

## Conservative Full-Coverage Baseline

`text_768_original + minibatch_k16` remains the conservative baseline because it uses the original 768D text embedding, covers all 100k rows, and avoids the additional assumptions introduced by PCA compression or auxiliary lexical features. Its silhouette is modest, but it is stable enough to anchor comparisons.

## Best Experimental Full-Coverage Model

`text_pca64_lexical + minibatch_k40` is selected as the best experimental full-coverage model. It improves the MiniBatch result after k-sweep and gives the best practical balance among silhouette, DBI, cluster-size distribution, fragmentation risk, and interpretability.

## Previous Experimental Candidate

`text_pca64_lexical + minibatch_k32` is retained as the previous candidate, but k40 supersedes it after the broader k-sweep.

## Why k96 Is Not Selected

`text_pca64_only + minibatch_k96` has attractive separation metrics, but it is not selected because k is too large for the final model role: it creates smaller clusters, increases fragmentation risk, and carries more publisher/metadata-driven warning risk.

## Representation Tuning

`text_768_l2 + minibatch_k40` improves DBI/stability in parts of the tuning grid, but it does not replace the experimental model because silhouette is lower and negative silhouette remains very high (`40.93%`).

## Diagnostic Baselines

`text-only gmm_k8` remains a compact probabilistic baseline. `text_pca64_lexical_calendar + clara_k16` is reflected as a true CLARA diagnostic baseline, but it is weaker than MiniBatch and is not selected.
