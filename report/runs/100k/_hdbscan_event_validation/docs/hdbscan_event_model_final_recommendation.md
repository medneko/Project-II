# HDBSCAN Event Model Final Recommendation

## Answers

1. Best PCA64 candidate can run full 100k: yes. Full 100k completed.
2. Full-run metrics remain strong: coverage=20.119%, n_clusters=219, silhouette_cosine=0.608580, DBI=1.026005.
3. Sample-level evidence is enough to keep this as the preferred event candidate: same-sample coverage=17.230%, n_clusters=113, silhouette_cosine=0.619671, DBI=0.980566.
4. Switch event model from text-only HDBSCAN minsize50 to PCA64 tuned HDBSCAN: yes. Text-only baseline reference: text-only hdbscan_minsize50 coverage=20.638%, n_clusters=105, silhouette=0.565234, DBI=0.986080. Same-sample PCA64 baseline check: PCA64 mcs50/eom same-sample coverage=23.196%, n_clusters=28, silhouette=0.223601, DBI=1.083749.
5. Report caveat: raw 768D HDBSCAN candidates remain unresolved because compact tuning timed out on `text_768_original` and `text_768_l2`; this recommendation is therefore for the validated PCA64 event-detector path, not a final claim that 768D cannot improve it.

## Recommended Conclusion Wording

Use `text_pca64_only + HDBSCAN(min_cluster_size=30, min_samples=20, leaf, euclidean)` as the event-detection HDBSCAN candidate. It improves cluster separation and event granularity versus the text-only minsize50 baseline while retaining comparable dense-event coverage for post-hoc review; full-run DBI is slightly worse, so keep that caveat visible. Caveat: raw 768D HDBSCAN variants timed out during compact tuning, so the conclusion is limited to validated PCA64 candidates.
