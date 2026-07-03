# Final Tuning Update

## Inputs Read

- `report\runs\100k\_k_metric_exploration`
- `report\runs\100k\_representation_tuning`
- `report\runs\100k\_hdbscan_tuning_compact`
- `report\runs\100k\_hdbscan_event_validation`
- `report\runs\100k\_multifeature\bounded_ablation\clara_true`
- `report\runs\100k\_final_analysis`

## Updated Decisions

- Conservative full-coverage baseline: `text_768_original + minibatch_k16`.
- Best experimental full-coverage model: `text_pca64_lexical + minibatch_k40`.
- Event/dense detection model: `text_pca64_only + HDBSCAN(min_cluster_size=30, min_samples=20, leaf)`.
- High-silhouette risky reference: `text_pca64_only + minibatch_k96` remains a reference only.
- CLARA true diagnostic baseline: `text_pca64_lexical_calendar + clara_k16`.

## Evidence Scope

- K-sweep rows read: 27.
- Representation tuning rows read: 44.
- Compact HDBSCAN rows completed: 36; feature spaces completed include text_pca64_lexical, text_pca64_only.
- Raw 768D HDBSCAN compact-grid rows remain missing because `text_768_original` and `text_768_l2` timed out.
- CLARA true was run through `sklearn_extra.cluster.CLARA`; it is not a fallback result.
