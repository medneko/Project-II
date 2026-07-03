# CLARA True Update

Old bounded-ablation CLARA fallback results were diagnostic only. CLARA was rerun under `.venv311` with `sklearn_extra.cluster.CLARA`, and every `clara_true` row records `CLARA true via sklearn_extra.cluster.CLARA`.

Best CLARA true result: `text_pca64_lexical_calendar + clara_k16` with silhouette 0.033399 and DBI 4.301401.

CLARA true still does not beat MiniBatch. It remains a diagnostic baseline rather than the selected final model.
