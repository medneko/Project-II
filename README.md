# Project II - Financial News Clustering on Multi-Feature Vector Spaces

This repository contains the code, experiment outputs, and LaTeX report for Project II. The project studies clustering methods for financial news headlines on vector spaces built mainly from text embeddings plus selected tabular auxiliary features.

The current version should be read as a multi-feature vector-space clustering project, not as a full multimodal system. It uses text embeddings and structured side features such as lexical, calendar, publisher, stock, and sentiment/risk signals. It does not currently include image, audio, or video modalities.

## Current Experimental Conclusions

The final report separates models by role instead of selecting one model from a single metric.

| Role | Model |
| --- | --- |
| Conservative full-coverage baseline | `text_768_original + minibatch_k16` |
| Selected experimental full-coverage model | `text_pca64_lexical + minibatch_k40` |
| Event / dense detection model | `text_pca64_only + HDBSCAN(min_cluster_size=30, min_samples=20, cluster_selection_method=leaf)` |
| Risky high-K reference | `text_pca64_only + minibatch_k96` |

Older hard KPI thresholds such as `silhouette >= 0.15` or `stability >= 0.7` are no longer treated as final acceptance conditions. The report uses a combined interpretation of silhouette, DBI, coverage, cluster balance, stability, negative silhouette, and metadata-dominance checks.

## Pipeline

The project pipeline is:

```text
preprocessing
-> embeddings
-> feature spaces
-> clustering
-> metrics
-> metadata post-check
-> report
```

Main algorithm families:

- `MiniBatchKMeans`
- `GaussianMixture` / GMM
- `HDBSCAN`
- `CLARA`

Main metric groups:

- silhouette score
- Davies-Bouldin Index (DBI)
- coverage and noise ratio
- ARI / NMI stability where available
- negative silhouette where available
- cluster balance
- metadata dominance checks for publisher and stock effects

PCA and UMAP figures are diagnostic visualizations only. They help inspect model behavior and support demos, but they are not used as the primary evidence for final model selection.

## Important Output Locations

- `report/runs/100k/_final_analysis/`: final comparison docs, tables, charts, and selected-model analysis.
- `report/runs/100k/_k_metric_exploration/`: MiniBatch `k` sweep summaries and stability checks.
- `report/runs/100k/_hdbscan_event_validation/`: validated HDBSCAN event/dense detection candidate.
- `report/runs/100k/_hdbscan_tuning_compact/`: compact HDBSCAN tuning summaries.
- `report/runs/100k/_multifeature/bounded_ablation/`: bounded auxiliary-feature ablation outputs.
- `report_latex/`: report source for Overleaf / XeLaTeX.

The LaTeX report entry point is:

```text
report_latex/main.tex
```

Compile it with XeLaTeX.

## Large Artifacts

Large experiment artifacts should not be copied into the LaTeX project and should not be edited casually:

- `data/*.npy`
- `data/embeddings_clean.npy`
- `data/embeddings_100k.npy`
- `data/embeddings_10k.npy`
- large label CSV files
- `report/results_10k_approved/`
- `report/results_100k_approved/`

The report uses curated summaries, selected charts, and generated LaTeX tables rather than embedding large intermediate artifacts.

## Reproducibility Notes

Some full pipeline steps are expensive, especially embedding generation and large HDBSCAN sweeps. The report therefore relies on existing validated CSV, Markdown, PNG, and LaTeX artifacts under `report/runs/100k/`. Do not rerun clustering or embedding generation unless the experiment design explicitly requires it.
