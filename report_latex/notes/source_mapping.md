# Source mapping

- `report/runs/100k/_final_analysis/metrics/final_model_comparison_updated.csv` -> `tables/final_model_comparison.tex`, sections 06-08.
- `report/runs/100k/_k_metric_exploration/metrics/k_sweep_summary.csv` -> `tables/k_sweep_decision.tex`, section 05.
- `report/runs/100k/_representation_tuning/metrics/representation_tuning_summary.csv` -> `tables/representation_tuning_summary.tex`, sections 05 and 07.
- `report/runs/100k/_hdbscan_event_validation/metrics/hdbscan_event_validation_summary.csv` -> `tables/hdbscan_event_validation.tex`, sections 06 and 07.
- `report/runs/100k/_final_analysis/tables/extended_algorithm_comparison.tex` -> `tables/extended_algorithm_comparison.tex`, appendix.
- `report/runs/100k/_final_analysis/docs/report_ready_algorithm_comparison.tex` -> `sections/report_ready_algorithm_comparison.tex`, section 06.
- `report/runs/100k/_final_analysis/docs/algorithm_family_comments.md` -> `notes/algorithm_family_comments.md`.

Large `.npy` files, large label CSV files, approved result folders, and intermediate artifacts are intentionally excluded from `report_latex/`.

## Copied figures

- `report/runs/100k/_k_metric_exploration/charts/text_pca64_lexical/silhouette_vs_k.png` -> `figures/k_sweep_silhouette_text_pca64_lexical.png`
- `report/runs/100k/_k_metric_exploration/charts/text_pca64_lexical/dbi_vs_k.png` -> `figures/k_sweep_dbi_text_pca64_lexical.png`
- `report/runs/100k/_k_metric_exploration/charts/text_pca64_lexical/cluster_size_balance_vs_k.png` -> `figures/k_sweep_cluster_balance_text_pca64_lexical.png`
- `report/runs/100k/_representation_tuning/charts/silhouette_vs_k.png` -> `figures/representation_silhouette_vs_k.png`
- `report/runs/100k/_representation_tuning/charts/dbi_vs_k.png` -> `figures/representation_dbi_vs_k.png`
- `report/runs/100k/_representation_tuning/charts/negative_silhouette_vs_k.png` -> `figures/representation_negative_silhouette_vs_k.png`
- `report/runs/100k/_final_analysis/best_model/text_pca64_lexical_minibatch_k32/images/best_model_umap_2d.png` -> `figures/selected_minibatch_umap_2d.png`
- `report/runs/100k/_final_analysis/best_model/text_pca64_lexical_minibatch_k32/images/cluster_balance.png` -> `figures/selected_minibatch_cluster_balance.png`
- `report/runs/100k/_final_analysis/charts/algorithm_comparison/full_coverage_silhouette_comparison.png` -> `figures/full_coverage_silhouette_comparison.png`
- `report/runs/100k/_final_analysis/charts/algorithm_comparison/full_coverage_dbi_comparison.png` -> `figures/full_coverage_dbi_comparison.png`
- `report/runs/100k/_final_analysis/charts/algorithm_comparison/model_scatter_silhouette_vs_dbi.png` -> `figures/model_scatter_silhouette_vs_dbi.png`
- `report/runs/100k/_final_analysis/charts/algorithm_comparison/event_models_coverage_comparison.png` -> `figures/event_models_coverage_comparison.png`
- `report/runs/100k/_final_analysis/charts/algorithm_comparison/event_models_cluster_count_comparison.png` -> `figures/event_models_cluster_count_comparison.png`
- `report/runs/100k/_final_analysis/charts/algorithm_comparison/stability_ari_comparison.png` -> `figures/stability_ari_comparison.png`
- `report/runs/100k/_final_analysis/charts/algorithm_comparison/metadata_warning_comparison.png` -> `figures/metadata_warning_comparison.png`
- `report/runs/100k/_final_analysis/charts/algorithm_comparison/cluster_balance_largest_cluster_pct.png` -> `figures/cluster_balance_largest_cluster_pct.png`
