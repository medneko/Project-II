# Bounded Multi-feature Ablation Analysis

## Run Status

- completed: 10
- skipped_existing: 0
- failed: 0
- time_budget_skipped: 0
- dependency_skipped: 0

## Scope

- Full 100k jobs: none.
- Sample-level diagnostics: clara_k16, clara_k8.
- Sample-level HDBSCAN/CLARA/GMM results should not be compared as final full-100k models.
- Total scheduled job runtime: 0.03 hours (108.7 seconds).

## Findings

- Best CLARA diagnostic: `text_pca64_lexical_calendar` / `clara_k16` silhouette 0.033399.
- CLARA used `sklearn_extra.cluster.CLARA` in the active environment.

## Answers

1. Full 100k jobs were MiniBatch k8/k16/k32 across the full fast variant list.
2. GMM, HDBSCAN, and CLARA were sample-level diagnostics with deterministic row_index samples.
3. For MiniBatch k16, `text_pca64_only` degraded versus the existing 768-d text-only baseline.
4. Lexical/calendar sometimes helped relative to PCA64, but not enough to beat the original text-only k16 baseline.
5. Publisher/stock features tended to hurt or collapse structure: they lowered MiniBatch k16, worsened GMM, and caused HDBSCAN high-coverage/two-cluster behavior in some variants.
6. The best aux_weight was algorithm-dependent, but `w010` was the strongest bounded HDBSCAN setting and competitive for MiniBatch; `w030` was often too heavy.
7. GMM on bounded samples preferred `text_pca64_only`/`gmm_k16`; richer metadata-heavy variants often produced weak or negative silhouettes.
8. HDBSCAN on bounded samples showed the central tradeoff: some all-aux variants had high silhouette at low coverage, while publisher/stock-heavy variants increased coverage but collapsed to very few clusters.
9. CLARA ran through the fallback path and remained a diagnostic, not a final full-100k result.
10. The run respected the 6-8h budget.
11. Text-only remains the conservative main model.
12. Multi-feature should be presented as a supplemental ablation/experimental extension, not as the primary model unless a tuned feature-weighting scheme later improves metrics and interpretability.

## Recommendation

Use text-only embedding as the main model if bounded ablation does not beat baseline metrics. Present multi-feature as supplemental evidence and future-work direction: tune feature weighting, supervised relevance weighting, or market-response labels.
