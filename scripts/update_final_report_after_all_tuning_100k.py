#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DOCS = {
    "tuning": "final_tuning_update.md",
    "selection": "final_model_selection_updated.md",
    "event": "final_event_model_update.md",
    "conclusion": "report_ready_conclusion_updated.md",
}
OUTPUT_METRICS_CSV = "final_model_comparison_updated.csv"
OUTPUT_METRICS_MD = "final_model_comparison_updated.md"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Update final 100k report docs after all tuning and event validation.")
    p.add_argument("--runs-root", default="report/runs/100k")
    p.add_argument("--final-analysis", default="report/runs/100k/_final_analysis")
    p.add_argument("--k-exploration", default="report/runs/100k/_k_metric_exploration")
    p.add_argument("--representation-tuning", default="report/runs/100k/_representation_tuning")
    p.add_argument("--hdbscan-compact", default="report/runs/100k/_hdbscan_tuning_compact")
    p.add_argument("--hdbscan-validation", default="report/runs/100k/_hdbscan_event_validation")
    p.add_argument("--clara-true", default="report/runs/100k/_multifeature/bounded_ablation/clara_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_write_path(path: Path, final_analysis: Path, overwrite: bool) -> Path:
    if not is_relative_to(path, final_analysis):
        raise ValueError(f"Refusing to write outside final analysis root: {path}")
    if path.exists() and path.is_file() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input CSV: {path}")
    return pd.read_csv(path)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append([format_value(row[col]) for col in cols])
    widths = [len(str(col)) for col in cols]
    for row in rows:
        widths = [max(widths[i], len(row[i])) for i in range(len(cols))]
    header = "| " + " | ".join(str(cols[i]).ljust(widths[i]) for i in range(len(cols))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def format_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.6g}"
    return str(value)


def first_row(df: pd.DataFrame, description: str, **equals) -> pd.Series:
    sub = df.copy()
    for col, value in equals.items():
        if col not in sub.columns:
            raise KeyError(f"{description}: missing column {col}")
        sub = sub[sub[col].astype(str) == str(value)]
    if sub.empty:
        raise ValueError(f"Could not find row for {description}: {equals}")
    return sub.iloc[0]


def mean_stability(stability: pd.DataFrame, feature_space: str, k: int) -> tuple[float, float]:
    if stability.empty or not {"feature_space", "k", "ARI", "NMI"}.issubset(stability.columns):
        return math.nan, math.nan
    sub = stability[(stability["feature_space"].astype(str) == feature_space) & (pd.to_numeric(stability["k"], errors="coerce") == k)]
    if sub.empty:
        return math.nan, math.nan
    return float(sub["ARI"].mean()), float(sub["NMI"].mean())


def numeric(row: pd.Series, *names: str) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    return math.nan


def text_value(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return str(row[name])
    return ""


def minibatch_row(
    row: pd.Series,
    role: str,
    selected: str,
    selected_role: str,
    reason: str,
    warnings: str,
    stability: tuple[float, float],
) -> dict:
    k = int(numeric(row, "k", "n_clusters"))
    feature_space = text_value(row, "feature_space", "variant")
    return {
        "model_role": role,
        "selected_for_final": selected,
        "selected_role": selected_role,
        "feature_space": feature_space,
        "algorithm": f"minibatch_k{k}",
        "params": f"k={k}",
        "fit_n_rows": int(numeric(row, "fit_n_rows")),
        "eval_n_rows": int(numeric(row, "eval_n_rows")),
        "is_sample_run": False,
        "coverage_pct": numeric(row, "coverage_pct"),
        "n_clusters": int(numeric(row, "n_clusters", "k")),
        "silhouette_cosine": numeric(row, "silhouette_cosine", "silhouette"),
        "silhouette_euclidean": numeric(row, "silhouette_euclidean"),
        "dbi": numeric(row, "dbi"),
        "negative_silhouette_pct": numeric(row, "negative_silhouette_pct"),
        "largest_cluster_pct": numeric(row, "largest_cluster_pct"),
        "min_cluster_size": numeric(row, "min_cluster_size"),
        "stability_ari_mean": stability[0],
        "stability_nmi_mean": stability[1],
        "warnings": warnings,
        "reason": reason,
    }


def gmm_row(row: pd.Series) -> dict:
    return {
        "model_role": "compact probabilistic baseline",
        "selected_for_final": "yes",
        "selected_role": "compact_probabilistic_baseline",
        "feature_space": "text_768_original",
        "algorithm": "gmm_k8",
        "params": "k=8; covariance=diag",
        "fit_n_rows": int(numeric(row, "fit_n_rows", "n_points_labeled")),
        "eval_n_rows": int(numeric(row, "eval_n_rows", "n_embedding_points")),
        "is_sample_run": False,
        "coverage_pct": numeric(row, "coverage_pct"),
        "n_clusters": int(numeric(row, "n_clusters")),
        "silhouette_cosine": numeric(row, "silhouette_cosine", "silhouette"),
        "silhouette_euclidean": numeric(row, "silhouette_euclidean"),
        "dbi": numeric(row, "dbi"),
        "negative_silhouette_pct": math.nan,
        "largest_cluster_pct": math.nan,
        "min_cluster_size": math.nan,
        "stability_ari_mean": math.nan,
        "stability_nmi_mean": math.nan,
        "warnings": "text-only probabilistic reference, not the selected main model",
        "reason": "Kept as a compact probabilistic baseline for comparison.",
    }


def hdbscan_row(row: pd.Series, role: str, selected: str, selected_role: str, reason: str, warnings: str) -> dict:
    mcs = text_value(row, "min_cluster_size")
    ms = text_value(row, "min_samples")
    method = text_value(row, "cluster_selection_method")
    params = f"min_cluster_size={mcs}; min_samples={ms}; method={method}; metric=euclidean"
    return {
        "model_role": role,
        "selected_for_final": selected,
        "selected_role": selected_role,
        "feature_space": text_value(row, "feature_space"),
        "algorithm": "hdbscan",
        "params": params,
        "fit_n_rows": int(numeric(row, "fit_n_rows")),
        "eval_n_rows": int(numeric(row, "metric_sample_size_used", "fit_n_rows")),
        "is_sample_run": not bool(row.get("is_full_100k", False)),
        "coverage_pct": numeric(row, "coverage_pct"),
        "n_clusters": int(numeric(row, "n_clusters")),
        "silhouette_cosine": numeric(row, "silhouette_cosine"),
        "silhouette_euclidean": numeric(row, "silhouette_euclidean"),
        "dbi": numeric(row, "dbi"),
        "negative_silhouette_pct": numeric(row, "negative_silhouette_pct"),
        "largest_cluster_pct": numeric(row, "largest_cluster_pct"),
        "min_cluster_size": numeric(row, "min_assigned_cluster_size", "min_cluster_size"),
        "stability_ari_mean": math.nan,
        "stability_nmi_mean": math.nan,
        "warnings": warnings,
        "reason": reason,
    }


def clara_row(row: pd.Series) -> dict:
    return {
        "model_role": "CLARA true diagnostic baseline",
        "selected_for_final": "no",
        "selected_role": "diagnostic_baseline",
        "feature_space": text_value(row, "feature_space", "variant"),
        "algorithm": text_value(row, "algorithm"),
        "params": "sklearn_extra.cluster.CLARA; n_sampling=2048; n_sampling_iter=5",
        "fit_n_rows": int(numeric(row, "fit_n_rows")),
        "eval_n_rows": int(numeric(row, "eval_n_rows")),
        "is_sample_run": bool(row.get("is_sample_run", True)),
        "coverage_pct": numeric(row, "coverage_pct"),
        "n_clusters": int(numeric(row, "n_clusters")),
        "silhouette_cosine": numeric(row, "silhouette", "silhouette_cosine"),
        "silhouette_euclidean": math.nan,
        "dbi": numeric(row, "dbi"),
        "negative_silhouette_pct": math.nan,
        "largest_cluster_pct": math.nan,
        "min_cluster_size": math.nan,
        "stability_ari_mean": math.nan,
        "stability_nmi_mean": math.nan,
        "warnings": "true CLARA run, no fallback; sample-level diagnostic only",
        "reason": "Verified as a diagnostic baseline but clearly weaker than MiniBatch on silhouette and DBI.",
    }


def build_comparison(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    k_summary = read_required_csv(Path(args.k_exploration) / "metrics" / "k_sweep_summary.csv")
    k_stability = read_required_csv(Path(args.k_exploration) / "metrics" / "k_sweep_stability.csv")
    rep_summary = read_required_csv(Path(args.representation_tuning) / "metrics" / "representation_tuning_summary.csv")
    rep_stability = read_required_csv(Path(args.representation_tuning) / "metrics" / "representation_tuning_stability.csv")
    hdbscan_summary = read_required_csv(Path(args.hdbscan_validation) / "metrics" / "hdbscan_event_validation_summary.csv")
    hdbscan_compact = read_required_csv(Path(args.hdbscan_compact) / "metrics" / "hdbscan_tuning_summary.csv")
    clara_summary = read_required_csv(Path(args.clara_true) / "metrics" / "bounded_ablation_summary.csv")
    old_final = read_required_csv(Path(args.final_analysis) / "metrics" / "final_model_comparison.csv")

    text_k16 = first_row(k_summary, "text_768_original minibatch_k16", feature_space="text_768_original", k=16)
    lexical_k32 = first_row(k_summary, "text_pca64_lexical minibatch_k32", feature_space="text_pca64_lexical", k=32)
    lexical_k40 = first_row(k_summary, "text_pca64_lexical minibatch_k40", feature_space="text_pca64_lexical", k=40)
    pca64_k96 = first_row(k_summary, "text_pca64_only minibatch_k96", feature_space="text_pca64_only", k=96)
    rep_l2_k40 = first_row(rep_summary, "text_768_l2 minibatch_k40", feature_space="text_768_l2", k=40)
    gmm = first_row(old_final, "text-only gmm_k8", model_role="compact probabilistic baseline")
    hdbscan_old = first_row(hdbscan_summary, "historical text-only hdbscan", model_role="historical_text_only_baseline")
    hdbscan_new = first_row(hdbscan_summary, "selected PCA64 HDBSCAN", model_role="candidate_validation")
    clara_best = first_row(clara_summary, "best CLARA true", feature_space="text_pca64_lexical_calendar", algorithm="clara_k16")

    rows = [
        minibatch_row(
            text_k16,
            "conservative full-coverage baseline",
            "yes",
            "conservative_full_coverage_baseline",
            "Stable 768D text embedding baseline with complete coverage and moderate k.",
            text_value(text_k16, "notes"),
            mean_stability(k_stability, "text_768_original", 16),
        ),
        minibatch_row(
            lexical_k32,
            "previous experimental candidate",
            "no",
            "previous_experimental_candidate",
            "Earlier lexical PCA64 candidate; superseded by k40 after k-sweep.",
            text_value(lexical_k32, "notes"),
            mean_stability(k_stability, "text_pca64_lexical", 32),
        ),
        minibatch_row(
            lexical_k40,
            "best experimental full-coverage model",
            "yes",
            "best_experimental_full_coverage_model",
            "Best balance of silhouette, DBI, cluster size, fragmentation risk, and interpretability.",
            text_value(lexical_k40, "notes"),
            mean_stability(k_stability, "text_pca64_lexical", 40),
        ),
        minibatch_row(
            pca64_k96,
            "high-silhouette risky reference",
            "no",
            "risk_reference",
            "Not selected despite strong silhouette/DBI because k96 increases fragmentation and metadata-driven cluster risk.",
            text_value(pca64_k96, "notes"),
            mean_stability(k_stability, "text_pca64_only", 96),
        ),
        minibatch_row(
            rep_l2_k40,
            "tuned representation candidate, not selected",
            "no",
            "representation_tuning_reference",
            "L2 tuning improves DBI/stability in places but has lower silhouette and very high negative silhouette.",
            text_value(rep_l2_k40, "notes"),
            mean_stability(rep_stability, "text_768_l2", 40),
        ),
        gmm_row(gmm),
        hdbscan_row(
            hdbscan_old,
            "previous event model baseline",
            "no",
            "previous_event_dense_detection_model",
            "Previous text-only dense-event baseline, replaced after PCA64 validation.",
            "DBI remains slightly better than new full PCA64 HDBSCAN, but silhouette and event granularity are weaker.",
        ),
        hdbscan_row(
            hdbscan_new,
            "selected event/dense detection model",
            "yes",
            "event_dense_detection_model",
            "Official event/dense detection model after full 100k validation.",
            "DBI full-run is slightly worse than text-only baseline; 768D HDBSCAN grid timed out.",
        ),
        clara_row(clara_best),
    ]
    comparison = pd.DataFrame(rows)
    inputs = {
        "k_summary": k_summary,
        "representation_summary": rep_summary,
        "hdbscan_summary": hdbscan_summary,
        "hdbscan_compact": hdbscan_compact,
        "clara_summary": clara_summary,
        "old_final": old_final,
    }
    return comparison, inputs


def by_role(comparison: pd.DataFrame, role: str) -> pd.Series:
    return comparison[comparison["model_role"] == role].iloc[0]


def write_docs(comparison: pd.DataFrame, inputs: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    final_root = Path(args.final_analysis)
    docs_dir = final_root / "docs"
    metrics_dir = final_root / "metrics"
    csv_path = check_write_path(metrics_dir / OUTPUT_METRICS_CSV, final_root, args.overwrite)
    md_path = check_write_path(metrics_dir / OUTPUT_METRICS_MD, final_root, args.overwrite)
    comparison.to_csv(csv_path, index=False)

    compact_completed = len(inputs["hdbscan_compact"])
    compact_spaces = ", ".join(sorted(inputs["hdbscan_compact"]["feature_space"].astype(str).unique()))
    read_sources = [
        str(Path(args.k_exploration)),
        str(Path(args.representation_tuning)),
        str(Path(args.hdbscan_compact)),
        str(Path(args.hdbscan_validation)),
        str(Path(args.clara_true)),
        str(Path(args.final_analysis)),
    ]

    md_lines = [
        "# Final Model Comparison Updated",
        "",
        dataframe_to_markdown(comparison),
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    conservative = by_role(comparison, "conservative full-coverage baseline")
    experimental = by_role(comparison, "best experimental full-coverage model")
    previous_exp = by_role(comparison, "previous experimental candidate")
    risky = by_role(comparison, "high-silhouette risky reference")
    rep = by_role(comparison, "tuned representation candidate, not selected")
    old_event = by_role(comparison, "previous event model baseline")
    new_event = by_role(comparison, "selected event/dense detection model")
    clara = by_role(comparison, "CLARA true diagnostic baseline")

    tuning_lines = [
        "# Final Tuning Update",
        "",
        "## Inputs Read",
        "",
        *[f"- `{src}`" for src in read_sources],
        "",
        "## Updated Decisions",
        "",
        f"- Conservative full-coverage baseline: `{conservative['feature_space']} + {conservative['algorithm']}`.",
        f"- Best experimental full-coverage model: `{experimental['feature_space']} + {experimental['algorithm']}`.",
        f"- Event/dense detection model: `{new_event['feature_space']} + HDBSCAN(min_cluster_size=30, min_samples=20, leaf)`.",
        f"- High-silhouette risky reference: `{risky['feature_space']} + {risky['algorithm']}` remains a reference only.",
        f"- CLARA true diagnostic baseline: `{clara['feature_space']} + {clara['algorithm']}`.",
        "",
        "## Evidence Scope",
        "",
        f"- K-sweep rows read: {len(inputs['k_summary'])}.",
        f"- Representation tuning rows read: {len(inputs['representation_summary'])}.",
        f"- Compact HDBSCAN rows completed: {compact_completed}; feature spaces completed include {compact_spaces}.",
        "- Raw 768D HDBSCAN compact-grid rows remain missing because `text_768_original` and `text_768_l2` timed out.",
        "- CLARA true was run through `sklearn_extra.cluster.CLARA`; it is not a fallback result.",
    ]
    check_write_path(docs_dir / OUTPUT_DOCS["tuning"], final_root, args.overwrite).write_text(
        "\n".join(tuning_lines) + "\n", encoding="utf-8"
    )

    selection_lines = [
        "# Final Model Selection Updated",
        "",
        "## Conservative Full-Coverage Baseline",
        "",
        f"`{conservative['feature_space']} + {conservative['algorithm']}` remains the conservative baseline because it uses the original 768D text embedding, covers all 100k rows, and avoids the additional assumptions introduced by PCA compression or auxiliary lexical features. Its silhouette is modest, but it is stable enough to anchor comparisons.",
        "",
        "## Best Experimental Full-Coverage Model",
        "",
        f"`{experimental['feature_space']} + {experimental['algorithm']}` is selected as the best experimental full-coverage model. It improves the MiniBatch result after k-sweep and gives the best practical balance among silhouette, DBI, cluster-size distribution, fragmentation risk, and interpretability.",
        "",
        "## Previous Experimental Candidate",
        "",
        f"`{previous_exp['feature_space']} + {previous_exp['algorithm']}` is retained as the previous candidate, but k40 supersedes it after the broader k-sweep.",
        "",
        "## Why k96 Is Not Selected",
        "",
        f"`{risky['feature_space']} + {risky['algorithm']}` has attractive separation metrics, but it is not selected because k is too large for the final model role: it creates smaller clusters, increases fragmentation risk, and carries more publisher/metadata-driven warning risk.",
        "",
        "## Representation Tuning",
        "",
        f"`{rep['feature_space']} + {rep['algorithm']}` improves DBI/stability in parts of the tuning grid, but it does not replace the experimental model because silhouette is lower and negative silhouette remains very high (`{format_value(rep['negative_silhouette_pct'])}%`).",
        "",
        "## Diagnostic Baselines",
        "",
        f"`text-only gmm_k8` remains a compact probabilistic baseline. `{clara['feature_space']} + {clara['algorithm']}` is reflected as a true CLARA diagnostic baseline, but it is weaker than MiniBatch and is not selected.",
    ]
    check_write_path(docs_dir / OUTPUT_DOCS["selection"], final_root, args.overwrite).write_text(
        "\n".join(selection_lines) + "\n", encoding="utf-8"
    )

    event_lines = [
        "# Final Event Model Update",
        "",
        "## Decision",
        "",
        "`text_pca64_only + HDBSCAN(min_cluster_size=30, min_samples=20, cluster_selection_method=leaf)` is now the official event/dense detection model.",
        "",
        "## Baseline vs Updated Event Model",
        "",
        f"- Previous baseline: `text-only hdbscan_minsize50`, coverage `{format_value(old_event['coverage_pct'])}%`, clusters `{format_value(old_event['n_clusters'])}`, silhouette cosine `{format_value(old_event['silhouette_cosine'])}`, DBI `{format_value(old_event['dbi'])}`.",
        f"- Updated model: `text_pca64_only + HDBSCAN(mcs=30, ms=20, leaf)`, coverage `{format_value(new_event['coverage_pct'])}%`, noise `{format_value(100.0 - float(new_event['coverage_pct']))}%`, clusters `{format_value(new_event['n_clusters'])}`, silhouette cosine `{format_value(new_event['silhouette_cosine'])}`, silhouette euclidean `{format_value(new_event['silhouette_euclidean'])}`, DBI `{format_value(new_event['dbi'])}`, negative silhouette `{format_value(new_event['negative_silhouette_pct'])}%`.",
        "- The updated HDBSCAN model is not a full-coverage clustering model. It is selected specifically for dense/event detection.",
        "- The new model improves silhouette and negative-silhouette behavior while retaining coverage close to the text-only HDBSCAN baseline.",
        "- Caveat: full-run DBI is slightly worse than the text-only baseline (`1.026005` vs `0.986080`).",
        "- Caveat: the raw 768D HDBSCAN grid timed out, so this conclusion is limited to the validated PCA64 path and does not prove that 768D cannot improve it.",
    ]
    check_write_path(docs_dir / OUTPUT_DOCS["event"], final_root, args.overwrite).write_text(
        "\n".join(event_lines) + "\n", encoding="utf-8"
    )

    conclusion_lines = [
        "# Kết luận cập nhật sẵn cho báo cáo",
        "",
        "Sau các bước representation tuning, k-sweep, kiểm chứng CLARA true và validation riêng cho HDBSCAN, kết quả cho thấy embedding văn bản gốc vẫn là baseline ổn định nhất cho vai trò conservative full-coverage. Mô hình `text_768_original + minibatch_k16` được giữ làm mốc so sánh chính vì bao phủ toàn bộ 100k quan sát và không phụ thuộc vào đặc trưng phụ.",
        "",
        "Với nhóm mô hình full-coverage thực nghiệm, đặc trưng lexical multi-feature giúp cải thiện MiniBatch khi chọn số cụm hợp lý. Mô hình `text_pca64_lexical + minibatch_k40` được chọn làm experimental full-coverage model vì cân bằng tốt giữa silhouette, DBI, phân bố kích thước cụm, rủi ro fragmentation và khả năng diễn giải. Ngược lại, `text_pca64_only + minibatch_k96` không được chọn dù có một số chỉ số tách cụm tốt, vì k quá lớn dễ tạo cụm nhỏ, làm tăng fragmentation và rủi ro cụm bị chi phối bởi metadata/publisher.",
        "",
        "Representation tuning cho thấy L2-normalization có thể cải thiện một phần stability và DBI, đặc biệt với `text_768_l2 + minibatch_k40`, nhưng mức silhouette thấp hơn và tỷ lệ negative silhouette cao khiến hướng này chưa đủ để thay thế mô hình experimental đã chọn.",
        "",
        "Đối với dense/event detection, mô hình được cập nhật chính thức sang `text_pca64_only + HDBSCAN(min_cluster_size=30, min_samples=20, leaf)`. Mô hình này chạy được trên full 100k, đạt coverage 20.119%, 219 cụm, silhouette cosine 0.608580, negative silhouette 0.69%, không có collapse warning và không quá sparse. Đây không phải mô hình full-coverage, mà là bộ phát hiện các cụm sự kiện dày và rõ hơn.",
        "",
        "CLARA true đã được kiểm chứng bằng `sklearn_extra.cluster.CLARA` và không dùng fallback. Kết quả tốt nhất là `text_pca64_lexical_calendar + clara_k16`, nhưng silhouette khoảng 0.033399 và DBI khoảng 4.301401 cho thấy CLARA chỉ nên giữ vai trò diagnostic baseline, yếu hơn MiniBatch cho kết luận chính.",
        "",
        "Các hạn chế cần nêu rõ gồm: stability chưa quá cao, một số cụm có publisher dominance, lưới HDBSCAN 768D bị timeout nên kết luận event model chỉ áp dụng cho validated PCA64 path, và thí nghiệm chưa có nhãn market-response hoặc multimodal thực sự để đánh giá tác động thị trường.",
    ]
    check_write_path(docs_dir / OUTPUT_DOCS["conclusion"], final_root, args.overwrite).write_text(
        "\n".join(conclusion_lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_root = Path(args.runs_root)
    final_analysis = Path(args.final_analysis)
    if not is_relative_to(final_analysis, runs_root):
        raise ValueError(f"--final-analysis must stay under --runs-root: {final_analysis}")
    comparison, inputs = build_comparison(args)
    write_docs(comparison, inputs, args)
    print(f"Wrote updated final docs and metrics to {final_analysis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
