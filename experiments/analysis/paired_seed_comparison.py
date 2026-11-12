"""Paired-seed comparison of the corrected baseline and reliability method.

Uses the already completed DR-5 runs. Pairing is by fold and seed; uncertainty
intervals summarize the five fold-level mean differences, since seeds within a
fold share the same held-out examples.
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("accuracy", "macro_f1", "qwk", "nll", "ece")
LOWER_IS_BETTER = {"nll", "ece"}


def paired_summary(baseline, proposed, labels_per_class):
    baseline = baseline.loc[baseline.labels_per_class == labels_per_class].copy()
    proposed = proposed.loc[proposed.labels_per_class == labels_per_class].copy()
    keys = ["fold", "seed"]
    if baseline.duplicated(keys).any() or proposed.duplicated(keys).any():
        raise ValueError("Each method must have one row per fold/seed at the chosen budget")
    joined = baseline.merge(proposed, on=keys, suffixes=("_baseline", "_proposed"),
                            validate="one_to_one")
    if len(joined) != len(baseline) or len(joined) != len(proposed):
        raise ValueError("Baseline and proposed runs do not have identical fold/seed pairs")
    if not np.array_equal(joined.total_human_queries_baseline,
                          joined.total_human_queries_proposed):
        raise ValueError("Methods have unequal total annotation costs")
    if not np.array_equal(joined.active_human_queries_baseline,
                          joined.active_human_queries_proposed):
        raise ValueError("Methods have unequal active annotation costs")

    detail = joined[keys + ["total_human_queries_baseline"]].rename(
        columns={"total_human_queries_baseline": "total_human_queries"})
    summary_rows = []
    for metric in METRICS:
        base = joined[f"{metric}_baseline"]
        prop = joined[f"{metric}_proposed"]
        delta = prop - base
        detail[f"{metric}_baseline"] = base
        detail[f"{metric}_proposed"] = prop
        detail[f"{metric}_difference_proposed_minus_baseline"] = delta
        fold_delta = delta.groupby(joined.fold).mean()
        critical = t.ppf(0.975, df=len(fold_delta) - 1)
        half_width = critical * fold_delta.std(ddof=1) / np.sqrt(len(fold_delta))
        summary_rows.append({
            "metric": metric,
            "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
            "baseline_mean": float(base.mean()),
            "proposed_mean": float(prop.mean()),
            "mean_paired_difference": float(delta.mean()),
            "paired_run_sd": float(delta.std(ddof=1)),
            "fold_mean_ci95_low": float(fold_delta.mean() - half_width),
            "fold_mean_ci95_high": float(fold_delta.mean() + half_width),
            "folds_proposed_better": int((fold_delta < 0).sum() if metric in LOWER_IS_BETTER
                                          else (fold_delta > 0).sum()),
            "folds": int(len(fold_delta)),
            "paired_runs": int(len(delta)),
        })
    return detail, pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/baseline/metrics.csv")
    parser.add_argument("--proposed", type=Path, default=ROOT / "results/reliability/metrics.csv")
    parser.add_argument("--labels-per-class", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results/paired_seed_comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    detail, summary = paired_summary(pd.read_csv(args.baseline), pd.read_csv(args.proposed),
                                     args.labels_per_class)
    detail.to_csv(args.output / "paired_runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    accuracy = summary.set_index("metric").loc["accuracy"]
    report = [
        "# Paired-seed comparison: corrected MWAC vs reliability-aware MWAC",
        "",
        f"Endpoint: {args.labels_per_class} queried labels per class; "
        f"{len(detail)} matched fold/seed runs; "
        f"{int(detail.total_human_queries.iloc[0])} total human queries per run.",
        "",
        f"Mean accuracy was {accuracy.baseline_mean:.3%} for the baseline and "
        f"{accuracy.proposed_mean:.3%} for the proposed method. The paired mean "
        f"difference was {accuracy.mean_paired_difference:+.3%} "
        f"(95% fold-level t interval {accuracy.fold_mean_ci95_low:+.3%} to "
        f"{accuracy.fold_mean_ci95_high:+.3%}); the proposed method was higher "
        f"in {int(accuracy.folds_proposed_better)} of {int(accuracy.folds)} folds.",
        "",
        "Seeds are paired within each fold and share its test examples. The "
        "interval therefore uses five fold-level averages, not 15 independent "
        "test estimates. Results are exploratory because these folds have been "
        "inspected previously and the source data lacks patient identifiers and "
        "backbone provenance. A gain in accuracy alone does not establish "
        "reliability; the separate error-risk and calibration analyses remain "
        "necessary. The proposed method also does not beat the random-acquisition "
        "control in the existing comparison.",
        "",
        "Full matched-run results: `paired_runs.csv`; metric summary: `summary.csv`.",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
