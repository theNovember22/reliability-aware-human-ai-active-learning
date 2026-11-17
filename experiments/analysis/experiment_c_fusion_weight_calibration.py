"""Explore train-only LOO calibration of machine/expert fusion weights."""

from pathlib import Path
import argparse
import json
import hashlib
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dr5.core import load_csv, metrics  # noqa: E402
from src.dr5.budget import fit_fusion, fit_fusion_loglinear  # noqa: E402

DATA = ROOT / "data" / "DR-5"
TRACE_PATH = ROOT / "results" / "experiment_c_reliability_weighted_fusion" / "acquisition_traces.json"
NEURAL_RESULTS = ROOT / "results" / "experiment_c_reliability_weighted_fusion" / "per_fold_budget.csv"
OUT = ROOT / "results" / "experiment_c_fusion_weight_calibration"
METHOD_FIXED = "queried-confusion fusion, AI weight=1"
METHOD_TUNED = "LOO-tuned AI and expert weights"
METRICS = ("accuracy", "kl_divergence")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--traces", type=Path, default=TRACE_PATH)
    p.add_argument("--neural-results", type=Path, default=NEURAL_RESULTS)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    return p.parse_args()


def measure(probabilities, labels):
    # KL(one-hot truth || prediction) is the categorical NLL for this task.
    result = metrics(torch.as_tensor(probabilities, dtype=torch.float64), labels)
    return {"accuracy": float(result["accuracy"]), "kl_divergence": float(result["nll"])}


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    trace_data = json.loads(args.traces.read_text())
    trace_by_key = {(r["fold"], r["seed"]): r for r in trace_data}
    neural = pd.read_csv(args.neural_results)
    neural = neural[(neural.method == "fixed_fusion_weight") & neural.fold.isin(args.folds) & neural.seed.isin(args.seeds)]
    rows, details = [], []
    for fold in args.folds:
        train = load_csv(args.data_dir / f"model_output_{fold}_train.csv")
        test = load_csv(args.data_dir / f"model_output_{fold}_test.csv")
        for seed in args.seeds:
            trace = trace_by_key[(fold, seed)]
            for round_data in trace["rounds"]:
                budget = round_data["labels_per_class"]
                ids = np.asarray(round_data["selected_ids"], dtype=int)
                y, h, ai = (train.labels.numpy()[ids], train.human.numpy()[ids], train.probs.numpy()[ids])

                # Both fusion models see exactly the same queried labels.
                fixed = fit_fusion(ai, y, h)
                tuned = fit_fusion_loglinear(ai, y, h)
                fixed_metrics = measure(fixed.predict(test.probs.numpy(), test.human.numpy()), test.labels)
                tuned_metrics = measure(tuned.predict(test.probs.numpy(), test.human.numpy()), test.labels)
                for method, fit, result in ((METHOD_FIXED, fixed, fixed_metrics),
                                            (METHOD_TUNED, tuned, tuned_metrics)):
                    rows.append({"fold": fold, "seed": seed, "labels_per_class": budget,
                                 "total_queries": len(ids), "method": method,
                                 "machine_strength": fit.machine_strength,
                                 "expert_strength": fit.strength, "prior_mass": fit.mass,
                                 **result})
                details.append({"fold": fold, "seed": seed, "labels_per_class": budget,
                                "selected_ids": ids.tolist(), "fixed_ai_strength": fixed.machine_strength,
                                "fixed_expert_strength": fixed.strength,
                                "tuned_ai_strength": tuned.machine_strength,
                                "tuned_expert_strength": tuned.strength,
                                "tuned_prior_mass": tuned.mass,
                                "loo_nll": tuned.loo_nll})

            # Previously run neural evaluator on these same acquired samples.
            existing = neural[(neural.fold == fold) & (neural.seed == seed)]
            for _, record in existing.iterrows():
                rows.append({"fold": fold, "seed": seed, "labels_per_class": int(record.labels_per_class),
                             "total_queries": int(record.total_queries),
                             "method": "neural evaluator, fixed product fusion",
                             "machine_strength": 1.0, "expert_strength": 1.0, "prior_mass": np.nan,
                             "accuracy": float(record.accuracy),
                             "kl_divergence": float(record.kl_divergence)})

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "per_fold_budget.csv", index=False)
    summary = frame.groupby(["method", "labels_per_class"])[list(METRICS)].agg(["mean", "std"])
    summary.to_csv(args.output / "summary.csv")
    pd.DataFrame(details).to_csv(args.output / "fitted_weights.csv", index=False)

    pair_rows, pair_summary = [], []
    comparisons = ((METHOD_TUNED, METHOD_FIXED),
                   (METHOD_TUNED, "neural evaluator, fixed product fusion"),
                   (METHOD_FIXED, "neural evaluator, fixed product fusion"))
    for left, right in comparisons:
        left_df = frame[frame.method == left].set_index(["fold", "seed", "labels_per_class"])
        right_df = frame[frame.method == right].set_index(["fold", "seed", "labels_per_class"])
        for budget in range(2, 9):
            l = left_df.xs(budget, level="labels_per_class")
            r = right_df.xs(budget, level="labels_per_class")
            diff = l[list(METRICS)] - r[list(METRICS)]
            for (fold, seed), values in diff.iterrows():
                for metric, value in values.items():
                    pair_rows.append({"comparison": f"{left} - {right}", "fold": fold,
                                      "seed": seed, "labels_per_class": budget,
                                      "total_queries": budget * 5, "metric": metric,
                                      "difference": float(value)})
            for metric in METRICS:
                values = diff[metric].to_numpy()
                pair_summary.append({"comparison": f"{left} - {right}",
                                     "labels_per_class": budget, "total_queries": budget * 5,
                                     "metric": metric, "mean_difference": float(values.mean()),
                                     "std_difference": float(values.std(ddof=1)),
                                     "paired_ttest_pvalue": float(ttest_rel(l[metric], r[metric]).pvalue),
                                     "n_paired_runs": len(values)})
    pd.DataFrame(pair_rows).to_csv(args.output / "paired_differences.csv", index=False)
    pd.DataFrame(pair_summary).to_csv(args.output / "paired_difference_summary.csv", index=False)
    (args.output / "acquisition_traces.json").write_text(json.dumps(details, indent=2))

    final = frame[frame.labels_per_class == 8]
    final_table = final.groupby("method")[list(METRICS)].agg(["mean", "std"])
    final_table.columns = [f"{metric}_{stat}" for metric, stat in final_table.columns]
    report = {
        "experiment": "C follow-up: LOO-calibrated AI/expert fusion weights",
        "question": "Can train-only leave-one-out selection of AI and queried-expert fusion weights improve final fusion accuracy/KL?",
        "selection": "Reuses the exact MWAC + AI entropy acquisition IDs from the completed Experiment B/C trace for every fold, seed and budget.",
        "methods": [METHOD_FIXED, METHOD_TUNED, "neural evaluator, fixed product fusion"],
        "weight_grid": {"machine_alpha": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        "expert_beta": [0.0, 0.5, 1.0, 1.5, 2.0], "prior_mass": [1.0, 5.0, 20.0]},
        "selection_criterion": "Leave-one-out NLL on acquired human-label pairs. Each held-out response is removed from both confusion counts and the shared accuracy prior.",
        "test_usage": "Test labels and test human responses are used only for the established final fusion evaluation. Neither enters weight fitting or acquisition.",
        "kl_definition": "KL(one-hot true label || prediction), equal to categorical NLL.",
        "final_40_query_summary": final_table.to_dict(orient="index"),
        "limitations": ["Previously inspected test folds are exploratory, not confirmatory.",
                        "The 15 runs share five test folds; SD and paired p-values are descriptive.",
                        "The LOO-selected weight grid is a new model variant; any further grid changes require nested training-only validation."],
        "source_sha256": {"experiment": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "budget_core": hashlib.sha256((ROOT / "src/dr5/budget.py").read_bytes()).hexdigest()},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(final_table.round(4).to_string())
    print(f"\nSaved fusion-weight calibration outputs to {args.output}")


if __name__ == "__main__":
    main(parse_args())
