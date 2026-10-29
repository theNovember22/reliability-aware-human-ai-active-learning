
%cd /kaggle/working/reliability-aware-human-ai-active-learning

from pathlib import Path

script_path = Path("experiments/analysis/error_risk_baselines.py")
script_path.parent.mkdir(parents=True, exist_ok=True)

code = r'''
"""Exploratory AI-error detection baselines for DR-5.

Evaluates maximum-softmax-probability confidence, entropy, and margin.
Existing test folds were inspected in prior experiments; results are
exploratory and must not be treated as a confirmatory evaluation.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "DR-5"
OUT = ROOT / "results" / "error_risk_baselines"


def load_fold(fold):
    path = DATA / f"model_output_{fold}_test.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Missing test file: {path}")

    df = pd.read_csv(path)

    required = ["Logits", "True Label"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    probabilities = np.stack([
        np.fromstring(str(value).strip().strip("[]"), sep=",")
        for value in df["Logits"]
    ]).astype(np.float64)

    labels = df["True Label"].to_numpy(dtype=int)

    if probabilities.shape != (len(labels), 5):
        raise ValueError(
            f"Expected {len(labels)} rows of 5 probabilities; "
            f"got shape {probabilities.shape} in {path}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities in {path}")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"Probabilities outside [0, 1] in {path}")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(
            f"Rows do not sum to 1 in {path}. "
            "Check the stored probability format; do not softmax blindly."
        )

    if not np.isin(labels, np.arange(5)).all():
        raise ValueError(f"Unexpected class labels in {path}")

    return probabilities, labels


def confidence_ece(confidence, correct, bins=15):
    """Expected calibration error for confidence predicting correctness."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for index in range(bins):
        if index == bins - 1:
            mask = (
                (confidence >= edges[index])
                & (confidence <= edges[index + 1])
            )
        else:
            mask = (
                (confidence >= edges[index])
                & (confidence < edges[index + 1])
            )

        if mask.any():
            ece += mask.mean() * abs(
                correct[mask].mean() - confidence[mask].mean()
            )

    return float(ece)


def evaluate_fold(fold):
    probabilities, labels = load_fold(fold)

    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(int)
    errors = 1 - correct

    sorted_probabilities = np.sort(probabilities, axis=1)
    top_probability = sorted_probabilities[:, -1]
    second_probability = sorted_probabilities[:, -2]

    margin = top_probability - second_probability

    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    )

    # Higher risk scores must indicate greater estimated error risk.
    risk_scores = {
        "confidence_risk": 1.0 - top_probability,
        "entropy_risk": entropy,
        "margin_risk": -margin,
    }

    row = {
        "fold": fold,
        "n": len(labels),
        "ai_accuracy": float(correct.mean()),
        "error_rate": float(errors.mean()),
        "error_count": int(errors.sum()),
        "confidence_brier": float(
            brier_score_loss(correct, top_probability)
        ),
        "confidence_ece": confidence_ece(top_probability, correct),
    }

    for name, risk in risk_scores.items():
        row[f"{name}_auroc"] = (
            float(roc_auc_score(errors, risk))
            if len(np.unique(errors)) == 2
            else float("nan")
        )

        row[f"{name}_average_precision"] = (
            float(average_precision_score(errors, risk))
            if errors.sum() > 0
            else float("nan")
        )

    # Random-ranking reference: theoretical AUROC is 0.5.
    row["random_auroc_reference"] = 0.5
    row["random_ap_reference"] = float(errors.mean())

    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [evaluate_fold(fold) for fold in range(5)]
    results = pd.DataFrame(rows)

    results.to_csv(OUT / "per_fold.csv", index=False)

    metric_columns = [
        column for column in results.columns
        if column not in ("fold", "n", "error_count")
    ]

    summary = results[metric_columns].agg(["mean", "std"]).T
    summary.columns = ["mean_across_folds", "std_across_folds"]
    summary.to_csv(OUT / "summary.csv")

    report = {
        "purpose": "Exploratory AI-error detection baseline evaluation",
        "data": "Existing DR-5 test folds",
        "fold_results": rows,
        "mean_metrics": {
            key: float(value)
            for key, value in results[metric_columns].mean().items()
        },
        "limitations": [
            "The existing test folds were inspected in prior experiments.",
            "These results are exploratory, not confirmatory.",
            "No learned risk model is fitted in this script.",
            "Patient-level split independence is not established by this audit.",
            "Mutual information is not included because this script does not "
            "load MC Dropout predictions.",
        ],
    }

    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True)
    )

    print("\n=== EXPERIMENT A: PER-FOLD RESULTS ===")
    print(results.round(4).to_string(index=False))

    print("\n=== MEAN AND STANDARD DEVIATION ACROSS FOLDS ===")
    print(summary.round(4).to_string())

    print(f"\nSaved results to: {OUT}")


if __name__ == "__main__":
    main()
'''

script_path.write_text(code, encoding="utf-8")
print(f"Created: {script_path}")

# Execute the script.
import subprocess
import sys

subprocess.run(
    [sys.executable, str(script_path)],
    check=True,
)
