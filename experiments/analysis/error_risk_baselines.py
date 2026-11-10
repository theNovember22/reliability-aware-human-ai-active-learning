"""Experiment A: compare error-risk scores on DR-5.

The learned score is fit using only each fold's training CSV. A stratified
training/validation split is used for development assessment. Existing test
CSVs are reported as exploratory because they have already been inspected.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "DR-5"
OUT = ROOT / "results" / "error_risk_baselines"
SEED = 42


def parse_vector(value):
    return np.fromstring(str(value).strip().strip("[]"), sep=",")


def load_fold(fold, split):
    path = DATA / f"model_output_{fold}_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file: {path}")
    frame = pd.read_csv(path)
    required = ["Logits", "True Label"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    probs = np.stack([parse_vector(value) for value in frame["Logits"]])
    labels = frame["True Label"].to_numpy(dtype=int)
    if probs.shape != (len(frame), 5):
        raise ValueError(f"{path}: expected five probabilities per row")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError(f"{path}: invalid probability values")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{path}: probability rows do not sum to one")
    if not np.isin(labels, np.arange(5)).all():
        raise ValueError(f"{path}: labels must be integers 0 through 4")

    pred = probs.argmax(axis=1)
    error = (pred != labels).astype(int)
    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1]
    margin = confidence - sorted_probs[:, -2]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    # Fixed, interpretable baseline selected a priori: one minus max probability.
    baseline_risks = {
        "confidence": 1.0 - confidence,
        "entropy": entropy,
        "margin": -margin,
    }
    # No Human Label is read or used as a feature.
    features = np.column_stack([probs, 1.0 - confidence, entropy, -margin])
    return {"error": error, "baseline_risks": baseline_risks, "features": features}


def ece_binary(target, score, bins=10):
    """Equal-width expected calibration error for predicted error probability."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = ((score >= edges[i]) & (score <= edges[i + 1])) if i == bins - 1 else (
            (score >= edges[i]) & (score < edges[i + 1])
        )
        if mask.any():
            total += mask.mean() * abs(target[mask].mean() - score[mask].mean())
    return float(total)


def score_metrics(target, score, calibrated_probability=False):
    result = {
        "n": int(len(target)),
        "error_rate": float(target.mean()),
        # Ranking scores are not probabilities. Only report probability
        # calibration metrics for the learned probability output.
        "brier": float(brier_score_loss(target, score)) if calibrated_probability else float("nan"),
        "ece_10bin": ece_binary(target, score) if calibrated_probability else float("nan"),
        "auroc": float(roc_auc_score(target, score)) if np.unique(target).size == 2 else float("nan"),
        "average_precision": float(average_precision_score(target, score)) if target.sum() else float("nan"),
    }
    # Fraction of errors remaining after referring the highest-risk cases.
    for fraction in (0.1, 0.2, 0.3):
        n_refer = int(np.ceil(fraction * len(target)))
        # Refer the highest-risk cases; retain the lowest-risk remainder.
        keep = np.argsort(score)[: len(target) - n_refer]
        result[f"residual_error_rate_referral_{int(fraction * 100)}pct"] = (
            float(target[keep].mean()) if len(keep) else float("nan")
        )
        result[f"coverage_after_referral_{int(fraction * 100)}pct"] = float(len(keep) / len(target))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    fold_notes = []
    for fold in range(5):
        train = load_fold(fold, "train")
        test = load_fold(fold, "test")
        dev_idx, val_idx = train_test_split(
            np.arange(len(train["error"])), test_size=0.20,
            random_state=SEED, stratify=train["error"],
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, random_state=SEED),
        )
        model.fit(train["features"][dev_idx], train["error"][dev_idx])

        for split_name, data, indices in (
            ("validation", train, val_idx),
            ("test_exploratory", test, np.arange(len(test["error"]))),
        ):
            target = data["error"][indices]
            scores = {name: values[indices] for name, values in data["baseline_risks"].items()}
            scores["learned_probability"] = model.predict_proba(data["features"][indices])[:, 1]
            for name, score in scores.items():
                rows.append({
                    "fold": fold,
                    "split": split_name,
                    "score": name,
                    **score_metrics(target, score, calibrated_probability=(name == "learned_probability")),
                })
        fold_notes.append({
            "fold": fold,
            "development_n": int(len(dev_idx)),
            "validation_n": int(len(val_idx)),
            "test_n": int(len(test["error"])),
            "validation_error_rate": float(train["error"][val_idx].mean()),
            "test_error_rate": float(test["error"].mean()),
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "per_fold.csv", index=False)
    metric_columns = [column for column in results.columns if column not in {"fold", "split", "score", "n"}]
    summary = results.groupby(["split", "score"])[metric_columns].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")
    report = {
        "experiment": "A: AI error-risk prediction",
        "target": "1 when argmax of stored AI probabilities differs from True Label",
        "candidate_score": "learned_probability: logistic regression over probabilities and fixed uncertainty features",
        "baseline_preferred_for_now": "confidence (1 - maximum class probability); previous validation results favored simple uncertainty baselines",
        "protocol": "Per fold, fit StandardScaler and fixed-C logistic regression on a stratified 80% subset of training CSV; assess on held-out 20% validation subset.",
        "test_status": "test_exploratory only: prior test inspection means it is not an untouched confirmatory set",
        "calibration": "Brier score and 10-bin equal-width ECE; ECE is bin-dependent",
        "ranking_and_triage": "AUROC, average precision, and residual error rate after referring the top 10/20/30% risk cases",
        "folds": fold_notes,
        "limitations": [
            "No untouched final test set is available; do not claim confirmatory superiority.",
            "CSV files contain no patient or image IDs, so patient-disjointness cannot be verified.",
            "The learned probability is an uncalibrated logistic regression estimate; calibration metrics describe its held-out validation performance.",
            "The candidate score is not established as superior; use validation results to choose and freeze a score before a new test evaluation.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(results.groupby(["split", "score"])[["auroc", "average_precision", "brier", "ece_10bin"]].mean().round(4))
    print(f"\nSaved results to: {OUT}")


if __name__ == "__main__":
    main()
