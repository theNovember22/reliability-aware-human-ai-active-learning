"""Experiment A: learned AI-error risk estimation on DR-5.

Inputs never include Human Label.
Model selection uses only an internal development/validation split.
Existing test folds are evaluated for exploratory comparison only.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "DR-5"
OUT = ROOT / "results" / "learned_error_risk"
SEED = 42


def parse_vector(value):
    """Parse a comma-separated vector stored in a CSV cell."""
    return np.fromstring(str(value).strip().strip("[]"), sep=",")


def load_split(fold, split):
    path = DATA / f"model_output_{fold}_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = ["Logits", "True Label", "Feat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    probs = np.stack([parse_vector(v) for v in df["Logits"]]).astype(float)
    features = np.stack([parse_vector(v) for v in df["Feat"]]).astype(float)
    labels = df["True Label"].to_numpy(dtype=int)

    if probs.shape != (len(df), 5):
        raise ValueError(f"{path}: expected five probabilities per row")
    if features.shape != (len(df), 512):
        raise ValueError(f"{path}: expected 512 features per row")
    if not np.isfinite(probs).all() or not np.isfinite(features).all():
        raise ValueError(f"{path}: NaN or infinite values found")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError(f"{path}: probabilities outside [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{path}: probability rows do not sum to 1")
    if not np.isin(labels, np.arange(5)).all():
        raise ValueError(f"{path}: unexpected true-label values")

    # Target: 1 means AI is wrong; 0 means AI is correct.
    error = (probs.argmax(axis=1) != labels).astype(int)

    confidence = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropy = -np.sum(
        probs * np.log(np.clip(probs, 1e-12, 1.0)),
        axis=1,
    )

    uncertainty = np.column_stack([
        1.0 - confidence,
        entropy,
        -margin,
    ])

    combined = np.column_stack([probs, uncertainty, features])

    # Human Label is deliberately never read as a model input.
    return {
        "uncertainty": uncertainty,
        "features": features,
        "combined": combined,
        "error": error,
        "confidence_risk": 1.0 - confidence,
        "entropy_risk": entropy,
        "margin_risk": -margin,
    }


def calculate_metrics(y, risk):
    result = {
        "error_rate": float(np.mean(y)),
        "n": int(len(y)),
    }

    if len(np.unique(y)) < 2:
        result["auroc"] = float("nan")
    else:
        result["auroc"] = float(roc_auc_score(y, risk))

    result["average_precision"] = (
        float(average_precision_score(y, risk))
        if np.sum(y) > 0 else float("nan")
    )

    return result


def fit_model(x_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=3000,
            random_state=SEED,
        ),
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model_set(fold, split_name, data, indices, models):
    y = data["error"][indices]
    rows = []

    # Untrained standard uncertainty baselines.
    baselines = {
        "baseline_confidence": data["confidence_risk"][indices],
        "baseline_entropy": data["entropy_risk"][indices],
        "baseline_margin": data["margin_risk"][indices],
    }

    for name, risk in baselines.items():
        row = {
            "fold": fold,
            "split": split_name,
            "model": name,
            **calculate_metrics(y, risk),
        }
        rows.append(row)

    for name, model in models.items():
        x = data[name][indices]
        risk = model.predict_proba(x)[:, 1]

        row = {
            "fold": fold,
            "split": split_name,
            "model": f"learned_{name}",
            **calculate_metrics(y, risk),
            "brier": float(brier_score_loss(y, risk)),
        }
        rows.append(row)

    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    fold_notes = []

    for fold in range(5):
        train_data = load_split(fold, "train")
        test_data = load_split(fold, "test")

        y = train_data["error"]
        indices = np.arange(len(y))

        dev_idx, val_idx = train_test_split(
            indices,
            test_size=0.20,
            random_state=SEED,
            stratify=y,
        )

        if len(np.unique(y[dev_idx])) < 2:
            raise ValueError(f"Fold {fold}: development split has one class")

        models = {}
        for name in ["uncertainty", "features", "combined"]:
            models[name] = fit_model(
                train_data[name][dev_idx],
                y[dev_idx],
            )

        # Internal validation is used for development assessment.
        all_rows.extend(
            evaluate_model_set(
                fold, "validation", train_data, val_idx, models
            )
        )

        # Previously inspected test folds: exploratory results only.
        test_indices = np.arange(len(test_data["error"]))
        all_rows.extend(
            evaluate_model_set(
                fold, "test_exploratory", test_data,
                test_indices, models
            )
        )

        fold_notes.append({
            "fold": fold,
            "development_n": int(len(dev_idx)),
            "validation_n": int(len(val_idx)),
            "test_n": int(len(test_indices)),
            "development_error_rate": float(y[dev_idx].mean()),
            "validation_error_rate": float(y[val_idx].mean()),
            "test_error_rate": float(test_data["error"].mean()),
            "features_used": [
                "uncertainty: confidence risk, entropy, negative margin",
                "features: 512-dimensional Feat",
                "combined: 5 probabilities + 3 uncertainty measures + Feat",
            ],
            "human_label_used_as_input": False,
        })

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT / "per_fold_metrics.csv", index=False)

    summary = (
        results.groupby(["split", "model"])[
            ["auroc", "average_precision", "error_rate", "brier"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(OUT / "summary_metrics.csv", index=False)

    report = {
        "experiment": "A: learned AI-error risk estimation",
        "target": "1 if argmax(AI probabilities) differs from True Label",
        "development_protocol": (
            "Within each training fold, stratified 80/20 development/"
            "validation split. StandardScaler and LogisticRegression are "
            "fitted only on development data. C=1.0 fixed in advance."
        ),
        "test_status": (
            "Existing test folds have been examined in prior experiments; "
            "test_exploratory metrics are not confirmatory."
        ),
        "folds": fold_notes,
        "limitations": [
            "No untouched final test set is available in this experiment.",
            "Patient-level split independence is not established.",
            "Logistic regression is a simple first learned estimator.",
            "Results are not evidence of causal or clinical reliability.",
        ],
    }

    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    print("\n=== PER-FOLD METRICS ===")
    print(results.round(4).to_string(index=False))

    print("\n=== MEAN AND STD BY SPLIT AND MODEL ===")
    print(summary.round(4).to_string(index=False))

    print(f"\nFiles saved in: {OUT}")


if __name__ == "__main__":
    main()
