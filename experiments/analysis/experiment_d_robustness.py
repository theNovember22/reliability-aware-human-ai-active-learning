"""Exploratory robustness checks for error-risk signals on DR-5.

D1 and D2 evaluate cached AI errors on the five existing test folds. D3 uses
saved human-AI evaluator predictions across evaluator seeds. The cached AI
backbone outputs themselves are fixed within each fold and their seed
provenance is unavailable, so D3 does not claim backbone-seed robustness.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "DR-5"
BASELINE_PREDICTIONS = ROOT / "results" / "baseline"
OUT = ROOT / "results" / "experiment_d_robustness"
SEED = 42


def parse_vector(value):
    return np.fromstring(str(value).strip().strip("[]"), sep=",")


def load_csv(fold, split):
    frame = pd.read_csv(DATA / f"model_output_{fold}_{split}.csv")
    probs = np.stack([parse_vector(x) for x in frame["Logits"]]).astype(float)
    features = np.stack([parse_vector(x) for x in frame["Feat"]]).astype(float)
    labels = frame["True Label"].to_numpy(dtype=int)
    if probs.shape != (len(frame), 5) or features.shape != (len(frame), 512):
        raise ValueError(f"Fold {fold} {split}: expected 5 probabilities and 512 features")
    if not np.isfinite(probs).all() or not np.isfinite(features).all():
        raise ValueError(f"Fold {fold} {split}: non-finite data")
    if not np.allclose(probs.sum(1), 1., atol=1e-5):
        raise ValueError(f"Fold {fold} {split}: probabilities do not sum to one")
    pred = probs.argmax(1)
    error = (pred != labels).astype(int)
    sorted_probs = np.sort(probs, axis=1)
    confidence = probs.max(1)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.)), axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    uncertainty = np.column_stack([1. - confidence, entropy, -margin])
    combined = np.column_stack([probs, uncertainty, features])
    return {
        "probs": probs, "features": features, "labels": labels, "pred": pred,
        "error": error, "absolute_error": np.abs(pred - labels),
        "confidence": 1. - confidence, "entropy": entropy, "margin": -margin,
        "uncertainty": uncertainty, "combined": combined,
    }


def fit_risk_models(train):
    dev, _ = train_test_split(
        np.arange(len(train["error"])), test_size=.20, random_state=SEED,
        stratify=train["error"],
    )
    models = {}
    for name in ("uncertainty", "features", "combined"):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1., max_iter=3000, random_state=SEED)
        )
        model.fit(train[name][dev], train["error"][dev])
        models[name] = model
    return models


def auc_or_nan(target, score):
    return float(roc_auc_score(target, score)) if np.unique(target).size == 2 else float("nan")


def ap_or_nan(target, score):
    return float(average_precision_score(target, score)) if target.sum() else float("nan")


def d1_class_specific():
    rows = []
    for fold in range(5):
        train = load_csv(fold, "train")
        test = load_csv(fold, "test")
        models = fit_risk_models(train)
        risks = {
            "confidence": test["confidence"], "entropy": test["entropy"],
            "margin": test["margin"],
            "learned_uncertainty": models["uncertainty"].predict_proba(test["uncertainty"])[:, 1],
            "learned_features": models["features"].predict_proba(test["features"])[:, 1],
            "learned_combined": models["combined"].predict_proba(test["combined"])[:, 1],
        }
        for class_id in range(5):
            mask = test["labels"] == class_id
            y = test["error"][mask]
            for name, score in risks.items():
                rows.append({
                    "fold": fold, "true_class": class_id, "score": name,
                    "n": int(mask.sum()), "error_count": int(y.sum()),
                    "error_rate": float(y.mean()), "auroc": auc_or_nan(y, score[mask]),
                    "average_precision": ap_or_nan(y, score[mask]),
                })
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(OUT / "d1_per_fold_class.csv", index=False)
    return per_fold


def d2_ordinal_severity():
    detail_rows, summary_rows = [], []
    severity_names = {0: "correct", 1: "minor_1_grade", 2: "severe_2plus_grades"}
    for fold in range(5):
        train = load_csv(fold, "train")
        test = load_csv(fold, "test")
        models = fit_risk_models(train)
        risks = {
            "confidence": test["confidence"], "entropy": test["entropy"],
            "margin": test["margin"],
            "learned_uncertainty": models["uncertainty"].predict_proba(test["uncertainty"])[:, 1],
            "learned_features": models["features"].predict_proba(test["features"])[:, 1],
            "learned_combined": models["combined"].predict_proba(test["combined"])[:, 1],
        }
        distance = test["absolute_error"]
        severe = (distance >= 2).astype(int)
        for name, score in risks.items():
            rho = spearmanr(score, distance).statistic if np.unique(distance).size > 1 else float("nan")
            summary_rows.append({
                "fold": fold, "score": name, "n": len(distance),
                "correct_count": int(np.sum(distance == 0)),
                "minor_error_count": int(np.sum(distance == 1)),
                "severe_error_count": int(np.sum(distance >= 2)),
                "severe_error_rate": float(severe.mean()),
                "severe_vs_other_auroc": auc_or_nan(severe, score),
                "severe_average_precision": ap_or_nan(severe, score),
                "spearman_risk_vs_ordinal_distance": float(rho),
            })
            for severity, label in severity_names.items():
                mask = distance == severity if severity < 2 else distance >= 2
                detail_rows.append({
                    "fold": fold, "score": name, "severity": label,
                    "n": int(mask.sum()), "mean_risk": float(score[mask].mean()) if mask.any() else float("nan"),
                })
    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)
    summary.to_csv(OUT / "d2_per_fold_severity_metrics.csv", index=False)
    details.to_csv(OUT / "d2_severity_score_means.csv", index=False)
    return summary, details


def d3_evaluator_seed_variation():
    rows = []
    for fold in range(5):
        expected_labels = load_csv(fold, "test")["labels"]
        for seed in (1, 2, 3):
            path = BASELINE_PREDICTIONS / f"pred_fold{fold}_seed{seed}_budget8.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved evaluator prediction: {path}")
            with np.load(path, allow_pickle=False) as archive:
                probs, labels = archive["probabilities"].astype(float), archive["labels"].astype(int)
            if probs.shape != (len(labels), 5) or not np.array_equal(labels, expected_labels):
                raise ValueError(f"{path}: predictions do not align with the test fold")
            pred = probs.argmax(1)
            error = (pred != labels).astype(int)
            absolute_error = np.abs(pred - labels)
            confidence = probs.max(1)
            sorted_probs = np.sort(probs, axis=1)
            scores = {
                "confidence": 1. - confidence,
                "entropy": -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.)), axis=1),
                "margin": -(sorted_probs[:, -1] - sorted_probs[:, -2]),
            }
            for name, score in scores.items():
                rows.append({
                    "fold": fold, "evaluator_seed": seed, "score": name,
                    "n": len(labels), "error_rate": float(error.mean()),
                    "error_auroc": auc_or_nan(error, score),
                    "error_average_precision": ap_or_nan(error, score),
                    "severe_error_rate": float(np.mean(absolute_error >= 2)),
                    "severe_error_auroc": auc_or_nan((absolute_error >= 2).astype(int), score),
                })
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "d3_per_fold_evaluator_seed.csv", index=False)
    summary = results.groupby(["evaluator_seed", "score"])[
        ["error_rate", "error_auroc", "error_average_precision", "severe_error_rate", "severe_error_auroc"]
    ].agg(["mean", "std"])
    summary.to_csv(OUT / "d3_seed_summary.csv")
    return results, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d1 = d1_class_specific()
    d2, severity_means = d2_ordinal_severity()
    d3, d3_summary = d3_evaluator_seed_variation()

    report = {
        "experiment": "D: robustness and generalization checks",
        "status": "Exploratory analyses completed on available cached data.",
        "D1": {
            "question": "Does risk rank errors within each true DR grade?",
            "metrics": "Class-conditional AUROC and average precision, reported per fold and class.",
            "macro_class_auroc": d1.groupby(["true_class", "score"]).auroc.mean().groupby("score").mean().to_dict(),
        },
        "D2": {
            "question": "Does risk rise with ordinal error severity?",
            "severity": "correct=0 grades off; minor=1 grade off; severe=2 or more grades off.",
            "metrics": "Severe-vs-other AUROC/AP, Spearman risk vs absolute grade distance, and mean risk by severity.",
            "mean_severe_error_auroc": d2.groupby("score").severe_vs_other_auroc.mean().to_dict(),
            "mean_spearman_risk_vs_distance": d2.groupby("score").spearman_risk_vs_ordinal_distance.mean().to_dict(),
            "mean_risk_by_severity": severity_means.groupby(["score", "severity"]).mean_risk.mean().unstack().to_dict(),
        },
        "D3": {
            "question": "Does the reliability ranking vary across saved evaluator random seeds?",
            "model_scope": "Human-AI fusion evaluator seeds 1-3 at budget 8; this is not image-backbone seed variation.",
            "metrics": "Error rate, error AUROC/AP, severe-error rate and severe-error AUROC per fold and seed.",
            "mean_error_auroc_by_seed": d3.groupby(["evaluator_seed", "score"]).error_auroc.mean().unstack("score").to_dict(),
        },
        "limitations": [
            "All five existing test folds have been inspected; results are exploratory, not confirmatory.",
            "No patient/image IDs or backbone training provenance are supplied.",
            "Cached AI outputs have no documented model-seed provenance; D3 uses evaluator seeds only.",
            "Class-specific analyses condition on true class for diagnostic analysis; true class is unavailable to a deployed referral rule.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print("D1 mean class-conditional AUROC")
    print(d1.groupby(["true_class", "score"]).auroc.mean().unstack("score").round(3).to_string())
    print("\nD2 severity ranking")
    print(d2.groupby("score")[["severe_vs_other_auroc", "spearman_risk_vs_ordinal_distance"]].mean().round(3).to_string())
    print("\nD3 error AUROC by evaluator seed")
    print(d3.groupby(["evaluator_seed", "score"]).error_auroc.mean().unstack("score").round(3).to_string())
    print(f"\nSaved Experiment D outputs to: {OUT}")


if __name__ == "__main__":
    main()
