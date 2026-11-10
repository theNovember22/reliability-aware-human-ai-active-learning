"""Experiment B: error remaining after referring high-risk AI predictions.

Uses fixed cached AI probabilities and correctness labels. It does not use
Human Label or claim post-review accuracy. Existing test folds are exploratory
because they were inspected in earlier work. MI for the AI is unavailable:
the repository stores only one probability vector per case, not MC samples.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "DR-5"
OUT = ROOT / "results" / "experiment_b_referral"
SEED = 42
RANDOM_REPEATS = 1000
REFERRAL_GRID = np.arange(0.0, 0.801, 0.05)


def parse_vector(value):
    return np.fromstring(str(value).strip().strip("[]"), sep=",")


def load_split(fold, split):
    path = DATA / f"model_output_{fold}_{split}.csv"
    frame = pd.read_csv(path)
    required = ["Logits", "True Label", "Feat"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    probs = np.stack([parse_vector(x) for x in frame["Logits"]]).astype(float)
    features = np.stack([parse_vector(x) for x in frame["Feat"]]).astype(float)
    labels = frame["True Label"].to_numpy(dtype=int)
    if probs.shape != (len(frame), 5) or features.shape != (len(frame), 512):
        raise ValueError(f"{path}: expected 5 probabilities and 512 features per row")
    if not np.isfinite(probs).all() or not np.isfinite(features).all():
        raise ValueError(f"{path}: non-finite input")
    if not np.allclose(probs.sum(1), 1.0, atol=1e-5):
        raise ValueError(f"{path}: probabilities do not sum to one")
    error = (probs.argmax(1) != labels).astype(int)
    confidence = probs.max(1)
    ordered = np.sort(probs, axis=1)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    uncertainty = np.column_stack([1.0 - confidence, entropy, -(ordered[:, -1] - ordered[:, -2])])
    combined = np.column_stack([probs, uncertainty, features])
    return {
        "error": error,
        "confidence": uncertainty[:, 0],
        "entropy": uncertainty[:, 1],
        "margin": uncertainty[:, 2],
        "uncertainty": uncertainty,
        "features": features,
        "combined": combined,
    }


def fit_risk_models(train):
    indices = np.arange(len(train["error"]))
    dev, _ = train_test_split(indices, test_size=.20, random_state=SEED,
                              stratify=train["error"])
    models = {}
    for name in ("uncertainty", "features", "combined"):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, random_state=SEED),
        )
        model.fit(train[name][dev], train["error"][dev])
        models[name] = model
    return models


def residual_error(errors, risk, referral_fraction):
    """Error fraction among the cases left to automatic handling."""
    n_refer = int(np.ceil(referral_fraction * len(errors)))
    n_keep = len(errors) - n_refer
    if n_keep == 0:
        return float("nan")
    order = np.argsort(-risk, kind="stable")
    remaining = errors[order[n_refer:]]
    return float(remaining.mean())


def random_curve(errors, grid, rng):
    values = np.empty((RANDOM_REPEATS, len(grid)), dtype=float)
    for repeat in range(RANDOM_REPEATS):
        order = rng.permutation(len(errors))
        for j, fraction in enumerate(grid):
            n_refer = int(np.ceil(fraction * len(errors)))
            remaining = errors[order[n_refer:]]
            values[repeat, j] = remaining.mean() if len(remaining) else np.nan
    return np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=1)


def write_curve_svg(curves, path):
    width, height = 900, 590
    left, right, top, bottom = 90, 250, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = REFERRAL_GRID
    all_values = [np.asarray(v["mean"]) for v in curves.values()]
    upper = [np.asarray(v["mean"]) + np.asarray(v.get("std", np.zeros(len(xs)))) for v in curves.values()]
    ymax = min(1.0, max(.55, float(np.nanmax(upper)) + .04))
    ymin = 0.0
    xmap = lambda x: left + x / .8 * plot_w
    ymap = lambda y: top + (ymax - y) / (ymax - ymin) * plot_h
    colors = {
        "Random referral": "#7c8798", "Confidence": "#1976d2", "Entropy": "#d97706",
        "Margin": "#16a085", "Learned uncertainty": "#8e44ad",
        "Learned features": "#c0392b", "Learned combined": "#34495e",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.axis{stroke:#444;stroke-width:1}</style>',
        '<text x="90" y="30" font-size="20" font-weight="bold">Experiment B: referral rate vs residual error</text>',
    ]
    for tick in np.arange(0, ymax + 1e-9, .1):
        yy = ymap(tick)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text class="small" x="{left-12}" y="{yy+4:.1f}" text-anchor="end">{tick:.0%}</text>')
    for tick in np.arange(0, .801, .1):
        xx = xmap(tick)
        lines.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_h}" stroke="#f1f3f5"/>')
        lines.append(f'<text class="small" x="{xx:.1f}" y="{top+plot_h+22}" text-anchor="middle">{tick:.0%}</text>')
    lines += [
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle">Cases referred to humans</text>',
        f'<text transform="translate(23 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Error among automatically handled cases</text>',
    ]
    # Random line includes the one-standard-deviation band over simulation draws/folds.
    random = curves["Random referral"]
    lo = np.clip(np.asarray(random["mean"]) - np.asarray(random["std"]), 0, ymax)
    hi = np.clip(np.asarray(random["mean"]) + np.asarray(random["std"]), 0, ymax)
    band_points = [f'{xmap(x):.1f},{ymap(y):.1f}' for x, y in zip(xs, hi)]
    band_points += [f'{xmap(x):.1f},{ymap(y):.1f}' for x, y in zip(xs[::-1], lo[::-1])]
    lines.append(f'<polygon points="{" ".join(band_points)}" fill="{colors["Random referral"]}" opacity="0.13"/>')
    for name, values in curves.items():
        pts = " ".join(f'{xmap(x):.1f},{ymap(y):.1f}' for x, y in zip(xs, values["mean"]))
        dash = ' stroke-dasharray="6 5"' if name == "Random referral" else ""
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{colors[name]}" stroke-width="2.8"{dash}/>')
    legend_x, legend_y = left + plot_w + 25, top + 12
    for idx, name in enumerate(curves):
        yy = legend_y + idx * 27
        dash = ' stroke-dasharray="5 4"' if name == "Random referral" else ""
        lines.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x+25}" y2="{yy}" stroke="{colors[name]}" stroke-width="3"{dash}/>')
        lines.append(f'<text class="small" x="{legend_x+33}" y="{yy+4}">{name}</text>')
    lines.append('<text class="small" x="90" y="565">Lines average five folds. Shaded random band: ±1 SD over 1,000 random draws per fold.</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    curve_values = {name: [] for name in (
        "confidence", "entropy", "margin", "learned_uncertainty", "learned_features", "learned_combined"
    )}
    random_means, random_stds = [], []
    fold_notes = []

    for fold in range(5):
        train = load_split(fold, "train")
        test = load_split(fold, "test")
        models = fit_risk_models(train)
        scores = {
            "confidence": test["confidence"],
            "entropy": test["entropy"],
            "margin": test["margin"],
            "learned_uncertainty": models["uncertainty"].predict_proba(test["uncertainty"])[:, 1],
            "learned_features": models["features"].predict_proba(test["features"])[:, 1],
            "learned_combined": models["combined"].predict_proba(test["combined"])[:, 1],
        }
        for name, risk in scores.items():
            curve = [residual_error(test["error"], risk, rate) for rate in REFERRAL_GRID]
            curve_values[name].append(curve)
            for rate in (.1, .2, .3):
                rows.append({
                    "fold": fold, "split": "test_exploratory", "strategy": name,
                    "referral_rate": rate,
                    "n_referred": int(np.ceil(rate * len(test["error"]))),
                    "coverage": 1.0 - int(np.ceil(rate * len(test["error"]))) / len(test["error"]),
                    "residual_error_rate": residual_error(test["error"], risk, rate),
                    "baseline_error_rate": float(test["error"].mean()),
                })
        rm, rs = random_curve(test["error"], REFERRAL_GRID, np.random.default_rng(SEED + fold))
        random_means.append(rm)
        random_stds.append(rs)
        for rate in (.1, .2, .3):
            idx = int(np.argmin(np.abs(REFERRAL_GRID - rate)))
            rows.append({
                "fold": fold, "split": "test_exploratory", "strategy": "random",
                "referral_rate": rate,
                "n_referred": int(np.ceil(rate * len(test["error"]))),
                "coverage": 1.0 - int(np.ceil(rate * len(test["error"]))) / len(test["error"]),
                "residual_error_rate": float(rm[idx]),
                "random_residual_sd": float(rs[idx]),
                "baseline_error_rate": float(test["error"].mean()),
            })
        fold_notes.append({
            "fold": fold, "n": int(len(test["error"])),
            "error_rate": float(test["error"].mean()),
            "mi_available": False,
        })

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(OUT / "per_fold_referral.csv", index=False)
    summary = per_fold.groupby(["strategy", "referral_rate"])[
        ["residual_error_rate", "baseline_error_rate", "coverage"]
    ].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")

    curves = {name.replace("_", " ").title(): {"mean": np.mean(v, axis=0), "std": np.std(v, axis=0, ddof=1)}
              for name, v in curve_values.items()}
    curves["Random referral"] = {"mean": np.mean(random_means, axis=0),
                                 "std": np.sqrt(np.mean(np.asarray(random_stds) ** 2, axis=0)
                                                + np.var(np.asarray(random_means), axis=0, ddof=1))}
    # Use readable strategy labels in the plot legend.
    curves = {
        "Confidence": curves["Confidence"], "Entropy": curves["Entropy"],
        "Margin": curves["Margin"], "Learned uncertainty": curves["Learned Uncertainty"],
        "Learned features": curves["Learned Features"], "Learned combined": curves["Learned Combined"],
        "Random referral": curves["Random referral"],
    }
    write_curve_svg(curves, OUT / "referral_curve.svg")
    curve_frame = pd.DataFrame({"referral_rate": REFERRAL_GRID, **{k: v["mean"] for k, v in curves.items()}})
    curve_frame.to_csv(OUT / "referral_curve.csv", index=False)
    report = {
        "experiment": "B: human referral and residual AI error",
        "estimand": "Error rate among cases not referred, assuming the AI prediction is used unchanged for those cases.",
        "referral_rates": [0.1, 0.2, 0.3],
        "curve_grid": REFERRAL_GRID.tolist(),
        "data": "Existing DR-5 cached train/test CSVs; five 200-row test folds.",
        "test_status": "Exploratory only because these test folds were previously inspected.",
        "random_baseline": f"{RANDOM_REPEATS} random without-replacement referrals per fold and referral rate; curve band combines simulation and fold variation.",
        "models": "Three fixed C=1 logistic regressions fit on stratified 80% subsets of each training fold: uncertainty-only, 512 features-only, and combined.",
        "human_labels": "Human Label is not read; this estimates residual AI error, not post-review accuracy or expert correction benefit.",
        "mi_status": "Unavailable: the dataset stores one AI probability vector per case and no MC probability samples or AI model checkpoints. Existing MC artifacts are evaluator/fusion predictions that depend on human labels, so they are not a valid pre-referral AI-MI score.",
        "folds": fold_notes,
        "limitations": [
            "No actual expert corrections are simulated; no claim of improved final accuracy is made.",
            "No patient/image IDs are available to verify patient-disjointness.",
            "The five folds may not be treated as 1,000 unique patients because sample IDs are absent.",
            "MI cannot be evaluated without saved AI MC-dropout probability samples or access to the AI model and images.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(summary.round(4).to_string())
    print(f"\nSaved Experiment B outputs to: {OUT}")


if __name__ == "__main__":
    main()
