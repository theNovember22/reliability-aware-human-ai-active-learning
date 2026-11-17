"""Experiment B: entropy-ranked MWAC active annotation.

Within each class, draw a candidate pool, form the MWAC median-window set
using the current evaluator, and query the highest-AI-entropy case in that
window. Retrain the evaluator after each batch of human annotations.
The supplied folds are exploratory and do not support confirmatory claims.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dr5.core import load_csv, train_evaluator, initial_ids, predict_samples, metrics

DATA = ROOT / "data" / "DR-5"
OUT = ROOT / "results" / "experiment_b_referral"
SEEDS = (1, 2, 3)
INITIAL_PER_CLASS = 2
MAX_LABELS_PER_CLASS = 8
CANDIDATES_PER_CLASS = 100
MWAC_WINDOW_START = 55
MWAC_WINDOW_LENGTH = 5
EPOCHS = 100


def entropy_risk(probabilities):
    """Shannon entropy of the frozen AI probability vector (higher = riskier)."""
    p = np.clip(np.asarray(probabilities), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def acquire_entropy_mwac(model, data, pool_ids, selected, rng):
    """Select one human label per class: highest entropy inside its MWAC window."""
    additions, diagnostics = [], []
    y = data.labels.cpu().numpy()
    for cls in range(5):
        available = np.asarray(
            [i for i in pool_ids if y[i] == cls and i not in selected], dtype=int
        )
        if not len(available):
            raise ValueError(f"Pool exhausted for class {cls}")
        ids = rng.choice(available, min(CANDIDATES_PER_CLASS, len(available)), replace=False)

        # MWAC score is evaluated with pseudo expert response h=y, as in the
        # repository's acquisition implementation. No unqueried Human Label is read.
        pseudo = data.labels[ids]
        with torch.no_grad():
            model.eval()
            evaluator_p = model(pseudo, data.features[ids]).softmax(-1)
            boundary = (evaluator_p.gather(1, pseudo[:, None]).squeeze(1) - .5).abs().cpu().numpy()
        order = np.argsort(-boundary, kind="stable")
        start = min(MWAC_WINDOW_START, max(0, len(ids) - MWAC_WINDOW_LENGTH))
        window = order[start:start + MWAC_WINDOW_LENGTH]

        risk = entropy_risk(data.probs[ids].detach().cpu().numpy())
        highest = np.flatnonzero(np.isclose(risk[window], risk[window].max()))
        local = int(window[rng.choice(highest)])
        chosen = int(ids[local])
        additions.append(chosen)
        diagnostics.append({
            "class": cls,
            "candidate_count": int(len(ids)),
            "mwac_window_start": int(start),
            "mwac_window_length": int(len(window)),
            "selected_id": chosen,
            "mwac_boundary_score": float(boundary[local]),
            "ai_entropy_risk": float(risk[local]),
        })
    return additions, diagnostics


def evaluate(model, data, seed):
    with torch.no_grad():
        fused = predict_samples(model, data.features, data.human, data.probs, passes=1, seed=seed).mean(0)
    return metrics(fused, data.labels)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, traces = [], []
    for fold in range(5):
        full_train = load_csv(DATA / f"model_output_{fold}_train.csv")
        test = load_csv(DATA / f"model_output_{fold}_test.csv")
        train = full_train
        labels = train.labels.cpu().numpy()
        pool_ids = np.arange(len(labels))
        if min(np.sum(labels[pool_ids] == c) for c in range(5)) < MAX_LABELS_PER_CLASS:
            raise ValueError(f"Fold {fold}: insufficient examples for the requested budget")

        for seed in SEEDS:
            run_seed = fold * 1000 + seed
            rng = np.random.default_rng(run_seed)
            selected = initial_ids(train.labels, pool_ids, INITIAL_PER_CLASS, rng)
            run_trace = []
            for budget in range(INITIAL_PER_CLASS, MAX_LABELS_PER_CLASS + 1):
                # Only selected human responses are used for training.
                model, loss = train_evaluator(train, selected, run_seed, epochs=EPOCHS)
                scores = evaluate(model, test, run_seed + budget)
                rows.append({
                    "fold": fold, "seed": seed, "labels_per_class": budget,
                    "active_human_queries": len(selected), "train_loss": loss,
                    "selection": "MWAC median-window then highest AI entropy",
                    "test_status": "exploratory",
                    **{k: v for k, v in scores.items() if k not in ("confusion_matrix", "per_class_recall")},
                })
                event = {"labels_per_class": budget, "selected_ids": list(selected), "metrics": {
                    k: v for k, v in scores.items() if k not in ("confusion_matrix", "per_class_recall")
                }}
                if budget < MAX_LABELS_PER_CLASS:
                    additions, diagnostics = acquire_entropy_mwac(
                        model, train, pool_ids, selected, rng
                    )
                    # This is the simulated human annotation step: labels become
                    # visible to training only after the samples are selected.
                    selected.extend(additions)
                    event.update({"new_annotations": additions, "acquisition": diagnostics})
                run_trace.append(event)
            traces.append({"fold": fold, "seed": seed, "rounds": run_trace})

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "per_fold_budget.csv", index=False)
    frame.groupby("labels_per_class")[
        ["accuracy", "macro_f1", "balanced_accuracy", "qwk", "nll", "ece", "active_human_queries"]
    ].agg(["mean", "std"]).to_csv(OUT / "summary.csv")
    report = {
        "experiment": "B: entropy-ranked MWAC active annotation",
        "workflow": [
            "cached AI probabilities and features",
            "train evaluator on currently selected human annotations",
            "MWAC class-balanced candidate sampling and median-window filtering",
            "rank only median-window candidates by AI Shannon entropy",
            "annotate the highest-entropy candidate per class",
            "add revealed human labels and retrain evaluator",
        ],
        "risk_score": "AI predictive entropy: -sum(p * log(p)); higher values rank as higher risk. This is the same entropy score as Experiment A.",
        "mwac": f"Per class, sample up to {CANDIDATES_PER_CLASS} unqueried cases; sort by descending abs(evaluator probability for true class - 0.5); use zero-based window start {MWAC_WINDOW_START}, length {MWAC_WINDOW_LENGTH}.",
        "budget": f"Start with {INITIAL_PER_CLASS} selected human labels per class and acquire one per class per round through {MAX_LABELS_PER_CLASS}.",
        "test_status": "Exploratory: the existing five test folds have been inspected previously.",
        "data_note": "All supplied train rows are used as the acquisition pool, matching the known-ground-truth ActiveHAI setting. Only queried Human Label values are used to train the evaluator. Test fusion evaluation uses the supplied test Human Label values.",
        "limitations": [
            "The experiment uses cached AI outputs and does not retrain the image model.",
            "No patient/image IDs are available to verify patient-disjointness.",
            "The test folds are not an untouched confirmatory set.",
            "Entropy measures predictive uncertainty and is used as the Experiment A risk ranking; it is not a calibrated probability of error.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    (OUT / "acquisition_traces.json").write_text(json.dumps(traces, indent=2))
    print(frame.groupby("labels_per_class")["accuracy"].agg(["mean", "std"]).round(4).to_string())
    print(f"\nSaved Experiment B outputs to: {OUT}")


if __name__ == "__main__":
    main()
