"""Experiment C: use evaluator MC-dropout reliability to adapt fusion weight."""

from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dr5.core import (  # noqa: E402
    Evaluator, initial_ids, load_csv, mutual_information, predict_samples,
    train_evaluator,
)
from experiments.analysis.experiment_b_referral import acquire_entropy_mwac  # noqa: E402

DATA = ROOT / "data" / "DR-5"
OUT = ROOT / "results" / "experiment_c_reliability_weighted_fusion"
METHOD_FIXED = "fixed_fusion_weight"
METHOD_ADAPTIVE = "reliability_adaptive_weight"
METHODS = (METHOD_FIXED, METHOD_ADAPTIVE)
METRIC_COLUMNS = ("accuracy", "kl_divergence")
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--mc-passes", type=int, default=20)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args()


def weighted_mc_fusion(samples, ai_probs):
    """Return fixed and MI-adaptive fused predictions from identical MC samples."""
    ai_log = ai_probs.clamp_min(EPS).log()
    mi = mutual_information(samples)
    max_mi = float(np.log(ai_probs.shape[1]))
    evaluator_reliability = (1.0 - mi / max_mi).clamp(0.0, 1.0)

    # `samples` are existing unit-weight product-fusion probabilities:
    # log(q) - log(p_AI) equals evaluator logits up to a class-constant term.
    # Class constants disappear in softmax, so this recovers the same
    # evaluator contribution without changing the evaluator architecture.
    evaluator_delta = samples.clamp_min(EPS).log() - ai_log.unsqueeze(0)
    adaptive_logits = ai_log.unsqueeze(0) + evaluator_reliability[None, :, None] * evaluator_delta
    adaptive_samples = F.softmax(adaptive_logits, dim=-1)
    return samples.mean(0), adaptive_samples.mean(0), evaluator_reliability, mi


def score(probabilities, labels):
    p = probabilities.clamp_min(EPS)
    y = labels.long()
    accuracy = float(p.argmax(-1).eq(y).double().mean())
    target = F.one_hot(y, num_classes=p.shape[1]).to(dtype=p.dtype)
    # KL(one-hot true label || predicted distribution); exactly equal to NLL.
    kl = F.kl_div(p.log(), target, reduction="batchmean")
    return {"accuracy": accuracy, "kl_divergence": float(kl)}


def svg_budget_plot(frame, metric, title, y_label, path, higher_is_better=True):
    width, height = 1120, 590
    left, right, top, bottom = 90, 360, 65, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    budgets = sorted(frame.labels_per_class.unique())
    queries = [int(b) * 5 for b in budgets]
    palette = {METHOD_FIXED: "#667085", METHOD_ADAPTIVE: "#1683d8"}
    legend = {METHOD_FIXED: "Fixed evaluator weight = 1", METHOD_ADAPTIVE: "MI-based adaptive evaluator weight"}
    ymax = max(float(frame[metric].max()) * 1.18, 0.05)
    if metric == "accuracy":
        ymax = 1.0
    xmap = lambda x: left + x / max(queries) * plot_w
    ymap = lambda y: top + (ymax - y) / ymax * plot_h
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#18212f}.small{font-size:13px}</style>',
           f'<text x="{left}" y="33" font-size="21" font-weight="bold">{title}</text>']
    for i in range(6):
        val = ymax * i / 5
        yy = ymap(val)
        svg.extend([f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e7ebf0"/>',
                    f'<text class="small" x="{left-10}" y="{yy+4:.1f}" text-anchor="end">{val:.2f}</text>'])
    for q in queries:
        xx = xmap(q)
        svg.extend([f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_h}" stroke="#f1f3f6"/>',
                    f'<text class="small" x="{xx:.1f}" y="{top+plot_h+22}" text-anchor="middle">{q}</text>'])
    svg.extend([f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#344054"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#344054"/>',
                f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-size="15">Total development expert queries</text>',
                f'<text transform="translate(25 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-size="15">{y_label}</text>'])
    for method in METHODS:
        stats = frame[frame.method == method].groupby("labels_per_class")[metric].agg(["mean", "std"]).reindex(budgets)
        pts = []
        for b, mean, std in zip(budgets, stats["mean"], stats["std"].fillna(0)):
            xx, yy = xmap(b * 5), ymap(float(mean))
            err = float(std) / ymax * plot_h
            svg.append(f'<line x1="{xx:.1f}" y1="{yy-err:.1f}" x2="{xx:.1f}" y2="{yy+err:.1f}" stroke="{palette[method]}" stroke-width="1.5"/>')
            pts.append(f"{xx:.1f},{yy:.1f}")
        dash = ' stroke-dasharray="7 4"' if method == METHOD_ADAPTIVE else ""
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{palette[method]}" stroke-width="3"{dash}/>')
        for point in pts:
            xx, yy = point.split(",")
            if method == METHOD_ADAPTIVE:
                svg.append(f'<rect x="{float(xx)-3.5:.1f}" y="{float(yy)-3.5:.1f}" width="7" height="7" fill="{palette[method]}"/>')
            else:
                svg.append(f'<circle cx="{xx}" cy="{yy}" r="4" fill="{palette[method]}"/>')
    for i, method in enumerate(METHODS):
        yy = top + 25 + i * 34
        svg.extend([f'<line x1="{left+plot_w+20}" y1="{yy}" x2="{left+plot_w+52}" y2="{yy}" stroke="{palette[method]}" stroke-width="3"/>',
                    f'<text class="small" x="{left+plot_w+60}" y="{yy+4}">{legend[method]}</text>'])
    unique_runs = frame[["labels_per_class", "fold", "seed"]].drop_duplicates()
    n = int(unique_runs.groupby("labels_per_class").size().max())
    svg.append(f'<text class="small" x="{left}" y="{height-4}">Mean ± SD across fold-seed runs (n={n} per budget).</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def render_png(svg_png_pairs):
    if sys.platform == "darwin":
        swift = r'''import AppKit
import Foundation
let args = Array(CommandLine.arguments.dropFirst())
for i in stride(from: 0, to: args.count, by: 2) {
  let source = URL(fileURLWithPath: args[i])
  let target = URL(fileURLWithPath: args[i + 1])
  guard let image = NSImage(contentsOf: source) else { fatalError("Could not load SVG") }
  let size = image.size
  guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width), pixelsHigh: Int(size.height),
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
        let context = NSGraphicsContext(bitmapImageRep: bitmap) else { fatalError("Could not create bitmap") }
  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = context
  NSColor.white.setFill()
  NSRect(origin: .zero, size: size).fill()
  image.draw(in: NSRect(origin: .zero, size: size), from: .zero, operation: .sourceOver, fraction: 1)
  context.flushGraphics()
  NSGraphicsContext.restoreGraphicsState()
  guard let png = bitmap.representation(using: .png, properties: [:]) else { fatalError("Could not encode PNG") }
  try png.write(to: target)
}
'''
        cache = Path(tempfile.gettempdir()) / "experiment_c_fusion_swift_module_cache"
        subprocess.run(["swift", "-module-cache-path", str(cache), "-e", swift,
                        *[str(p) for pair in svg_png_pairs for p in pair]], check=True)
    else:
        raise RuntimeError("PNG plot conversion currently requires the repository's macOS environment")


def main(args):
    if args.epochs < 1 or args.mc_passes < 2:
        raise ValueError("epochs must be positive and MC Dropout needs at least two passes")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if not set(args.folds) <= set(range(5)) or not args.folds:
        raise ValueError("folds must be selected from 0 through 4")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, traces, reliability_rows = [], [], []

    for fold in args.folds:
        train = load_csv(args.data_dir / f"model_output_{fold}_train.csv").to(args.device)
        pool_ids = np.arange(len(train.labels))
        for seed in args.seeds:
            run_seed = fold * 1000 + seed
            rng = np.random.default_rng(run_seed)
            selected = initial_ids(train.labels, pool_ids, 2, rng)
            checkpoint_states, run_trace = [], []

            # Follow Experiment B's entropy-ranked MWAC query sequence once.
            # The same evaluator checkpoints and acquired labels are evaluated
            # under fixed and reliability-adaptive fusion weights.
            for budget in range(2, 9):
                model, train_loss = train_evaluator(train, selected, run_seed, epochs=args.epochs)
                checkpoint_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                event = {"labels_per_class": budget, "total_queries": len(selected),
                         "selected_ids": list(selected), "train_loss": train_loss}
                if budget < 8:
                    additions, signals = acquire_entropy_mwac(model, train, pool_ids, selected, rng)
                    selected.extend(additions)
                    event.update({"newly_queried_ids": additions, "acquisition": signals})
                run_trace.append(event)

            # Test labels enter only the final metric calculation. They are not
            # used for acquisition, MI, reliability weights, or evaluator fit.
            test = load_csv(args.data_dir / f"model_output_{fold}_test.csv").to(args.device)
            model = Evaluator(train.features.shape[1], dropout=0.1).to(args.device)
            for budget, state in zip(range(2, 9), checkpoint_states):
                model.load_state_dict(state)
                samples = predict_samples(
                    model, test.features, test.human, test.probs,
                    passes=args.mc_passes, seed=run_seed + 70_000 + budget,
                )
                fixed, adaptive, weights, mi = weighted_mc_fusion(samples, test.probs)
                for method, probabilities in ((METHOD_FIXED, fixed), (METHOD_ADAPTIVE, adaptive)):
                    results = score(probabilities, test.labels)
                    rows.append({"fold": fold, "seed": seed, "method": method,
                                 "labels_per_class": budget, "total_queries": budget * 5,
                                 **results})
                reliability_rows.append({
                    "fold": fold, "seed": seed, "labels_per_class": budget,
                    "mean_mi_nats": float(mi.mean()), "std_mi_nats": float(mi.std(unbiased=False)),
                    "mean_evaluator_weight": float(weights.mean()),
                    "min_evaluator_weight": float(weights.min()),
                    "max_evaluator_weight": float(weights.max()),
                })
            traces.append({"fold": fold, "seed": seed, "initial_ids": run_trace[0]["selected_ids"], "rounds": run_trace})
            del test, model

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "per_fold_budget.csv", index=False)
    summary = frame.groupby(["method", "labels_per_class"])[list(METRIC_COLUMNS)].agg(["mean", "std"])
    summary.to_csv(args.output / "summary.csv")
    reliability_frame = pd.DataFrame(reliability_rows)
    reliability_frame.to_csv(args.output / "reliability_weights.csv", index=False)

    details = []
    aggregates = []
    for budget in range(2, 9):
        fixed = frame[(frame.method == METHOD_FIXED) & (frame.labels_per_class == budget)].set_index(["fold", "seed"])
        adaptive = frame[(frame.method == METHOD_ADAPTIVE) & (frame.labels_per_class == budget)].set_index(["fold", "seed"])
        diff = adaptive[list(METRIC_COLUMNS)] - fixed[list(METRIC_COLUMNS)]
        t_by_metric = {}
        for metric in METRIC_COLUMNS:
            t_by_metric[metric] = float(ttest_rel(adaptive[metric], fixed[metric]).pvalue) if len(diff) >= 2 else float("nan")
        for (fold, seed), record in diff.iterrows():
            for metric, value in record.items():
                details.append({"comparison": f"{METHOD_ADAPTIVE} - {METHOD_FIXED}",
                                "fold": fold, "seed": seed, "labels_per_class": budget,
                                "total_queries": budget * 5, "metric": metric,
                                "difference": float(value)})
        for metric in METRIC_COLUMNS:
            values = diff[metric].to_numpy()
            aggregates.append({"comparison": f"{METHOD_ADAPTIVE} - {METHOD_FIXED}",
                               "labels_per_class": budget, "total_queries": budget * 5,
                               "metric": metric, "mean_difference": float(values.mean()),
                               "std_difference": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                               "paired_ttest_pvalue": t_by_metric[metric],
                               "n_paired_runs": int(len(values))})
    pd.DataFrame(details).to_csv(args.output / "paired_differences.csv", index=False)
    pd.DataFrame(aggregates).to_csv(args.output / "paired_difference_summary.csv", index=False)
    (args.output / "acquisition_traces.json").write_text(json.dumps(traces, indent=2))

    svg_pairs = []
    for metric, title, y_label in (("accuracy", "Accuracy vs Expert Query Budget", "Fusion accuracy"),
                                   ("kl_divergence", "KL Divergence vs Expert Query Budget", "KL(one-hot true label || fusion)") ):
        stem = "accuracy_vs_budget" if metric == "accuracy" else "kl_divergence_vs_budget"
        svg = args.output / f"{stem}.svg"
        png = args.output / f"{stem}.png"
        svg_budget_plot(frame, metric, f"Experiment C: {title}", y_label, svg)
        svg_pairs.append((svg, png))
    render_png(svg_pairs)

    final = frame[frame.labels_per_class == 8]
    final_summary = final.groupby("method")[list(METRIC_COLUMNS)].agg(["mean", "std"])
    final_summary.columns = [f"{metric}_{stat}" for metric, stat in final_summary.columns]
    final_summary = final_summary.to_dict(orient="index")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = "unavailable"
    report = {
        "experiment": "C: evaluator-reliability-adaptive Human-AI fusion",
        "question": "Does an MC-dropout evaluator reliability estimate improve Human-AI fusion by changing the evaluator's contribution per case?",
        "dataset": "DR-5 cached AI probability, feature, true-label, and human-label CSVs",
        "folds": args.folds, "seeds": args.seeds,
        "budgets_total_queries": [10, 15, 20, 25, 30, 35, 40],
        "active_annotation_policy": "The same Experiment B MWAC median-window then highest-AI-entropy acquisition policy is used once per fold/seed. Both fusion conditions share the exact evaluator checkpoints and acquired labels.",
        "fixed_fusion": "Existing product-rule fusion with evaluator-logit coefficient beta=1, averaged over MC-dropout passes.",
        "reliability_adaptive_fusion": "Per-case evaluator coefficient beta_i = clip(1 - MI_i/log(5), 0, 1). Higher evaluator MC mutual information lowers its fusion contribution; AI probability contribution remains coefficient 1.",
        "reliability_metric": "Existing evaluator MC-dropout mutual information: H(mean pass probability) - mean(pass entropy), implemented by src.dr5.core.mutual_information; 20 forward passes by default.",
        "kl_definition": "Mean KL(one-hot True Label || fused predictive distribution); for one-hot labels this equals categorical NLL.",
        "metrics": ["accuracy", "kl_divergence"],
        "query_cost": "10 to 40 development human labels, with one query per class per round. Test Human Label is required only as the input to the already-established fusion evaluation and is not used for acquisition, evaluator fitting, or reliability weights.",
        "evaluator_training_epochs": args.epochs,
        "seed_schedule": {"run_seed": "fold*1000 + seed", "initial_sampling": "run_seed", "evaluator_initialization": "run_seed at each budget", "test_mc_dropout": "run_seed + 70000 + labels_per_class"},
        "final_40_query_summary": final_summary,
        "paired_test": "Two-sided paired t-tests over fold/seed runs at each budget; descriptive only, no multiplicity adjustment.",
        "limitations": [
            "The reliability weight is a predeclared monotone heuristic, not a calibrated probability that the evaluator is correct.",
            "The evaluator is still trained with the existing beta=1 fusion objective; only inference-time fusion weights are adapted.",
            "The existing five test folds have been inspected and are exploratory; patient IDs are unavailable.",
            "KL to one-hot truth is mathematically the same information as NLL, so these metrics are not independent evidence.",
        ],
        "git_commit": commit,
        "source_hashes": {"experiment_c": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "experiment_b": hashlib.sha256((ROOT / "experiments/analysis/experiment_b_referral.py").read_bytes()).hexdigest(),
                          "dr5_core": hashlib.sha256((ROOT / "src/dr5/core.py").read_bytes()).hexdigest()},
        "input_hashes": {f"fold{fold}_{split}": hashlib.sha256((args.data_dir / f"model_output_{fold}_{split}.csv").read_bytes()).hexdigest()
                         for fold in args.folds for split in ("train", "test")},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("Final 40-query mean metrics:")
    print(final.groupby("method")[list(METRIC_COLUMNS)].agg(["mean", "std"]).round(4).to_string())
    print(f"\nSaved Experiment C outputs to: {args.output}")


if __name__ == "__main__":
    main(parse_args())
