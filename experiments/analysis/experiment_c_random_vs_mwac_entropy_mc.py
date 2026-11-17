"""Paired DR-5 acquisition ablation: random, MWAC+AI entropy, MWAC+MC MI."""

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
import scipy
from scipy.stats import ttest_rel
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dr5.core import (  # noqa: E402
    load_csv, train_evaluator, initial_ids, predict_samples, reliability_signals,
    entropy, metrics,
)

DATA = ROOT / "data" / "DR-5"
DEFAULT_OUT = ROOT / "results" / "experiment_c_random_vs_mwac_entropy_mc"
METHODS = ("random", "mwac_ai_entropy", "mwac_mc_mi")
METHOD_LABELS = {
    "random": "Random",
    "mwac_ai_entropy": "MWAC + AI Entropy",
    "mwac_mc_mi": "MWAC + MC Dropout (MI)",
}
METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "qwk", "nll", "ece")
PAIRS = (
    ("mwac_ai_entropy", "random"),
    ("mwac_mc_mi", "random"),
    ("mwac_mc_mi", "mwac_ai_entropy"),
)


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def ai_entropy(probs):
    return entropy(probs).detach().cpu().numpy()


def acquire_random(data, pool_ids, selected, rng):
    """One random unqueried row per true class; no Human Label enters selection."""
    y = data.labels.cpu().numpy()
    additions, details = [], []
    for cls in range(5):
        available = np.asarray([i for i in pool_ids if y[i] == cls and i not in selected], dtype=int)
        if not len(available):
            raise ValueError(f"Pool exhausted for class {cls}")
        chosen = int(rng.choice(available))
        additions.append(chosen)
        details.append({"class": cls, "candidate_count": int(len(available)), "selected_id": chosen})
    return additions, details


@torch.no_grad()
def acquire_mwac(model, data, pool_ids, selected, rng, method, mc_passes, seed):
    """Use Experiment B MWAC filtering; rank its window by entropy or MC MI."""
    additions, details = [], []
    y = data.labels.cpu().numpy()
    for cls in range(5):
        available = np.asarray([i for i in pool_ids if y[i] == cls and i not in selected], dtype=int)
        if not len(available):
            raise ValueError(f"Pool exhausted for class {cls}")
        ids = rng.choice(available, min(100, len(available)), replace=False)
        pseudo_human = data.labels[ids]

        # Keep Experiment B's MWAC score, descending sort and median window.
        model.eval()
        evaluator_p = model(pseudo_human, data.features[ids]).softmax(-1)
        boundary = (evaluator_p.gather(1, pseudo_human[:, None]).squeeze(1) - .5).abs().cpu().numpy()
        order = np.argsort(-boundary, kind="stable")
        window_start = min(55, max(0, len(ids) - 5))
        window = order[window_start:window_start + 5]

        if method == "mwac_ai_entropy":
            scores = ai_entropy(data.probs[ids])[window]
            metric_name = "AI predictive Shannon entropy"
        elif method == "mwac_mc_mi":
            # Existing implementation returns MC probability samples. Its
            # reliability_signals function computes MI = H(mean p) - mean H(p).
            samples = predict_samples(
                model, data.features[ids[window]], pseudo_human[window], data.probs[ids[window]],
                passes=mc_passes, seed=seed + cls,
            )
            signals = reliability_signals(samples, pseudo_human[window])
            scores = signals["mi"].detach().cpu().numpy()
            metric_name = "Mutual information (existing reliability_signals MI)"
        else:
            raise ValueError(f"Unknown MWAC method: {method}")

        local_window = np.flatnonzero(np.isclose(scores, scores.max()))
        window_pos = int(rng.choice(local_window))
        local = int(window[window_pos])
        chosen = int(ids[local])
        additions.append(chosen)
        details.append({
            "class": cls,
            "candidate_count": int(len(ids)),
            "mwac_window_start": int(window_start),
            "mwac_window_length": int(len(window)),
            "mwac_boundary_score": float(boundary[local]),
            "uncertainty_metric": metric_name,
            "uncertainty_score": float(scores[window_pos]),
            "selected_id": chosen,
        })
    return additions, details


def evaluate(model, test, seed):
    # Same deterministic single-pass fusion evaluation for all methods.
    with torch.no_grad():
        fused = predict_samples(model, test.features, test.human, test.probs, passes=1, seed=seed)[0]
    result = metrics(fused, test.labels)
    return {name: float(result[name]) for name in METRICS}


def paired_comparisons(frame):
    detail_rows = []
    grouped = []
    idx = ["fold", "seed", "labels_per_class"]
    for left, right in PAIRS:
        left_df = frame[frame.method == left].set_index(idx)
        right_df = frame[frame.method == right].set_index(idx)
        differences = left_df[list(METRICS)] - right_df[list(METRICS)]
        for (fold, seed, budget), values in differences.iterrows():
            for metric_name, value in values.items():
                detail_rows.append({
                    "comparison": f"{left} - {right}", "fold": fold, "seed": seed,
                    "labels_per_class": budget, "total_queries": budget * 5,
                    "metric": metric_name, "difference": float(value),
                })
        for (budget, metric_name), values in differences.stack().groupby(level=[2, 3]):
            vals = values.to_numpy(dtype=float)
            if len(vals) >= 2:
                t_result = ttest_rel(
                    left_df.xs(budget, level="labels_per_class")[metric_name].to_numpy(),
                    right_df.xs(budget, level="labels_per_class")[metric_name].to_numpy(),
                )
                p_value = float(t_result.pvalue)
                std_difference = float(vals.std(ddof=1))
            else:
                p_value = float("nan")
                std_difference = float("nan")
            grouped.append({
                "comparison": f"{left} - {right}", "labels_per_class": int(budget),
                "total_queries": int(budget) * 5, "metric": metric_name,
                "mean_difference": float(vals.mean()),
                "std_difference": std_difference,
                "paired_ttest_pvalue": p_value,
                "n_paired_runs": int(len(vals)),
            })
    detail = pd.DataFrame(detail_rows)
    aggregate = pd.DataFrame(grouped)
    return detail, aggregate


def _svg_line_chart(frame, metric_name, y_label, title, path, y_max=1.0):
    width, height = 1000, 640
    left, right, top, bottom = 100, 280, 72, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    budgets = sorted(frame.labels_per_class.unique())
    queries = [b * 5 for b in budgets]
    colors = {"random": "#697586", "mwac_ai_entropy": "#1976d2", "mwac_mc_mi": "#d97706"}
    labels = {"random": "Random", "mwac_ai_entropy": "MWAC + AI Entropy", "mwac_mc_mi": "MWAC + MC Dropout (MI)"}
    xmax = max(queries)
    xmap = lambda value: left + value / xmax * plot_w
    ymap = lambda value: top + (y_max - value) / y_max * plot_h
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#18212f}.small{font-size:13px}.axis{stroke:#344054;stroke-width:1.5}</style>',
        f'<text x="{left}" y="35" font-size="22" font-weight="bold">{title}</text>',
    ]
    for tick_i in range(0, 6):
        value = y_max * tick_i / 5
        yy = ymap(value)
        svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e7ebf0"/>')
        svg.append(f'<text class="small" x="{left-12}" y="{yy+4:.1f}" text-anchor="end">{value:.2f}</text>')
    for q in queries:
        xx = xmap(q)
        svg.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_h}" stroke="#f1f3f6"/>')
        svg.append(f'<text class="small" x="{xx:.1f}" y="{top+plot_h+23}" text-anchor="middle">{q}</text>')
    svg.extend([
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-size="16">Total Expert Queries</text>',
        f'<text transform="translate(26 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-size="16">{y_label}</text>',
    ])
    for method in METHODS:
        stat = frame[frame.method == method].groupby("labels_per_class")[metric_name].agg(["mean", "std"]).reindex(budgets)
        points = []
        for budget, mean, std in zip(budgets, stat["mean"], stat["std"].fillna(0)):
            xx, yy = xmap(budget * 5), ymap(float(mean))
            err = float(std) / y_max * plot_h
            svg.append(f'<line x1="{xx:.1f}" y1="{yy-err:.1f}" x2="{xx:.1f}" y2="{yy+err:.1f}" stroke="{colors[method]}" stroke-width="1.6"/>')
            svg.append(f'<line x1="{xx-5:.1f}" y1="{yy-err:.1f}" x2="{xx+5:.1f}" y2="{yy-err:.1f}" stroke="{colors[method]}"/>')
            svg.append(f'<line x1="{xx-5:.1f}" y1="{yy+err:.1f}" x2="{xx+5:.1f}" y2="{yy+err:.1f}" stroke="{colors[method]}"/>')
            points.append(f"{xx:.1f},{yy:.1f}")
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[method]}" stroke-width="3"/>')
        for point in points:
            xx, yy = point.split(",")
            svg.append(f'<circle cx="{xx}" cy="{yy}" r="4.5" fill="{colors[method]}"/>')
    for i, method in enumerate(METHODS):
        yy = top + 20 + i * 30
        svg.append(f'<line x1="{left+plot_w+25}" y1="{yy}" x2="{left+plot_w+55}" y2="{yy}" stroke="{colors[method]}" stroke-width="3"/>')
        svg.append(f'<text class="small" x="{left+plot_w+65}" y="{yy+4}">{labels[method]}</text>')
    svg.append(f'<text class="small" x="{left}" y="{height-5}">Points and error bars show mean ± SD across fold-seed runs (full run: n=15 per budget).</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def _svg_final_chart(frame, path):
    width, height = 1180, 620
    colors = {"random": "#697586", "mwac_ai_entropy": "#1976d2", "mwac_mc_mi": "#d97706"}
    labels = {"random": "Random", "mwac_ai_entropy": "MWAC + AI Entropy", "mwac_mc_mi": "MWAC + MC Dropout (MI)"}
    panels = [("accuracy", "Accuracy", 1.0), ("macro_f1", "Macro-F1", 1.0), ("ece", "ECE (lower is better)", 0.5)]
    stats = frame[frame.labels_per_class == frame.labels_per_class.max()].groupby("method")[list(m[0] for m in panels)].agg(["mean", "std"]).reindex(METHODS)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#18212f}.small{font-size:12px}</style>',
           '<text x="35" y="35" font-size="22" font-weight="bold">Final Budget: 40 Expert Queries</text>']
    panel_w, top, base, chart_h = 350, 95, 500, 350
    for panel_idx, (metric, title, ymax) in enumerate(panels):
        x0 = 35 + panel_idx * 380
        svg.append(f'<text x="{x0+panel_w/2}" y="75" text-anchor="middle" font-size="17" font-weight="bold">{title}</text>')
        svg.append(f'<line x1="{x0+35}" y1="{base}" x2="{x0+panel_w-5}" y2="{base}" stroke="#344054"/>')
        for i, method in enumerate(METHODS):
            mean = float(stats.loc[method, (metric, "mean")])
            std = float(stats.loc[method, (metric, "std")])
            std = 0.0 if not np.isfinite(std) else std
            bar_w = 56
            bx = x0 + 55 + i * 91
            bh = mean / ymax * chart_h
            by = base - bh
            svg.append(f'<rect x="{bx}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{colors[method]}"/>')
            ey = std / ymax * chart_h
            svg.append(f'<line x1="{bx+bar_w/2}" y1="{by-ey:.1f}" x2="{bx+bar_w/2}" y2="{by+ey:.1f}" stroke="#111827" stroke-width="1.5"/>')
            svg.append(f'<text class="small" x="{bx+bar_w/2}" y="{base+18}" text-anchor="middle">{labels[method].replace("MWAC + ", "")}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def render_pngs(out):
    # macOS's system AppKit can rasterize SVG without adding Python packages.
    svg_png_pairs = [(out / f"{name}.svg", out / f"{name}.png") for name in
                     ("accuracy_vs_budget", "macro_f1_vs_budget", "ece_vs_budget", "final_budget_comparison")]
    if sys.platform == "darwin":
        swift = r'''import AppKit
import Foundation
let args = Array(CommandLine.arguments.dropFirst())
for i in stride(from: 0, to: args.count, by: 2) {
  let source = URL(fileURLWithPath: args[i])
  let target = URL(fileURLWithPath: args[i + 1])
  guard let image = NSImage(contentsOf: source) else { fatalError("Could not load SVG: \(source.path)") }
  let size = image.size
  guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width), pixelsHigh: Int(size.height),
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
        let context = NSGraphicsContext(bitmapImageRep: bitmap) else { fatalError("Could not create bitmap: \(source.path)") }
  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = context
  NSColor.white.setFill()
  NSRect(origin: .zero, size: size).fill()
  image.draw(in: NSRect(origin: .zero, size: size), from: .zero, operation: .sourceOver, fraction: 1)
  context.flushGraphics()
  NSGraphicsContext.restoreGraphicsState()
  guard let png = bitmap.representation(using: .png, properties: [:]) else { fatalError("Could not encode PNG: \(source.path)") }
  try png.write(to: target)
}
'''
        cache = Path(tempfile.gettempdir()) / "experiment_c_swift_module_cache"
        subprocess.run(["swift", "-module-cache-path", str(cache), "-e", swift,
                        *[str(p) for pair in svg_png_pairs for p in pair]], check=True)
    else:
        try:
            import cairosvg
        except ImportError as exc:
            raise RuntimeError("PNG export needs AppKit on macOS or cairosvg on other platforms") from exc
        for source, target in svg_png_pairs:
            cairosvg.svg2png(url=str(source), write_to=str(target), output_width=2000)


def plot_budget(frame, out):
    _svg_line_chart(frame, "accuracy", "Fusion Accuracy", "Experiment C: Accuracy vs Expert Query Budget", out / "accuracy_vs_budget.svg")
    _svg_line_chart(frame, "macro_f1", "Macro-F1", "Macro-F1 vs Expert Query Budget", out / "macro_f1_vs_budget.svg")
    _svg_line_chart(frame, "ece", "ECE (lower is better)", "Calibration Error vs Expert Query Budget", out / "ece_vs_budget.svg", y_max=max(.2, float(frame.ece.max()) * 1.2))
    _svg_final_chart(frame, out / "final_budget_comparison.svg")
    render_pngs(out)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    if args.epochs < 1 or args.mc_passes < 2:
        raise ValueError("epochs must be positive and MC Dropout needs at least two passes")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, traces, initial_by_run = [], [], {}
    for fold in args.folds:
        train = load_csv(args.data_dir / f"model_output_{fold}_train.csv").to(args.device)
        test = load_csv(args.data_dir / f"model_output_{fold}_test.csv").to(args.device)
        pool_ids = np.arange(len(train.labels))
        for seed in args.seeds:
            run_seed = fold * 1000 + seed
            init_rng = np.random.default_rng(run_seed)
            common_initial = initial_ids(train.labels, pool_ids, 2, init_rng)
            initial_by_run[f"fold{fold}_seed{seed}"] = common_initial
            for method_index, method in enumerate(METHODS):
                # Independent stream per method, paired initial examples and
                # identical model initialization at each budget.
                rng = np.random.default_rng(run_seed + 100_000 * (method_index + 1))
                selected = list(common_initial)
                method_trace = []
                for budget in range(2, 9):
                    model, train_loss = train_evaluator(train, selected, run_seed, epochs=args.epochs)
                    score_row = evaluate(model, test, run_seed + budget)
                    rows.append({
                        "fold": fold, "seed": seed, "method": method,
                        "labels_per_class": budget, "total_queries": budget * 5,
                        "active_human_queries": len(selected), "train_loss": train_loss,
                        **score_row,
                    })
                    event = {
                        "labels_per_class": budget, "total_queries": budget * 5,
                        "selected_ids": list(selected), "train_loss": train_loss,
                        "test_metrics": score_row,
                    }
                    if budget < 8:
                        if method == "random":
                            new_ids, detail = acquire_random(train, pool_ids, selected, rng)
                        else:
                            new_ids, detail = acquire_mwac(
                                model, train, pool_ids, selected, rng, method,
                                args.mc_passes, run_seed + budget + 300_000 * method_index,
                            )
                        # Human Labels are read only by train_evaluator after
                        # these training IDs are selected.
                        selected.extend(new_ids)
                        event.update({"newly_queried_ids": new_ids, "acquisition": detail})
                    method_trace.append(event)
                traces.append({"fold": fold, "seed": seed, "method": method, "rounds": method_trace})

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "per_fold_budget.csv", index=False)
    summary = frame.groupby(["method", "labels_per_class"])[list(METRICS)].agg(["mean", "std"])
    summary.to_csv(args.output / "summary.csv")
    paired_detail, paired_summary = paired_comparisons(frame)
    # The requested per fold/seed differences and their mean/SD are both kept.
    paired_detail.to_csv(args.output / "paired_differences.csv", index=False)
    paired_summary.to_csv(args.output / "paired_difference_summary.csv", index=False)
    (args.output / "acquisition_traces.json").write_text(json.dumps({
        "common_initial_ids_by_fold_seed": initial_by_run, "runs": traces,
    }, indent=2))
    plot_budget(frame, args.output)

    final_budget = int(frame.labels_per_class.max())
    final_rows = frame[frame.labels_per_class == final_budget]
    final_summary = final_rows.groupby("method")[list(METRICS)].agg(["mean", "std"])
    final_summary.columns = [f"{metric}_{stat}" for metric, stat in final_summary.columns]
    final_summary = final_summary.reindex(METHODS)
    final_pair_summary = paired_summary[paired_summary.labels_per_class == final_budget].to_dict("records")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = "unavailable"
    report = {
        "experiment": "C: random vs MWAC + AI entropy vs MWAC + MC Dropout",
        "research_question": "At equal expert query budgets, does entropy or MC Dropout uncertainty acquisition outperform random acquisition?",
        "dataset": "DR-5 cached train/test CSVs",
        "folds": args.folds, "seeds": args.seeds,
        "initial_labels_per_class": 2,
        "query_budgets_total": [10, 15, 20, 25, 30, 35, 40],
        "candidate_pool_size": 100,
        "mwac_window_start": 55, "mwac_window_length": 5,
        "mc_dropout_forward_passes": args.mc_passes,
        "mc_dropout_uncertainty_metric": "Mutual information, computed by existing reliability_signals as H(mean dropout prediction) - mean(H(each dropout prediction)); no failure-frequency blend is used.",
        "ai_entropy_metric": "Predictive Shannon entropy of the cached AI probability vector: -sum(p * log(p)).",
        "evaluator_training_epochs": args.epochs,
        "evaluator_and_fusion": "Existing src.dr5.core train_evaluator, predict_samples and product-rule fusion; test evaluation uses the same deterministic single-pass fusion for all three methods.",
        "method_details": {
            "random": "One unqueried training case per class sampled uniformly at each round.",
            "mwac_ai_entropy": "Experiment B MWAC candidate sampling, boundary ranking and window, then highest cached-AI predictive entropy in the window.",
            "mwac_mc_mi": "Same MWAC filtering and window, then highest existing MC-dropout mutual information in the window.",
        },
        "paired_initial_ids": "Identical across all three methods for each fold/seed; written in acquisition_traces.json.",
        "test_leakage_controls": "Acquisition reads training-fold features, AI probabilities and known class labels only. Human Label is consumed only after a training example is queried. Test data is loaded for evaluation only; test labels and Human Labels do not enter acquisition or evaluator training.",
        "test_inference": "Existing fusion evaluation requires test Human Label as the evaluator input; these values are used only to calculate the common test metric, never for selection.",
        "random_seeds": {"initial_ids": "fold*1000 + seed", "evaluator_initialization": "fold*1000 + seed at each budget for every method", "method_acquisition_rng": "fold*1000 + seed + 100000*(method index + 1)", "mc_sampling": "run seed + budget + 300000*method index + class"},
        "git_commit": commit,
        "source_sha256": {
            "experiment_c": sha256(Path(__file__)),
            "dr5_core": sha256(ROOT / "src" / "dr5" / "core.py"),
            "experiment_b": sha256(ROOT / "experiments" / "analysis" / "experiment_b_referral.py"),
        },
        "input_sha256": {
            f"fold{fold}_{split}": sha256(args.data_dir / f"model_output_{fold}_{split}.csv")
            for fold in args.folds for split in ("train", "test")
        },
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "final_40_query_summary": final_summary.to_dict(orient="index"),
        "final_40_query_paired_differences": final_pair_summary,
        "paired_test": "Two-sided paired t-tests over fold-seed runs are reported per budget for descriptive context; n=15 in the full run and no multiplicity adjustment. Interpret cautiously.",
        "limitations": [
            "Existing folds were inspected in prior experiments and are exploratory, not an untouched confirmatory test set.",
            "The dataset has no patient/image identifiers, so patient-disjointness cannot be independently verified.",
            "MC mutual information reflects evaluator dropout uncertainty conditional on pseudo human response h=y and cached AI probabilities; it is not a calibrated clinical error probability.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("Final 40-query mean metrics:")
    print(final_rows.groupby("method")[list(METRICS)].mean().reindex(METHODS).round(4).to_string())
    print(f"\nOutputs saved to {args.output}")


if __name__ == "__main__":
    main(args_parser().parse_args())
