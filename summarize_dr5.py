"""Rebuild the exploratory report from saved experiment results."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    'baseline_uncalibrated': 'MWAC, no AI calibration',
    'baseline': 'Corrected MWAC + AI temperature',
    'mc': 'MWAC + MC inference',
    'reliability': 'Reliability-aware MWAC + MC',
    'random': 'Random acquisition',
    'reliability_calibrated': 'Reliability + fusion calibration',
}


def main():
    frames = {key: pd.read_csv(ROOT/'results'/key/'metrics.csv') for key in VARIANTS}
    for name, d in frames.items():
        if len(d) != 105 or d.duplicated(['fold','seed','labels_per_class']).any():
            raise ValueError(f'{name}: expected 15 runs at seven budgets')
    lines = ['# Measured DR5 results', '',
        'Executed on the supplied cached features, CPU, 2026-10-05. Five folds × three seeds, '
        '100 epochs/round. These are exploratory measurements, not a verified reproduction of the paper.', '',
        '## Endpoint: eight queried doctor labels per class', '',
        'Accuracy is mean ± sample standard deviation across 15 runs. ECE uses 15 equal-width bins. '
        'Repeated seeds share test folds; the standard deviation is descriptive, not a confidence interval.', '',
        '| Method | Accuracy (%) | Macro-F1 | QWK | ECE ↓ | NLL ↓ | Total development doctor queries |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in VARIANTS.items():
        d=frames[name].query('labels_per_class == 8')
        lines.append(f"| {label} | {100*d.accuracy.mean():.2f} ± {100*d.accuracy.std():.2f} | "
                     f"{d.macro_f1.mean():.4f} | {d.qwk.mean():.4f} | {d.ece.mean():.4f} | "
                     f"{d.nll.mean():.4f} | {int(d.total_human_queries.iloc[0])} |")
    b=frames['baseline'].query('labels_per_class == 8')
    lines.extend(['',f"Frozen AI mean accuracy: **{100*b.ai_accuracy.mean():.2f}%**. "
        f"Human mean accuracy: **{100*b.human_accuracy.mean():.2f}%**.", '',
        'The reliability variant improves corrected MWAC by **1.17 percentage points**, '
        'but trails random acquisition by **1.67 points**. This does not establish superiority '
        'of reliability-aware acquisition. The uncalibrated-AI control also exceeds the calibrated '
        'MWAC baseline, so AI temperature fitting was not uniformly beneficial.', '',
        'Post-fusion temperature scaling reduces mean ECE from **0.1917 to 0.1159**, '
        'while leaving accuracy unchanged. It consumes 100 extra doctor labels, for a total '
        'of 140 rather than 40. This is not an equal-cost calibration comparison.', '',
        '## Accuracy by active annotation budget', '',
        '| Labels/class | Total active queries | MWAC + AI TS | MC | Reliability | Random |',
        '|---:|---:|---:|---:|---:|---:|'])
    for budget in range(2,9):
        values=[100*frames[k].query('labels_per_class == @budget').accuracy.mean()
                for k in ['baseline','mc','reliability','random']]
        lines.append(f'| {budget} | {5*budget} | '+' | '.join(f'{x:.2f}%' for x in values)+' |')
    lines.extend(['', '## Paired fold comparison at eight labels/class', '',
        'Each fold value averages the same three seeds.', '',
        '| Fold | MWAC | Reliability | Difference (percentage points) |', '|---:|---:|---:|---:|'])
    r=frames['reliability'].query('labels_per_class == 8')
    for fold in range(5):
        bv=b.query('fold == @fold').accuracy.mean()*100
        rv=r.query('fold == @fold').accuracy.mean()*100
        lines.append(f'| {fold} | {bv:.2f}% | {rv:.2f}% | {rv-bv:+.2f} |')
    lines.extend(['', '## Validation and limitations', '',
        '- Five unit checks passed: product-fusion loss/gradients; MC variance and state/RNG restoration; '
        'temperature fitting; ECE boundaries; balanced, disjoint and reproducible acquisition.',
        '- Audit passed for all ten CSVs: finite values, normalized five-class probabilities, '
        '512 features, balanced labels, no identical train/test feature rows within a fold.',
        '- Baseline/MC query histories matched exactly. Reliability/fusion-calibrated query histories '
        'also matched exactly. This isolates those inference/calibration comparisons.',
        '- No patient IDs, raw images, backbone checkpoints or feature-generation scripts are supplied. '
        'Exact feature duplicate checks cannot rule out patient or backbone leakage.',
        '- Original historical script fails in this environment at its unused torchvision import. '
        'It was not run to completion, and its test-selected accuracy is not reported as a valid baseline.',
        '- Stopping demonstration (fold 0, seed 1) stopped at five labels/class, using 25 active '
        'queries plus 50 validation queries. Its 58% accuracy is a single functionality check, '
        'not evidence that stopping improves performance.',
        '- The paper reports 59.98% at eight labels/class; our corrected protocol differs in several '
        'material ways and uses three rather than ten repeats/fold. Do not present their difference '
        'as a like-for-like gain or loss.',
        '- Do not tune methods against these now-observed test results. Use nested validation '
        'and a fresh evaluation protocol for the next comparison.', '',
        'See `methodology.md` for equations, query accounting and the next experiments. '
        'The `results/` folders contain full predictions, checkpoints, manifests and metrics. '
        'Run `python summarize_dr5.py` to regenerate this report.', ''])
    (ROOT/'docs/results_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:20]))


if __name__ == '__main__':
    main()
