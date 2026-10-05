# Measured DR5 results

Executed on the supplied cached features, CPU, 2026-10-05. Five folds × three seeds, 100 epochs/round. These are exploratory measurements, not a verified reproduction of the paper.

## Endpoint: eight queried doctor labels per class

Accuracy is mean ± sample standard deviation across 15 runs. ECE uses 15 equal-width bins. Repeated seeds share test folds; the standard deviation is descriptive, not a confidence interval.

| Method | Accuracy (%) | Macro-F1 | QWK | ECE ↓ | NLL ↓ | Total development doctor queries |
|---|---:|---:|---:|---:|---:|---:|
| MWAC, no AI calibration | 57.70 ± 2.88 | 0.5755 | 0.6740 | 0.1486 | 1.1844 | 40 |
| Corrected MWAC + AI temperature | 56.50 ± 3.84 | 0.5603 | 0.6719 | 0.2001 | 1.3091 | 40 |
| MWAC + MC inference | 56.67 ± 3.76 | 0.5624 | 0.6734 | 0.1967 | 1.3039 | 40 |
| Reliability-aware MWAC + MC | 57.67 ± 4.17 | 0.5716 | 0.6694 | 0.1917 | 1.2906 | 40 |
| Random acquisition | 59.33 ± 4.84 | 0.5908 | 0.6707 | 0.1843 | 1.2888 | 40 |
| Reliability + fusion calibration | 57.67 ± 4.17 | 0.5716 | 0.6694 | 0.1159 | 1.1111 | 140 |

Frozen AI mean accuracy: **48.90%**. Human mean accuracy: **51.80%**.

The reliability variant improves corrected MWAC by **1.17 percentage points**, but trails random acquisition by **1.67 points**. This does not establish superiority of reliability-aware acquisition. The uncalibrated-AI control also exceeds the calibrated MWAC baseline, so AI temperature fitting was not uniformly beneficial.

Post-fusion temperature scaling reduces mean ECE from **0.1917 to 0.1159**, while leaving accuracy unchanged. It consumes 100 extra doctor labels, for a total of 140 rather than 40. This is not an equal-cost calibration comparison.

## Accuracy by active annotation budget

| Labels/class | Total active queries | MWAC + AI TS | MC | Reliability | Random |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 53.83% | 53.87% | 53.87% | 53.83% |
| 3 | 15 | 55.93% | 56.03% | 55.57% | 55.43% |
| 4 | 20 | 56.83% | 56.60% | 55.63% | 56.23% |
| 5 | 25 | 56.30% | 56.40% | 56.17% | 57.37% |
| 6 | 30 | 55.90% | 56.03% | 57.20% | 58.57% |
| 7 | 35 | 56.23% | 56.40% | 57.63% | 58.77% |
| 8 | 40 | 56.50% | 56.67% | 57.67% | 59.33% |

## Paired fold comparison at eight labels/class

Each fold value averages the same three seeds.

| Fold | MWAC | Reliability | Difference (percentage points) |
|---:|---:|---:|---:|
| 0 | 59.00% | 59.33% | +0.33 |
| 1 | 50.33% | 51.17% | +0.83 |
| 2 | 57.83% | 59.83% | +2.00 |
| 3 | 56.17% | 60.00% | +3.83 |
| 4 | 59.17% | 58.00% | -1.17 |

## Validation and limitations

- Five unit checks passed: product-fusion loss/gradients; MC variance and state/RNG restoration; temperature fitting; ECE boundaries; balanced, disjoint and reproducible acquisition.
- Audit passed for all ten CSVs: finite values, normalized five-class probabilities, 512 features, balanced labels, no identical train/test feature rows within a fold.
- Baseline/MC query histories matched exactly. Reliability/fusion-calibrated query histories also matched exactly. This isolates those inference/calibration comparisons.
- No patient IDs, raw images, backbone checkpoints or feature-generation scripts are supplied. Exact feature duplicate checks cannot rule out patient or backbone leakage.
- Original historical script fails in this environment at its unused torchvision import. It was not run to completion, and its test-selected accuracy is not reported as a valid baseline.
- Stopping demonstration (fold 0, seed 1) stopped at five labels/class, using 25 active queries plus 50 validation queries. Its 58% accuracy is a single functionality check, not evidence that stopping improves performance.
- The paper reports 59.98% at eight labels/class; our corrected protocol differs in several material ways and uses three rather than ten repeats/fold. Do not present their difference as a like-for-like gain or loss.
- Do not tune methods against these now-observed test results. Use nested validation and a fresh evaluation protocol for the next comparison.

See `methodology.md` for equations, query accounting and the next experiments. The `results/` folders contain full predictions, checkpoints, manifests and metrics. Run `python summarize_dr5.py` to regenerate this report.
