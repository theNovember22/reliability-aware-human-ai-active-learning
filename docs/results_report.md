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

## Experiment A: predicting AI errors

`experiments/analysis/error_risk_baselines.py` evaluates whether fixed AI
uncertainty scores rank incorrect predictions above correct ones. For each of
five folds, a logistic regression over the five probabilities and three
derived uncertainty features is fit on 80% of the training CSV and assessed
on the remaining 20%. No human labels are used as inputs. Metrics below are
the mean of the five fold metrics; fold standard deviations are in
`results/error_risk_baselines/summary.csv`.

| Split | Score | AUROC ↑ | Average precision ↑ | Error rate after 10% referral | after 20% | after 30% |
|---|---|---:|---:|---:|---:|---:|
| Validation | Confidence risk (1 − max probability) | 0.715 | 0.588 | 36.7% | 31.6% | 29.3% |
| Validation | Negative top-two margin | 0.711 | 0.590 | 36.0% | 32.7% | 29.1% |
| Validation | Entropy | 0.677 | 0.544 | 37.4% | 33.4% | 31.3% |
| Validation | Learned probability | 0.709 | 0.568 | 36.8% | 33.9% | 30.2% |
| Test, exploratory | Confidence risk | 0.592 | 0.590 | 49.7% | 48.3% | 47.3% |
| Test, exploratory | Negative top-two margin | 0.566 | 0.543 | 51.7% | 50.6% | 48.9% |
| Test, exploratory | Entropy | 0.606 | 0.601 | 49.7% | 48.6% | 46.9% |
| Test, exploratory | Learned probability | 0.573 | 0.562 | 50.7% | 49.8% | 49.0% |

The separate 512-feature learned-risk analysis (`experiments/analysis/learned_error_risk.py`)
also compared learned uncertainty, feature-only, and combined models. Mean AUROC
for validation / exploratory test was **0.711 / 0.578** for learned uncertainty,
**0.546 / 0.503** for features alone, and **0.648 / 0.533** for combined.
The learned uncertainty model's validation Brier score was 0.207 (test 0.262);
the combined model's was 0.280 (test 0.370). The 512 image features alone and
their combination with uncertainty failed to improve the ranking baselines.

Validation error prevalence is 38.9%; at 30% referral, confidence and margin
reduce the remaining error rate to about 29%. Test error prevalence is 51.1%,
and entropy has the strongest exploratory ranking (AUROC 0.606); referral
reduces remaining error only to 46.9% at 30%. The learned score does not beat
the simple uncertainty baselines on either split. Its validation Brier score
is 0.207 and 10-bin ECE is 0.080; exploratory test values worsen to 0.265 and
0.127. Calibration values are reported only for the learned probability,
because raw confidence, entropy and margin are ranking scores rather than
estimated error probabilities.

**Decision:** use confidence risk as the simple provisional score based on
validation results; do not claim the proposed learned score is superior.
Experiment A has been completed as an exploratory analysis on the available
CSVs, but not as a confirmatory experiment: the test folds were previously
inspected, and the files have no patient/image IDs. Confirmatory completion
requires a fresh untouched test set with verifiable split provenance.

## Experiment B: entropy-ranked MWAC active annotation

`experiments/analysis/experiment_b_referral.py` now follows an iterative active
annotation workflow. It starts with two human annotations per class, fits the
evaluator, forms a class-balanced MWAC candidate set and its median window,
then ranks only that window by AI predictive entropy (the same risk score used
in Experiment A). The highest-entropy case in each class is sent for human
annotation. Its human label is added to the training set before refitting the
evaluator at the next budget. Test evaluator metrics are saved at each budget.

Results are written to `results/experiment_b_referral/per_fold_budget.csv`,
`summary.csv`, `report.json`, and `acquisition_traces.json`. The five existing
test folds have been inspected previously, so these results are exploratory.
Entropy is a ranking score, not a calibrated error probability. The simulation
uses the known ground-truth class to form the MWAC candidate pool, as assumed
by the ActiveHAI setting; only selected cases' human labels are revealed to
the evaluator during acquisition.

The observed test accuracy across five folds and three evaluator seeds was:

| Human labels per class | Accuracy (mean ± SD) |
|---:|---:|
| 2 | 48.5% ± 7.1% |
| 3 | 52.7% ± 4.6% |
| 4 | 54.3% ± 3.6% |
| 5 | 56.5% ± 4.1% |
| 6 | 57.4% ± 3.0% |
| 7 | 58.2% ± 3.2% |
| 8 | 58.9% ± 3.5% |

These are measurements along one acquisition policy, not a comparison proving
that entropy improves over another policy.

## Experiment C: reliability-adaptive fusion

`experiments/analysis/experiment_c_reliability_weighted_fusion.py` tests the
thesis question directly: does a per-case evaluator reliability estimate
improve Human–AI fusion when it changes the evaluator's contribution? Both
conditions use the exact same Experiment B entropy-ranked MWAC acquisition
path, acquired labels, trained evaluator checkpoints, MC-dropout samples, and
test set. The fixed condition uses evaluator coefficient `beta=1`; the
adaptive condition uses `beta_i = clip(1 - MI_i/log(5), 0, 1)`. MI comes from
the existing MC-dropout implementation and does not use test labels.

KL is calculated as `KL(one-hot true label || fused prediction)`. With a
one-hot categorical target, this is mathematically the same as NLL.

Mean test metrics (± SD across 15 fold-seed runs):

| Labels/class | Total queries | Fixed accuracy | Adaptive accuracy | Fixed KL | Adaptive KL |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 48.33% ± 7.15% | 48.37% ± 7.11% | 1.5022 ± 0.2030 | 1.4989 ± 0.2020 |
| 3 | 15 | 52.73% ± 4.94% | 52.70% ± 4.92% | 1.3870 ± 0.1447 | 1.3842 ± 0.1442 |
| 4 | 20 | 54.33% ± 3.89% | 54.40% ± 3.85% | 1.3426 ± 0.1402 | 1.3402 ± 0.1395 |
| 5 | 25 | 56.27% ± 4.37% | 56.30% ± 4.35% | 1.2815 ± 0.1406 | 1.2795 ± 0.1400 |
| 6 | 30 | 57.87% ± 2.98% | 57.90% ± 2.87% | 1.2318 ± 0.1071 | 1.2301 ± 0.1064 |
| 7 | 35 | 58.20% ± 3.26% | 58.20% ± 3.35% | 1.1887 ± 0.1085 | 1.1873 ± 0.1079 |
| 8 | 40 | 58.57% ± 3.54% | 58.60% ± 3.59% | 1.1671 ± 0.1056 | 1.1659 ± 0.1050 |

At 40 queries, the paired adaptive-minus-fixed accuracy difference was
**+0.03 percentage points** (SD 0.23 pp; paired p=0.582). The KL difference
was **−0.00122** (SD 0.00078; paired p=0.00003, unadjusted). The observed mean
MI at the endpoint was 0.0083 nats, corresponding to a mean evaluator weight
of 0.9948. Thus the predeclared weight rule changed predictions only slightly:
it reduced KL, but did not produce a meaningful accuracy gain in these runs.
The KL result is not independent of NLL, and the existing folds are
exploratory; these results do not establish improved final accuracy.

Full fold-seed metrics and paired differences, reliability weights, traces,
and figures are in `results/experiment_c_reliability_weighted_fusion/`.

### Train-only weight-tuning follow-up

I also tested leave-one-out tuning of both machine and queried-expert log
weights (`alpha` and `beta`) on the same Experiment B query IDs. The candidate
grid and confusion-prior mass were selected by LOO NLL using acquired training
responses only. This **did not improve** the result:

| Fusion method | Accuracy at 40 queries | KL to one-hot truth |
|---|---:|---:|
| Queried-confusion fusion, fixed AI weight | 59.37% ± 2.62% | 1.0625 ± 0.0574 |
| LOO-tuned AI and expert weights | 57.20% ± 3.11% | 1.2422 ± 0.0900 |
| Neural evaluator, fixed product fusion | 58.57% ± 3.53% | 1.1671 ± 0.1056 |

The LOO-tuned version lost 2.17 percentage points versus the fixed-AI
confusion fusion (paired p=0.0022, unadjusted). The fitted AI weight often
landed near the top of its grid, but that train-only LOO preference did not
transfer to the explored test folds. Increasing fusion weights based on the
same small queried sample is therefore not a reliable way to force the target.
Details are in `results/experiment_c_fusion_weight_calibration/`.

The strongest previously measured candidate remains the separate
budget-preserving diverse-acquisition fusion result: 60.97% ± 1.89% in
[`docs/budget_accuracy.md`](budget_accuracy.md). This is +4.47 percentage
points versus the earlier 56.50% corrected MWAC + AI-temperature result in
this report, but only +2.37 points versus the current 58.60% adaptive-fusion
result. Those runs change acquisition and/or fusion method, so this gap does
not isolate the reliability weight and does not establish a 4–5 point causal
gain.
