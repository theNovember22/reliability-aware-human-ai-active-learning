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

## Experiment B: risk-guided referral

`experiments/analysis/experiment_b_referral.py` simulates sending the highest-risk
AI predictions to a human and measures only errors among the cases left for
automatic handling. The table reports mean residual error over five existing
test folds (200 cases/fold); the baseline AI error rate is 51.1%.

| Referral score | Residual error at 10% referral | 20% | 30% |
|---|---:|---:|---:|
| Random referral | 51.1% | 51.1% | 51.1% |
| Confidence risk (1 − max probability) | 49.7% | 48.3% | 47.3% |
| Entropy | 49.7% | 48.6% | 46.9% |
| Negative top-two margin | 51.7% | 50.6% | 48.9% |
| Learned uncertainty | 50.2% | 49.9% | 48.3% |
| Learned features | 50.4% | 49.9% | 49.9% |
| Learned combined | 49.9% | 50.5% | 50.0% |

The plot and full referral-rate grid are in
`results/experiment_b_referral/referral_curve.svg` and
`results/experiment_b_referral/referral_curve.csv`. The random curve averages
1,000 without-replacement referral draws per fold; its shaded band shows ±1 SD.
At 30% referral, entropy has the lowest observed residual error (46.9%), a
4.2 percentage-point reduction from the 51.1% baseline. These are exploratory
results on previously inspected test folds. They do not measure corrected
predictions or final accuracy after human review.

MI could not be included as an AI referral score: the cached files contain one
AI probability vector per case, without MC samples or the AI model/image inputs
needed to generate them. Existing MC-dropout outputs are for the human-AI
fusion evaluator and depend on a human label, so they are not an available
pre-review routing signal. MI remains a follow-up once AI MC predictions are
saved for each case.

## Experiment D: robustness checks

`experiments/analysis/experiment_d_robustness.py` ran three exploratory checks
on the available folds and saved the full per-fold tables under
`results/experiment_d_robustness/`.

### D1: class-specific AI error ranking

AUROC is calculated within each true grade, so class prevalence does not make
one grade's errors look like another grade's. Mean AUROC across the five
existing folds:

| True grade | Confidence risk | Entropy | Learned uncertainty | Learned combined |
|---:|---:|---:|---:|---:|
| 0 | 0.668 | 0.703 | 0.639 | 0.539 |
| 1 | 0.449 | 0.471 | 0.463 | 0.415 |
| 2 | 0.339 | 0.265 | 0.403 | 0.465 |
| 3 | 0.719 | 0.731 | 0.683 | 0.601 |
| 4 | 0.627 | 0.643 | 0.595 | 0.554 |

The simple scores rank errors for grades 0, 3, and 4, but perform below chance
within grades 1 and 2. Grade 2 has a high error rate (70.5%), yet confidence
and entropy rank which grade-2 examples fail poorly. This indicates strong
class dependence in the current risk behavior.

### D2: ordinal mistake severity

Errors are minor when the predicted grade is one step from truth and severe
when it is at least two steps away. Severe cases make up 16.9% of the test
examples. Entropy has the best mean severe-versus-other AUROC (0.687); its
Spearman correlation with absolute grade distance is 0.231. Confidence risk
gives AUROC 0.647 and correlation 0.194. For confidence, mean risk rises from
0.433 on correct predictions to 0.467 for minor errors and 0.522 for severe
errors; entropy rises from 1.018 to 1.078 to 1.221. The learned feature-only
score is nearly flat across these severity groups.

### D3: evaluator-seed variation

The cached AI backbone outputs do not have documented seed provenance. For a
limited model-variation check, I evaluated confidence/entropy/margin on the
saved human-AI evaluator predictions at budget 8 across evaluator seeds 1–3.
Mean error AUROC by seed was 0.680/0.678/0.680 for entropy and
0.684/0.671/0.689 for confidence. Error rates were 43.0%, 43.6%, and 43.9%.
The ranking results are fairly similar across these three evaluator seeds,
but this does not establish robustness to different retinal AI backbones.

All D results use previously inspected test folds without patient IDs or split
provenance; they are descriptive and exploratory. D3 varies the downstream
evaluator seed, not the image-model seed. Class-conditional calculations use
true grade for post-hoc analysis and are not deployable class-specific routing
rules.
