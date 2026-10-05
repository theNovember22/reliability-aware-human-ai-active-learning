# Method, diagnosis, and improvement plan

## Sources and scope

1. Zhao et al., *ActiveHAI: Active Collection Based Human-AI Diagnosis with
   Limited Expert Predictions*, IJCAI 2025, pp. 4282–4289. Supplied `0477.pdf`.
   https://www.ijcai.org/proceedings/2025/477
2. Moustapha, Marelli and Sudret, *Active learning for structural reliability:
   Survey, general framework and benchmark*, Structural Safety 96 (2022), 102174.
   https://arxiv.org/abs/2106.01713 and DOI 10.1016/j.strusafe.2021.102174.
   The supplied HTML contains the abstract; the framework was also checked
   against the authors' public arXiv version.
3. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
   https://proceedings.mlr.press/v70/guo17a.html

ActiveHAI assumes pool ground-truth labels are already known. The acquisition
cost is an additional individual doctor's prediction. Using pool class labels
for stratified sampling and pseudo h=y therefore follows this task. It would
not be legitimate in ordinary active learning with unknown ground truth.
Evaluation also assumes a doctor prediction is available for every test case;
reported acquisition budgets concern evaluator development, not clinical usage.

## Problems in the supplied code

| Location | Finding | Corrected runner |
|---|---|---|
| `experiments/baseline/train_dr5.py` | Chooses best checkpoint from test accuracy; averages the five best test epochs | Fixed final epoch; no test-driven selection |
| Same file, acquisition block | Feeds actual unqueried `h_labels` to evaluator | Acquisition function accepts only features, known y and AI probabilities; uses pseudo h=y |
| Same file, training loss | Applies CrossEntropyLoss to normalized fusion probabilities | Stable log-product fusion and NLL |
| Same file, settings | Config epochs are overridden to 50, then the all-label branch to 1 | One explicit CLI epoch setting; all-label comparison is not implemented |
| Same file, metric names | `test_num_m` measures confusion-matrix fusion, not AI alone | Separate AI, human and evaluator-fusion metrics |
| Two MC dropout helpers | Different return signatures, no restoration of mode, not integrated into baseline | One new tested implementation used by the new runner |
| `experiments/analysis/reliability_signal_ablation.py` | Only prints settings; does not execute the ablation | Executable modes with saved predictions and traces |
| `src/baseline/utils.py` | `Evaluator(1280)` silently resets dimension to 512 | Uses validated feature dimension from data |
| Same file | Temperature optimizer's warning path refers to unimported `warnings` | Bounded scalar log-temperature fit |
| Repository setup | Empty requirements and README; feature-only training imports torchvision | Minimal feature-only dependencies and Windows instructions |

The old implementation is retained to make the differences reviewable. The new
runner is not a byte-for-byte reproduction of it. Corrected NLL, holdout AI
calibration, fixed epochs and removal of query/test leakage change the protocol.
The paper's equation for acquisition omits absolute-value notation, while its
following prose describes an absolute difference; this implementation follows
that prose and the existing repository's absolute-difference convention.

## Experiment protocol

Each supplied fold has 800 train and 200 test rows with exactly equal class
counts. Each feature vector has 512 values. Although named `Logits`, the AI
column contains probabilities. Raw images and backbone weights are not needed
for these evaluator experiments, and are not present.

Reserve 20 training rows per class (100 total) for AI temperature fitting,
using a fixed seed 10000+fold. The remaining 700 rows are the active pool.
The baseline never reads the calibration rows' doctor labels. Reserve additional
validation rows only when explicitly requested. These partitions are shared
across methods and seeds and are saved in each checkpoint. All labels needed
for AI calibration are assumed known under the ActiveHAI setting.

Train a fresh evaluator each round with 2,3,...,8 acquired doctor labels per
class, using the same initialization seed within a run. Optimize full-batch
NLL for 100 epochs with Adam, learning rate 3e-4 and cosine scheduling. Train
dropout is 0.1. Features and AI predictions stay fixed. At acquisition, draw
up to 100 available samples per true class and rank by
`abs(E(x, pseudo_h=y)[y] - 0.5)`. Original MWAC randomly chooses within the
zero-based window [55,60). The new small-pool guard moves the window into
the available range rather than returning an empty slice.

Use three seeds on each of the five supplied folds (15 runs per variant).
This is fewer repeats than the paper's ten per fold. Report every budget;
the highlighted endpoint of eight labels/class was fixed before the comparison.
No hyperparameters were tuned using these test results. These results must
now be treated as observed exploratory evidence, not a fresh test set for
iterative method tuning. Future optimization needs nested train-only validation
and ultimately an untouched evaluation set.

## Four modules in detail

### 1. Surrogate and stable fusion

The evaluator is `z_h = Linear(Dropout(feature + Embedding(h)))`.
For AI probabilities p_a, compute

`log p_combined = log_softmax(log(clamp(p_a)) + z_h)`.

NLL of the true grade implements the product rule without underflow. Evaluator
logits can be used directly because their softmax normalizer cancels in fusion.
The feature extractor is frozen by construction: its outputs are CSV inputs.

### 2. Reliability estimation

At MC inference, keep the model in evaluation mode except dropout, then restore
all prior mode flags and RNG state. Average **per-pass fused probabilities**.
MC passes are reproducible and cannot change later training initialization.
The AI backbone has no stochastic passes here; uncertainty is only that of the
evaluator head conditional on the cached features and probabilities.

For each pass t, define margin
`g_t(x,y) = p_t(y) - max_{k != y} p_t(k)`.
Record its mean/std and the fraction of passes with g_t <= 0. Compute mutual
information as `H(mean_t p_t) - mean_t H(p_t)`.

In the unqueried pool we must use pseudo h=y. Consequently the dropout failure
frequency is a **counterfactual model-disagreement proxy**, not the actual
doctor-error probability or a calibrated estimate of real-world clinical risk.
Dropout draws sample model masks, not the structural paper's input distribution.
The analogy is the modular design, not mathematical equivalence to AK-MCS.

Optional validation uses actual held-out doctor labels to measure empirical
combined error. Its development annotation cost is counted explicitly.

### 3. Acquisition

The experimental reliability variant retains class balance and the MWAC
window. Within that window use

`utility = 0.5 * normalized_rank(MI) + 0.5 * 4*pf*(1-pf)`.

Select one utility maximizer per class, breaking ties with the seeded RNG.
This favors epistemic uncertainty and stochastic decision-boundary ambiguity,
not simply the most confidently wrong example. The formula and equal weighting
are a new heuristic, not a formula from either source paper. The current
evaluator is often uncertain for reasons that do not make a useful query;
the random-control result demonstrates that distinction.

### 4. Stopping

The default is a hard query budget, needed for equal-cost comparisons. Optional
validation-NLL plateau stopping halts after `stop-patience` rounds without a
`stop-min-delta` improvement. This stops acquisition and reports the current
model; it does not retroactively select the best test result or restore an
earlier budget. The stopping example is a functionality demonstration, not a
fair 40-query comparison or a validated convergence guarantee.

## Calibration and query accounting

ECE means expected calibration error, here computed in 15 equal-width bins.
It is an evaluation statistic. Temperature scaling fits one positive scalar
by NLL minimization on reserved data. There is no claim it always improves
test ECE, especially on small or distribution-shifted calibration sets.

AI calibration needs known y but no extra doctor predictions. The supplied
CSV provenance does not establish whether that calibration partition was
held out when the backbone itself was trained. Its favorable training behavior
may therefore fail to transfer. The no-AI-calibration control is included.

Optional fusion calibration fits a scalar on the averaged fused distribution
using 100 additional held-out doctor predictions. It leaves the argmax
unchanged and cannot improve top-1 accuracy on a fixed distribution. At the
eight-label endpoint its total development cost is 40+100=140 doctor queries.
Do not compare its calibration gain as if it cost only 40 queries. Future work
should compare 140-query uncalibrated training as a cost-matched control.

## How to pursue higher accuracy next

1. **Use random acquisition as the mandatory benchmark.** The current window
   restriction may exclude useful examples. Test a wider window and a fixed
   random-exploration fraction against the same seeds and total query cost.
   Choose these settings using nested validation, not the test accuracies here.
2. **Improve the reliability target.** With a development set of queried doctor
   labels, estimate disagreement or expected fusion benefit across possible
   human labels, rather than assuming h=y for every candidate. Evaluate whether
   the score predicts actual reduction in held-out fusion NLL or error.
3. **Test feature diversity.** In the uncertain candidate set, select examples
   far from already queried features. Fit any normalization/PCA using only
   the permitted training split. This can reduce redundant expert queries.
4. **Control small-sample overfitting.** Validate embedding/linear weight decay,
   lower-dimensional feature projections, and a smoothed confusion-matrix
   prior. With only 10–40 queried labels, extra model complexity can hurt.
5. **Validate the fusion strength.** Compare calibrated log-linear fusion
   `log p = alpha*log p_AI + beta*log p_h` against the fixed product. Learn the
   weights only from a counted validation set; alpha/beta can change accuracy,
   unlike scalar post-fusion temperature scaling.
6. **Report more than accuracy.** Keep macro-F1, per-grade recall, quadratic
   weighted kappa, NLL, Brier score and ECE. DR grades are ordered and the sample
   is artificially balanced; its accuracy is not a population prevalence estimate.
7. **Verify raw-data provenance before a thesis claim.** Obtain image IDs,
   patient IDs, fold membership and feature-extraction/backbone training scripts.
   Split paired eyes/patients together and generate true out-of-fold AI outputs.
   To improve the frozen AI's 48.9% accuracy itself requires the images and
   backbone-training pipeline, beyond the evaluator data provided here.

Do not promise a target such as 80–90% from MC dropout or the four-module
framework. A defensible project contribution is an explicit acquisition
objective, leakage-controlled protocol, cost accounting and repeatable evidence.
