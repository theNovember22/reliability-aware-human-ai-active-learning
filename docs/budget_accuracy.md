# Higher DR5 accuracy with 40 development expert predictions

Implemented and measured on 2026-10-06, CPU / Python 3.11. The new entry point is
`run_budget_dr5.py`; implementation is `src/dr5/budget.py`. Historical code and
the corrected neural baseline remain available for comparison.

## Measured result

All four variants, five folds and three seeds were trained and sealed before
any new test evaluation. Fixed endpoint: eight responses per grade, 40 total.
Every row below is the mean over the same 15 fold/seed combinations.

| Machine component | Acquisition | Accuracy (%) | Macro-F1 | ECE | Development queries |
|---|---|---:|---:|---:|---:|
| Frozen AI + smoothed expert fusion | Reliability + diversity + exploration | **60.97 +/- 1.89** | 0.6069 | 0.1201 | 40 |
| Frozen AI + smoothed expert fusion | Random | 60.43 +/- 2.96 | 0.6012 | 0.1040 | 40 |
| Feature probe + smoothed expert fusion | Reliability + diversity + exploration | 59.83 +/- 2.30 | 0.5966 | 0.1499 | 40 |
| Feature probe + smoothed expert fusion | Random | 58.47 +/- 3.42 | 0.5839 | 0.1322 | 40 |

The target of more than 60% was reached as an **exploratory mean**, not in every
fold or as a guaranteed future accuracy. SD is across runs, not a confidence
interval: seeds share test cases. No settings were changed in response to these
four variants' test scores. Choosing which variant to highlight is nevertheless
post-evaluation; a fresh evaluation set is needed for a confirmatory claim.

Previously measured neural reliability/MWAC was 57.67%; neural random sampling
was 59.33%. The new best mean is respectively +3.30 and +1.64 percentage points.
Those historical comparisons change fusion, acquisition and AI calibration, so
they do not isolate one cause. The new frozen variants both use the **raw frozen
AI probabilities**, with no AI temperature fitting, and differ only in query
acquisition. Their comparison isolates the new query policy within this fusion
model: only +0.53 points (rounding the displayed means gives +0.54).

## Fold-level paired check

Each entry averages three seeds. This is the appropriate descriptive pairing;
the 15 runs are not 15 independent test sets.

| Fold | Reliability/diversity (%) | Random (%) | Difference (points) |
|---|---:|---:|---:|
| 0 | 60.83 | 61.17 | -0.33 |
| 1 | 58.50 | 55.17 | +3.33 |
| 2 | 60.00 | 61.17 | -1.17 |
| 3 | 62.83 | 62.83 | 0.00 |
| 4 | 62.67 | 61.83 | +0.83 |

The active policy wins on two folds, loses on two, and ties on one. **This does
not establish a reliable advantage over random acquisition.** The compact
fusion model is the more promising practical change to investigate further.

## What changed

### 1. A small expert-reliability model

Instead of learning an embedding and feature classifier from 10-40 responses,
estimate a 5-by-5 likelihood matrix `q[y,h] = P(expert says h | true grade y)`.
Use counts from **queried pairs only**. Smooth each row toward a shared accuracy
prior. The shared accuracy is `(number correct + 1)/(queries + 2)`; remaining
prior mass is uniform across the four wrong classes. This avoids zero
probabilities and pools information across grades when counts are tiny.

The posterior concentration matrix is `counts + mass * prior`. Its normalized
rows define q. This is an empirical-Bayes approximation; it does not model all
uncertainty in the estimated prior or claim calibrated clinical reliability.

### 2. Budget-preserving fusion selection

Predict using:

`P(y | x,h) = softmax(log P_machine(y|x) + beta * log q[y,h])`.

Select prior mass from `{1,5,20}` and beta from `{0,0.5,1,1.5,2}` using
leave-one-out NLL on the already-queried pairs. In each LOO fit, remove the
held-out response from **both** confusion counts and the shared prior. Refit
using all queried pairs after selecting the two scalars. LOO reuses labels;
it does not request any new expert responses. Beta can alter class predictions,
unlike scalar post-fusion temperature scaling.

These 15 candidates were fixed before evaluation. LOO is a development
criterion on adaptively selected examples, not an unbiased held-out score.

### 3. Reliability, diversity and exploration

Start with the same two random expert queries per true grade as the existing
runner. Add one query per grade at each round, up to eight per grade.

For each unqueried candidate, consider all five possible expert responses.
Draw 64 confusion matrices from the fitted row-wise Dirichlet distributions.
Compute mutual information of the fused predictions across those draws,
conditional on each hypothetical response. Average across responses using
estimated `P(h | known y)`. No actual unqueried expert response is supplied to
the acquisition function. Use unit fusion strength when scoring candidates
so that a temporary fitted beta of zero does not suppress all learning.

Combine the rank of this expected information score with the rank of feature
distance from the nearest already-queried example of the same grade, with equal
weights. Standardize features using training-pool statistics, then L2 normalize;
distance is cosine distance. Average ranks handle tied scores. With probability
0.25, choose uniformly at random instead. This is a new heuristic inspired by
uncertainty and coverage, not a reproduction of core-set or structural-reliability
algorithms. Its uncertainty score is a model proxy, not the probability of a
clinical diagnostic failure.

### 4. Fixed stopping and explicit costs

The stop is 40 unique expert queries, not a test-driven accuracy target.
No extra human validation or fusion-calibration set is requested. All selected
IDs, responses, selected hyperparameters and acquisition reasons are saved.
The ledger tracks all cumulative queries, including labels used for LOO.

**The ground-truth training diagnoses are assumed known**, as in this project's
ActiveHAI task. Forty queries means additional doctor predictions for developing
the fusion model. Evaluation still consumes a doctor prediction for every test
case (200 per fold); do not describe this as 40 doctor responses for the entire
training-plus-deployment workflow. Independent experimental variants simulate
alternative 40-query workflows on cached responses; running all variants with
new live experts could cost more than 40 distinct responses in total.

## Optional machine-head experiment

`--machine probe` fits a regularized multiclass logistic-regression head to
known training grades and cached features. It trains on the 700-row acquisition
pool and selects C from `{0.001,0.01,0.1,1}` and a probability blend weight from
`{0,0.25,0.5,0.75,1}` using NLL on the reserved 100 known-label rows. No doctor
responses from that split are used. Feature normalization is fitted on the
fitting partition only.

Fusion tuning and acquisition use five-fold out-of-fold machine predictions
for the pool. Each row's target is excluded from the fit generating its OOF
prediction. The held-out calibration split is separate from those CV folds.
Final evaluation uses the machine head trained on all 700 pool rows.

This changes the machine head without retraining the image backbone. It did
not improve the measured results; it is retained as a reported negative ablation.
Any unknown leakage in the supplied frozen predictions/features remains unresolved.

## Reproduce

From the project root, create a Python 3.11 virtual environment if needed and
install `requirements-tested.txt` (exact tested versions) or `requirements.txt`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_budget_dr5.py --phase train --output results/my_budget_v2
.\.venv\Scripts\python.exe run_budget_dr5.py --phase evaluate --output results/my_budget_v2
```

Default: four variants x five folds x three seeds. `--phase all` performs the
same two stages in one command; it also completes **all** training before any
test loading. To run only the frozen reliability/diversity variant:

```powershell
.\.venv\Scripts\python.exe run_budget_dr5.py --machine frozen --acquisition diverse --output results/my_frozen_diverse
```

Use a new output directory each time. The evaluate phase reads the saved run
configuration, checks the training/source hashes, and refuses incomplete,
modified or already-evaluated experiments. JSON fusion states and NPZ machine
weights are sufficient to reconstruct the models without pickle. Predictions,
metrics, traces, data hashes and a generated report are in
[`results/budget_v2/`](../results/budget_v2/report.md).

Ten tests passed: the five original checks plus likelihood stability, honest
LOO refitting, balanced deterministic acquisition, probe serialization and
normalization, and a full 40-query run whose results remain identical after
changing every unqueried doctor label. That integration test runs with no test
CSV present and also checks that changed sealed artifacts cannot be evaluated.

## What to do next

Use this as a fixed candidate protocol on new patient-disjoint folds with
documented feature/backbone provenance. Include the new random-query fusion
control. Do not keep editing the method against these same test scores until
the target increases. If a more complex contextual expert-reliability model is
tried, select it using only the same queried-label budget or explicitly charge
any additional development responses.

These five test folds have already been inspected in earlier experiments.
Neither the 60.97% mean nor the small active/random difference is a fresh
confirmatory result, a patient-level generalization guarantee, or a clinical
performance claim.

## Background sources

- [ActiveHAI, IJCAI 2025](https://www.ijcai.org/proceedings/2025/477): task context
  of collecting limited additional expert predictions and combining decisions.
- [Sener and Savarese, A Core-Set Approach](https://arxiv.org/abs/1708.00489):
  background motivation for diversity/coverage in active selection. The query
  rule implemented here is different and is evaluated as its own heuristic.
