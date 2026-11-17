# Reliability-aware Human–AI Active Learning (DR-5)

This repository studies how to use a limited number of doctor annotations to
learn and evaluate human–AI fusion for five-class retinal grading (DR-5). It
contains a corrected neural evaluator, risk-ranking and active-annotation
experiments, reliability-aware fusion analyses, and a budget-preserving
confusion-model fusion baseline.

The project uses **cached AI probabilities and 512-dimensional image features**.
It trains evaluator/fusion components; it does not train the retinal image
backbone or process raw images.

## Current findings

All reported DR-5 test folds have been inspected in prior analyses. Results
below are exploratory and are not confirmatory estimates of performance on new
patients.

| Result | Accuracy at 40 development queries | Interpretation |
|---|---:|---|
| Best measured pipeline: frozen AI + smoothed expert fusion + reliability/diversity acquisition | **60.97% ± 1.89%** | Best observed mean in the budget-preserving study; exploratory |
| Matched random acquisition for that fusion pipeline | 60.43% ± 2.96% | Active-policy difference was small and mixed across folds |
| Experiment C, fixed neural product fusion | 58.57% ± 3.53% | Same Experiment B query sequence as the adaptive-weight result |
| Experiment C, MC-reliability-adaptive fusion | 58.60% ± 3.59% | Only +0.03 percentage points over fixed fusion; no meaningful accuracy gain |
| Experiment C follow-up, leave-one-out tuned AI/expert weights | 57.20% ± 3.11% | Lower than fixed-weight queried-confusion fusion (59.37%) |

The 60.97% result is **4.47 percentage points above** the earlier 56.50%
corrected MWAC + AI-temperature result, but it is only 2.37 points above the
current 58.60% Experiment C result. The methods use different fusion and
acquisition protocols, so this comparison does not isolate reliability
weighting or establish a 4–5 point causal improvement. See
[`docs/results_report.md`](docs/results_report.md) and
[`docs/budget_accuracy.md`](docs/budget_accuracy.md) for details.

## Data and protocol

The supplied `data/DR-5/` directory contains five train/test folds. Each CSV
has `Logits`, `True Label`, `Human Label`, and `Feat` columns. Despite its
column name, `Logits` contains normalized five-class probabilities. `Feat`
contains the cached 512-value representation.

The active-learning experiments start with **2 queried human labels per
class**, then add one case per class per round through **8 labels per class**
(40 development queries). Under the ActiveHAI setup, the true training grade is
known for pool stratification and MWAC selection; the costly label is the
doctor’s response. Queried doctor responses are used only after those cases
are selected. Evaluation also uses each test case’s supplied human response as
the fusion input; it is never used for acquisition or evaluator training.

No patient/image IDs, raw images, image-backbone checkpoints, or feature
extraction provenance are included. Patient-disjointness and possible
backbone-level leakage cannot be verified from the supplied CSVs.

## Setup

The project is feature-only and can run on CPU. Python 3.11 is the tested
environment; `requirements-tested.txt` records the tested package versions.

### Windows / VS Code

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Choose `.venv` with **Python: Select Interpreter** in VS Code. Python 3.12 can
also be used with `py -3.12` if the installed PyTorch build supports it.

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Optional test command for the project’s unit checks:

```bash
python -m unittest discover -s tests -v
```

## Main experiment entry points

### Corrected neural MWAC baseline

[`run_dr5.py`](run_dr5.py) trains the existing evaluator with normalized
product-rule fusion. It supports `baseline`, `random`, `mc`, and
`reliability` acquisition/inference modes. Outputs go to a new directory for
each run.

```bash
python run_dr5.py --audit-only --output results/my_audit
python run_dr5.py --mode baseline --output results/my_baseline
python run_dr5.py --mode random --output results/my_random
python run_dr5.py --mode mc --output results/my_mc
python run_dr5.py --mode reliability --output results/my_reliability
```

The neural evaluator is
`z_h = Linear(Dropout(feature + Embedding(human_label)))`. Fusion is
`softmax(log(p_AI) + z_h)`. MC Dropout estimates evaluator uncertainty
conditional on the cached AI probabilities and the supplied human response.
The unqueried-pool failure-frequency signal uses pseudo `human_label = true
grade`; it is a counterfactual model proxy, not observed clinical reliability.

### Experiment A: AI error-risk ranking

[`experiments/analysis/error_risk_baselines.py`](experiments/analysis/error_risk_baselines.py)
compares confidence, entropy, margin and a learned risk score for ranking
incorrect cached AI predictions. It evaluates error ranking, not human–AI
fusion accuracy.

```bash
python experiments/analysis/error_risk_baselines.py
```

### Experiment B: entropy-ranked MWAC active annotation

[`experiments/analysis/experiment_b_referral.py`](experiments/analysis/experiment_b_referral.py)
implements the active-annotation loop:

1. Train the evaluator on currently queried doctor responses.
2. For each true class, sample up to 100 unqueried training cases.
3. Rank with the MWAC evaluator boundary score and retain its median window
   (start 55, length 5).
4. Rank those candidates by **AI predictive Shannon entropy** and query the
   highest-entropy case per class.
5. Add the acquired human responses, retrain, and evaluate fusion at each
   budget from 10 to 40 queries.

```bash
python experiments/analysis/experiment_b_referral.py
```

The reported endpoint is 58.9% mean accuracy across the five folds and three
seeds for that single acquisition policy. It does not demonstrate that entropy
selection beats another policy.

### Experiment C: reliability-adaptive fusion

The current thesis Experiment C holds the Experiment B query sequence and
evaluator checkpoints fixed, then compares:

- **Fixed fusion:** evaluator logit coefficient `beta = 1`.
- **Adaptive fusion:** per-case coefficient
  `beta_i = clip(1 - MI_i / log(5), 0, 1)`, where MI is mutual information
  from the repository’s existing 20-pass MC Dropout implementation.

The test labels do not enter the MI estimate or the adaptive weights. Metrics
are accuracy and `KL(one-hot true label || fused probabilities)`. For one-hot
classification labels, this KL is numerically equal to NLL.

```bash
python experiments/analysis/experiment_c_reliability_weighted_fusion.py
```

At 40 queries, adaptive fusion measured 58.60% ± 3.59% versus 58.57% ± 3.53%
for fixed fusion. Mean MI was 0.0083 nats, so the mean evaluator weight was
0.9948. The result reduced KL slightly but did not meaningfully increase
accuracy.

The follow-up
[`experiments/analysis/experiment_c_fusion_weight_calibration.py`](experiments/analysis/experiment_c_fusion_weight_calibration.py)
reuses the same queried examples and tunes AI/expert fusion weights using
leave-one-out NLL on acquired training responses only. Its tuned-weight result
was 57.20%, lower than fixed-weight queried-confusion fusion at 59.37%; this
follow-up is a negative result, not evidence for stronger fusion weights.

Tracked C result artifacts are in:

- [`results/experiment_c_reliability_weighted_fusion/`](results/experiment_c_reliability_weighted_fusion/)
- [`results/experiment_c_fusion_weight_calibration/`](results/experiment_c_fusion_weight_calibration/)

## Results, interpretation and reproducibility

- [`docs/results_report.md`](docs/results_report.md): consolidated A–C results,
  paired comparisons and limitations.
- [`docs/methodology.md`](docs/methodology.md): protocol, equations, query
  accounting and research interpretation.
- Experiment reports record seeds, data/source hashes, weight choices, and
  paired fold/seed metrics where applicable.
- Use the existing test folds for reporting only. Do not tune new settings on
  them; the next confirmatory study needs patient-level split identifiers and
  a fresh untouched test set.

The project’s evidence does **not** establish a 4–5 percentage-point gain from
the current reliability-adaptive fusion rule. The 60.97% result is a separate
pipeline and should not be presented as a causal improvement from Experiment C.
