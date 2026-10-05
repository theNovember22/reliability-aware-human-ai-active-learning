# DR5: reproducible ActiveHAI baseline and reliability experiments

For the new 40-query accuracy experiments, start with **`run_budget_dr5.py`**
and [the budget-preserving method and results](docs/budget_accuracy.md).
The frozen-AI + reliability/diversity variant measured **60.97% mean accuracy**
across five folds and three seeds, versus **60.43%** for its random-query control.
This meets 60% on the existing exploratory folds; it is not a guarantee on new
data or proof that active acquisition reliably beats random.

Use **`run_dr5.py`** for the corrected neural MWAC baseline. The previous scripts in `experiments/baseline/`
are preserved as historical code; they contain test-driven selection and are
not the recommended experiment entry point.

This project uses the repository's frozen 512-dimensional DR5 features,
five-class AI probabilities, ground-truth grades, and doctor predictions.
It trains the **human evaluator**, not the retinal image backbone.

## Start in VS Code on Windows

1. Extract the supplied project ZIP and open its `dr5-project` folder in VS Code
   (the folder containing this README and `run_dr5.py`). If working from a Git
   checkout, open that repository folder instead.
2. Install Python 3.11 or 3.12 and the VS Code Python extension.
3. In the VS Code terminal, run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_dr5.py --folds 0 --seeds 1 --mode baseline --output results/my_first_baseline
```

Use `py -3.12` instead if that is the version installed. No activation or
PowerShell execution-policy changes are needed. Select `.venv` with
**Python: Select Interpreter**. A baseline launch configuration is included
under Run and Debug; choose a fresh output folder for each run.

On Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_dr5.py --folds 0 --seeds 1 --output results/my_first_baseline
```

CPU is sufficient for these cached features. `--device cuda` is optional if
your installed PyTorch supports your GPU. Image training is not included.

## New budget-preserving accuracy experiment

The new runner replaces the high-dimensional evaluator with a smoothed expert
confusion model, tunes fusion using only already-queried responses, and combines
expected information with feature diversity and 25% random exploration.
The default endpoint is **40 development expert predictions**, with no extra
expert responses for validation/calibration. A doctor prediction is still
required for each case at evaluation/deployment, as in the original protocol.

```powershell
# Single-fold smoke run (not an estimate of the five-fold mean)
.\.venv\Scripts\python.exe run_budget_dr5.py --folds 0 --seeds 1 --machine frozen --acquisition diverse --output results/my_budget_smoke

# Full predeclared comparison: four variants x five folds x three seeds
.\.venv\Scripts\python.exe run_budget_dr5.py --phase train --output results/my_budget_comparison
.\.venv\Scripts\python.exe run_budget_dr5.py --phase evaluate --output results/my_budget_comparison
```

The defaults compare frozen/probe machine predictions and random/diverse
acquisition. All training completes before test evaluation. Completed outputs
cannot be overwritten. Each new run needs a fresh output directory.
The feature probe consumes existing known ground-truth grades, not additional
expert responses; it did not improve this comparison. To reproduce the tested
environment, install `requirements-tested.txt` in a Python 3.11 environment.

Saved [results](results/budget_v2/report.md), query traces, models, predictions,
source/data hashes and paired-fold comparisons are in `results/budget_v2/`.

## Reproduce the measured experiments

Defaults: five folds, seeds 1/2/3, 100 epochs per acquisition round, Adam at
3e-4 with cosine decay, dropout 0.1, 2 through 8 human predictions per class.
Every variant uses the same initial queries and reserved calibration partition.

```bash
python run_dr5.py --audit-only --output results/my_audit
python run_dr5.py --mode baseline --output results/my_baseline
python run_dr5.py --mode random --output results/my_random
python run_dr5.py --mode mc --output results/my_mc
python run_dr5.py --mode reliability --output results/my_reliability
python run_dr5.py --mode baseline --ai-calibration none --output results/my_no_ai_calibration
python run_dr5.py --mode reliability --fusion-calibration --output results/my_fusion_calibration
```

`baseline` is a corrected MWAC implementation. `mc` changes inference to
Monte Carlo dropout while retaining baseline acquisition. `reliability`
also changes the choice within the MWAC window. `random` is a necessary
acquisition control with identical training and label budgets.

Each run writes per-round checkpoints, predictions, sample-selection traces,
configuration/data/source hashes, metrics and summary CSVs. Test results are
calculated only after training and acquisition decisions finish for a run.
Existing output folders with a manifest are rejected to prevent overwriting.

## Four-module research adaptation

| Structural reliability module | DR5 implementation |
|---|---|
| Surrogate | Frozen image features plus a learned doctor-label embedding and linear evaluator, trained with normalized-product NLL |
| Reliability estimation | MC dropout mutual information and stochastic decision margins; optional observed validation error |
| Learning function | Class-balanced MWAC, optionally selecting high epistemic/boundary uncertainty inside its window |
| Stopping criterion | Fixed expert-query budget by default; optional validation-NLL plateau with explicitly charged human labels |

This is an adaptation of the framework, **not** a reproduction of structural
failure probability estimation or an established clinical reliability model.
Read [the methodological details](docs/methodology.md) before interpreting it.

Optional plateau demonstration:

```bash
python run_dr5.py --mode reliability --folds 0 --seeds 1 --validation-per-class 10 --stop-patience 2 --output results/my_stopping_demo
```

That run costs 50 additional validation doctor predictions. Fusion calibration
costs another 100 predictions with the default calibration split. These costs
are always included in `total_human_queries`; they are not free annotations.

## What actually improved?

See [the measured results](docs/results_report.md) and
[the diagnosis and next experiments](docs/methodology.md). The current reliability
heuristic modestly improves over corrected MWAC, but **random acquisition is
stronger in the completed comparison**. Temperature calibration improves the
fusion probability calibration, not its argmax accuracy. Do not claim a
validated new state of the art from this experiment.

The current audit cannot establish patient-disjoint splits or whether the
cached features were extracted from correctly held-out backbone models:
the CSVs have no patient/image IDs or extraction provenance.
