# Streamlit app for Reliability-aware Human–AI Active Learning

This app wraps the repository's existing `src/dr5/core.py` evaluator, fusion, MC Dropout, and reliability-signal functions. It does not train the model inside the UI.

## 1. Put the app in your repository

Copy `app.py` into your repository under `streamlit_app/app.py`. Install Streamlit and Matplotlib in the same environment as the repo's ML requirements:

```bash
pip install streamlit matplotlib
```

Alternatively, merge the dependencies from this folder's `requirements.txt` into the repository requirements.

## 2. Train a checkpoint first

From the repository root, train the current evaluator and save checkpoints:

```bash
python run_dr5.py --folds 0 --seeds 1 --mode reliability --output results/streamlit_demo
```

This produces checkpoints such as `results/streamlit_demo/fold0_seed1_budget8.pt`. The code expects a checkpoint from the current `run_dr5.py`, not a historical `experiments/baseline` checkpoint.

Ensure `data/DR-5/model_output_0_train.csv` and `model_output_0_test.csv` exist. The supplied repository may not include the DR-5 CSVs or trained checkpoints because they are local experiment artifacts.

## 3. Launch

From the repository root:

```bash
streamlit run streamlit_app/app.py
```

In the sidebar, set **Repository root** to the root folder containing `src/dr5/core.py`. In **Run evaluator**, choose the trained `.pt` checkpoint and upload a matching-format CSV, for example `data/DR-5/model_output_0_test.csv`.

## CSV schema

The exact column names and order are:

- `Logits`: comma-separated string of 5 **normalized probabilities** (despite the column name)
- `True Label`: integer class 0–4
- `Human Label`: integer class 0–4
- `Feat`: comma-separated string of 512 cached feature values

The current repository does not supply a raw-image-to-feature extraction pipeline, so this app evaluates cached DR-5 rows rather than arbitrary retinal image uploads.

## Important interpretation note

The `failure-frequency proxy` is computed from stochastic margins against the CSV's true label. It is a model disagreement/uncertainty statistic, not a calibrated probability of clinical failure. Use a held-out CSV for honest evaluation; avoid reporting metrics on rows used to train/select the checkpoint.
