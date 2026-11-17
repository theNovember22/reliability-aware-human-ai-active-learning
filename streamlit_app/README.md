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

In the sidebar, set **Repository root** to the root folder containing `src/dr5/core.py`. For labeled evaluation, use **Evaluate labeled CSV** and upload a matching-format CSV. For prediction on new cached cases, use **Infer new cases**, choose a trained `.pt` checkpoint, and upload a CSV without ground-truth labels.

### Inference CSV

The inference tab expects these columns, in this order:

- `Logits`: comma-separated string of 5 normalized AI probabilities
- `Human Label`: integer class 0–4
- `Feat`: comma-separated string of 512 cached feature values

It returns the AI prediction, human–AI fusion prediction, fusion confidence, MC mutual information, and per-class fusion probabilities. It does not calculate accuracy because ground-truth labels are not needed for inference. The current repository has no raw-image-to-feature extractor, so retinal images cannot be uploaded directly.

## CSV schema

The exact column names and order are:

- `Logits`: comma-separated string of 5 **normalized probabilities** (despite the column name)
- `True Label`: integer class 0–4
- `Human Label`: integer class 0–4
- `Feat`: comma-separated string of 512 cached feature values

The current repository does not supply a raw-image-to-feature extraction pipeline, so inference requires cached DR-5 feature vectors rather than arbitrary retinal image uploads.

## Important interpretation note

The `failure-frequency proxy` is computed from stochastic margins against the CSV's true label. It is a model disagreement/uncertainty statistic, not a calibrated probability of clinical failure. Use a held-out CSV for honest evaluation; avoid reporting metrics on rows used to train/select the checkpoint.
