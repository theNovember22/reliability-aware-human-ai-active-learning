from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Reliability-aware Human-AI Active Learning", page_icon="🧠", layout="wide")
REPO_ROOT = Path(st.sidebar.text_input("Repository root", value=str(ROOT.parent)))

# Import the project's real evaluator / fusion / MC-dropout implementation when present.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.dr5.core import Evaluator, predict_samples, reliability_signals, temperature_scale, load_csv
    CORE_IMPORT_ERROR = None
except Exception as exc:
    Evaluator = predict_samples = reliability_signals = temperature_scale = load_csv = None
    CORE_IMPORT_ERROR = str(exc)

try:
    from src.dr5.budget import BudgetFusion
    BUDGET_IMPORT_ERROR = None
except Exception as exc:
    BudgetFusion = None
    BUDGET_IMPORT_ERROR = str(exc)

with st.sidebar:
    st.markdown("### :material/blur_on: ActiveHAI lab")
    st.caption("Reliability-aware human–AI diagnosis")
    st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)
    st.subheader("Inference settings")
    device_choice = st.selectbox("Device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"])
    mc_passes = st.slider("MC Dropout passes", min_value=2, max_value=100, value=20, step=2)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1)
    st.divider()
    st.markdown("**Input expectations**")
    st.write("CSV columns: `Logits`, `True Label`, `Human Label`, `Feat`.")
    st.write("Five classes (0–4), normalized AI probabilities, and 512-dimensional cached features.")

def inject_material_theme() -> None:
    st.html("""
    <style>
      :root {{ --m3-primary:#6750A4; --m3-on-primary:#FFFFFF; --m3-primary-container:#EADDFF; --m3-on-primary-container:#21005D; --m3-page:#F7F5FA; --m3-surface:#FFFFFF; --m3-surface-low:#F1EDF6; --m3-surface-high:#EAE4F0; --m3-outline:#D7D0DF; --m3-ink:#1D1B20; --m3-variant:#49454F; --m3-shadow:rgba(43,31,63,.09); }}
      html, body, [data-testid="stAppViewContainer"], .stApp {{ color:var(--m3-ink) !important; background:var(--m3-page) !important; }}
      [data-testid="stMain"], section.main, [data-testid="stMainBlockContainer"] {{ background:transparent !important; color:var(--m3-ink) !important; }}
      [data-testid="stHeader"] {{ background:transparent !important; }}
      [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{ background:#F1EDF6 !important; border-right:1px solid var(--m3-outline); }}
      .block-container {{ max-width:1440px; padding-top:2rem; padding-bottom:4rem; }}
      h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {{ color:var(--m3-ink); }}
      [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p {{ color:var(--m3-variant) !important; }}
      .sidebar-rule {{ height:1px; margin:.8rem 0 1rem; background:var(--m3-outline); }}
      .m3-hero {{ min-height:184px; display:flex; flex-direction:column; justify-content:center; padding:1.8rem 2rem; margin:0 0 1.1rem; border:1px solid #D7C8F3; border-radius:28px; background:var(--m3-primary-container); box-shadow:0 8px 26px var(--m3-shadow); }}
      .m3-sidecard {{ min-height:184px; display:flex; flex-direction:column; justify-content:center; padding:1.5rem; border:1px solid var(--m3-outline); border-radius:28px; background:var(--m3-surface); box-shadow:0 8px 26px var(--m3-shadow); }}
      .m3-kicker {{ color:var(--m3-primary); font-size:.76rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }}
      .m3-hero h1 {{ margin:.45rem 0; color:var(--m3-on-primary-container); font-size:clamp(1.85rem, 3.4vw, 2.8rem); letter-spacing:-.035em; }}
      .m3-hero p, .m3-sidecard p {{ margin:0; color:var(--m3-variant); font-size:1rem; }}
      .m3-sidecard strong {{ color:var(--m3-primary); font-size:1.45rem; }}
      [data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap:.25rem; padding:.35rem; border:0 !important; border-radius:24px; background:var(--m3-surface-low) !important; }}
      [data-testid="stTabs"] [data-baseweb="tab-border"] {{ background:transparent !important; }}
      [data-testid="stTabs"] [data-baseweb="tab"] {{ border-radius:18px; color:var(--m3-variant); padding:.65rem 1rem; }}
      [data-testid="stTabs"] [aria-selected="true"] {{ color:var(--m3-on-primary-container) !important; background:var(--m3-primary-container) !important; }}
      [data-testid="stMetric"] {{ padding:1rem; border:1px solid var(--m3-outline); border-radius:24px; background:var(--m3-surface); box-shadow:0 4px 18px var(--m3-shadow); }}
      [data-testid="stMetricLabel"] {{ color:var(--m3-variant); }} [data-testid="stMetricValue"] {{ color:var(--m3-primary); }}
      [data-testid="stVerticalBlockBorderWrapper"] {{ border-color:var(--m3-outline); border-radius:24px; background:var(--m3-surface); box-shadow:0 4px 18px var(--m3-shadow); }}
      .stButton > button, [data-testid="stDownloadButton"] button {{ min-height:2.8rem; border-radius:999px; border:1px solid var(--m3-outline); box-shadow:none; }}
      .stButton > button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {{ background:var(--m3-primary) !important; border:0 !important; color:var(--m3-on-primary) !important; box-shadow:0 3px 8px var(--m3-shadow); }}
      .stButton > button:not([kind="primary"]), [data-testid="stDownloadButton"] button {{ background:var(--m3-surface-low) !important; color:var(--m3-primary) !important; border:1px solid var(--m3-outline) !important; }}
      [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stFileUploader"] section {{ background:var(--m3-surface) !important; border-color:var(--m3-outline) !important; border-radius:16px; }}
      [data-testid="stDataFrame"], [data-testid="stTable"] {{ border:1px solid var(--m3-outline); border-radius:18px; overflow:hidden; }}
      [data-testid="stAlert"] {{ border-radius:20px; border:1px solid #D7C8F3; background:var(--m3-primary-container) !important; }}
      code {{ color:var(--m3-primary); background:#F1EDF6; }}
    </style>
    """)


inject_material_theme()

hero_main, hero_support = st.columns([1.618, 1], vertical_alignment="center")
with hero_main:
    st.html("""
    <section class="m3-hero">
      <div class="m3-kicker">ActiveHAI · DR-5 research dashboard</div>
      <h1>Reliability-aware human–AI diagnosis</h1>
      <p>Explore evaluator fusion, reliability signals, and the structural-reliability active-learning extension.</p>
    </section>
    """)
with hero_support:
    st.html("""
    <section class="m3-sidecard">
      <div class="m3-kicker">Research workspace</div>
      <p><strong>5 folds · 3 seeds</strong></p>
      <p>Inference, budget fusion, and experiment summaries in one place.</p>
    </section>
    """)

if CORE_IMPORT_ERROR:
    st.error("Couldn't import the repository model code. Check the Repository root in the sidebar and install the requirements.")
    st.code(CORE_IMPORT_ERROR)
    st.stop()


def parse_csv(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded)
    required = ["Logits", "True Label", "Human Label", "Feat"]
    if list(df.columns) != required:
        raise ValueError(f"Expected exactly these columns in this order: {required}")
    if df.empty:
        raise ValueError("CSV is empty.")
    return df


def vector_from_cell(value, name: str, expected: int | None = None) -> np.ndarray:
    try:
        arr = np.asarray([float(part.strip()) for part in str(value).split(",")], dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Could not parse {name} as a comma-separated numeric vector.") from exc
    if expected is not None and len(arr) != expected:
        raise ValueError(f"{name} must contain {expected} values; found {len(arr)}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def load_checkpoint(path: Path, device: torch.device):
    # Checkpoints are produced by run_dr5.py and contain state_dict + feature_dim + dropout.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older PyTorch
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint should be a run_dr5.py checkpoint containing 'state_dict'.")
    feature_dim = int(checkpoint.get("feature_dim", 512))
    dropout = float(checkpoint.get("dropout", 0.1))
    model = Evaluator(feature_dim=feature_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def find_default_checkpoint(repo_root: Path) -> Path:
    """Return a usable bundled checkpoint, preferring the Streamlit demo run."""
    preferred = repo_root / "results" / "streamlit_demo" / "fold0_seed1_budget8.pt"
    if preferred.is_file():
        return preferred

    candidates = sorted((repo_root / "results").glob("*/fold0_seed1_budget8.pt"))
    if candidates:
        return candidates[0]

    return repo_root / "results" / "dr5" / "fold0_seed1_budget8.pt"


def compute_predictions(model, ai_probs, features, human_labels, true_labels, passes, seed, device,
                        ai_temperature=1.0, fusion_temperature=1.0):
    ai_t = torch.tensor(ai_probs, dtype=torch.float32, device=device)
    calibrated_ai_t = temperature_scale(ai_t, float(ai_temperature))
    feat_t = torch.tensor(features, dtype=torch.float32, device=device)
    human_t = torch.tensor(human_labels, dtype=torch.long, device=device)
    y_t = torch.tensor(true_labels, dtype=torch.long, device=device)
    samples = predict_samples(model, feat_t, human_t, calibrated_ai_t, passes=passes, seed=int(seed))
    mean_probs = temperature_scale(samples.mean(dim=0), float(fusion_temperature))
    sig = reliability_signals(samples, y_t)
    return mean_probs.cpu().numpy(), {
        "mi": sig["mi"].cpu().numpy(),
        "failure_frequency": sig["failure_frequency"].cpu().numpy(),
        "margin_mean": sig["margin_mean"].cpu().numpy(),
        "margin_std": sig["margin_std"].cpu().numpy(),
    }, ai_t.cpu().numpy()


model_tab, budget_run_tab, budget_tab, sample_tab, metrics_tab = st.tabs([
    "Run evaluator", "Run budget fusion", "Reported 60.97% result", "Inspect CSV", "How to use / limitations"
])

with model_tab:
    st.subheader("Run a trained evaluator checkpoint")
    default_ckpt = find_default_checkpoint(REPO_ROOT)
    ckpt_path_text = st.text_input("Checkpoint file (.pt)", value=str(default_ckpt))
    checkpoint_file = Path(ckpt_path_text).expanduser()
    uploaded_csv = st.file_uploader("Upload a DR-5 CSV", type=["csv"], key="run_csv")
    csv_path_text = st.text_input("Or use a CSV path on this machine (optional)", value="")
    use_path = bool(csv_path_text.strip())
    if st.button("Run model", type="primary", width="stretch"):
        try:
            if not checkpoint_file.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
            if use_path:
                csv_df = pd.read_csv(Path(csv_path_text).expanduser())
            elif uploaded_csv is not None:
                csv_df = pd.read_csv(uploaded_csv)
            else:
                raise ValueError("Upload a CSV or enter a CSV path.")
            required = ["Logits", "True Label", "Human Label", "Feat"]
            if list(csv_df.columns) != required:
                raise ValueError(f"Expected columns exactly: {required}")
            if csv_df.empty:
                raise ValueError("CSV has no rows.")
            probs = np.stack([vector_from_cell(v, "Logits", 5) for v in csv_df["Logits"]])
            features = np.stack([vector_from_cell(v, "Feat", None) for v in csv_df["Feat"]])
            if features.shape[1] != 512:
                raise ValueError(f"Expected 512 feature values per row; found {features.shape[1]}.")
            if (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
                raise ValueError("The Logits column must actually contain normalized AI probabilities (sum to 1).")
            y = csv_df["True Label"].to_numpy(dtype=int)
            h = csv_df["Human Label"].to_numpy(dtype=int)
            if not np.isin(y, np.arange(5)).all() or not np.isin(h, np.arange(5)).all():
                raise ValueError("True Label and Human Label must be integers 0–4.")
            device = torch.device("cuda" if device_choice == "cuda" and torch.cuda.is_available() else "cpu")
            model, checkpoint = load_checkpoint(checkpoint_file, device)
            record = checkpoint.get("record", {})
            ai_temperature = float(record.get("ai_temperature", 1.0))
            fusion_temperature = float(record.get("fusion_temperature", 1.0))
            with st.spinner("Running evaluator and MC Dropout passes…"):
                fused, signals, ai_probs = compute_predictions(
                    model, probs, features, h, y, int(mc_passes), int(seed), device,
                    ai_temperature=ai_temperature, fusion_temperature=fusion_temperature,
                )
            df_out = csv_df.copy()
            df_out["AI prediction"] = ai_probs.argmax(axis=1)
            df_out["Evaluator fusion prediction"] = fused.argmax(axis=1)
            df_out["Fusion confidence"] = fused.max(axis=1)
            df_out["MI"] = signals["mi"]
            df_out["Failure-frequency proxy"] = signals["failure_frequency"]
            df_out["Mean margin"] = signals["margin_mean"]
            df_out["Margin std"] = signals["margin_std"]
            ai_acc = float((ai_probs.argmax(1) == y).mean())
            fusion_acc = float((fused.argmax(1) == y).mean())
            human_acc = float((h == y).mean())
            mean_mi = float(np.mean(signals["mi"]))
            mean_failure = float(np.mean(signals["failure_frequency"]))
            st.session_state["dr5_output"] = df_out
            st.session_state["dr5_summary"] = {
                "rows": len(df_out), "ai_accuracy": ai_acc, "human_accuracy": human_acc,
                "fusion_accuracy": fusion_acc, "mean_mi": mean_mi, "mean_failure": mean_failure,
                "checkpoint": str(checkpoint_file), "mode": checkpoint.get("config", {}).get("mode", "unknown"),
                "ai_temperature": ai_temperature, "fusion_temperature": fusion_temperature,
            }
        except Exception as exc:
            st.error(f"Could not run inference: {exc}")

    if "dr5_output" in st.session_state:
        df_out = st.session_state["dr5_output"]
        summary = st.session_state["dr5_summary"]
        st.success(f"Inference completed for {summary['rows']} samples.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI accuracy", f"{summary['ai_accuracy']:.1%}")
        c2.metric("Human accuracy", f"{summary['human_accuracy']:.1%}")
        c3.metric("Human–AI fusion accuracy", f"{summary['fusion_accuracy']:.1%}")
        c4.metric("Mean MI", f"{summary['mean_mi']:.4f}")
        st.caption(f"Checkpoint: {summary['checkpoint']} · Model mode in checkpoint: {summary['mode']}")
        st.warning("Failure-frequency is calculated against the supplied True Label and is a stochastic disagreement proxy, not a calibrated clinical failure probability. The CSV must contain valid cached feature vectors and labels.")
        st.subheader("Sample-level predictions and reliability signals")
        st.dataframe(df_out[["True Label", "Human Label", "AI prediction", "Evaluator fusion prediction", "Fusion confidence", "MI", "Failure-frequency proxy", "Mean margin", "Margin std"]], width="stretch", hide_index=True)
        st.download_button("Download predictions CSV", data=df_out.to_csv(index=False).encode("utf-8"), file_name="dr5_streamlit_predictions.csv", mime="text/csv")
        fig, ax = plt.subplots()
        ax.hist(df_out["MI"], bins=25)
        ax.set_xlabel("Mutual information")
        ax.set_ylabel("Sample count")
        ax.set_title("MC Dropout epistemic uncertainty")
        st.pyplot(fig)
        fig2, ax2 = plt.subplots()
        ax2.hist(df_out["Failure-frequency proxy"], bins=np.linspace(0, 1, 11), edgecolor="black")
        ax2.set_xlabel("Failure-frequency proxy")
        ax2.set_ylabel("Sample count")
        ax2.set_title("Stochastic margin crossing frequency")
        st.pyplot(fig2)
        st.subheader("Confusion matrix: fusion predictions")
        cm = pd.crosstab(pd.Series(df_out["True Label"], name="True label"), pd.Series(df_out["Evaluator fusion prediction"], name="Predicted label"), dropna=False).reindex(index=range(5), columns=range(5), fill_value=0)
        st.dataframe(cm, width="stretch")


with budget_run_tab:
    st.subheader("Run the 60.97% experiment method on an uploaded CSV")
    budget_input, budget_method = st.columns([1, 1.618], vertical_alignment="top")
    with budget_input:
        with st.container(border=True):
            st.markdown("#### Evaluation input")
            budget_csv = st.file_uploader("Upload test CSV", type=["csv"], key="budget_run_csv")
            budget_fold = st.selectbox("Test fold", list(range(5)), key="budget_fold")
            budget_seed = st.selectbox("Saved development seed", [1, 2, 3], key="budget_seed")
            run_budget = st.button("Run budget fusion", type="primary", width="stretch")
    with budget_method:
        with st.container(border=True):
            st.markdown("#### Fusion method")
            st.write(
                "Uses the saved frozen-AI and smoothed expert-fusion model from "
                "`results/budget_v2`. The selected fold must match your uploaded test CSV."
            )
            st.caption("40 development doctor predictions · 8 per class · five DR-5 classes")

    if run_budget:
        try:
            if BUDGET_IMPORT_ERROR:
                raise RuntimeError(f"Could not import budget fusion code: {BUDGET_IMPORT_ERROR}")
            if budget_csv is None:
                raise ValueError("Upload a test CSV first.")
            budget_df = pd.read_csv(budget_csv)
            required = ["Logits", "True Label", "Human Label", "Feat"]
            if list(budget_df.columns) != required or budget_df.empty:
                raise ValueError(f"Expected nonempty columns exactly: {required}")
            machine = np.stack([vector_from_cell(v, "Logits", 5) for v in budget_df["Logits"]])
            human = budget_df["Human Label"].to_numpy(dtype=int)
            labels = budget_df["True Label"].to_numpy(dtype=int)
            if np.stack([vector_from_cell(v, "Feat", 512) for v in budget_df["Feat"]]).shape[1] != 512:
                raise ValueError("Feat must contain 512 values per row.")
            run_path = REPO_ROOT / "results" / "budget_v2" / f"frozen_diverse_fold{budget_fold}_seed{budget_seed}.json"
            if not run_path.is_file():
                raise FileNotFoundError(f"Saved budget run not found: {run_path}")
            run = json.loads(run_path.read_text(encoding="utf-8"))
            endpoint = next(r for r in run["rounds"] if r["labels_per_class"] == 8)
            fusion = BudgetFusion.from_dict(endpoint["fusion"])
            prediction_probs = fusion.predict(machine, human)
            predictions = prediction_probs.argmax(axis=1)
            result_accuracy = float(np.mean(predictions == labels))
            st.session_state["budget_output"] = budget_df.assign(
                **{"Budget fusion prediction": predictions, "Budget fusion confidence": prediction_probs.max(axis=1)}
            )
            st.session_state["budget_summary"] = {
                "accuracy": result_accuracy, "rows": len(budget_df), "run": str(run_path),
            }
        except Exception as exc:
            st.error(f"Could not run budget fusion: {exc}")

    if "budget_output" in st.session_state:
        result = st.session_state["budget_summary"]
        st.metric("Uploaded CSV accuracy", f"{result['accuracy']:.2%}")
        st.caption(f"Evaluated {result['rows']} rows using {result['run']}")
        st.info(
            "The published 60.97% is the mean over 15 matching fold/seed test runs. "
            "One uploaded CSV represents one test fold, so its accuracy may differ."
        )
        st.dataframe(st.session_state["budget_output"], width="stretch", hide_index=True)


with budget_tab:
    st.subheader("Budget-preserving experiment")
    st.write(
        "This is the repository's measured 60.97% result from `run_budget_dr5.py`, "
        "not the single-checkpoint neural evaluator above."
    )
    budget_metrics_path = REPO_ROOT / "results" / "budget_v2" / "metrics.csv"
    if not budget_metrics_path.is_file():
        st.error(f"Budget experiment results not found: {budget_metrics_path}")
    else:
        budget_df = pd.read_csv(budget_metrics_path)
        endpoint = budget_df[budget_df["labels_per_class"] == 8].copy()
        endpoint["accuracy"] = endpoint["accuracy"].astype(float)
        summary_df = (
            endpoint.groupby(["machine", "acquisition"], as_index=False)
            .agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), runs=("accuracy", "size"))
        )
        summary_df["accuracy"] *= 100
        summary_df["accuracy_sd"] *= 100
        diverse = summary_df[
            (summary_df["machine"] == "frozen") & (summary_df["acquisition"] == "diverse")
        ]
        if not diverse.empty:
            row = diverse.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Frozen + diverse accuracy", f"{row['accuracy']:.2f}%")
            c2.metric("Across fold/seed runs", int(row["runs"]))
            c3.metric("Development queries", "40")

        st.dataframe(
            summary_df.rename(columns={
                "machine": "Machine", "acquisition": "Acquisition",
                "accuracy": "Accuracy (%)", "accuracy_sd": "SD (%)", "runs": "Runs",
            }).round({"Accuracy (%)": 2, "SD (%)": 2}),
            width="stretch", hide_index=True,
        )
        st.info(
            "Interpretation: 60.97% is the mean across five folds and three seeds "
            "at the eight-labels-per-class endpoint. It is an exploratory result, "
            "not a guarantee for one uploaded CSV. Each test case also uses its "
            "expert label for evaluation."
        )
        st.caption(f"Source: {budget_metrics_path} · Full report: results/budget_v2/report.md")

with sample_tab:
    st.subheader("Inspect the cached DR-5 format")
    st.write("This tab checks a file's shape and input fields without running the trained model.")
    inspect_upload = st.file_uploader("Upload CSV to inspect", type=["csv"], key="inspect_csv")
    inspect_path = st.text_input("Or CSV path (optional)", value="", key="inspect_path")
    if st.button("Inspect CSV"):
        try:
            if inspect_path.strip():
                dfi = pd.read_csv(Path(inspect_path).expanduser())
            elif inspect_upload is not None:
                dfi = pd.read_csv(inspect_upload)
            else:
                raise ValueError("Upload a file or enter a path.")
            required = ["Logits", "True Label", "Human Label", "Feat"]
            if list(dfi.columns) != required:
                raise ValueError(f"Expected columns: {required}")
            pp = np.stack([vector_from_cell(v, "Logits", 5) for v in dfi["Logits"]])
            xx = np.stack([vector_from_cell(v, "Feat") for v in dfi["Feat"]])
            st.metric("Rows", len(dfi))
            a, b = st.columns(2)
            a.metric("AI probability dimensions", pp.shape[1])
            b.metric("Feature dimensions", xx.shape[1])
            st.write("Probabilities normalized:", bool(np.allclose(pp.sum(1), 1, atol=1e-5)))
            st.write("AI accuracy:", f"{(pp.argmax(1) == dfi['True Label'].to_numpy(dtype=int)).mean():.1%}")
            st.write("Human accuracy:", f"{(dfi['Human Label'].to_numpy(dtype=int) == dfi['True Label'].to_numpy(dtype=int)).mean():.1%}")
            st.dataframe(dfi.head(10), width="stretch")
        except Exception as exc:
            st.error(str(exc))

with metrics_tab:
    st.subheader("What this app does — and does not do")
    st.markdown("""
**It can:**
- Load a checkpoint saved by the repository's `run_dr5.py` training runner.
- Evaluate cached five-class AI probabilities, human labels, and 512-D features from the repository's CSV format.
- Run MC Dropout through the project's actual `predict_samples` and `reliability_signals` functions.
- Show AI, human, and fused accuracy on the uploaded labelled CSV, plus sample-level MI/margin signals.

**It cannot:**
- Accept a retinal image and produce features: the repository does not include the image backbone/feature extraction pipeline in this app.
- Run without a trained `.pt` evaluator checkpoint.
- Turn the reliability proxy into a calibrated clinical failure probability.

For an honest evaluation, use a held-out test CSV that was not used to train or select the checkpoint. Scores computed on training rows are not generalization estimates.
    """)
    st.code("""# from the cloned repository root\npython -m pip install -r requirements.txt\npython -m pip install streamlit matplotlib\nstreamlit run streamlit_app/app.py""", language="bash")
