"""AsymView: a simple dashboard for AsymMirai breast-asymmetry risk scores.

Run with:
    streamlit run src/app.py

Pick a de-identified sample study, run the model, and inspect the risk
score together with the localisation heatmaps. Research use only.
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from predict import AsymMiraiPredictor, gather_exams  # noqa: E402
from visualize import overlay_heatmap  # noqa: E402

DATA_DIR = os.environ.get("ASYMVIEW_DATA", "data/vindr")
META_PATH = os.path.join(DATA_DIR, "vindr_detection_v1_folds.csv")
WEIGHTS = os.environ.get("ASYMVIEW_WEIGHTS", "snapshots/trained_asymmirai.pt")
ASYM_DIR = os.environ.get("ASYMVIEW_CODE", "src/asymmirai")


@st.cache_data
def load_metadata():
    """Load the VinDr metadata table."""
    return pd.read_csv(META_PATH, low_memory=False)


@st.cache_data
def load_exams():
    """Return {exam_id: {(laterality, view): image_path}} for complete exams."""
    return gather_exams(DATA_DIR, load_metadata())


@st.cache_resource
def load_predictor():
    """Load the AsymMirai model once and reuse it across reruns."""
    return AsymMiraiPredictor(WEIGHTS, ASYM_DIR)


@st.cache_data(show_spinner="Running AsymMirai inference...")
def run_inference(exam_id):
    """Score one exam and return (risk, stats, heatmaps)."""
    exams = load_exams()
    risk, stats, heatmaps = load_predictor().predict(
        exams[exam_id], return_heatmaps=True
    )
    return risk, stats, heatmaps


def risk_band(risk):
    """Map a risk probability to an illustrative colour band and label."""
    if risk < 0.06:
        return "#2e7d32", "Low"
    if risk < 0.09:
        return "#f9a825", "Moderate"
    return "#c62828", "Elevated"


def render_overlay_figure(images, heatmaps):
    """Build a 2x2 matplotlib figure of the four views with heatmap overlays."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, view in enumerate(["CC", "MLO"]):
        for col, lat in enumerate(["L", "R"]):
            gray = cv2.imread(images[(lat, view)], cv2.IMREAD_GRAYSCALE)
            axes[row, col].imshow(overlay_heatmap(gray, heatmaps[view]))
            axes[row, col].set_title(f"{lat} {view}", fontsize=12)
            axes[row, col].axis("off")
    fig.tight_layout()
    return fig


def main():
    """Assemble the AsymView dashboard."""
    st.set_page_config(page_title="AsymView", page_icon="🩺", layout="wide")
    st.title("AsymView — AsymMirai asymmetry risk dashboard")

    # Persistent research-only warning.
    st.warning(
        "**Research use only.** AsymView is a course demonstration built on the "
        "AsymMirai research model. It is **not a medical device**, must not be "
        "used for diagnosis, and its scores are illustrative only."
    )

    exams = load_exams()
    metadata = load_metadata()
    if not exams:
        st.error(
            "No exam data found. Run `python scripts/download_data.py` first "
            "(or set ASYMVIEW_DATA)."
        )
        return

    # Sidebar controls.
    st.sidebar.header("Sample study")
    exam_ids = sorted(exams)
    labels = {eid: f"{eid[:12]}…" for eid in exam_ids}
    selected = st.sidebar.selectbox(
        "Select a de-identified exam", exam_ids, format_func=lambda e: labels[e]
    )
    show_overlay = st.sidebar.checkbox("Show asymmetry heatmap", value=True)
    st.sidebar.caption(
        f"{len(exam_ids)} public VinDr exams available. "
        "Model: AsymMirai (trained on MGH/EMBED)."
    )

    # Patient context from metadata (look up the L CC image row).
    example = metadata[
        metadata["image_id"] == os.path.basename(exams[selected][("L", "CC")])
    ].iloc[0]

    risk, stats, heatmaps = run_inference(selected)
    color, band = risk_band(risk)

    # Risk summary row.
    left, mid, right = st.columns([1.2, 1, 1])
    with left:
        st.markdown(
            f"""<div style="border-radius:12px;padding:18px;background:{color}22;
            border:2px solid {color};">
            <div style="font-size:14px;color:#555;">5-year asymmetry risk score</div>
            <div style="font-size:44px;font-weight:700;color:{color};">{risk:.1%}</div>
            <div style="font-size:16px;font-weight:600;color:{color};">{band} (illustrative)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mid:
        st.metric("BI-RADS", str(example["breast_birads"]))
        st.metric("Breast density", str(example["breast_density"]))
    with right:
        st.metric("CC asymmetry", f"{stats['CC_asymmetry']:.1f}")
        st.metric("MLO asymmetry", f"{stats['MLO_asymmetry']:.1f}")

    st.divider()

    # Heatmap overlays.
    if show_overlay:
        st.subheader("Asymmetry localisation")
        st.caption(
            "Warm colours mark regions where the left/right breast embeddings "
            "differ most — the model's basis for the risk score."
        )
        st.pyplot(render_overlay_figure(exams[selected], heatmaps))
    else:
        st.subheader("Mammogram views")
        cols = st.columns(4)
        for col, view in zip(cols, [("L", "CC"), ("R", "CC"), ("L", "MLO"), ("R", "MLO")]):
            col.image(exams[selected][view], caption=f"{view[0]} {view[1]}")

    st.divider()
    st.caption(
        "**Interpretation.** AsymMirai predicts *future* breast-cancer risk from "
        "left/right breast asymmetry; it is not designed to detect current lesions, "
        "so scores need not track BI-RADS. The public VinDr images differ from the "
        "model's training data (scanner, resolution, bit depth), so treat every "
        "score as illustrative."
    )


if __name__ == "__main__":
    main()
