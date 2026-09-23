"""AsymView: a small dashboard for AsymMirai breast-asymmetry predictions.

Pick one of the downloaded VinDr-Mammo sample studies or upload four images,
run the published AsymMirai model, and read the asymmetry score next to the
heatmap that produced it. Research demonstration only -- not a clinical tool.
"""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from asymview.data import VIEWS, list_studies, study_from_uploads
from asymview.model import load_model, preprocess

DISCLAIMER = (
    "**Research use only.** AsymMirai is a research model and this is a demonstration "
    "of it, not a medical device. It must not be used for diagnosis or to inform care. "
    "The VinDr-Mammo sample data carries no cancer follow-up, so scores shown here are "
    "uncalibrated and no accuracy claim is made."
)

# The public PNGs are 8-bit conversions, so absolute scores drift from the paper's.
BIT_DEPTH_NOTE = (
    "The public sample images are 8-bit conversions of the original DICOMs, which were "
    "16-bit. They are rescaled to the range the model expects, but the original DICOM "
    "windowing is gone, so absolute asymmetry values are not comparable to the paper's. "
    "Compare studies against each other, not against published numbers."
)


# Load the checkpoint once per session; it takes several seconds.
@st.cache_resource
def get_model():
    return load_model()


# Score one study, keyed by its image paths so repeat clicks are free.
@st.cache_data(show_spinner=False)
def score_study(image_paths, mirror_right):
    views = {key: preprocess(path) for key, path in image_paths}
    return get_model().score(views, mirror_right=mirror_right)


# Window a 16-bit mammogram into an 8-bit greyscale image for display.
def to_display(path):
    image = np.array(Image.open(path)).astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    scaled = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


# Blend the asymmetry heatmap over the image as a red wash.
def overlay_heatmap(path, heatmap, flip=False):
    # The map is computed in the left image's frame; mirror it back for a
    # mirrored right breast so it lands on the anatomy it actually describes.
    if flip:
        heatmap = heatmap[:, ::-1]

    grey = to_display(path)
    rgb = np.stack([grey] * 3, axis=-1).astype(np.float32)

    # Stretch the small heatmap up to the image and normalise it to 0..1.
    heat = np.array(
        Image.fromarray(heatmap.astype(np.float32)).resize(
            (grey.shape[1], grey.shape[0]), Image.BILINEAR
        )
    )
    spread = heat.max() - heat.min()
    heat = (heat - heat.min()) / spread if spread > 0 else np.zeros_like(heat)

    # Square the weight so only the strongest regions show through.
    alpha = (heat**2)[..., None] * 0.55
    red = np.array([255.0, 40.0, 40.0])
    return np.clip(rgb * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)


# Draw the score, probability and per-view image pairs for one result.
def render_result(result, images, mirrored=False):
    left, right = st.columns(2)
    left.metric("Mean asymmetry", f"{result.asymmetry:.1f}")
    right.metric("Model score", f"{result.probability:.3f}")
    st.caption(
        f"Views compared: {', '.join(result.views_used)}. "
        "The model score is a sigmoid of the asymmetry using the paper's fixed "
        "calibration constants; it is not a validated probability on this data. "
        + BIT_DEPTH_NOTE
    )

    # One row per view, left and right breast side by side with the heatmap.
    for view in result.views_used:
        st.subheader(f"{view} — peak asymmetry at cell {result.peaks[view]}")
        columns = st.columns(2)
        for column, laterality in zip(columns, ("L", "R")):
            column.image(
                overlay_heatmap(
                    images[(laterality, view)],
                    result.heatmaps[view],
                    flip=mirrored and laterality == "R",
                ),
                caption=f"{laterality} {view}",
                use_container_width=True,
            )


# Show the VinDr annotations that came with a sample study.
def render_metadata(study):
    if not study.birads:
        st.info("No VinDr annotations for uploaded images.")
        return

    st.subheader("VinDr annotations")
    st.caption(
        "Dataset annotations, shown for context. VinDr's own asymmetry findings are a "
        "qualitative cross-check, not the model's training target."
    )
    st.table(
        {
            "Breast": ["Left", "Right"],
            "BI-RADS": [study.birads.get("L", "—"), study.birads.get("R", "—")],
            "Density": [study.density.get("L", "—"), study.density.get("R", "—")],
            "Findings": [
                ", ".join(study.findings.get("L", [])) or "—",
                ", ".join(study.findings.get("R", [])) or "—",
            ],
        }
    )


# Save uploaded files to a temp directory and assemble them into a study.
def collect_uploads(uploaded):
    if len(uploaded) != 4:
        st.sidebar.warning("Upload exactly four images.")
        return None

    directory = Path(tempfile.mkdtemp())
    assignments = {}
    st.sidebar.caption("Assign each file to a view:")
    for index, upload in enumerate(uploaded):
        choice = st.sidebar.selectbox(
            upload.name,
            VIEWS,
            index=index,
            format_func=lambda view: f"{view[0]} {view[1]}",
            key=f"assign_{upload.name}",
        )
        path = directory / upload.name
        path.write_bytes(upload.getbuffer())
        assignments[choice] = path

    # A duplicate assignment silently drops a view, so require all four.
    if len(assignments) != 4:
        st.sidebar.warning("Each of the four views must be assigned exactly once.")
        return None
    return study_from_uploads(assignments)


# Let the user pick a downloaded sample study or bring their own images.
def choose_study():
    source = st.sidebar.radio("Study source", ["Sample studies", "Upload"])

    if source == "Upload":
        uploaded = st.sidebar.file_uploader(
            "Four images (L/R x CC/MLO)", type=["png"], accept_multiple_files=True
        )
        return collect_uploads(uploaded) if uploaded else None

    try:
        studies = list_studies()
    except FileNotFoundError as error:
        st.error(str(error))
        return None
    if not studies:
        st.error("No studies found in data/. Run `make data`.")
        return None

    return st.sidebar.selectbox(
        "Study",
        studies,
        format_func=lambda study: f"{study.study_id[:12]}… ({len(study.images)} images)",
    )


def main():
    st.set_page_config(page_title="AsymView", layout="wide")
    st.title("AsymView")
    st.caption("Breast asymmetry risk predictions from the AsymMirai model")
    st.warning(DISCLAIMER)

    study = choose_study()
    mirror_right = st.sidebar.checkbox(
        "Mirror right breast",
        value=False,
        help=(
            "The trained model does no mirroring, but VinDr stores left and right "
            "breasts facing opposite ways. Off matches upstream; on compares "
            "anatomically. Left as a knob because the sample data cannot settle it."
        ),
    )

    if study is None:
        st.info("Choose a sample study or upload four images to begin.")
        return

    # An incomplete study is still scorable from whichever pairs it has.
    if not study.is_complete:
        missing = [f"{lat} {view}" for lat, view in VIEWS if (lat, view) not in study.images]
        st.warning(f"Incomplete study; missing {', '.join(missing)}. Scoring the pairs present.")

    render_metadata(study)

    if st.button("Run AsymMirai", type="primary"):
        image_paths = tuple(sorted((key, str(path)) for key, path in study.images.items()))
        with st.spinner("Running inference on CPU (roughly 5-15 seconds)…"):
            try:
                result = score_study(image_paths, mirror_right)
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))
                return
        render_result(result, study.images, mirrored=mirror_right)


if __name__ == "__main__":
    main()
