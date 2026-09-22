"""Render AsymMirai asymmetry heatmaps on top of the mammogram views.

Produces one figure per exam showing the four views with the model's
localisation heatmap overlaid, so a prediction can be inspected visually.

Usage:
    python src/visualize.py --exam-id <id> --out-dir outputs/heatmaps
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from predict import AsymMiraiPredictor, gather_exams, VIEWS  # noqa: E402


def overlay_heatmap(gray, heatmap):
    """Resize a heatmap to the image size and blend it over the grayscale image."""
    h, w = gray.shape
    heat = cv2.resize(heatmap.astype(np.float32), (w, h))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base, 0.6, colored, 0.4, 0)


def render_exam(exam_id, images, predictor, out_dir):
    """Create a 2x2 overlay figure for one exam and save it as a PNG."""
    risk, _, heatmaps = predictor.predict(images, return_heatmaps=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, view in enumerate(["CC", "MLO"]):
        for col, lat in enumerate(["L", "R"]):
            path = images[(lat, view)]
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            axes[row, col].imshow(overlay_heatmap(gray, heatmaps[view]))
            axes[row, col].set_title(f"{lat} {view}")
            axes[row, col].axis("off")
    fig.suptitle(f"Exam {exam_id[:16]}  |  AsymMirai risk = {risk:.3f}", fontsize=13)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{exam_id}.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path, risk


def main():
    """Render overlays for the highest-risk exams."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/vindr")
    parser.add_argument("--metadata", default="data/vindr/vindr_detection_v1_folds.csv")
    parser.add_argument("--weights", default="snapshots/trained_asymmirai.pt")
    parser.add_argument("--asymmirai-dir", default="src/asymmirai")
    parser.add_argument("--predictions", default="outputs/predictions.csv")
    parser.add_argument("--out-dir", default="outputs/heatmaps")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, low_memory=False)
    exams = gather_exams(args.data_dir, metadata)
    predictor = AsymMiraiPredictor(args.weights, args.asymmirai_dir)

    predictions = pd.read_csv(args.predictions)
    top = predictions.head(args.top_n)["exam_id"].tolist()
    for exam_id in top:
        if exam_id not in exams:
            print(f"skip {exam_id}: missing views")
            continue
        path, risk = render_exam(exam_id, exams[exam_id], predictor, args.out_dir)
        print(f"wrote {path}  (risk={risk:.3f})")


if __name__ == "__main__":
    main()
