"""Run AsymMirai breast-cancer risk prediction on the public VinDr subset.

The trained AsymMirai model expects the four standard mammography views
(L CC, R CC, L MLO, R MLO) as 16-bit-scale grayscale images. This script
loads the public VinDr subset, runs the model on CPU (Apple Silicon) and
writes one risk score per exam to outputs/predictions.csv.

Usage:
    python src/predict.py \
        --data-dir data/vindr \
        --metadata data/vindr/vindr_detection_v1_folds.csv \
        --weights snapshots/trained_asymmirai.pt \
        --out outputs/predictions.csv
"""
import argparse
import glob
import os
import sys
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# AsymMirai was trained on 16-bit PNGs; these constants are from the repo.
IMG_MEAN = 7699.5
IMG_STD = 11765.06
TARGET_SIZE = (1664, 2048)
VIEWS = [("L", "CC"), ("R", "CC"), ("L", "MLO"), ("R", "MLO")]


def rescale_to_16bit(img, mode):
    """Rescale an 8-bit image to the 16-bit intensity range AsymMirai expects.

    The public VinDr subset is 8-bit, while AsymMirai was trained on 16-bit
    PNGs produced with DCMTK's -min-max-window. 'minmax16' reproduces that
    windowing per image; 'x257' is a plain linear 8->16 bit expansion.
    """
    img = img.astype(np.float32)
    if mode == "raw":
        return img
    if mode == "x257":
        return img * 257.0
    if mode == "standardize":
        return (img - img.mean()) / (img.std() + 1e-6) * IMG_STD + IMG_MEAN
    # Default: per-image min-max stretch to the full 16-bit range.
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo + 1e-6) * 65535.0


class AsymMiraiPredictor:
    """Thin wrapper around a pickled AsymMirai model for CPU inference."""

    def __init__(self, weights_path, asymmirai_dir, device="cpu", intensity="minmax16"):
        """Load the pickled model and prepare it for inference."""
        self.intensity = intensity
        # The pickled model references its original module names, so make
        # the vendored model definitions importable before unpickling.
        sys.path.insert(0, os.path.abspath(asymmirai_dir))
        # The original code hardcodes .cuda(); neutralise it so we can run on CPU.
        torch.Tensor.cuda = lambda self, *a, **k: self
        torch.nn.Module.cuda = lambda self, *a, **k: self

        self.device = device
        self.model = torch.load(
            weights_path, map_location=device, weights_only=False
        )
        self.model.eval()

    def preprocess(self, path):
        """Load one grayscale mammogram and return a normalised 3-channel tensor."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        # Rescale to the 16-bit intensity range the model was trained on.
        img = rescale_to_16bit(img, self.intensity)
        tensor = torch.tensor((img - IMG_MEAN) / IMG_STD)
        tensor = tensor.expand(1, 3, *tensor.shape).float()
        tensor = F.interpolate(
            tensor, size=TARGET_SIZE, mode="bilinear", align_corners=False
        )
        return tensor[0]

    def predict(self, images, return_heatmaps=False):
        """Return (risk_probability, per-view stats[, heatmaps]) for one exam.

        images: dict mapping (laterality, view) -> image path.
        """
        batch = [self.preprocess(images[v]).unsqueeze(0) for v in VIEWS]
        with torch.no_grad():
            output, other = self.model(*batch)

        # Column 1 of the softmax output is the cancer-risk probability.
        risk = float(output[0, 1])

        # Summarise the localisation heatmaps for each view group.
        stats, heatmaps = {}, {}
        for name, idx in (("CC", 0), ("MLO", 1)):
            heatmap = other[idx]["heatmap"]
            x_arg = other[idx]["x_argmin"]
            y_arg = other[idx]["y_argmin"]
            stats[f"{name}_asymmetry"] = float(heatmap.max())
            stats[f"{name}_x"] = int(x_arg.flatten()[0])
            stats[f"{name}_y"] = int(y_arg.flatten()[0])
            heatmaps[name] = heatmap[0].cpu().numpy()
        if return_heatmaps:
            return risk, stats, heatmaps
        return risk, stats


def gather_exams(data_dir, metadata):
    """Return {exam_id: {(laterality, view): image_path}} for complete exams."""
    exams = {}
    for folder in sorted(glob.glob(os.path.join(data_dir, "*/"))):
        images = {}
        for path in glob.glob(os.path.join(folder, "*.png")):
            match = metadata[metadata["image_id"] == os.path.basename(path)]
            if len(match) == 0:
                continue
            key = (match.iloc[0]["laterality"], match.iloc[0]["view"])
            images.setdefault(key, path)
        # Only keep exams that have all four standard views.
        if all(view in images for view in VIEWS):
            exams[os.path.basename(os.path.dirname(folder))] = images
    return exams


def main():
    """Run inference over every exam and save a results table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/vindr")
    parser.add_argument("--metadata", default="data/vindr/vindr_detection_v1_folds.csv")
    parser.add_argument("--weights", default="snapshots/trained_asymmirai.pt")
    parser.add_argument("--asymmirai-dir", default="src/asymmirai")
    parser.add_argument("--intensity", default="minmax16",
                        choices=["minmax16", "x257", "standardize", "raw"])
    parser.add_argument("--out", default="outputs/predictions.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    metadata = pd.read_csv(args.metadata, low_memory=False)
    exams = gather_exams(args.data_dir, metadata)
    print(f"Found {len(exams)} complete exams with all four views")

    predictor = AsymMiraiPredictor(
        args.weights, args.asymmirai_dir, intensity=args.intensity
    )

    rows = []
    for i, (exam_id, images) in enumerate(exams.items(), 1):
        risk, stats = predictor.predict(images)
        # Pull patient-level context from the metadata for the report.
        example = metadata[metadata["image_id"] == os.path.basename(
            images[("L", "CC")])].iloc[0]
        rows.append({
            "exam_id": exam_id,
            "patient_id": example["patient_id"],
            "risk_score": risk,
            "breast_birads": example["breast_birads"],
            "breast_density": example["breast_density"],
            **stats,
        })
        print(f"[{i}/{len(exams)}] {exam_id[:12]}  risk={risk:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result = pd.DataFrame(rows).sort_values("risk_score", ascending=False)
    result.to_csv(args.out, index=False)
    print(f"\nSaved {len(result)} predictions to {args.out}")
    print(result[["exam_id", "risk_score", "breast_birads"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
