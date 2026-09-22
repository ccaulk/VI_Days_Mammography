# VI_Days_Mammography

VI Days 2026 group project: breast cancer risk prediction from mammograms
using the [AsymMirai](https://github.com/jdonnelly36/AsymMirai) model on
publicly available data.

## Starting task

> Perform breast cancer risk prediction based on the AsymMirai model on
> selected mammograms that are publicly available.

This repository contains a minimal, reproducible pipeline that:

1. Downloads the public **VinDr** mammogram subset (36 exams × 4 standard
   views) and the pre-trained **AsymMirai** weights.
2. Runs the model on each exam and reports a **5-year cancer-risk score**
   plus the localisation of the most asymmetric region.
3. Renders **heatmap overlays** so each prediction can be inspected visually.

## Quick start

```bash
# 1. Create the environment (Apple Silicon friendly)
conda create -n agentic python=3.11 -y
conda activate agentic
pip install -r requirements.txt

# 2. Fetch the public data and the model weights (~150 MB total)
python scripts/download_data.py

# 3. Run predictions over every exam
python src/predict.py

# 4. Render heatmap overlays for the highest-risk exams
python src/visualize.py --top-n 5
```

Results are written to `outputs/predictions.csv` and
`outputs/heatmaps/<exam_id>.png`.

## Repository layout

```
src/
  predict.py        # main inference script (risk score + asymmetry stats)
  visualize.py      # heatmap overlay renderer
  asymmirai/        # vendored model definitions (MIT, from AsymMirai)
scripts/
  download_data.py  # fetches the public data and pre-trained weights
outputs/
  predictions.csv   # one risk score per exam
  heatmaps/         # per-exam visualisations
```

## Method

- Each exam is loaded as four grayscale views: `L CC`, `R CC`, `L MLO`,
  `R MLO`.
- AsymMirai was trained on 16-bit PNGs produced with DCMTK's
  `-min-max-window`. The public VinDr subset is 8-bit, so each image is
  rescaled to the full 16-bit range before the model's fixed
  `(x - 7699.5) / 11765.06` normalisation. This choice matters: without it
  the model collapses to a near-constant output.
- The model returns a softmax over `[no cancer, cancer]`; the second
  column is the risk score. It also returns a spatial asymmetry heatmap
  (left vs. right embedding difference) used for localisation.

## Important limitations

- **Domain shift.** AsymMirai was trained on Hologic images from a US
  hospital; the public VinDr images come from different scanners and were
  downscaled to 8-bit. Scores should be treated as illustrative, not
  clinical.
- **Small sample.** Only 36 exams are available in the public subset, all
  of which are low BI-RADS (mostly 1–2). There is no cancer-outcome
  follow-up in the metadata, so the scores cannot be validated here.
- **Research use only.** This is not a medical device and must not be used
  for diagnosis.

## Attribution

AsymMirai and its weights are released under the MIT license by the
original authors (JonDonnelly et al.). The model definitions under
`src/asymmirai/` are vendored from that project so this repository is
self-contained.
