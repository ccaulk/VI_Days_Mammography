# AsymView

Breast cancer risk prediction from screening mammograms using the published
**AsymMirai** model, with a small dashboard for reading the result.

> **Research use only.** AsymMirai is a research model and this is a
> demonstration of it, not a medical device. It must not be used for diagnosis
> or to inform anyone's care. The sample data carries no cancer follow-up, so
> the scores shown are uncalibrated and **no accuracy claim is made here**.

## Quick start

```bash
make setup     # Python 3.12 venv and CPU-only dependencies
make weights   # AsymMirai checkpoint (45 MB) from Duke Box
make data      # 36 de-identified VinDr-Mammo studies from Google Drive
make run       # open the dashboard
make test      # run the checks
```

Pick a sample study or upload four PNGs (left and right, CC and MLO), press
**Run AsymMirai**, and the app shows the asymmetry score, the model score, the
heatmap that produced them, and the study's VinDr annotations.

There is no GPU requirement. Inference takes roughly 5–15 seconds per study on
a CPU.

## How the model works

Mirai is a mammography risk model whose backbone is a ResNet that turns each
image into a grid of 512-dimensional embeddings. AsymMirai replaces Mirai's
transformer risk head with something you can look at: it compares the left
breast's embeddings against the right breast's at matching locations, and takes
the single largest local difference as the risk signal.

That difference map is the heatmap the dashboard draws. A high score means the
model found one region where the two breasts disagree strongly — which is what
makes the prediction inspectable rather than a bare number.

The model score is a sigmoid of the mean asymmetry using the paper's fixed
calibration constants. It is a monotone rescaling of the asymmetry, not a
validated probability on this dataset.

## What this repository contains

```
asymview/model.py    loads the published checkpoint, runs inference on CPU
asymview/data.py     pairs the sample PNGs into studies via the VinDr metadata
asymview/app.py      the Streamlit dashboard
scripts/             the two download scripts
docs/superpowers/    the design spec and implementation plan
```

`model.py` reimplements only AsymMirai's inference path. The published
checkpoint is a 2019 pickle of a whole PyTorch module graph that names modules
modern torch has deleted, so the file installs small compatibility stubs
before loading it, then uses the restored backbone and stretch vectors
directly. The original forward pass is CUDA-bound and is never called.

## A caveat about the sample images

The public sample PNGs are **8-bit** conversions of mammograms that were
originally 16-bit DICOM. AsymMirai's normalisation constants assume the 16-bit
range, so `preprocess` lifts 8-bit input back into it. That restores a usable
dynamic range, but the original DICOM windowing is gone for good.

The practical consequence: absolute asymmetry values from this repository are
**not comparable to the numbers in the AsymMirai paper**. Comparing studies in
this sample against each other is meaningful; comparing them to published
figures is not.

## An open question: mirroring

The trained checkpoint has `alignment_space = None` — it performs **no**
left/right mirroring — and upstream's own evaluation ran with image alignment
switched off. But VinDr stores left and right breasts facing opposite
directions, so running faithfully may compare one breast against a mirror image
of the other.

There is a further wrinkle: Mirai's own training pipeline included an
`align_to_left` transform, so the model may well have been trained on images
that were *already* aligned — which would make "no mirroring at comparison
time" correct there and wrong for raw VinDr.

Measured on four sample studies, mirroring consistently lowers the asymmetry
score, which is what you would expect if it removes a spurious misalignment:

| study | no mirror | peak | mirrored | peak |
|---|---|---|---|---|
| 00a369b4ec1e | 11.71 | (14, 0) | 10.53 | (21, 52) |
| 00a7a306c763 | 13.11 | (20, 0) | 11.74 | (5, 31) |
| 00ba2f2c0cd9 | 12.11 | (25, 0) | 10.08 | (33, 0) |
| 0a0c5108270e | 11.12 | (6, 14) | 10.44 | (26, 52) |

But it does not settle the matter. Without mirroring the peak pins to column 0
in three of four studies; with mirroring it mostly moves to column 52 — the
opposite edge. Either way the strongest differences tend to sit at a border
rather than in tissue, so some of the signal here is image framing, not
anatomy.

The app therefore defaults to faithful-to-upstream and offers a **mirror right
breast** toggle in the sidebar. This is a knob, not a settled question.
Answering it properly needs a cohort with cancer outcomes, which VinDr does not
provide.

## Data and weights

- Model: [AsymMirai](https://github.com/jdonnelly36/AsymMirai) —
  [checkpoint](https://duke.box.com/s/9uu9sarz6zizjkqj41iavgxz6zxwxz7c)
- Sample studies:
  [Drive folder](https://drive.google.com/drive/folders/1EnTqFhuDVcpSuCajsp-RqvfkQ5wHu58Z),
  a subset of
  [VinDr-Mammo as PNG](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)

Neither weights nor data are committed; both are fetched by the make targets.

## Attribution

AsymMirai (Donnelly et al.) and the Mirai codebase it forks are MIT licensed.
The two backbone classes reproduced in `asymview/model.py` carry their upstream
attribution in place.
