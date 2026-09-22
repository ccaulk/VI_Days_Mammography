# AsymView — VI Days Mammography

A local research dashboard built during the **VIGS Agentic AI workshop, 22 September 2026**. AsymView uses the published **AsymMirai** model to compare paired mammograms, display a model score and explore experimental asymmetry windows.

**Research only. Not validated for diagnosis, screening decisions or individual cancer-risk advice.** The workshop PNGs differ from the model's reference image domain. Successful inference does not establish accurate risk prediction.

## Features

- Select downloaded studies with metadata-based left/right and view assignments.
- Upload four de-identified grayscale PNGs; reject known filenames assigned to the wrong views.
- Run locally on **CPU or Apple Silicon GPU (Metal/MPS)**, with GPU selected when available.
- View paired images and optional experimental asymmetry windows, hidden by default.
- Export scores with input hashes, preprocessing settings, checkpoint digest, device and timing.
- Compare reference preprocessing against an explicitly unvalidated 8-bit scaling experiment.

The development dataset contains **36 studies / 144 images**. Images, metadata, model weights, generated results and virtual environments are **not committed**; obtain them using the steps below.

## Quick start

Requires **Python 3.11**, Git and network access for initial setup. Tested on Apple Silicon macOS.

```sh
git clone --branch codex git@github.com:ccaulk/VI_Days_Mammography.git
cd VI_Days_Mammography

python3.11 -m venv AsymView/.venv
AsymView/.venv/bin/python -m pip install -r AsymView/requirements.lock.txt
AsymView/.venv/bin/python setup_model.py

cd AsymView
.venv/bin/python prepare_samples.py --studies 3
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Open **http://127.0.0.1:8501**. Keep the terminal running while using the app. After setup, macOS users can also double-click `AsymView/Launch AsymView.command`.

`setup_model.py` clones upstream at a pinned commit, applies the CPU/MPS patch, downloads the complete official trained checkpoint and checks its published digest. It can be rerun and reuses verified assets. Packages are installed only in the virtual environment.

The exact development dependencies are in `AsymView/requirements.lock.txt`. Other platforms are not validated. CUDA is exposed by the CLI but has not been tested.

## Prepare the complete workshop data

From `AsymView/`:

```sh
.venv/bin/python prepare_samples.py --studies 36
```

If you already downloaded the data, use this layout and index it without downloading again:

```text
AsymView/data/
├── Metadata/vindr_detection_v1_folds.csv
├── <study_id>/
│   ├── <image_id>.png
│   └── ...
└── ...
```

```sh
.venv/bin/python catalog.py
```

The flat `data/metadata.csv` created by the downloader is also supported. `catalog.py` validates view mappings and writes `data/manifest.csv`. Refresh the dashboard and choose **Workshop samples**; these files do not need manual upload or labeling.

Downloading a different subset rewrites the generated manifest to that subset while retaining existing images. Run `catalog.py` to index all local studies again. The downloader reads a public Drive listing; if access or its format changes, use the manual-download route.

## Required views

| Input | Meaning |
| --- | --- |
| L-CC | Left breast, craniocaudal / top-down |
| R-CC | Right breast, craniocaudal / top-down |
| L-MLO | Left breast, mediolateral oblique / angled |
| R-MLO | Right breast, mediolateral oblique / angled |

Left/right refer to the **patient**, not the display. All four files must belong to one study. Use acquisition metadata rather than filename order or visual guesses. The sample selector includes a **Which image is which view?** table. Upload validation uses known filenames as lookup keys; renamed or unknown images require manual verification.

## Inference and tests

From `AsymView/`, after setting up the model and at least two sample studies:

```sh
# One study on CPU; omit --exam-id to process every study in the manifest.
.venv/bin/python inference.py --exam-id 0a0c5108270e814818c1ad002482ce74 --device cpu

# Apple Silicon GPU.
.venv/bin/python inference.py --exam-id 0a0c5108270e814818c1ad002482ce74 --device mps

# Optional preprocessing sensitivity experiment.
.venv/bin/python inference.py --exam-id 0a0c5108270e814818c1ad002482ce74 --mode scaled_8bit

.venv/bin/python -m unittest test_inference test_app -v
```

Results are saved to `results/<study>_<mode>_<device>.json`. CPU and GPU results remain separate. Uploaded studies use temporary local files, removed after inference; their results stay in the app session unless downloaded. **No image goes to an external model service, and no API key is needed.**

Tests cover upstream numerical equivalence, missing/duplicate views, duplicate content, blank/color images, invalid scaling, incorrect metadata assignments and dashboard behavior. GPU output was also compared directly with CPU output. A full browser file-upload interaction is not covered by the automated tests.

## Interpretation and limitations

The score is the checkpoint's positive output on a **0–1 scale**, not a calibrated personal risk percentage, BI-RADS score or validated time-specific prediction. No clinical threshold is applied.

Boxes approximately map the strongest pooled feature-difference window onto both images in a pair. They are **not tumour detections or lesion boundaries**. The model selects a maximum even when differences are weak or caused by positioning, orientation or background. Boxes are therefore experimental and off by default.

Reference preprocessing normalizes grayscale values as `(value - 7699.5) / 11765.06`, repeats three channels and resizes to **1664 × 2048 (height × width)** using bilinear interpolation. No additional cropping, flipping or registration is applied; this checkpoint has `alignment_space=None`.

The supplied images are resized **8-bit PNGs**, with a substantial intensity mismatch to that normalization. Multiplication by 257 is only a sensitivity experiment, not recovery of original DICOM values or a validated correction. Orientation and conversion history still need verification. The workshop metadata does not provide the longitudinal outcomes needed to validate future-risk performance.

## Workshop observations

| Study prefix | Reference normalization | Exploratory ×257 |
| --- | ---: | ---: |
| `0a0c510…` | 0.018129 | 0.052731 |
| `0a1dc22…` | 0.018126 | 0.051258 |
| `0a2a950…` | 0.018132 | 0.051753 |

These demonstrate execution and sensitivity to preprocessing, not clinical accuracy. One paired test took **1.72 s on MPS versus 2.56 s on CPU**, excluding startup and checkpoint loading, with an absolute score difference around **1.7 × 10⁻⁸**. Timings depend on hardware and warm-up.

See [workshop findings](AsymView/WORKSHOP_RESULTS.md) for the experiment history and presentation outline. Next scientific steps: verify conversion and orientation, then evaluate on appropriate longitudinal outcomes.

## Project layout

```text
setup_model.py             Fetch pinned code and verify the official checkpoint
AsymView/
  app.py                   Streamlit dashboard
  inference.py             Model loading, preprocessing and inference
  compat.py                Legacy checkpoint compatibility
  catalog.py               Metadata indexing and view checks
  prepare_samples.py       Public workshop subset downloader
  upstream-device.patch    Two forward-pass device portability changes
  test_inference.py        Numerical and input validation checks
  test_app.py              Dashboard and metadata regression checks
  requirements.lock.txt    Exact development dependencies
AsymMirai/                 Downloaded upstream checkout; ignored by Git
```

The compatibility loader preserves upstream forward methods for two legacy backbone classes, avoiding unrelated training imports. The model uses evaluation mode and moves its unregistered stretch tensors to the chosen device. The checkpoint uses legacy Python object serialization, so the app only accepts the fixed verified official model, not uploaded checkpoints.

## Sources and attribution

- [AsymMirai code](https://github.com/jdonnelly36/AsymMirai), Jon Donnelly and colleagues; pinned commit `88ac34a0cf5a8a8e2d301a4d05da625059633004`.
- [Complete trained checkpoint](https://duke.box.com/s/9uu9sarz6zizjkqj41iavgxz6zxwxz7c): `trained_asymmirai.pt`, 44,817,259 bytes; SHA-1 `76756e79a8f560b23ecf6586a3a5b53e8e53c10c`.
- [Mirai backbone link from the task](https://duke.app.box.com/s/g21ak9kfneudotokp9nmlp07r1bsdki7), which differs from the complete checkpoint used here.
- [Workshop data folder](https://drive.google.com/drive/folders/1EnTqFhuDVcpSuCajsp-RqvfkQ5wHu58Z).
- [VinDr-derived PNG dataset](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png).

Upstream-derived code retains the [AsymMirai MIT license notice](AsymView/UPSTREAM_LICENSE.txt). Data and checkpoint use remain subject to their respective source terms. Neither asset is redistributed in this branch.
