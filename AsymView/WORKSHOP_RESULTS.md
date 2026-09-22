# Mammography workshop results

## Outcome on 22 September 2026

Built AsymView, a local dashboard using the official trained AsymMirai checkpoint. Downloaded three complete workshop studies, matched twelve PNGs to left/right CC/MLO views using the supplied metadata, and ran inference on CPU. The dashboard supports sample selection, four-image upload, score display, model asymmetry windows and result export.

## Recorded inference results

| Study | Native PNG values with reference normalization | Exploratory values multiplied by 257 |
| --- | ---: | ---: |
| 0a0c5108270e814818c1ad002482ce74 | 0.018129 | 0.052731 |
| 0a1dc220382dfae8af7f04d91fc99cfb | 0.018126 | 0.051258 |
| 0a2a9501dc0c247951fb036ba1711277 | 0.018132 | 0.051753 |

These are model outputs, not clinical risk percentages. Each complete four-view run took approximately 2.2–2.8 seconds on this Mac's CPU in the recorded runs, excluding application startup and checkpoint loading. Detailed records are in `results/`.

## What the agent discovered and changed

1. The task's Box link contains the Mirai backbone. The repository README links separately to the complete trained AsymMirai checkpoint. Used the latter and verified its published digest.
2. The upstream environment and code assume an older CUDA setup. Created an isolated current environment, adapted two forward-pass device transfers, and added compatibility mappings for two legacy backbone classes. No model weights were retrained.
3. Input image names do not encode view labels. Used the supplied metadata to establish left/right CC/MLO assignments instead of guessing from appearance.
4. The provided images are resized 8-bit PNGs, while reference normalization operates at a very different intensity scale. Retained the reference pipeline and made the mismatch explicit. Added a separately labeled sensitivity experiment.
5. Added checks for missing views, duplicate inputs, blank images, unsupported color images and invalid outputs. Set the model to evaluation mode and retained provenance with every saved result.

## What the results support

The model and dashboard can execute end to end on the provided samples. The narrow score range under native preprocessing and its shift after rescaling demonstrate dependence on the input pipeline. They do not establish accurate risk prediction or show that rescaling is the correct adaptation.

The next scientific step is to verify the PNG conversion and orientation against the original data, agree a valid preprocessing procedure, and evaluate on data with suitable longitudinal outcomes. The agent should not decide that a plausible-looking numerical output establishes model validity.

## Verification completed

Eight automated checks passed: exact reference preprocessing equivalence for 8-bit and 16-bit inputs; backbone compatibility equivalence; device adaptation equivalence; missing/duplicate-view rejection; blank/RGB rejection; duplicate-content rejection; invalid scaling rejection; and dashboard selection, mode change, real inference and empty-upload behavior.

The local server starts successfully. Dashboard behavior was tested with Streamlit AppTest. Browser visual inspection could not be completed because macOS computer-control permissions were pending; layout still needs a human browser check. At that initial stage, GPU execution and a full uploaded-study UI interaction were not tested; GPU testing was completed in the follow-up below. The same inference backend is used for uploads and sample selection.

## Suggested short presentation narrative

1. **Task and scope** — make the published AsymMirai model run on public workshop mammograms and build a small research dashboard.
2. **Agent workflow** — trace the correct checkpoint, inspect code and metadata, adapt the runtime, implement the app, and test numerical equivalence.
3. **Demo** — select one study, inspect four views, run inference, toggle asymmetry windows and export the record.
4. **Key finding** — the supplied PNGs and reference intensity scale differ. Show the two-column score comparison; neither column is a validated clinical prediction.
5. **Reflection and next steps** — agents accelerated integration and exposed a data-contract issue. Domain review, preprocessing verification and outcome-based validation remain necessary.

## Next workshop actions

- Ask the organisers whether raw or 16-bit mammograms and the exact PNG conversion procedure are available.
- Verify whether the supplied PNGs have already been cropped, normalized or flipped to a common orientation.
- Ask which checkpoint and prediction interpretation are intended for the exercise.
- Run a small approved validation set with longitudinal follow-up before making any risk-performance claims.
- Rehearse the working local demo and retain the recorded outputs as a fallback for the 24 September presentation.

## Follow-up after the complete data download

Indexed all 36 local studies (144 images) and connected them to the dashboard without additional downloads. Corrected the interaction so known filenames in the wrong view slots block inference; all four screenshot assignments were wrong according to the supplied metadata. Added full view definitions and a filename mapping table.

Verified Apple MPS inference outside the restricted environment. A paired CPU/GPU test produced scores 0.018128989 and 0.018129006 (absolute difference 1.68e-8), with times 2.56 and 1.72 seconds in that run. The dashboard now defaults to Mac GPU when available, allows CPU selection and keeps results separate by device.

Asymmetry windows are hidden by default and labeled experimental. The model always chooses a maximum-difference window; that does not establish a lesion. Wrong view pairing, opposite orientation and the existing intensity mismatch can make the selected region uninformative. No cosmetic relocation of the boxes or unvalidated orientation correction was applied.
