# AsymView — Design

Date: 2026-09-22
Status: approved

## Purpose

Run the published AsymMirai breast-asymmetry risk model over publicly available
VinDr-Mammo screening studies and present the result in a small dashboard:
one score, one heatmap per view, the study's VinDr metadata, and a standing
research-only disclaimer.

AsymMirai (Donnelly et al.) replaces Mirai's transformer risk head with an
interpretable bilateral-asymmetry comparison: the Mirai ResNet backbone embeds
each image, the left and right embeddings are differenced at matching spatial
locations, and the largest local difference is the risk signal.

## Non-goals

- No training, fine-tuning or evaluation of model accuracy.
- No AUC or performance claim. VinDr-Mammo carries **no cancer follow-up**, so
  AsymMirai's 1–5 year risk cannot be validated on it and its score is
  uncalibrated here. The app states this; it does not paper over it.
- No clinical use. Research demonstration only.
- No port of Mirai's training code or the ~200-file `onconet` tree.

## Verified facts

Established by probing the published artifacts before writing this spec.

**Weights** — two public Box links, both downloadable without auth:

| File | Size | Contents |
|---|---|---|
| `trained_asymmirai.pt` | 44.8 MB | Trained `LocalizedDifModel`: backbone **and** head |
| `mgh_mammo_MIRAI_Base_May20_2019.p` | 50.7 MB | Mirai backbone only — **not required** |

Both are pre-torch-1.6 pickles of whole `nn.Module` objects, not state dicts.
They reference `onconet.models.*`, `mirai_localized_dif_head`,
`asymmetry_metrics`, and `torch.nn.backends.thnn` — the last of which modern
torch has deleted. Loading therefore requires stub modules registered in
`sys.modules` before `torch.load(..., map_location="cpu", weights_only=False)`.
This was tested and works; it emits `SourceChangeWarning`s, which are harmless.

**Trained configuration**, read off the checkpoint rather than guessed:

```
alignment_space   : None       # no left/right flip in the trained forward
use_stretch       : True
use_stretch_matrix: False      # per-channel VECTOR stretch, shape (512,)
flexible_asymmetry: True       # max-pool with stride 1
latent_h, latent_w: 5, 5
topk_for_heatmap  : None
learned_asym_mean : 40
learned_asym_std  : 10
backbone          : Sequential, 9 children, 11.2M params
```

**Preprocessing**, pinned from upstream `embed_explore.resize_and_normalize`:
read the 16-bit PNG unchanged, `(img - 7699.5) / 11765.06`, expand to 3
channels, bilinear resize to `(1664, 2048)` as `(H, W)`. The backbone is
stride-32, giving a `52 x 64` embedding; pooling kernel `(52//5, 64//5) =
(10, 12)` at stride 1 gives a `43 x 53` heatmap.

**Data** — the Drive folder is public and holds 36 studies of 4 PNGs each,
plus `Metadata/vindr_detection_v1_folds.csv`. That CSV maps `patient_id`
(the folder name) and `image_id` (the PNG name) to `laterality`, `view`,
`breast_birads`, `breast_density`, and per-finding flags including
`Asymmetry`, `Focal_Asymmetry` and `Global_Asymmetry`.

**Environment** — CPU only, no GPU. System Python is 3.14, which has no torch
wheels, so the project pins Python 3.12 through `uv`.

## Architecture

```
asymview/model.py    legacy-pickle shims, preprocessing, CPU inference
asymview/data.py     study discovery and pairing from the VinDr CSV
asymview/app.py      Streamlit dashboard
scripts/fetch_weights.py   Box  -> weights/
scripts/fetch_data.py      Drive -> data/
tests/test_asymview.py
Makefile, README.md, pyproject.toml
```

Three units with one job each, dependencies pointing one way:
`app.py -> {model.py, data.py}`, and `model.py` and `data.py` know nothing of
each other or of Streamlit. Both are usable from a plain script.

### `model.py`

Owns everything about the network. Public surface:

- `load_model(weights_path) -> AsymMirai` — installs the legacy shims once,
  unpickles onto CPU, reads the trained config off the object.
- `preprocess(path_or_array) -> Tensor[1,3,1664,2048]` — the upstream recipe.
- `AsymMirai.score(views, mirror_right=False) -> Result` where `views` maps
  `("L"|"R", "CC"|"MLO")` to a preprocessed tensor.

`Result` carries the mean asymmetry, the sigmoid probability, a per-view
heatmap, and the peak `(y, x)` cell per view.

The forward pass is reimplemented for CPU following
`LocalizedDifModel.forward`, using the config read from the checkpoint:

1. per view pair, optionally mirror the right image (see open question),
2. embed left and right through the backbone,
3. scale both by the view's stretch vector, `emb * params.view(1,-1,1,1)`,
4. `hybrid_asymmetry`: `|L-R|`, `max_pool2d(kernel=(H//5, W//5), stride=1)`,
   L2 over channels, then max over space — returning the scalar, the heatmap
   and the argmax location,
5. mean over the view pairs actually present,
6. `prob = sigmoid((mean_asym - 40) / 10)`.

Only the unpickled object's `backbone` and stretch tensors are used; its own
`forward` is never called, which is what keeps the CUDA-bound training code out.

### `data.py`

- `load_index(csv_path) -> DataFrame` — one row per image.
- `list_studies() -> list[Study]` — folders with all four of L/R x CC/MLO.
- `Study` — id, the four image paths, BI-RADS, density, finding flags.
- `study_from_uploads(files, assignments)` — the same shape from uploads, so
  the app has a single code path.

Studies missing a view are listed but marked incomplete; the model scores
whichever pairs exist, matching upstream's behaviour of skipping empty views.

### `app.py`

Sidebar: sample-study dropdown or a four-image upload, the mirror-right
toggle, and a Run button. Main panel: the score and probability, a CC row and
an MLO row showing each image with its asymmetry heatmap overlaid and the peak
cell marked, and a VinDr metadata panel — BI-RADS, density, and VinDr's own
asymmetry findings, which give a qualitative sanity check on an asymmetry
model even though they are not the model's target.

A research-only disclaimer is always visible, not behind an expander.

The model is cached with `@st.cache_resource` (roughly 10 s to load) and
scores with `@st.cache_data`. Inference is about 5–15 s per exam on CPU.

## Error handling

| Condition | Behaviour |
|---|---|
| Weights absent | Error naming the exact `make weights` command |
| Data absent | Study list empty, message naming `make data`; upload still works |
| Study missing views | Score the pairs present; state which were skipped |
| Upload not 16-bit PNG | Accept it, warn that normalisation assumes 16-bit |
| Unpickle fails | Raise with the checkpoint path and torch version |

## Testing

One runnable file, `tests/test_asymview.py`, asserting the properties that
break silently if the port drifts:

1. `preprocess` returns `[1,3,1664,2048]` and applies the exact mean/std.
2. `hybrid_asymmetry` of identical inputs is 0.
3. A planted local perturbation puts the heatmap peak in the expected cell.
4. Heatmap shape is `43 x 53` for a `52 x 64` embedding.
5. Given weights, an end-to-end study score is finite and its probability is
   in `(0, 1)` — skipped when weights are absent.

## Open question, deliberately left open

The trained model has `alignment_space = None`: it performs **no** mirroring,
and upstream's own evaluation ran with `align_images=False`. But VinDr stores
left and right breasts facing opposite directions, so a faithful run may be
comparing one breast against a mirror image of the other.

The app therefore defaults to faithful-to-upstream (no mirroring) and exposes
a **mirror right breast** toggle. This is a knob, not a resolved question, and
the README says so. Resolving it properly needs a labelled cohort, which VinDr
does not provide.

## Attribution

AsymMirai and the Mirai backbone it builds on are MIT licensed. The two
classes copied into the compatibility shim carry their upstream attribution.
