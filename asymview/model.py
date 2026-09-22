"""Load the published AsymMirai checkpoint and run its asymmetry inference on CPU.

AsymMirai replaces Mirai's transformer risk head with an interpretable bilateral
comparison: the Mirai ResNet backbone embeds each image, left and right embeddings
are differenced at matching locations, and the largest local difference is the score.

The published checkpoint is a pre-torch-1.6 pickle of a whole ``nn.Module`` graph
referencing modules that no longer exist, so loading it needs the compatibility
stubs installed below. Only the restored backbone and stretch vectors are used --
the original ``forward`` is CUDA-bound and is never called.

Upstream: https://github.com/jdonnelly36/AsymMirai (MIT).
"""

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "trained_asymmirai.pt"

# Preprocessing constants, fixed by upstream embed_explore.resize_and_normalize.
IMG_MEAN = 7699.5
IMG_STD = 11765.06
TARGET_SIZE = (1664, 2048)  # (height, width)

# Asymmetry constants, read from the trained checkpoint.
LATENT_H = 5
LATENT_W = 5
ASYM_MEAN = 40.0
ASYM_STD = 10.0


class BasicBlock(nn.Module):
    """Residual block from the Mirai backbone, needed to rebuild the pickled graph.

    Copied from onconet.models.blocks.basic_block (MIT). Only ``forward`` matters:
    unpickling restores attributes directly and never calls ``__init__``.
    """

    expansion = 1

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Downsampler(nn.Module):
    """Stem of the Mirai backbone (onconet.models.resnet_base, MIT).

    The pickled instance carries conv1, bn1, relu and maxpool, so ``forward`` is
    the standard ResNet stem applied in that order.
    """

    def forward(self, x):
        return self.maxpool(self.relu(self.bn1(self.conv1(x))))


class LocalizedDifModel(nn.Module):
    """Container for the restored AsymMirai object; its own forward is never used."""


# Register a throwaway module under `name` so the pickle can resolve it.
def _stub_module(name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


# Recreate the 2019 module graph the checkpoint refers to, once per process.
def _install_legacy_shims():
    if "mirai_localized_dif_head" in sys.modules:
        return

    for package in (
        "onconet",
        "onconet.models",
        "onconet.models.blocks",
        "onconet.models.pools",
        "torch.nn.backends",
    ):
        _stub_module(package)

    _stub_module("onconet.models.blocks.basic_block").BasicBlock = BasicBlock

    resnet_base = _stub_module("onconet.models.resnet_base")
    resnet_base.Downsampler = Downsampler
    resnet_base.ResNet = LocalizedDifModel

    # torch removed this backend hook after 1.x; the pickle still names it.
    _stub_module("torch.nn.backends.thnn")._get_thnn_function_backend = lambda *_: None

    _stub_module("mirai_localized_dif_head").LocalizedDifModel = LocalizedDifModel
    _stub_module("asymmetry_metrics").hybrid_asymmetry = hybrid_asymmetry


# Apply upstream's exact recipe: normalise, expand to 3 channels, resize.
def preprocess(source):
    if isinstance(source, (str, Path)):
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"could not read image: {source}")
    else:
        image = source

    if image.ndim == 3:
        image = image[..., 0]

    # The public VinDr PNGs are 8-bit, but the model's constants assume the
    # 16-bit DICOM range, so lift 8-bit input to that range first.
    if image.dtype == np.uint8:
        image = image.astype(np.float32) * 257.0  # 255 -> 65535

    normalised = torch.tensor((image.astype(np.float32) - IMG_MEAN) / IMG_STD)
    expanded = normalised.expand(1, 3, *normalised.shape).float()
    return F.interpolate(expanded, size=TARGET_SIZE, mode="bilinear", align_corners=False)


# Largest local left/right embedding difference, plus the map it came from.
def hybrid_asymmetry(left, right, latent_h=LATENT_H, latent_w=LATENT_W, **_):
    difference = torch.abs(left - right)

    # Stride-1 pooling ("flexible" in upstream) scans every window position.
    kernel = (difference.shape[-2] // latent_h, difference.shape[-1] // latent_w)
    pooled = F.max_pool2d(difference, kernel, stride=(1, 1))

    # Collapse the channel axis, then take the single strongest location.
    heatmap = torch.norm(pooled, dim=-3)
    scores = heatmap.flatten(start_dim=-2).max(dim=-1).values
    return scores, heatmap


@dataclass
class Result:
    """One study's asymmetry result: score, probability, and per-view heatmaps."""

    asymmetry: float
    probability: float
    heatmaps: dict
    peaks: dict
    views_used: list


class AsymMirai(nn.Module):
    """The trained AsymMirai backbone and stretch vectors, running on CPU."""

    def __init__(self, restored):
        super().__init__()
        # The checkpoint may wrap the backbone in DataParallel; unwrap if so.
        self.backbone = getattr(restored, "module", restored.backbone)
        self.backbone = getattr(self.backbone, "module", self.backbone)
        self.backbone.eval()
        self.stretch = {
            "CC": restored.cc_stretch_params.detach().cpu().view(1, -1, 1, 1),
            "MLO": restored.mlo_stretch_params.detach().cpu().view(1, -1, 1, 1),
        }

    # Embed one image and apply that view's learned per-channel scaling.
    def _embed(self, image, view):
        return self.backbone(image) * self.stretch[view]

    # Score one study: mean asymmetry over the view pairs that are present.
    @torch.no_grad()
    def score(self, views, mirror_right=False):
        asymmetries, heatmaps, peaks, used = [], {}, {}, []

        for view in ("CC", "MLO"):
            left, right = views.get(("L", view)), views.get(("R", view))
            if left is None or right is None:
                continue

            # The trained model has alignment_space=None, so mirroring is opt-in.
            if mirror_right:
                right = torch.flip(right, dims=[-1])

            score, heatmap = hybrid_asymmetry(self._embed(left, view), self._embed(right, view))

            asymmetries.append(score.item())
            heatmaps[view] = heatmap[0].numpy()
            flat = int(np.argmax(heatmaps[view]))
            peaks[view] = divmod(flat, heatmaps[view].shape[-1])
            used.append(view)

        if not asymmetries:
            raise ValueError("no complete left/right view pair to compare")

        # Upstream averages over available views, then calibrates with fixed stats.
        mean_asymmetry = float(np.mean(asymmetries))
        probability = float(torch.sigmoid(torch.tensor((mean_asymmetry - ASYM_MEAN) / ASYM_STD)))
        return Result(mean_asymmetry, probability, heatmaps, peaks, used)


# Restore the published checkpoint onto CPU and wrap it for inference.
def load_model(path=DEFAULT_WEIGHTS):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found at {path}; run `make weights`")

    _install_legacy_shims()
    restored = torch.load(path, map_location="cpu", weights_only=False)
    return AsymMirai(restored)
