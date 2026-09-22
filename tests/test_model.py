"""Checks on the AsymMirai inference port: preprocessing, asymmetry metric, end-to-end."""

from pathlib import Path

import numpy as np
import pytest
import torch

from asymview.model import (
    DEFAULT_WEIGHTS,
    IMG_MEAN,
    IMG_STD,
    TARGET_SIZE,
    hybrid_asymmetry,
    load_model,
    preprocess,
)


def test_preprocess_shape_and_normalisation():
    # A constant image lets us check the exact normalisation upstream uses.
    image = np.full((512, 256), 10000, dtype=np.uint16)
    tensor = preprocess(image)

    assert tensor.shape == (1, 3, *TARGET_SIZE)
    expected = (10000 - IMG_MEAN) / IMG_STD
    assert tensor.min().item() == pytest.approx(expected, rel=1e-4)
    assert tensor.max().item() == pytest.approx(expected, rel=1e-4)


def test_identical_embeddings_have_zero_asymmetry():
    embedding = torch.randn(1, 512, 52, 64)
    scores, heatmap = hybrid_asymmetry(embedding, embedding)

    assert scores.shape == (1,)
    assert scores.item() == pytest.approx(0.0, abs=1e-6)
    assert heatmap.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_heatmap_shape_matches_stride_one_pooling():
    # 52x64 embedding, kernel (52//5, 64//5) = (10, 12) at stride 1 -> 43x53.
    left = torch.zeros(1, 512, 52, 64)
    right = torch.zeros(1, 512, 52, 64)
    _, heatmap = hybrid_asymmetry(left, right)

    assert heatmap.shape == (1, 43, 53)


def test_peak_window_contains_planted_difference():
    # Plant one local difference; the peak window must cover where we put it.
    left = torch.zeros(1, 512, 52, 64)
    right = torch.zeros(1, 512, 52, 64)
    right[0, :, 20, 30] = 5.0

    _, heatmap = hybrid_asymmetry(left, right)
    flat = int(torch.argmax(heatmap[0]))
    peak_y, peak_x = divmod(flat, heatmap.shape[-1])

    # A window at (y, x) covers rows y..y+9 and columns x..x+11.
    assert peak_y <= 20 <= peak_y + 9
    assert peak_x <= 30 <= peak_x + 11


@pytest.mark.skipif(not Path(DEFAULT_WEIGHTS).exists(), reason="run `make weights` first")
def test_end_to_end_score_on_synthetic_study():
    model = load_model()
    left = preprocess(np.full((512, 256), 12000, dtype=np.uint16))
    right = preprocess(np.full((512, 256), 9000, dtype=np.uint16))
    views = {("L", "CC"): left, ("R", "CC"): right, ("L", "MLO"): left, ("R", "MLO"): right}

    result = model.score(views)

    assert np.isfinite(result.asymmetry)
    assert 0.0 < result.probability < 1.0
    assert sorted(result.views_used) == ["CC", "MLO"]
    assert result.heatmaps["CC"].shape == (43, 53)


def test_eight_bit_input_is_lifted_to_sixteen_bit_range():
    # The public VinDr PNGs are 8-bit; without rescaling the signal collapses.
    eight_bit = np.full((512, 256), 255, dtype=np.uint8)
    sixteen_bit = np.full((512, 256), 65535, dtype=np.uint16)

    assert preprocess(eight_bit).max().item() == pytest.approx(
        preprocess(sixteen_bit).max().item(), rel=1e-3
    )
