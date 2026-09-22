import ast
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from inference import ROOT, UPSTREAM, load_model, preprocess, read_image, run_study, study_paths


class InferenceTests(unittest.TestCase):
    """Check numerical equivalence and invalid-input handling."""
    # Limit CPU threading for repeatable, lightweight numerical checks.
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    # Verify preprocessing matches upstream function.
    def test_preprocessing_matches_upstream_function(self):
        tree = ast.parse((UPSTREAM / 'asymmetry_model/embed_explore.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resize_and_normalize')
        namespace = {'torch': torch, 'F': F}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream preprocessing>', 'exec'), namespace)
        rng = np.random.default_rng(4)
        for dtype, maximum in [(np.uint8, 255), (np.uint16, 65535)]:
            image = rng.integers(0, maximum, (40, 30), dtype=dtype)
            expected = namespace['resize_and_normalize'](image)
            torch.testing.assert_close(preprocess(image)[0], expected, rtol=0, atol=0)

    # Verify device patch matches original forward.
    def test_device_patch_matches_original_forward(self):
        source = subprocess.check_output(['git', '-C', str(UPSTREAM), 'show',
                       'HEAD:asymmetry_model/mirai_localized_dif_head.py'], text=True)
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LocalizedDifModel')
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<original forward>', 'exec'), namespace)
        model = load_model()
        torch.manual_seed(7)
        inputs = [torch.randn(1, 3, 192, 224) for _ in range(4)]
        with torch.inference_mode():
            actual, _ = model(*inputs)
            with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self):
                expected, _ = namespace['forward'](model, *inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Verify backbone compatibility matches upstream.
    def test_backbone_compatibility_matches_upstream(self):
        model = load_model()
        sources = {'Downsampler': 'onconet/models/resnet_base.py',
                   'BasicBlock': 'onconet/models/blocks/basic_block.py'}
        for name, relative in sources.items():
            node = next(n for n in ast.parse((UPSTREAM / relative).read_text()).body
                        if isinstance(n, ast.ClassDef) and n.name == name)
            forward = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
            ns = {}
            exec(compile(ast.Module(body=[forward], type_ignores=[]), '<original backbone>', 'exec'), ns)
            module = next(m for m in model.modules() if type(m).__name__ == name)
            inputs = torch.randn(1, module.conv1.in_channels, 40, 36)
            with torch.inference_mode():
                torch.testing.assert_close(module(inputs), ns['forward'](module, inputs), rtol=0, atol=0)

    # Verify rejects missing or duplicate view.
    def test_rejects_missing_or_duplicate_view(self):
        manifest = pd.read_csv(ROOT / 'data/manifest.csv')
        exam_id = manifest.exam_id.iloc[0]
        rows = manifest[manifest.exam_id == exam_id].copy()
        with self.assertRaises(ValueError):
            study_paths(rows.iloc[:-1], exam_id)
        rows.file_path = rows.file_path.iloc[0]
        with self.assertRaises(ValueError):
            study_paths(rows, exam_id)

    # Verify rejects blank and rgb images.
    def test_rejects_blank_and_rgb_images(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.png'
            for image in [np.zeros((20, 20), np.uint8), np.zeros((20, 20, 3), np.uint8)]:
                cv2.imwrite(str(path), image)
                with self.assertRaises(ValueError):
                    read_image(path)

    # Verify rejects duplicate content before model execution.
    def test_rejects_duplicate_content_before_model_execution(self):
        manifest = pd.read_csv(ROOT / 'data/manifest.csv')
        paths = study_paths(manifest, manifest.exam_id.iloc[0])
        paths['R-CC'] = paths['L-CC']
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            run_study(None, paths, 'invalid')

    # Verify scaling rejects 16bit.
    def test_scaling_rejects_16bit(self):
        with self.assertRaises(ValueError):
            preprocess(np.ones((20, 20), dtype=np.uint16), 'scaled_8bit')


if __name__ == '__main__':
    unittest.main()
