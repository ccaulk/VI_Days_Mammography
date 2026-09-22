"""Research inference using the official AsymMirai checkpoint and input size."""
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT.parent / 'AsymMirai'
sys.path.insert(0, str(UPSTREAM / 'asymmetry_model'))
import compat

CHECKPOINT = ROOT / 'models/trained_asymmirai.pt'
CHECKPOINT_SHA1 = '76756e79a8f560b23ecf6586a3a5b53e8e53c10c'
VIEWS = ('L-CC', 'R-CC', 'L-MLO', 'R-MLO')
MODES = {'reference': 'Reference normalization, native PNG values',
         'scaled_8bit': 'Exploratory 8-bit values multiplied by 257'}


# Verify and load the official model on the requested compute device.
def load_model(device='cpu'):
    if not CHECKPOINT.exists():
        raise FileNotFoundError('The official trained AsymMirai checkpoint is missing.')
    if hashlib.sha1(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA1:
        raise ValueError('Checkpoint is incomplete or differs from the verified official download.')
    # Full-object pickle is required by this legacy checkpoint. Only load the
    # fixed official file whose digest above was checked against Box metadata.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.serialization.SourceChangeWarning)
        model = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, pickle_module=compat)
    if isinstance(model.backbone, torch.nn.DataParallel):
        model.backbone = model.backbone.module
    model.to(device)
    # The original code stores learned stretch tensors outside parameters/buffers.
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if isinstance(value, torch.Tensor):
                setattr(module, name, value.detach().to(device))
    model.eval()
    return model


# Resolve exactly one distinct image for each required study view.
def study_paths(manifest, exam_id):
    rows = manifest[manifest.exam_id.astype(str) == str(exam_id)]
    paths = {}
    for view in VIEWS:
        side, position = view.split('-')
        values = rows[(rows.laterality == side) & (rows.view == position)].file_path.unique()
        if len(values) != 1:
            raise ValueError(f'{view}: expected exactly one image, found {len(values)}.')
        path = Path(values[0])
        paths[view] = path if path.is_absolute() else ROOT / path
    if len(set(p.resolve() for p in paths.values())) != 4:
        raise ValueError('Each view must use a different image.')
    return paths


# Read a nonblank grayscale PNG without changing its bit depth.
def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Cannot read image: {Path(path).name}')
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError('Expected a grayscale 8-bit or 16-bit PNG image.')
    if image.min() == image.max():
        raise ValueError('The image is blank or constant.')
    return image


# Apply the reference normalization or the explicit scaling experiment.
def preprocess(image, mode='reference'):
    if mode not in MODES:
        raise ValueError('Unknown preprocessing mode.')
    if mode == 'scaled_8bit' and image.dtype != np.uint8:
        raise ValueError('Exploratory scaling is only defined for 8-bit images.')
    values = image.astype(np.float64)
    if mode == 'scaled_8bit':
        values *= 257.0
    # Match embed_explore.resize_and_normalize(use_crop=False): same order,
    # mean, standard deviation, three channels, bilinear size and no alignment.
    tensor = torch.tensor((values - 7699.5) / 11765.06).expand(1, 3, *image.shape).float()
    return F.interpolate(tensor, size=(1664, 2048), mode='bilinear', align_corners=False)


# Analyze all four views and record the output with input provenance.
def run_study(model, paths, exam_id, mode='reference', device='cpu'):
    if device == 'mps':
        torch.mps.synchronize()
    started = time.perf_counter()
    images = {key: read_image(paths[key]) for key in VIEWS}
    hashes = {key: hashlib.sha256(paths[key].read_bytes()).hexdigest() for key in VIEWS}
    if len(set(hashes.values())) != 4:
        raise ValueError('Duplicate images detected; supply four distinct views.')
    tensors = [preprocess(images[key], mode).to(device) for key in VIEWS]
    with torch.inference_mode():
        output, details = model(*tensors)
    score = float(output[0, 1].cpu())
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Model returned an invalid score.')
    windows = {}
    for view, detail in zip(('CC', 'MLO'), details):
        heatmap = detail['heatmap'][0].cpu().numpy()
        row, col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # At input 1664x2048, the backbone gives 52x64 features. Flexible
        # max pooling uses floor(52/5) x floor(64/5) = 10x12 windows.
        windows[view] = {'heatmap': heatmap.tolist(),
                         'feature_row': int(row), 'feature_col': int(col),
                         'normalized_box': [float(col / 64), float(row / 52),
                                            float((col + 12) / 64), float((row + 10) / 52)]}
    if device == 'mps':
        torch.mps.synchronize()
    notices = ['Research only. Not a diagnosis or a calibrated individual cancer-risk estimate.',
               'The workshop subset does not establish longitudinal risk-prediction accuracy.',
               'Prediction windows are model asymmetry regions, not lesion segmentations.']
    if any(a.dtype == np.uint8 for a in images.values()):
        notices.append('8-bit, resized workshop PNGs differ from the reference intensity domain.')
    if mode == 'scaled_8bit':
        notices.append('Multiplying by 257 is an unvalidated sensitivity experiment, not recovery of original intensities.')
    return {'exam_id': str(exam_id), 'score': score, 'mode': mode,
            'mode_description': MODES[mode], 'device': device,
            'elapsed_seconds': round(time.perf_counter() - started, 3),
            'checkpoint_sha1': CHECKPOINT_SHA1, 'input_size': [1664, 2048],
            'model_eval_mode': True, 'alignment_space': model.alignment_space,
            'images': {key: {'filename': paths[key].name, 'shape': list(a.shape),
                             'dtype': str(a.dtype), 'min': int(a.min()), 'max': int(a.max()),
                             'sha256': hashes[key]}
                       for key, a in images.items()},
            'windows': windows, 'notices': notices}


# Run the command-line workflow using the selected options.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifest.csv')
    parser.add_argument('--exam-id')
    parser.add_argument('--mode', choices=MODES, default='reference')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = pd.read_csv(args.manifest)
    model = load_model(args.device)
    ids = [args.exam_id] if args.exam_id else manifest.exam_id.unique()
    (ROOT / 'results').mkdir(exist_ok=True)
    for exam_id in ids:
        result = run_study(model, study_paths(manifest, exam_id), exam_id, args.mode, args.device)
        path = ROOT / 'results' / f'{exam_id}_{args.mode}_{args.device}.json'
        path.write_text(json.dumps(result, indent=2))
        print(f'{exam_id}: {result["score"]:.6f}, {result["elapsed_seconds"]:.1f}s -> {path.name}', flush=True)


if __name__ == '__main__':
    main()
