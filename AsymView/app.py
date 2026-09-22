"""AsymView: local mammography research demonstration."""
import hashlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import torch

from inference import ROOT, VIEWS, MODES, load_model, read_image, run_study, study_paths
from catalog import metadata_index, assignment_errors

st.set_page_config(page_title='AsymView · Mammography research', page_icon='◐', layout='wide')
st.markdown('''<style>
 .block-container {max-width:1250px;padding-top:3.5rem;}
 h1 {letter-spacing:-.045em;font-size:3.2rem!important;}
 [data-testid="stMetric"] {background:#142633;border:1px solid #2a414c;border-radius:12px;padding:18px;}
 [data-testid="stImage"] img {border-radius:9px;}
 .eyebrow {color:#7cdac9;letter-spacing:.17em;font-size:.76rem;font-weight:600;}
 </style>''', unsafe_allow_html=True)


@st.cache_resource
# Cache the verified model separately for each compute device.
def cached_model(device):
    torch.set_num_threads(4)
    return load_model(device)


@st.cache_data
# Cache the workshop metadata used to check uploaded filenames.
def known_images():
    return metadata_index()


# Create a display-only image with an optional experimental box.
def preview(array, box=None):
    values = array.astype(float)
    low, high = float(values.min()), float(values.max())
    rendered = Image.fromarray(np.uint8(np.clip((values-low) / max(high-low, 1) * 255, 0, 255))).convert('RGB')
    rendered.thumbnail((550, 650))
    if box:
        w, h = rendered.size
        ImageDraw.Draw(rendered).rectangle((box[0]*w, box[1]*h, box[2]*w, box[3]*h), outline='#7cdac9', width=3)
    return rendered


st.sidebar.markdown('### ◐ AsymView')
st.sidebar.caption('VI Days 2026 · Agentic AI workshop')
source = st.sidebar.radio('Study source', ['Workshop samples', 'Upload a study'])
devices = ['mps', 'cpu'] if torch.backends.mps.is_available() else ['cpu']
device = st.sidebar.selectbox('Inference device', devices,
                             format_func=lambda x: 'Mac GPU (Apple Metal)' if x == 'mps' else 'Mac CPU')
mode = st.sidebar.selectbox('Preprocessing', list(MODES), format_func=lambda x: MODES[x])
st.sidebar.caption('Runs locally on this computer. Images are not sent to an AI service.')
st.sidebar.divider()
st.sidebar.markdown('[AsymMirai source](https://github.com/jdonnelly36/AsymMirai)')
st.sidebar.caption('Model by Donnelly and colleagues. Built on Mirai. See the local README for provenance and limitations.')

st.markdown('<div class="eyebrow">HEALTH INNOVATION · RESEARCH DEMONSTRATOR</div>', unsafe_allow_html=True)
st.title('A closer look at asymmetry')
st.write('Compare paired mammography views and explore the regions contributing to an AsymMirai model score.')
st.warning('Research only — not for diagnosis, screening decisions or individual cancer-risk advice. '
           'The workshop PNGs differ from the model’s reference data; scores are not calibrated clinical probabilities.')

paths, arrays, exam_id, upload_bytes = {}, {}, None, {}
ready = False
if source == 'Workshop samples':
    manifest_path = ROOT / 'data/manifest.csv'
    if not manifest_path.exists():
        st.info('No samples prepared yet. Run prepare_samples.py first.')
        st.stop()
    manifest = pd.read_csv(manifest_path)
    ids = manifest.exam_id.unique().tolist()
    st.caption(f'{len(ids)} downloaded studies available. View labels are assigned automatically from the dataset metadata.')
    exam_id = st.selectbox('Select a de-identified study', ids,
                          format_func=lambda x: f'Study {ids.index(x)+1:02d} · {x}')
    try:
        paths = study_paths(manifest, exam_id)
        arrays = {key: read_image(path) for key, path in paths.items()}
        ready = True
        with st.expander('Which image is which view?'):
            st.write('CC = craniocaudal (top-down). MLO = mediolateral oblique (angled). L/R refer to the patient’s left/right breast.')
            st.dataframe(pd.DataFrame([{'View': k, 'Filename': p.name} for k, p in paths.items()]),
                         hide_index=True, width='stretch')
    except ValueError as exc:
        st.error(str(exc))
else:
    st.info('For the downloaded dataset, choose Workshop samples: all views are assigned from metadata automatically.')
    st.write('CC = top-down view; MLO = angled view. L/R mean the patient’s left/right breast. '
             'For other images, use acquisition metadata; do not guess from appearance.')
    filenames = {}
    labels = {'L-CC': 'L-CC · Left, top-down', 'R-CC': 'R-CC · Right, top-down',
              'L-MLO': 'L-MLO · Left, angled', 'R-MLO': 'R-MLO · Right, angled'}
    for col, key in zip(st.columns(4), VIEWS):
        upload = col.file_uploader(labels[key], type=['png'], key=f'upload_{key}')
        if upload is not None:
            filenames[key] = upload.name
            try:
                data = upload.getvalue()
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    array = np.asarray(image)
                if array.ndim != 2 or array.dtype not in (np.uint8, np.uint16):
                    raise ValueError('Use grayscale 8-bit or 16-bit PNGs.')
                if array.min() == array.max():
                    raise ValueError('Image is blank or constant.')
                arrays[key], upload_bytes[key] = array, data
            except Exception as exc:
                col.error(f'{key}: {exc}')
    confirmed = st.checkbox('These four de-identified images belong to one study, and the view labels are correct.')
    ready = len(upload_bytes) == 4 and confirmed
    errors = assignment_errors(filenames, known_images())
    for error in errors:
        st.error(error)
    if errors:
        ready = False
    if len(upload_bytes) == 4:
        hashes = [hashlib.sha256(upload_bytes[key]).hexdigest() for key in VIEWS]
        if len(set(hashes)) != 4:
            st.error('Duplicate images detected. Supply four distinct views.')
            ready = False
        exam_id = 'upload-' + hashlib.sha256(''.join(hashes).encode()).hexdigest()[:12]

if mode == 'scaled_8bit':
    st.info('Sensitivity experiment: multiply 8-bit values by 257 before reference normalization. '
            'This does not recover the original mammogram intensities and is not a validated adaptation.')
    if arrays and any(a.dtype != np.uint8 for a in arrays.values()):
        st.error('This experiment requires four 8-bit images. Choose reference normalization for 16-bit images.')
        ready = False

key = f'{exam_id}_{mode}_{device}'
result = st.session_state.get(key)
saved = ROOT / 'results' / f'{key}.json'
if result is None and source == 'Workshop samples' and saved.exists():
    result = json.loads(saved.read_text())
    st.caption(f'Showing a saved {result["device"].upper()} result for this study and preprocessing setting.')

if st.button('Run AsymMirai', type='primary', disabled=not ready):
    try:
        with st.spinner('Analyzing the four views locally…'):
            model = cached_model(device)
            if source == 'Upload a study':
                with tempfile.TemporaryDirectory(prefix='asymview-') as folder:
                    paths = {}
                    for view, data in upload_bytes.items():
                        paths[view] = Path(folder) / f'{view}.png'
                        paths[view].write_bytes(data)
                    result = run_study(model, paths, exam_id, mode, device)
            else:
                result = run_study(model, paths, exam_id, mode, device)
                saved.parent.mkdir(exist_ok=True)
                saved.write_text(json.dumps(result, indent=2))
            st.session_state[key] = result
    except Exception as exc:
        st.error(f'Inference did not complete: {exc}')

if result:
    c1, c2, c3 = st.columns(3)
    c1.metric('AsymMirai model score', f'{result["score"]:.4f}')
    c2.metric('Paired views analyzed', '4 / 4')
    c3.metric('Inference time', f'{result["elapsed_seconds"]:.1f} s')
    st.caption('The score is the model’s positive output on a 0–1 scale. No clinical threshold is applied.')
    st.caption(f'Computed locally on {"Mac GPU (Apple Metal)" if result["device"] == "mps" else "Mac CPU"}.')

if arrays:
    st.subheader('Paired views')
    show_windows = st.toggle('Show experimental asymmetry windows', value=False, disabled=result is None)
    st.caption('Display contrast is adjusted for viewing only. Boxes mark the strongest pooled asymmetry '
               'window in each pair; they are not cancer detections or lesion boundaries.')
    if show_windows:
        st.warning('Experimental localization: the model always selects a maximum, even for weak or misleading '
                   'differences. Positioning, orientation and image preprocessing can dominate these boxes.')
    for view in ('CC', 'MLO'):
        for col, side in zip(st.columns(2), ('L', 'R')):
            name = f'{side}-{view}'
            if name in arrays:
                box = result['windows'][view]['normalized_box'] if result and show_windows else None
                col.image(preview(arrays[name], box), caption=f'{"Left" if side == "L" else "Right"} breast · {view}', width='content')

if result:
    st.download_button('Download result and provenance', json.dumps(result, indent=2),
                       file_name=f'{key}.json', mime='application/json')
    with st.expander('Input checks and interpretation'):
        st.dataframe(pd.DataFrame(result['images']).T[['shape', 'dtype', 'min', 'max']], width='stretch')
        for notice in result['notices']:
            st.write('• ' + notice)
        st.caption(f'Checkpoint SHA-1: {result["checkpoint_sha1"]} · Evaluation mode · {result["device"].upper()}')
    if source == 'Workshop samples':
        records = []
        for candidate in MODES:
            p = ROOT / 'results' / f'{exam_id}_{candidate}_{device}.json'
            if p.exists():
                r = json.loads(p.read_text())
                records.append({'Preprocessing': MODES[candidate], 'Model score': r['score']})
        if len(records) > 1:
            with st.expander('Preprocessing sensitivity'):
                st.dataframe(pd.DataFrame(records), hide_index=True, width='stretch')
                st.caption('Changes reflect preprocessing sensitivity, not evidence of improved accuracy.')

st.divider()
st.caption('AsymView · Local research prototype · AsymMirai / Mirai · VinDr-derived workshop samples')
