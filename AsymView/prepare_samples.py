"""Download a small public workshop subset and pair views using its metadata."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
FOLDER = '1EnTqFhuDVcpSuCajsp-RqvfkQ5wHu58Z'
METADATA_ID = '121ol8SvdX6lBNJNNzXPM5XrKhkvs_03m'


# Read public file identifiers and names from a workshop Drive folder.
def listing(folder_id):
    response = requests.get(f'https://drive.google.com/drive/folders/{folder_id}', timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    found = {}
    for node in soup.select('[data-id]'):
        label = node.select_one('[aria-label*="Shared"]')
        if label:
            name = label['aria-label'].split(' Shared')[0].removesuffix(' Image').removesuffix(' CSV')
            found[node['data-id']] = name
    return found


# Reuse a local asset or download its public Drive file.
def download(file_id, path):
    if path.exists() and path.stat().st_size > 1000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get('https://drive.google.com/uc',
                            params={'export': 'download', 'id': file_id}, timeout=120)
    response.raise_for_status()
    if 'text/html' in response.headers.get('content-type', ''):
        raise RuntimeError(f'Download returned a web page: {file_id}')
    path.write_bytes(response.content)


# Run the command-line workflow using the selected options.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--studies', type=int, default=3)
    args = parser.parse_args()
    download(METADATA_ID, ROOT / 'data/metadata.csv')
    metadata = pd.read_csv(ROOT / 'data/metadata.csv', low_memory=False)
    manifest, sources = [], []
    folders = [(i, n) for i, n in listing(FOLDER).items() if n != 'Metadata'][:args.studies]
    for folder_id, study_id in folders:
        print('Preparing', study_id, flush=True)
        for file_id, filename in listing(folder_id).items():
            if not filename.endswith('.png'):
                continue
            path = ROOT / 'data' / study_id / filename
            download(file_id, path)
            records = metadata[(metadata.patient_id == study_id) & (metadata.image_id == filename)]
            pairs = records[['laterality', 'view']].drop_duplicates()
            if len(pairs) != 1:
                raise ValueError(f'Missing or ambiguous view metadata for {filename}')
            row = pairs.iloc[0]
            manifest.append({'exam_id': study_id, 'laterality': row.laterality,
                             'view': row.view, 'file_path': str(path.relative_to(ROOT))})
            sources.append({'path': str(path.relative_to(ROOT)), 'drive_id': file_id,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(ROOT / 'data/manifest.csv', index=False)
    (ROOT / 'data/sources.json').write_text(json.dumps(sources, indent=2))
    print(f'Ready: {len(folders)} studies, {len(manifest)} images', flush=True)


if __name__ == '__main__':
    main()
