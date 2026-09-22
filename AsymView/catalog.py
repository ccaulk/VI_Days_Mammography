"""Index downloaded studies and check view labels against workshop metadata."""
from pathlib import Path
import pandas as pd
from inference import ROOT, VIEWS, study_paths


# Build unambiguous image-to-study and view mappings.
def metadata_index():
    path = ROOT / 'data/Metadata/vindr_detection_v1_folds.csv'
    if not path.exists():
        path = ROOT / 'data/metadata.csv'
    frame = pd.read_csv(path, low_memory=False)
    records = frame[['image_id', 'patient_id', 'laterality', 'view']].drop_duplicates()
    records = records[~records.image_id.duplicated(keep=False)]
    return {row.image_id: (str(row.patient_id), f'{row.laterality}-{row.view}')
            for row in records.itertuples(index=False)}


# Identify view labels or study memberships that conflict with metadata.
def assignment_errors(filenames, index):
    errors, studies = [], set()
    for assigned, name in filenames.items():
        known = index.get(Path(name).name)
        if known:
            study, correct = known
            studies.add(study)
            if assigned != correct:
                errors.append(f'{name}: metadata identifies {correct}, but it was assigned to {assigned}.')
    if len(studies) > 1:
        errors.append('The selected files belong to different studies according to the metadata.')
    return errors


# Validate all local study folders and write their manifest.
def index_downloads():
    index, rows = metadata_index(), []
    for path in sorted((ROOT / 'data').glob('*/*.png')):
        known = index.get(path.name)
        if not known:
            raise ValueError(f'No unambiguous metadata for {path.name}')
        study, label = known
        if path.parent.name != study:
            raise ValueError(f'Study folder conflicts with metadata: {path}')
        side, view = label.split('-')
        rows.append({'exam_id': study, 'laterality': side, 'view': view,
                     'file_path': str(path.relative_to(ROOT))})
    if not rows:
        raise ValueError('No downloaded PNGs found.')
    frame = pd.DataFrame(rows)
    for exam_id in frame.exam_id.unique():
        study_paths(frame, exam_id)
    frame.to_csv(ROOT / 'data/manifest.csv', index=False)
    print(f'Indexed {frame.exam_id.nunique()} complete studies / {len(frame)} images.')


if __name__ == '__main__':
    index_downloads()
