"""Pair the downloaded VinDr-Mammo PNGs into studies using the dataset's metadata CSV.

Each Drive folder is one study; its four PNGs are one image per laterality and view.
The metadata CSV maps every image to its laterality, view, BI-RADS, density and
finding categories. BI-RADS and density are recorded per breast, so both are kept
per laterality rather than collapsed to a single study-level value.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_CSV = "Metadata/vindr_detection_v1_folds.csv"

# The four standard screening views of a bilateral mammogram.
VIEWS = [("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")]


@dataclass
class Study:
    """One screening study: up to four images plus the VinDr annotations for them."""

    study_id: str
    images: dict
    birads: dict = field(default_factory=dict)
    density: dict = field(default_factory=dict)
    findings: dict = field(default_factory=dict)

    @property
    def is_complete(self):
        return set(self.images) == set(VIEWS)


# Read the VinDr metadata CSV that ships alongside the images.
def load_index(data_dir=DEFAULT_DATA_DIR):
    csv_path = Path(data_dir) / METADATA_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata CSV not found at {csv_path}; run `make data`")

    columns = [
        "patient_id", "image_id", "laterality", "view",
        "breast_birads", "breast_density", "finding_categories",
    ]
    return pd.read_csv(csv_path, usecols=columns).drop_duplicates(subset=["patient_id", "image_id"])


# Turn a list-shaped finding_categories cell into a plain list of strings.
def _parse_findings(value):
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]


# Group index rows into Study objects, keeping only folders present on disk.
def pair_images(index, data_dir):
    data_dir = Path(data_dir)
    studies = []

    for study_id, rows in index.groupby("patient_id"):
        images, birads, density, findings = {}, {}, {}, {}

        for row in rows.itertuples():
            key = (row.laterality, row.view)
            if key not in VIEWS:
                continue
            images[key] = data_dir / study_id / row.image_id
            birads[row.laterality] = row.breast_birads
            density[row.laterality] = row.breast_density
            findings[row.laterality] = _parse_findings(row.finding_categories)

        if images:
            studies.append(Study(study_id, images, birads, density, findings))

    return sorted(studies, key=lambda study: study.study_id)


# List the studies that were actually downloaded, richest first.
def list_studies(data_dir=DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)
    index = load_index(data_dir)

    # Only the sample subset was downloaded, so restrict the index to real folders.
    downloaded = {path.name for path in data_dir.iterdir() if path.is_dir() and path.name != "Metadata"}
    index = index[index["patient_id"].isin(downloaded)]

    return [study for study in pair_images(index, data_dir) if study.images]


# Build the same Study shape from user-supplied files, which carry no annotations.
def study_from_uploads(assignments):
    images = {key: Path(path) for key, path in assignments.items() if path is not None}
    return Study("uploaded study", images)
