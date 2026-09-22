"""Checks on study pairing: a study is four images keyed by laterality and view."""

from pathlib import Path

import pandas as pd
import pytest

from asymview.data import VIEWS, Study, list_studies, pair_images, study_from_uploads


def _index_frame():
    # Two studies: one complete, one missing its right MLO.
    rows = [
        ("s1", "a.png", "L", "CC", "BI-RADS 3", "DENSITY C", "['Mass']"),
        ("s1", "b.png", "L", "MLO", "BI-RADS 3", "DENSITY C", "['Mass']"),
        ("s1", "c.png", "R", "CC", "BI-RADS 1", "DENSITY C", "['No Finding']"),
        ("s1", "d.png", "R", "MLO", "BI-RADS 1", "DENSITY C", "['No Finding']"),
        ("s2", "e.png", "L", "CC", "BI-RADS 2", "DENSITY B", "['Asymmetry']"),
        ("s2", "f.png", "L", "MLO", "BI-RADS 2", "DENSITY B", "['Asymmetry']"),
        ("s2", "g.png", "R", "CC", "BI-RADS 2", "DENSITY B", "['No Finding']"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "patient_id", "image_id", "laterality", "view",
            "breast_birads", "breast_density", "finding_categories",
        ],
    )


def test_views_are_the_four_screening_views():
    assert VIEWS == [("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")]


def test_complete_study_pairs_all_four_views(tmp_path):
    studies = {s.study_id: s for s in pair_images(_index_frame(), tmp_path)}
    study = studies["s1"]

    assert study.is_complete
    assert set(study.images) == set(VIEWS)
    assert study.images[("L", "CC")] == tmp_path / "s1" / "a.png"
    assert study.birads["L"] == "BI-RADS 3"
    assert study.birads["R"] == "BI-RADS 1"
    assert study.density["L"] == "DENSITY C"


def test_incomplete_study_is_listed_but_flagged(tmp_path):
    studies = {s.study_id: s for s in pair_images(_index_frame(), tmp_path)}
    study = studies["s2"]

    assert not study.is_complete
    assert ("R", "MLO") not in study.images


def test_findings_are_parsed_per_laterality(tmp_path):
    studies = {s.study_id: s for s in pair_images(_index_frame(), tmp_path)}

    assert studies["s2"].findings["L"] == ["Asymmetry"]
    assert studies["s1"].findings["R"] == ["No Finding"]


def test_uploads_become_a_study_without_metadata(tmp_path):
    assignments = {view: tmp_path / f"{view[0]}{view[1]}.png" for view in VIEWS}
    study = study_from_uploads(assignments)

    assert study.is_complete
    assert study.study_id == "uploaded study"
    assert study.birads == {}


def test_missing_metadata_csv_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data"):
        list_studies(tmp_path)
