"""Score every downloaded VinDr-Mammo sample study and print a ranked table.

This is the command-line counterpart to the dashboard: it answers the plain
question "run AsymMirai over the public mammograms" without a browser.
Research use only; the scores are uncalibrated on this data.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from asymview.data import list_studies
from asymview.model import load_model, preprocess

FIELDS = ["study_id", "asymmetry", "score", "birads_L", "birads_R", "density", "findings"]


# Turn one scored study into the row we record for it.
def _row(study, result):
    return {
        "study_id": study.study_id,
        "asymmetry": round(result.asymmetry, 3),
        "score": round(result.probability, 4),
        "birads_L": study.birads.get("L", ""),
        "birads_R": study.birads.get("R", ""),
        "density": study.density.get("L", ""),
        "findings": "; ".join(
            sorted(set(study.findings.get("L", []) + study.findings.get("R", [])))
        ),
    }


# Score every study, appending each result so a long run can be interrupted.
def score_all(out_path, mirror_right=False):
    # Inference is the whole cost here, so let torch use every core.
    torch.set_num_threads(os.cpu_count() or 4)

    model = load_model()
    studies = list_studies()
    rows = []

    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        for index, study in enumerate(studies, start=1):
            views = {key: preprocess(path) for key, path in study.images.items()}
            result = model.score(views, mirror_right=mirror_right)
            row = _row(study, result)
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(
                f"  [{index}/{len(studies)}] {study.study_id[:12]}  asym={result.asymmetry:7.2f}",
                flush=True,
            )

    return sorted(rows, key=lambda row: row["asymmetry"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-right", action="store_true", help="mirror the right breast")
    parser.add_argument("--out", default="study_scores.csv", help="where to write the table")
    args = parser.parse_args()

    rows = score_all(args.out, mirror_right=args.mirror_right)

    print(f"\nwrote {args.out} ({len(rows)} studies)\n")
    print(f"{'study':14} {'asym':>8} {'score':>7}  findings")
    for row in rows[:5] + rows[-5:]:
        print(
            f"{row['study_id'][:12]:14} {row['asymmetry']:8.2f} "
            f"{row['score']:7.4f}  {row['findings'][:40]}"
        )
    print("\nResearch use only. VinDr has no cancer follow-up; scores are uncalibrated.")


if __name__ == "__main__":
    main()
