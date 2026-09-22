"""Download the de-identified VinDr-Mammo sample studies from public Google Drive."""

from pathlib import Path

import gdown

# Public Drive folder: 36 studies of 4 PNGs each, plus the VinDr metadata CSV.
FOLDER_URL = "https://drive.google.com/drive/folders/1EnTqFhuDVcpSuCajsp-RqvfkQ5wHu58Z"

DEST = Path(__file__).resolve().parent.parent / "data"


# Mirror the Drive folder into data/, skipping files already downloaded.
def fetch(dest: Path = DEST) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(FOLDER_URL, output=str(dest), quiet=False, use_cookies=False)
    studies = [p for p in dest.iterdir() if p.is_dir() and p.name != "Metadata"]
    print(f"{len(studies)} studies in {dest}")
    return dest


if __name__ == "__main__":
    fetch()
