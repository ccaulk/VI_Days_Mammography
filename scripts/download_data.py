"""Download everything needed to reproduce the AsymMirai risk predictions.

This fetches:
  1. The public VinDr mammogram subset (36 exams, 4 PNG views each) and its
     metadata CSV, shared via Google Drive.
  2. The pre-trained AsymMirai weights, shared via Duke Box.

Run from the repository root:
    python scripts/download_data.py
"""
import os
import re
import time
import urllib.request

# Public share links supplied with the course task.
GDRIVE_FOLDER = "1EnTqFhuDVcpSuCajsp-RqvfkQ5wHu58Z"
GDRIVE_METADATA = "121ol8SvdX6lBNJNNzXPM5XrKhkvs_03m"
BOX_BACKBONE = ("g21ak9kfneudotokp9nmlp07r1bsdki7", "f_1493874427149",
                "mgh_mammo_MIRAI_Base_May20_2019.p")
BOX_ASYM = ("9uu9sarz6zizjkqj41iavgxz6zxwxz7c", "f_1931188393685",
            "trained_asymmirai.pt")

DATA_DIR = os.path.join("data", "vindr")
SNAP_DIR = "snapshots"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def fetch(url):
    """Return the raw bytes of a URL using a browser user-agent."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def list_folder(folder_id):
    """Return [(entry_id, name)] for a public Google Drive folder."""
    html = fetch(
        f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    ).decode("utf-8", "ignore")
    return re.findall(
        r'id="entry-([^"]+)".*?flip-entry-title">([^<]*)</div>', html, re.S
    )


def download_gdrive(file_id, dest):
    """Download one Google Drive file if it is not already present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    data = fetch(f"https://drive.google.com/uc?export=download&id={file_id}")
    if data[:8] == b"<!DOCTYPE" or data[:5] == b"<html":
        raise RuntimeError(f"Got an HTML page instead of file {file_id}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as handle:
        handle.write(data)


def download_box(shared_name, file_id, dest):
    """Download one Duke Box shared file if it is not already present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    url = ("https://duke.app.box.com/index.php?rm=box_download_shared_file"
           f"&shared_name={shared_name}&file_id={file_id}")
    data = fetch(url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as handle:
        handle.write(data)


def download_dataset():
    """Download the metadata CSV and every exam image in the shared folder."""
    metadata_path = os.path.join(DATA_DIR, "vindr_detection_v1_folds.csv")
    print("Downloading metadata CSV...")
    download_gdrive(GDRIVE_METADATA, metadata_path)

    entries = [(i, n) for i, n in list_folder(GDRIVE_FOLDER) if n != "Metadata"]
    print(f"Downloading {len(entries)} exam folders...")
    for idx, (folder_id, name) in enumerate(entries, 1):
        for file_id, fname in list_folder(folder_id):
            dest = os.path.join(DATA_DIR, folder_id, fname)
            download_gdrive(file_id, dest)
        print(f"  [{idx}/{len(entries)}] {name}")
        time.sleep(0.2)


def download_weights():
    """Download the pre-trained AsymMirai weights."""
    print("Downloading AsymMirai weights...")
    download_box(*BOX_BACKBONE, os.path.join(SNAP_DIR, BOX_BACKBONE[2]))
    download_box(*BOX_ASYM, os.path.join(SNAP_DIR, BOX_ASYM[2]))


if __name__ == "__main__":
    download_dataset()
    download_weights()
    print("\nAll downloads complete.")
