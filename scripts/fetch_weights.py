"""Download the published AsymMirai checkpoint from its public Box link."""

import sys
import urllib.request
from pathlib import Path

# Public Box share for trained_asymmirai.pt (44.8 MB), from the AsymMirai README.
SHARED_NAME = "9uu9sarz6zizjkqj41iavgxz6zxwxz7c"
FILE_ID = "f_1931188393685"
URL = (
    "https://duke.app.box.com/index.php?rm=box_download_shared_file"
    f"&shared_name={SHARED_NAME}&file_id={FILE_ID}"
)
# Box serves the file only to a browser-like client.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

DEST = Path(__file__).resolve().parent.parent / "weights" / "trained_asymmirai.pt"


# Stream the checkpoint to disk, skipping the download if it is already there.
def fetch(dest: Path = DEST) -> Path:
    if dest.exists() and dest.stat().st_size > 40_000_000:
        print(f"already present: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    print(f"downloading {dest.name} ...")
    with urllib.request.urlopen(request) as response, open(dest, "wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)

    size = dest.stat().st_size
    if size < 40_000_000:
        dest.unlink()
        sys.exit(f"download looks truncated ({size} bytes); check the Box link")

    print(f"saved {dest} ({size / 1e6:.1f} MB)")
    return dest


if __name__ == "__main__":
    fetch()
