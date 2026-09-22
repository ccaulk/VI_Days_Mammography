"""Fetch pinned upstream code and the verified official AsymMirai checkpoint."""
import hashlib
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'AsymMirai'
COMMIT = '88ac34a0cf5a8a8e2d301a4d05da625059633004'
MODEL_URL = 'https://duke.box.com/shared/static/9uu9sarz6zizjkqj41iavgxz6zxwxz7c.pt'
MODEL_SHA1 = '76756e79a8f560b23ecf6586a3a5b53e8e53c10c'


# Run Git without shell interpolation and fail on unexpected errors.
def git(*args):
    return subprocess.check_output(['git', '-C', str(UPSTREAM), *args], text=True).strip()


# Pin upstream and apply the device portability patch once.
def prepare_code():
    if not UPSTREAM.exists():
        subprocess.run(['git', 'clone', 'https://github.com/jdonnelly36/AsymMirai.git',
                        str(UPSTREAM)], check=True)
        git('checkout', COMMIT)
    if git('rev-parse', 'HEAD') != COMMIT:
        raise RuntimeError(f'AsymMirai must be at {COMMIT}; existing checkout left unchanged.')
    patch = ROOT / 'AsymView/upstream-device.patch'
    applied = subprocess.run(['git', '-C', str(UPSTREAM), 'apply', '--unidiff-zero', '--reverse', '--check', str(patch)],
                             capture_output=True)
    if applied.returncode != 0:
        git('apply', '--unidiff-zero', '--check', str(patch))
        git('apply', '--unidiff-zero', str(patch))
    print('Pinned upstream code and device patch ready.', flush=True)


# Install only the official checkpoint after checking its published digest.
def prepare_checkpoint():
    target = ROOT / 'AsymView/models/trained_asymmirai.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha1(target.read_bytes()).hexdigest() != MODEL_SHA1:
            raise RuntimeError('Existing checkpoint has an unexpected digest; file left unchanged.')
        print('Verified existing official checkpoint.', flush=True)
        return
    partial = target.with_suffix('.pt.part')
    print('Downloading official trained checkpoint (about 45 MB)…', flush=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response, partial.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if hashlib.sha1(partial.read_bytes()).hexdigest() != MODEL_SHA1:
            raise RuntimeError('Download checksum mismatch; model was not installed.')
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    print('Official trained checkpoint verified.', flush=True)


# Prepare runtime assets independently of the Python environment.
def main():
    prepare_code()
    prepare_checkpoint()


if __name__ == '__main__':
    main()
