"""Download fixed encoder files and verify SHA-256 (standard library only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(entry, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == entry['bytes'] and checksum(target) == entry['sha256']:
            print('Verified existing file:', target)
            return
        raise ValueError(f'Existing file does not match; move it aside before retrying: {target}')
    fd, name = tempfile.mkstemp(prefix=target.name + '.', suffix='.part', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as f, urllib.request.urlopen(entry['url'], timeout=120) as r:
            while True:
                chunk = r.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if Path(name).stat().st_size != entry['bytes'] or checksum(name) != entry['sha256']:
            raise ValueError('Downloaded file failed size/SHA-256 verification: ' + entry['url'])
        Path(name).replace(target)
        print('Downloaded and verified:', target)
    finally:
        Path(name).unlink(missing_ok=True)


def main():
    manifest = json.loads((ROOT / 'environments/encoder-files.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=manifest)
    parser.add_argument('--out-dir', default='pretrained')
    args = parser.parse_args()
    spec = manifest[args.model]
    root = Path(args.out_dir)
    for entry in spec['files']:
        download(entry, root / entry['path'])
    if 'extract' in spec:
        item = spec['extract']
        with zipfile.ZipFile(root / item['archive']) as z:
            matches = [n for n in z.namelist() if n == item['member']]
            if len(matches) != 1:
                raise ValueError('Expected archive member missing')
            # Extract only this known file; never trust archive paths.
            target = root / item['path']
            data = z.read(matches[0])
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Extracted weight checksum mismatch')
            if target.exists() and checksum(target) != item['sha256']:
                raise ValueError('Existing extracted weight differs: ' + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
            print('Verified extracted weight:', target)


if __name__ == '__main__':
    main()
