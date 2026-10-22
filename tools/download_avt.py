"""Download original AVT archives from public Figshare metadata and verify MD5.

No credentials, private mirror, alternate host, or manual file URL is accepted.
Archive extraction is a separate explicit step.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ARTICLE_API = 'https://api.figshare.com/v2/articles/14806362'
EXPECTED_ARCHIVES = {'Dongyang.zip', 'KiTS.zip', 'Rider.zip'}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_md5(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(entry, destination):
    name = entry['name']
    if name not in EXPECTED_ARCHIVES or Path(name).name != name:
        raise ValueError('Unexpected official archive filename: ' + str(name))
    size = int(entry['size'])
    expected_md5 = entry['computed_md5'].lower()
    if len(expected_md5) != 32 or any(c not in '0123456789abcdef' for c in expected_md5):
        raise ValueError('Invalid checksum in official metadata: ' + name)
    target = destination / name
    partial = destination / (name + '.part')
    if target.exists():
        if target.stat().st_size != size or file_md5(target) != expected_md5:
            raise RuntimeError('Existing archive failed size/MD5 validation: ' + name)
        print(json.dumps({'archive': name, 'state': 'verified_existing'}), flush=True)
        return
    if partial.exists() and partial.stat().st_size > size:
        raise RuntimeError('Partial archive exceeds official size; inspect it: ' + name)
    began = time.monotonic()
    initial = partial.stat().st_size if partial.exists() else 0
    last_print = 0.
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        headers = {'Range': 'bytes=%d-' % offset} if offset else {}
        request = urllib.request.Request(entry['download_url'], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith('bytes %d-' % offset)):
                    raise RuntimeError('Official endpoint did not honor byte-range resume; partial retained')
                with partial.open('ab' if offset else 'wb') as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(chunk)
                        offset += len(chunk)
                        if offset > size:
                            raise RuntimeError('Download exceeded official size: ' + name)
                        now = time.monotonic()
                        if now - last_print >= 10:
                            rate = (offset - initial) / max(now - began, .001)
                            print(json.dumps({'archive': name, 'bytes': offset, 'total_bytes': size,
                                              'MiB_per_second': round(rate / 2 ** 20, 2),
                                              'remaining_minutes': round((size - offset) / max(rate, 1.) / 60., 2)}), flush=True)
                            last_print = now
            if offset != size:
                raise RuntimeError('Incomplete official response: ' + name)
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 3 * attempt))
    if not partial.exists() or partial.stat().st_size != size or file_md5(partial) != expected_md5:
        raise RuntimeError('Downloaded archive failed size/MD5 validation: ' + name)
    partial.replace(target)
    print(json.dumps({'archive': name, 'state': 'verified', 'bytes': size, 'md5': expected_md5}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2, 3))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE_API, timeout=60) as response:
        metadata = json.load(response)
    entries = [entry for entry in metadata['files'] if entry['name'] in EXPECTED_ARCHIVES]
    if {entry['name'] for entry in entries} != EXPECTED_ARCHIVES or len(entries) != 3:
        raise RuntimeError('Official article archive inventory changed; inspect its metadata')
    public_record = {'article_api': ARTICLE_API, 'doi': metadata.get('doi'),
                     'title': metadata.get('title'), 'license': metadata.get('license'),
                     'checked_utc': datetime.now(timezone.utc).isoformat(),
                     'files': [{key: entry[key] for key in ('name', 'size', 'computed_md5', 'download_url')} for entry in entries]}
    write_json(args.destination / 'official_figshare_metadata.json', public_record)
    print(json.dumps(public_record, indent=2), flush=True)
    if args.metadata_only:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda entry: download_archive(entry, args.destination), entries))
    write_json(args.destination / 'DOWNLOAD_COMPLETE.json', {'state': 'all_archives_verified',
               'article_api': ARTICLE_API, 'files': public_record['files']})


if __name__ == '__main__':
    main()
