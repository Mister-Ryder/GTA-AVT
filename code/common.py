"""Small, explicit reproducibility and native-grid IO helpers."""
import hashlib
import json
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.tmp')
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    part.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    root = Path(__file__).parent
    selected = [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                and (p.suffix == '.py' or ('historical' in p.relative_to(root).parts
                                           and p.suffix.lower() in ('.json', '.txt', '.md')))]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(selected)}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_trusted_checkpoint(path, map_location):
    # torch1.11 lacks weights_only; torch>=2.6 defaults True and rejects RNG state.
    # This function is only for checkpoints generated in these owned run folders.
    import inspect
    import torch
    kwargs = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(str(path), **kwargs)
