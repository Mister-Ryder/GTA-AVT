"""Portable provenance/byte-identity helpers; never modify frozen code/.

Python3.8 compatible. Torch is imported only by functions that need it, so
cache/schema checks and all --help paths work in a NumPy-only environment.
"""
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
from collections import OrderedDict
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CODE = PROJECT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from common import atomic_json, load_json, sha256, source_hashes

SCHEMA = 'GTA-AVT-reference-weights-v1'
DATA_SCHEMA = 'GTA-AVT-canonical-data-v1'
FROZEN_SPLIT_SHA256 = 'df9647da8b331260fb9f66b294eced01180c1eaf812de27b5148b6800a344a3b'
REFERENCE_PYTHON = (3, 8)
REFERENCE_TORCH = (1, 11)
FIXED_INFERENCE = dict(threshold=.5, patch_size=128, overlap=.5, batch_size=2,
                       condition_mode='predicted', postprocessing='none', grid='original CT')
NONIDENTITY_CASE_FIELDS = ('cache_dir', 'image', 'label', 'preprocessing_seconds')
SPATIAL_FIELDS = ('shape_zyx', 'full_shape_zyx', 'crop_start_zyx', 'crop_stop_zyx',
                  'resampled_origin', 'resampled_direction', 'resampled_spacing', 'resampled_size',
                  'original_origin', 'original_direction', 'original_spacing', 'original_size')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    return sha256(Path(path))


def require_new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output directory is not empty; artifacts are immutable: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def frozen_patient_ids():
    choices = [PROJECT / 'configs/bounded_20261022/split_seed20261022.json',
               PROJECT / 'configs/split_seed20261022.json',
               PROJECT / 'results/local_geometry/split_seed20261022.json']
    path = next((p for p in choices if p.exists()), None)
    if path is None or file_hash(path) != FROZEN_SPLIT_SHA256:
        raise ValueError('Exact frozen patient split file is missing or changed')
    split = load_json(path)
    prefix = {'D': 'Dongyang', 'K': 'KiTS', 'R': 'Rider'}
    return {name: [prefix[x[0]] + '_' + x for x in split[name]] for name in ('train', 'val', 'test')}


def canonical_cache_identity(manifest_path, verify_bytes=True):
    """Verify all56 cached and raw pairs, then remove location/timing fields.

    The only excluded case fields are cache_dir, image/label path strings and
    preprocessing_seconds. Their image/label SHA256 identities remain included.
    ALL other manifest and per-case metadata remain identity-bearing.
    """
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    splits = frozen_patient_ids()
    if not manifest.get('complete') or manifest.get('split_sha256') != FROZEN_SPLIT_SHA256 or manifest.get('splits') != splits:
        raise ValueError('Require the complete exact frozen56-case split, including ID order')
    cases = manifest.get('cases', [])
    ids = [r['id'] for r in cases]
    if len(ids) != 56 or len(set(ids)) != 56 or set(ids) != set(sum(splits.values(), [])):
        raise ValueError('Cache must contain every frozen patient exactly once')
    current = source_hashes()
    expected_prepare = {k: current[k] for k in ('prepare.py', 'geometry.py', 'common.py', 'intensity.py')}
    canonical_cases = []
    verified_files = 0
    verified_bytes = 0
    for record in sorted(cases, key=lambda r: r['id']):
        missing = set(SPATIAL_FIELDS) - set(record)
        if missing:
            raise ValueError('Spatial metadata missing for%s:%s' % (record['id'], sorted(missing)))
        training = record['id'] in splits['train']
        if record.get('geometry_training_only') != training or record.get('preprocessing_source_sha256') != expected_prepare:
            raise ValueError('Geometry scope or preprocessing source changed: ' + record['id'])
        expected_names = {'image.npy', 'mask.npy'} | ({'geometry.npz'} if training else set())
        hashes = record.get('cache_file_sha256', {})
        if set(hashes) != expected_names:
            raise ValueError('Unexpected/missing cache byte identities: ' + record['id'])
        paths = [(Path(record['cache_dir']) / name, expected) for name, expected in sorted(hashes.items())]
        paths += [(Path(record[name]), record[name + '_sha256']) for name in ('image', 'label')]
        for path, expected in paths:
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError('Invalid SHA256 identity')
            if verify_bytes:
                if not path.is_file() or file_hash(path) != expected:
                    raise ValueError('Actual cache/source byte mismatch: ' + str(path))
                verified_files += 1
                verified_bytes += path.stat().st_size
        canonical_cases.append({k: v for k, v in record.items() if k not in NONIDENTITY_CASE_FIELDS})
    canonical = {k: v for k, v in manifest.items() if k != 'cases'}
    canonical.update(schema=DATA_SCHEMA, cases=canonical_cases)
    return dict(schema=DATA_SCHEMA, canonical_sha256=canonical_digest(canonical), canonical=canonical,
                excluded_case_fields=list(NONIDENTITY_CASE_FIELDS),
                verification=dict(actual_bytes_verified=bool(verify_bytes), file_count=verified_files,
                                  bytes=verified_bytes, manifest_sha256=file_hash(manifest_path)))


def runtime_environment():
    import importlib.metadata as metadata
    import torch
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-image', 'SimpleITK'):
        packages[name] = metadata.version(name)
    cuda = torch.cuda.is_available()
    return dict(python=sys.version, python_version=list(sys.version_info[:3]), executable=sys.executable,
                hostname=platform.node(),
                platform=platform.platform(), packages=packages, torch_cuda=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(), cuda_available=cuda,
                devices=[dict(index=i, name=torch.cuda.get_device_name(i),
                              capability=list(torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())] if cuda else [],
                backend_settings=dict(matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                    cudnn_tf32=bool(torch.backends.cudnn.allow_tf32), cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                    cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                    deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))


def verify_training_environment_receipt(path, current):
    receipt = load_json(path)
    for name, version in current['packages'].items():
        if receipt.get('packages', {}).get(name) != version:
            raise ValueError('Exporter is not running in the recorded training software environment: ' + name)
    if receipt.get('python') != current['python']:
        raise ValueError('Training Python receipt differs from exporter Python')
    return dict(sha256=file_hash(path), value=receipt,
                basis='Provided training-runtime receipt; software versions verified at export. Backend flags are not retrospectively inferred.')


def require_reference_runtime():
    import torch
    version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if sys.version_info[:2] != REFERENCE_PYTHON or version != REFERENCE_TORCH:
        raise RuntimeError('Reference inference requires Python3.8 + torch1.11 exactly; no runtime fallback')


def configure_inference_runtime():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')


def selected_execution_device(device):
    import torch
    device = torch.device(device)
    if device.type == 'cpu':
        return dict(type='cpu', index=None, name=platform.processor() or 'CPU')
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Unsupported/unavailable selected execution device')
    index = torch.cuda.current_device() if device.index is None else device.index
    return dict(type='cuda', index=index, name=torch.cuda.get_device_name(index),
                capability=list(torch.cuda.get_device_capability(index)),
                total_memory_bytes=torch.cuda.get_device_properties(index).total_memory)


def array_descriptor(array):
    array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise ValueError('Object arrays are forbidden')
    return dict(dtype=array.dtype.str, shape=list(array.shape),
                sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def tensor_descriptor(tensor):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError('Only ordinary dense tensor state_dict entries are supported')
    value = tensor.detach().cpu().contiguous()
    if sys.byteorder != 'little':
        raise ValueError('Reference tensor byte contract requires little-endian machines')
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.int16).numpy().tobytes(order='C')
    else:
        raw = value.numpy().tobytes(order='C')
    return dict(dtype=str(value.dtype), shape=list(value.shape), numel=value.numel(),
                contiguous_cpu_bytes_sha256=hashlib.sha256(raw).hexdigest(), byte_order='little')


def tensor_inventory(state):
    import torch
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Expected a nonempty tensor-only state_dict, with no nested model/optimizer/RNG objects')
    return {k: tensor_descriptor(v) for k, v in sorted(state.items())}


def load_tensor_bundle(bundle):
    import torch
    bundle = Path(bundle)
    provenance = load_json(bundle / 'provenance.json')
    if provenance.get('schema') != SCHEMA:
        raise ValueError('Unsupported reference-weight schema')
    weights = bundle / provenance['weights_file']
    if weights.parent != bundle or file_hash(weights) != provenance['weights_sha256']:
        raise ValueError('Exported weights file SHA256 mismatch')
    if source_hashes() != provenance['original_source_sha256']:
        raise ValueError('Current frozen code differs from training/export source')
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = True
    state = torch.load(str(weights), **kwargs)
    if tensor_inventory(state) != provenance['tensor_inventory']:
        raise ValueError('Tensor dtype/shape/bytes differ from exported provenance')
    return provenance, state


def load_strict_model(provenance, state, device):
    import torch
    from model import GTAAVT
    recipe = provenance['recipe']
    model = GTAAVT(recipe['method'], recipe['width']).to(device)
    model.load_state_dict(state, strict=True)
    if tensor_inventory(model.state_dict()) != provenance['tensor_inventory']:
        raise ValueError('Strict model load changed tensor dtype/shape/bytes')
    model.eval()
    return model


def assert_same_data(provenance, identity):
    if identity['canonical_sha256'] != provenance['canonical_data_identity']['canonical_sha256']:
        raise ValueError('Reference cache/source data identity differs from training cache')
    if identity['canonical'] != provenance['canonical_data_identity']['canonical']:
        raise ValueError('Canonical data payload differs despite recorded hash')


def evaluation_cases(provenance, split, allow_formal_test, intervention='none'):
    frozen = provenance['frozen_patient_ids']
    if frozen != frozen_patient_ids():
        raise ValueError('Exported frozen patient IDs differ from current exact split')
    if split == 'test':
        if intervention != 'none':
            raise ValueError('Runtime mechanism interventions are validation-only')
        if not allow_formal_test or provenance['stage'] != 'formal' or not provenance['final_test_eligible']:
            raise ValueError('Test requires completed eligible formal training AND --allow-formal-test; pilots are never eligible')
        if provenance['original_run_record']['validation_ids'] != frozen['val']:
            raise ValueError('Final-test model was not selected using all frozen validation cases')
        return frozen['test']
    if split != 'val' or allow_formal_test:
        raise ValueError('--allow-formal-test is only valid together with --split test')
    # A pilot keeps its predeclared validation subset; this does not expand it.
    selected = provenance['original_run_record']['validation_ids']
    if not selected or not set(selected).issubset(frozen['val']) or len(selected) != len(set(selected)):
        raise ValueError('Invalid original validation selection')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    env = sub.add_parser('capture-environment')
    env.add_argument('--output', required=True)
    identity = sub.add_parser('cache-identity')
    identity.add_argument('--manifest', required=True)
    identity.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise ValueError('Refusing to replace an existing receipt')
    if args.command == 'capture-environment':
        atomic_json(path, runtime_environment())
    else:
        atomic_json(path, canonical_cache_identity(args.manifest, verify_bytes=True))


if __name__ == '__main__':
    main()
