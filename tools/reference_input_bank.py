"""TRAIN-only common input arrays and FP32 cross-runtime logits diagnostics.

Build once, copy the exact bank to both runtimes, generate logits in each, then
compare. No thresholds/tolerances are tuned, no val/test inputs are permitted.
"""
import argparse
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from reference_weights_support import (
    atomic_json, load_json, file_hash, canonical_digest, source_hashes, canonical_cache_identity,
    frozen_patient_ids, array_descriptor, require_new_directory, load_tensor_bundle, load_strict_model,
    runtime_environment, require_reference_runtime, configure_inference_runtime, utc)
from reference_weights_support import selected_execution_device

BANK_SCHEMA = 'GTA-AVT-train-input-bank-v1'
LOGITS_SCHEMA = 'GTA-AVT-fp32-bank-logits-v1'
CASE_IDS = ['Dongyang_D1', 'KiTS_K2', 'Rider_R3']
PATCH = 96
SEED = 20261022


def build_bank(args):
    from data import normalize
    identity = canonical_cache_identity(args.manifest, verify_bytes=True)
    manifest = load_json(args.manifest)
    cases = {r['id']: r for r in manifest['cases']}
    frozen = frozen_patient_ids()
    if not set(CASE_IDS).issubset(frozen['train']):
        raise ValueError('Predeclared bank cases are not all frozen training patients')
    inputs, coordinates, samples = [], [], []
    for index, case_id in enumerate(CASE_IDS):
        record = cases[case_id]
        image = np.load(Path(record['cache_dir']) / 'image.npy', mmap_mode='r', allow_pickle=False)
        shape = np.asarray(image.shape)
        high = np.maximum(shape - PATCH, 0)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
        starts = [('centre', high // 2), ('seeded_uniform', np.asarray([rng.integers(int(n) + 1) for n in high]))]
        for policy, start in starts:
            slices = tuple(slice(int(s), min(int(s) + PATCH, int(n))) for s, n in zip(start, shape))
            raw = np.asarray(image[slices], np.float32)
            raw = np.pad(raw, [(0, PATCH - n) for n in raw.shape], constant_values=-1024)
            sample = np.ascontiguousarray(normalize(raw)[None], dtype=np.float32)
            centre = np.asarray(record['crop_start_zyx']) + start + PATCH / 2.
            coordinate = np.asarray(2. * centre / np.maximum(np.asarray(record['full_shape_zyx']) - 1., 1.) - 1., np.float32)
            inputs.append(sample)
            coordinates.append(coordinate)
            samples.append(dict(index=len(samples), case_id=case_id, split='train', start_policy=policy,
                start_zyx=start.astype(int).tolist(), cache_image_sha256=record['cache_file_sha256']['image.npy'],
                original_image_sha256=record['image_sha256'], input=array_descriptor(sample), coordinate=array_descriptor(coordinate)))
    image_array = np.ascontiguousarray(np.stack(inputs), dtype=np.float32)
    coordinate_array = np.ascontiguousarray(np.stack(coordinates), dtype=np.float32)
    output = require_new_directory(args.output)
    bank_file = output / 'inputs.npz'
    np.savez_compressed(bank_file, image=image_array, coordinate=coordinate_array)
    metadata = dict(schema=BANK_SCHEMA, created_utc=utc(), train_only=True, seed=SEED, patch_size=PATCH,
        case_ids=CASE_IDS, samples=samples, frozen_patient_ids=frozen, frozen_source_sha256=source_hashes(),
        canonical_data_sha256=identity['canonical_sha256'], manifest_sha256=file_hash(args.manifest),
        arrays_file=bank_file.name, arrays_file_sha256=file_hash(bank_file),
        arrays=dict(image=array_descriptor(image_array), coordinate=array_descriptor(coordinate_array)),
        selection='Two image-only starts per fixed training case: cache-volume centre and seeded uniform; no GT geometry/mask used',
        normalization='Imported frozen data.normalize; identical crop-coordinate expression and padding to frozen inference.py',
        builder_tool_sha256={Path(__file__).name: file_hash(__file__),
                             'reference_weights_support.py': file_hash(Path(__file__).with_name('reference_weights_support.py'))})
    atomic_json(output / 'bank.json', metadata)
    load_bank(output)
    print({'built': str(output), 'train_cases': CASE_IDS, 'samples': len(samples),
           'heldout_model_inputs_or_predictions_used': False, 'all56_file_bytes_verified_for_identity_only': True}, flush=True)


def load_bank(path):
    path = Path(path)
    metadata = load_json(path / 'bank.json')
    if metadata.get('schema') != BANK_SCHEMA or not metadata.get('train_only') or metadata.get('patch_size') != PATCH:
        raise ValueError('Unsupported or non-training input bank')
    if metadata.get('frozen_patient_ids') != frozen_patient_ids() or metadata.get('frozen_source_sha256') != source_hashes():
        raise ValueError('Bank frozen patient/source identity changed')
    if metadata.get('case_ids') != CASE_IDS:
        raise ValueError('Bank is outside predeclared training-case allowlist')
    if any(s['case_id'] not in frozen_patient_ids()['train'] or s.get('split') != 'train' for s in metadata['samples']):
        raise ValueError('Held-out input detected in bank')
    file = path / metadata['arrays_file']
    if file.parent != path or file_hash(file) != metadata['arrays_file_sha256']:
        raise ValueError('Saved input-bank byte identity changed')
    with np.load(file, allow_pickle=False) as arrays:
        if set(arrays.files) != {'image', 'coordinate'}:
            raise ValueError('Unexpected bank arrays')
        images, coordinates = arrays['image'], arrays['coordinate']
    if images.dtype != np.float32 or images.shape != (6, 1, PATCH, PATCH, PATCH) or coordinates.dtype != np.float32 or coordinates.shape != (6, 3):
        raise ValueError('Input bank shape/dtype differs from fixed protocol')
    for name, value in [('image', images), ('coordinate', coordinates)]:
        if array_descriptor(value) != metadata['arrays'][name] or not np.isfinite(value).all():
            raise ValueError('Saved bank array dtype/shape/hash/finite check failed: ' + name)
    if len(metadata['samples']) != 6:
        raise ValueError('Expected exactly six bank samples')
    for index, sample in enumerate(metadata['samples']):
        if sample['index'] != index or array_descriptor(images[index]) != sample['input'] or array_descriptor(coordinates[index]) != sample['coordinate']:
            raise ValueError('Per-sample saved input hash differs')
    return metadata, images, coordinates


def generate_logits(args):
    import torch
    provenance, state = load_tensor_bundle(args.bundle)
    bank, images, coordinates = load_bank(args.bank)
    if bank['canonical_data_sha256'] != provenance['canonical_data_identity']['canonical_sha256']:
        raise ValueError('Bank and exported model were not based on the same verified common cache/source bytes')
    if args.role == 'reference':
        require_reference_runtime()
    current = runtime_environment()
    if args.role == 'training':
        receipt = provenance['training_environment_receipt']['value']
        if receipt.get('python') != current['python'] or receipt.get('packages') != current['packages']:
            raise ValueError('Original logits must run in the recorded training software environment')
    configure_inference_runtime()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    torch.set_num_threads(provenance['recipe'].get('cpu_threads', 4))
    model = load_strict_model(provenance, state, device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), 2):
            image = torch.from_numpy(images[start:start + 2]).to(device)
            coordinate = torch.from_numpy(coordinates[start:start + 2]).to(device)
            # Explicitly FP32, even on the training runtime that used AMP.
            with torch.cuda.amp.autocast(enabled=False):
                value = model(image, coordinate, condition_mode='predicted')['logits']
            if value.dtype != torch.float32:
                raise ValueError('FP32 logit protocol returned a different tensor dtype')
            array = value.detach().cpu().contiguous().numpy()
            if array.shape != (len(image), 1, PATCH, PATCH, PATCH) or not np.isfinite(array).all():
                raise ValueError('Invalid logit shape or nonfinite value')
            outputs.append(array)
    logits = np.ascontiguousarray(np.concatenate(outputs), dtype=np.float32)
    output = require_new_directory(args.output)
    file = output / 'logits.npz'
    np.savez_compressed(file, logits=logits)
    record = dict(schema=LOGITS_SCHEMA, created_utc=utc(), role=args.role, precision='FP32; autocast=False; both TF32 backends disabled',
        batch_size=2, condition_mode='predicted', runtime=runtime_environment(),
        selected_execution_device=selected_execution_device(device), autocast_enabled=False,
        method=provenance['method'], stage=provenance['stage'], selected_step=provenance['selected_step'],
        weights_sha256=provenance['weights_sha256'], original_checkpoint_sha256=provenance['original_checkpoint_sha256'],
        tensor_inventory_sha256=canonical_digest(provenance['tensor_inventory']),
        weight_provenance_sha256=file_hash(Path(args.bundle) / 'provenance.json'),
        bank_json_sha256=file_hash(Path(args.bank) / 'bank.json'), bank_arrays_sha256=bank['arrays_file_sha256'],
        input_arrays=bank['arrays'], samples=bank['samples'], train_only=True,
        frozen_source_sha256=source_hashes(), canonical_data_sha256=bank['canonical_data_sha256'],
        logits_file=file.name, logits_file_sha256=file_hash(file), logits=array_descriptor(logits),
        tool_sha256={Path(__file__).name: file_hash(__file__),
                     'reference_weights_support.py': file_hash(Path(__file__).with_name('reference_weights_support.py'))})
    atomic_json(output / 'logits_provenance.json', record)
    print({'generated_fp32_logits': str(output), 'role': args.role, 'samples': len(images), 'heldout_used': False}, flush=True)


def load_logits(directory):
    directory = Path(directory)
    record = load_json(directory / 'logits_provenance.json')
    if record.get('schema') != LOGITS_SCHEMA or not record.get('train_only') or record.get('precision') != 'FP32; autocast=False; both TF32 backends disabled':
        raise ValueError('Invalid FP32 diagnostic logit receipt')
    file = directory / record['logits_file']
    if file.parent != directory or file_hash(file) != record['logits_file_sha256']:
        raise ValueError('Logit artifact file hash mismatch')
    with np.load(file, allow_pickle=False) as stream:
        if stream.files != ['logits']:
            raise ValueError('Unexpected logit arrays')
        logits = stream['logits']
    if logits.dtype != np.float32 or logits.shape != (6, 1, PATCH, PATCH, PATCH) or not np.isfinite(logits).all() or array_descriptor(logits) != record['logits']:
        raise ValueError('Logit dtype/shape/hash/finite check failed')
    backend = record['runtime']['backend_settings']
    if backend['matmul_tf32'] or backend['cudnn_tf32'] or record.get('autocast_enabled') is not False:
        raise ValueError('TF32 was enabled in a claimed FP32 diagnostic')
    return record, logits


def compare_logits(args):
    bank, _, _ = load_bank(args.bank)
    original, a = load_logits(args.training_logits)
    reference, b = load_logits(args.reference_logits)
    if original['role'] != 'training' or reference['role'] != 'reference':
        raise ValueError('Expected explicitly labelled training and reference runtime outputs')
    if reference['runtime']['python_version'][:2] != [3, 8] or reference['runtime']['packages']['torch'].split('+')[0].split('.')[:2] != ['1', '11']:
        raise ValueError('Comparison reference is not Python3.8/torch1.11')
    fields = ('method', 'stage', 'selected_step', 'weights_sha256', 'original_checkpoint_sha256',
              'tensor_inventory_sha256', 'weight_provenance_sha256', 'bank_json_sha256', 'bank_arrays_sha256',
              'input_arrays', 'samples', 'frozen_source_sha256', 'canonical_data_sha256', 'batch_size', 'condition_mode')
    for field in fields:
        if original[field] != reference[field]:
            raise ValueError('Cross-runtime identity mismatch: ' + field)
    if original['bank_json_sha256'] != file_hash(Path(args.bank) / 'bank.json') or original['bank_arrays_sha256'] != bank['arrays_file_sha256']:
        raise ValueError('Comparison did not use these exact saved input arrays')
    rows = []
    total = 0.
    count = 0
    maximum = 0.
    for index, sample in enumerate(bank['samples']):
        delta = np.abs(a[index].astype(np.float64) - b[index].astype(np.float64))
        local_max = float(delta.max())
        total += float(delta.sum())
        count += delta.size
        maximum = max(maximum, local_max)
        rows.append(dict(index=index, case_id=sample['case_id'], split='train', start_policy=sample['start_policy'],
                         max_absolute_logit_difference=local_max, mean_absolute_logit_difference=float(delta.mean()),
                         values=int(delta.size)))
    output = Path(args.output)
    if output.exists():
        raise ValueError('Comparison receipt is immutable')
    atomic_json(output, dict(schema='GTA-AVT-cross-runtime-logit-comparison-v1', created_utc=utc(),
        diagnostic_only=True, tuning_performed=False, tolerance_selected=False, validation_or_test_inputs=False,
        method=original['method'], stage=original['stage'], selected_step=original['selected_step'],
        weights_sha256=original['weights_sha256'], original_checkpoint_sha256=original['original_checkpoint_sha256'],
        bank_arrays_sha256=bank['arrays_file_sha256'], bank_json_sha256=original['bank_json_sha256'],
        training_runtime=original['runtime'], reference_runtime=reference['runtime'],
        training_execution_device=original['selected_execution_device'], reference_execution_device=reference['selected_execution_device'],
        training_logits_receipt_sha256=file_hash(Path(args.training_logits) / 'logits_provenance.json'),
        reference_logits_receipt_sha256=file_hash(Path(args.reference_logits) / 'logits_provenance.json'),
        max_absolute_logit_difference=maximum, mean_absolute_logit_difference=total / count,
        values=count, samples=rows,
        interpretation='Descriptive FP32 numerical bridge only; differences are reported without tuning. This is not segmentation performance or a trained-method comparison.'))
    print({'max_absolute_logit_difference': maximum, 'mean_absolute_logit_difference': total / count,
           'heldout_inputs': False, 'tuning': False}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--manifest', required=True)
    build.add_argument('--output', required=True)
    logits = sub.add_parser('logits')
    logits.add_argument('--bundle', required=True)
    logits.add_argument('--bank', required=True)
    logits.add_argument('--output', required=True)
    logits.add_argument('--role', choices=('training', 'reference'), required=True)
    logits.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    compare = sub.add_parser('compare')
    compare.add_argument('--bank', required=True)
    compare.add_argument('--training-logits', required=True)
    compare.add_argument('--reference-logits', required=True)
    compare.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build_bank(args)
    elif args.command == 'logits':
        generate_logits(args)
    else:
        compare_logits(args)


if __name__ == '__main__':
    main()
