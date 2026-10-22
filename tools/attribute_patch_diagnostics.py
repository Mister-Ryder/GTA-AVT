"""Bounded validation-only attribute faithfulness after genuine formal exports.

Nothing executes without --execute-frozen-validation. There is no test option,
training, tuning, cache mutation, model selection or automatic invocation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from reference_weights_support import (
    atomic_json, load_json, file_hash, canonical_digest, source_hashes,
    canonical_cache_identity, assert_same_data, require_reference_runtime,
    configure_inference_runtime, runtime_environment, selected_execution_device,
    load_tensor_bundle, load_strict_model, require_new_directory, frozen_patient_ids,
    array_descriptor, utc)

SEED = 20261022
PATCH = 128
BATCH = 2
METHODS = {'C': 'faithful_concat', 'A': 'faithful_soft', 'constant': 'faithful_soft_constant'}
COUNTS = {'CT_uniform': 8, 'GT_fine_centred': 4, 'GT_junction_centred': 4, 'GT_large_nonjunction_centred': 4}


def validate_formal_bundle(path, expected_method, current_runtime):
    provenance, state = load_tensor_bundle(path)
    receipt = load_json(Path(path) / 'export_verified.json')
    if receipt.get('state') != 'verified' or receipt.get('weights_sha256') != provenance['weights_sha256']:
        raise ValueError('Export lacks a genuine verified tensor receipt')
    if receipt.get('provenance_sha256') != file_hash(Path(path) / 'provenance.json'):
        raise ValueError('Export provenance changed after genuine checkpoint verification')
    record = provenance['original_run_record']
    complete = provenance['original_complete_record']
    recipe = provenance['recipe']
    fixed = frozen_patient_ids()
    if provenance['method'] != expected_method or recipe['method'] != expected_method or provenance['stage'] != 'formal' or recipe['stage'] != 'formal':
        raise ValueError('Require the named completed formal C/A/constant bundle, never a pilot/bridge substitute')
    if recipe['seed'] != SEED or not provenance['final_test_eligible'] or not record['selection_valid']:
        raise ValueError('Bundle is outside the single-seed eligible formal comparison')
    if complete.get('state') != 'complete' or complete.get('steps') != recipe['max_steps'] or complete.get('stage') != 'formal':
        raise ValueError('Formal training was not genuinely complete')
    if record['validation_ids'] != fixed['val'] or provenance['frozen_patient_ids'] != fixed:
        raise ValueError('Formal model was not selected using the exact frozen8validation patients')
    if record.get('test_used') is not False or complete.get('test_used') is not False:
        raise ValueError('Require explicit no-test-use provenance')
    if record['recipe'] != recipe or record['source_sha256'] != provenance['original_source_sha256']:
        raise ValueError('Export source/recipe differs from its genuine training record')
    if complete.get('best_val_dice') != provenance['selected_validation_dice']:
        raise ValueError('Exported model is not the completed formal validation best')
    training = provenance['training_environment_receipt']['value']
    if training['python'] != current_runtime['python'] or training['packages'] != current_runtime['packages']:
        raise ValueError('Use the original common main/reference software environment for all3models')
    return provenance, state


def stable_rng(case_id, stratum):
    word = int.from_bytes(hashlib.sha256((case_id + ':' + stratum).encode()).digest()[:4], 'little')
    return np.random.default_rng(np.random.SeedSequence([SEED, word]))


def build_geometry(record, directory):
    """Exact frozen prepare_case geometry arithmetic, in a separate output."""
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    from geometry import sparse_geometry
    started = time.time()
    cache = Path(record['cache_dir'])
    # Only this validation patient's actual arrays/label are read. Test records
    # are never passed here; canonical identity earlier used metadata only.
    for path, expected in [(cache / 'image.npy', record['cache_file_sha256']['image.npy']),
                           (cache / 'mask.npy', record['cache_file_sha256']['mask.npy']),
                           (Path(record['label']), record['label_sha256'])]:
        if file_hash(path) != expected:
            raise ValueError('Validation cache/source byte identity changed:' + str(path))
    mask = np.load(cache / 'mask.npy', mmap_mode='r', allow_pickle=False)
    label = sitk.ReadImage(record['label'])
    native_mask = sitk.GetArrayFromImage(label) > 0
    native_radius = ndi.distance_transform_edt(native_mask, sampling=np.asarray(label.GetSpacing())[::-1]).astype(np.float32)
    radius_image = sitk.GetImageFromArray(native_radius)
    radius_image.CopyInformation(label)
    grid = sitk.Image(record['resampled_size'], sitk.sitkFloat32)
    grid.SetOrigin(record['resampled_origin'])
    grid.SetDirection(record['resampled_direction'])
    grid.SetSpacing(record['resampled_spacing'])
    radius = sitk.GetArrayFromImage(sitk.Resample(radius_image, grid, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32))
    slices = tuple(slice(a, b) for a, b in zip(record['crop_start_zyx'], record['crop_stop_zyx']))
    geometry = sparse_geometry(mask > 0, np.asarray([record['spacing_mm']] * 3), radius_field=radius[slices])
    path = directory / (record['id'] + '_geometry.npz')
    np.savez_compressed(path, **geometry)
    receipt = {'case_id': record['id'], 'split': 'val', 'source_label_sha256': record['label_sha256'],
               'cache_mask_sha256': record['cache_file_sha256']['mask.npy'],
               'geometry_file': path.name, 'geometry_file_sha256': file_hash(path),
               'geometry_valid': bool(np.asarray(geometry['geometry_valid']).item()),
               'resampled_centerline_support': len(geometry['points']),
               'fine_support': int((geometry['radius'] <= 2.).sum()),
               'junction_support': int((geometry['structure'] == 2).sum()),
               'unknown_structure_support': int((geometry['structure'] == 255).sum()),
               'seconds': time.time() - started}
    return geometry, mask, receipt


def unique_starts(record, geometry):
    shape = np.asarray(record['shape_zyx'])
    high = np.maximum(shape - PATCH, 0)
    points, radius, structure = geometry['points'], geometry['radius'], geometry['structure']
    groups = {'GT_fine_centred': np.flatnonzero(radius <= 2.),
              'GT_junction_centred': np.flatnonzero(structure == 2),
              'GT_large_nonjunction_centred': np.flatnonzero((radius > 2.) & (structure == 1))}
    bank, receipts = {}, {}
    for stratum, requested in COUNTS.items():
        rng = stable_rng(record['id'], stratum)
        starts = []
        if stratum == 'CT_uniform':
            possible = math.prod([int(n) + 1 for n in high])
            for _ in range(512):
                start = tuple(int(rng.integers(int(n) + 1)) for n in high)
                if start not in starts:
                    starts.append(start)
                if len(starts) >= min(requested, possible):
                    break
            candidate_support = None
        else:
            candidates = groups[stratum]
            candidate_support = len(candidates)
            for index in rng.permutation(candidates):
                start = tuple(int(value) for value in np.clip(points[index] - PATCH // 2, 0, high))
                if start not in starts:
                    starts.append(start)
                if len(starts) == requested:
                    break
        for start in starts:
            bank.setdefault(start, []).append(stratum)
        receipts[stratum] = {'requested': requested, 'unique_selected': len(starts),
                             'candidate_centerline_support': candidate_support,
                             'unsupported': len(starts) == 0}
    return bank, receipts


def image_patch(record, start):
    from data import normalize
    image = np.load(Path(record['cache_dir']) / 'image.npy', mmap_mode='r', allow_pickle=False)
    slices = tuple(slice(int(s), min(int(s) + PATCH, int(n))) for s, n in zip(start, image.shape))
    array = np.asarray(image[slices], np.float32)
    array = np.pad(array, [(0, PATCH - n) for n in array.shape], constant_values=-1024)
    return np.ascontiguousarray(normalize(array)[None], dtype=np.float32)


def make_bank(manifest, output):
    from geometry import patch_histograms
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    records = {row['id']: row for row in manifest['cases']}
    geometry_dir = output / 'validation_geometry'
    geometry_dir.mkdir()
    patches, case_receipts = [], []
    for case_id in frozen_patient_ids()['val']:
        record = records[case_id]
        if record['geometry_training_only'] or (Path(record['cache_dir']) / 'geometry.npz').exists():
            raise ValueError('Frozen validation cache unexpectedly contains training geometry')
        geometry, mask, receipt = build_geometry(record, geometry_dir)
        starts, sampling = unique_starts(record, geometry)
        for start, strata in starts.items():
            scale, structure = patch_histograms(geometry, start, [PATCH] * 3)
            slices = tuple(slice(int(s), min(int(s) + PATCH, int(n))) for s, n in zip(start, mask.shape))
            mask_positive = bool(np.any(mask[slices]))
            shared_valid = not (mask_positive and scale[0] == 1.)
            points = geometry['points']
            inside = ((points >= np.asarray(start)) & (points < np.asarray(start) + PATCH)).all(1)
            values = geometry['structure'][inside]
            item = {'patch_id': '%s_%03d' % (case_id, len([row for row in patches if row['case_id'] == case_id])),
                'case_id': case_id, 'cohort': record['cohort'], 'split': 'val', 'strata': strata,
                'start_zyx': list(start), 'patch_size': PATCH,
                'CT_input': array_descriptor(image_patch(record, start)),
                'target_scale': scale.tolist(), 'target_structure': structure.tolist(),
                'mask_positive': mask_positive, 'shared_attribute_valid': shared_valid,
                'scale_valid': bool(shared_valid and scale.sum() > 0),
                'structure_valid': bool(shared_valid and structure.sum() > 0),
                'GT_centerline_support': int(inside.sum()),
                'GT_fine_support': int((geometry['radius'][inside] <= 2.).sum()),
                'GT_structure_valid_support': int(np.isin(values, [1, 2]).sum()),
                'GT_junction_support': int((values == 2).sum()),
                'GT_structure_unknown_support': int((values == 255).sum())}
            patches.append(item)
        receipt['sampling'] = sampling
        receipt['unique_patch_count_across_strata'] = len(starts)
        case_receipts.append(receipt)
        atomic_json(output / 'bank_progress.json', {'state': 'building_validation_bank', 'cases': len(case_receipts),
                    'patches': len(patches), 'test_used': False})
    if len(patches) > 160:
        raise AssertionError('Bounded validation patch budget exceeded')
    metadata = {'schema': 'GTA-AVT-validation-attribute-bank-v1', 'seed': SEED, 'patch_size': PATCH,
                'case_ids': frozen_patient_ids()['val'], 'case_receipts': case_receipts,
                'patches': patches, 'unique_patches': len(patches), 'maximum_patches': 160,
                'shared_across_models': True, 'stored_CT_patch_arrays': False,
                'geometry_selection': 'Frozen GT used only for explicit diagnostic strata/targets; never model inputs',
                'sampling_counts_per_case': COUNTS, 'test_used': False}
    atomic_json(output / 'validation_attribute_bank.json', metadata)
    return metadata


def predict_bank(provenance, state, bank, manifest, device):
    import torch
    model = load_strict_model(provenance, state, device)
    records = {row['id']: row for row in manifest['cases']}
    rows = []
    started = time.time()
    with torch.no_grad():
        for offset in range(0, len(bank['patches']), BATCH):
            selected = bank['patches'][offset:offset + BATCH]
            images = []
            for sample in selected:
                image = image_patch(records[sample['case_id']], sample['start_zyx'])
                if array_descriptor(image) != sample['CT_input']:
                    raise ValueError('Shared CT patch changed between model diagnostic passes')
                images.append(image)
            tensor = torch.from_numpy(np.stack(images)).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                attributes = model.classifier(tensor)
                predicted = {'scale': attributes['generation'].float().softmax(1),
                             'structure': attributes['structure'].float().softmax(1)}
                if provenance['method'] == 'faithful_concat':
                    used = None
                else:
                    scale, structure = model.conditions(attributes, 'predicted')
                    used = {'scale': scale.float(), 'structure': structure.float()}
            for index, sample in enumerate(selected):
                row = {'patch_id': sample['patch_id'], 'case_id': sample['case_id'], 'strata': sample['strata'],
                       'classifier_probabilities': {name: value[index].cpu().tolist() for name, value in predicted.items()},
                       'conditional_compositions_consumed': None if used is None else
                           {name: value[index].cpu().tolist() for name, value in used.items()},
                       'classifier_heads_consumed_by_conditional_arithmetic': provenance['method'] == 'faithful_soft',
                       'conditional_arithmetic': ('none; base attribute-feature fusion only' if used is None else
                            'fixed uniform5/3 composition' if provenance['method'] == 'faithful_soft_constant' else
                            'predicted composition/availability in original GAAM/SAAM operators')}
                if not all(np.isfinite(vector).all() for vector in row['classifier_probabilities'].values()):
                    raise ValueError('Nonfinite classifier probabilities')
                rows.append(row)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return rows, time.time() - started


def spearman(predicted, target):
    from scipy.stats import rankdata
    if len(target) < 3:
        return {'rho': None, 'reason': 'fewer_than3valid_patches'}
    if np.ptp(target) == 0:
        return {'rho': None, 'reason': 'constant_GT_composition'}
    if np.ptp(predicted) == 0:
        return {'rho': None, 'reason': 'constant_prediction'}
    # Rank correlation only: no inferential p-value is computed or retained.
    value = float(np.corrcoef(rankdata(predicted), rankdata(target))[0, 1])
    return {'rho': value if math.isfinite(value) else None, 'reason': 'descriptive_only_no_p_value'}


def score_bank(predictions, bank, method):
    lookup = {row['patch_id']: row for row in bank['patches']}
    scores = []
    for case_id in bank['case_ids']:
        for stratum in COUNTS:
            selected = [row for row in predictions if row['case_id'] == case_id and stratum in row['strata']]
            for kind in ('classifier_probabilities', 'conditional_compositions_consumed'):
                for head, component in (('scale', 1), ('structure', 2)):
                    valid = [row for row in selected if lookup[row['patch_id']][head + '_valid'] and row[kind] is not None]
                    row = {'method': method, 'case_id': case_id, 'stratum': stratum, 'prediction_kind': kind, 'head': head,
                           'selected_unique_patches': len(selected), 'valid_patches': len(valid),
                           'GT_centerline_support_in_selected_patches': sum(lookup[item['patch_id']]['GT_centerline_support'] for item in selected),
                           'support_note': 'overlapping patch support is descriptive, not independent nodes/patients'}
                    if not valid:
                        row.update(composition_MAE=None, composition_SSE=None, fine_or_junction_MAE=None,
                                   spearman={'rho': None, 'reason': 'no_valid_or_consumed_composition'})
                    else:
                        predicted = np.asarray([item[kind][head] for item in valid])
                        targets = np.asarray([lookup[item['patch_id']]['target_' + head] for item in valid])
                        difference = predicted - targets
                        row.update(composition_MAE=float(np.abs(difference).mean(1).mean()),
                                   composition_SSE=float((difference ** 2).sum(1).mean()),
                                   fine_or_junction_MAE=float(np.abs(difference[:, component]).mean()),
                                   spearman=spearman(predicted[:, component], targets[:, component]))
                    scores.append(row)
    aggregates = []
    for stratum in COUNTS:
        for kind in ('classifier_probabilities', 'conditional_compositions_consumed'):
            for head in ('scale', 'structure'):
                selected = [row for row in scores if row['stratum'] == stratum and row['prediction_kind'] == kind and row['head'] == head]
                aggregate = {'method': method, 'stratum': stratum, 'prediction_kind': kind, 'head': head,
                             'aggregation': 'equal weight per validation patient, after within-case patch averaging',
                             'patients_total': 8, 'patients_with_valid_support': sum(row['valid_patches'] > 0 for row in selected)}
                for metric in ('composition_MAE', 'composition_SSE', 'fine_or_junction_MAE'):
                    values = [row[metric] for row in selected if row[metric] is not None]
                    aggregate[metric] = {'mean_patients': statistics.mean(values) if values else None,
                                         'median_patients': statistics.median(values) if values else None,
                                         'defined_patient_count': len(values)}
                rhos = [row['spearman']['rho'] for row in selected if row['spearman']['rho'] is not None]
                aggregate['descriptive_within_case_spearman'] = {'median_patients': statistics.median(rhos) if rhos else None,
                                                               'defined_patient_count': len(rhos), 'p_values': 'not computed'}
                aggregates.append(aggregate)
    return scores, aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-c', required=True)
    parser.add_argument('--bundle-a', required=True)
    parser.add_argument('--bundle-constant', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--execute-frozen-validation', action='store_true')
    args = parser.parse_args()
    if not args.execute_frozen_validation:
        print(json.dumps({'state': 'not_executed', 'required_explicit_flag': '--execute-frozen-validation',
            'requires': 'three genuine completed formal C/A/constant export bundles in Python3.8/torch1.11 reference environment',
            'maximum_validation_patches': 160, 'models': METHODS, 'test_option': False,
            'training_or_cache_changes': False}, indent=2))
        return
    import torch
    require_reference_runtime()
    configure_inference_runtime()
    current = runtime_environment()
    paths = {'C': args.bundle_c, 'A': args.bundle_a, 'constant': args.bundle_constant}
    bundles = {label: validate_formal_bundle(path, METHODS[label], current) for label, path in paths.items()}
    # Verify metadata identity without opening ANY train/test source array.
    identity = canonical_cache_identity(args.manifest, verify_bytes=False)
    for provenance, state in bundles.values():
        assert_same_data(provenance, identity)
    recipes = [{key: value for key, value in provenance['recipe'].items() if key != 'method'} for provenance, _ in bundles.values()]
    if any(recipe != recipes[0] for recipe in recipes[1:]):
        raise ValueError('Main formal models do not share the exact comparison recipe')
    manifest = load_json(args.manifest)
    if manifest['splits'] != frozen_patient_ids():
        raise ValueError('Exact frozen validation split changed')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic fallback')
    torch.set_num_threads(4)
    output = require_new_directory(args.output)
    protocol = {'schema': 'GTA-AVT-formal-validation-attribute-faithfulness-v1', 'created_utc': utc(),
        'explicitly_invoked': True, 'split': 'val', 'case_ids': frozen_patient_ids()['val'], 'seed': SEED,
        'sampling': COUNTS, 'patch_size': PATCH, 'batch_size': BATCH, 'maximum_shared_patches': 160,
        'test_GT_opened': False, 'test_inputs_predictions_or_metrics_used': False,
        'training_or_model_selection_or_adaptation_changed': False, 'cache_modified': False,
        'full_segmentation_forward_executed': False, 'classifier_only_inference': True,
        'autocast_enabled': device.type == 'cuda', 'reference_environment': current,
        'selected_execution_device': selected_execution_device(device),
        'frozen_source_sha256': source_hashes(), 'canonical_data_sha256': identity['canonical_sha256'],
        'cache_manifest_sha256': file_hash(args.manifest),
        'bundles': {label: {'method': provenance['method'], 'selected_step': provenance['selected_step'],
             'weights_sha256': provenance['weights_sha256'], 'original_checkpoint_sha256': provenance['original_checkpoint_sha256'],
             'provenance_sha256': file_hash(Path(paths[label]) / 'provenance.json')}
             for label, (provenance, state) in bundles.items()},
        'tool_sha256': {Path(__file__).name: file_hash(__file__),
                       'reference_weights_support.py': file_hash(Path(__file__).with_name('reference_weights_support.py'))},
        'interpretation': 'Validation composition fidelity of operational GT proxies, not test segmentation or calibrated uncertainty; no post-diagnostic tuning.'}
    atomic_json(output / 'diagnostic_protocol.json', protocol)
    try:
        bank = make_bank(manifest, output)
        all_aggregates, timings = {}, {}
        for label, (provenance, state) in bundles.items():
            predictions, seconds = predict_bank(provenance, state, bank, manifest, device)
            case_scores, aggregates = score_bank(predictions, bank, provenance['method'])
            atomic_json(output / (label + '_patch_predictions.json'), {'method': provenance['method'], 'predictions': predictions})
            atomic_json(output / (label + '_case_scores.json'), {'method': provenance['method'], 'case_scores': case_scores})
            all_aggregates[label], timings[label] = aggregates, seconds
        atomic_json(output / 'summary.json', {'state': 'complete', 'protocol': protocol,
            'bank_sha256': file_hash(output / 'validation_attribute_bank.json'), 'unique_patches': bank['unique_patches'],
            'classifier_forward_samples': bank['unique_patches'] * 3, 'inference_seconds': timings,
            'case_aggregates': all_aggregates, 'test_used': False, 'no_changes_after_viewing_diagnostic': True})
        print({'completed_validation_only_attribute_diagnostic': str(output), 'unique_shared_patches': bank['unique_patches'],
               'classifier_forward_samples': bank['unique_patches'] * 3, 'test_used': False}, flush=True)
    except BaseException as error:
        atomic_json(output / 'failed.json', {'state': 'failed', 'type': type(error).__name__, 'reason': str(error),
                    'test_used': False, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
