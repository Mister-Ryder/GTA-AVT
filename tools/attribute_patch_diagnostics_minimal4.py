"""Portable, explicitly invoked four-bundle VAL-only classifier diagnostic.

The index declares reproduction roles; generic exports do not prove physical
server identity. Default planning imports only the standard library and opens
no index, bundle, manifest or array. No training, TEST or checkpoint selection.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

SCHEMA = 'GTA-AVT-minimal4-portable-bundle-index-v1'
DIAGNOSTIC_SCHEMA = 'GTA-AVT-minimal4-portable-attribute-faithfulness-v1'
ROLE_ASSIGNMENT = 'declared_for_this_reproduction_not_proved_server_identity'
SLOTS = {
    'main_C_concat': ('main', 'C_concat', 'faithful_concat'),
    'main_A_soft': ('main', 'A_soft', 'faithful_soft'),
    'ablations_A_soft_platform_bridge': ('ablations', 'A_soft_platform_bridge', 'faithful_soft'),
    'ablations_A_soft_constant': ('ablations', 'A_soft_constant', 'faithful_soft_constant'),
}
FILES = ('weights_tensor_only.pt', 'provenance.json', 'export_verified.json')
SOURCE_SHA256 = '7c5b29ff069f5b33fe2212d721945c6b2e468bc5d15a9f87f6f12e1661f7fe96'
SPLIT_SHA256 = 'df9647da8b331260fb9f66b294eced01180c1eaf812de27b5148b6800a344a3b'
OLD_HELPER_SHA256 = 'e2c2ddb0bc16bb3cb34508e1e99355a312d6757a5da3955f9917b0c5d87f6293'
SUPPORT_SHA256 = 'f76d205fe3e8eb889017fc21d0a29d6e267667f9867c9629848e537e7e4d9d9c'
EXPORTER_SHA256 = 'a6dd6ee31fb67b1456df0351f1f9e9fd8963e86838ec6799b1a5640e6560a1e9'
PURE_FUNCTIONS = ('stable_rng', 'build_geometry', 'unique_starts', 'image_patch',
                  'make_bank', 'predict_bank', 'spearman', 'score_bank')
PACKAGES = {'torch': '1.11.0+cu113', 'numpy': '1.22.4', 'scipy': '1.10.1',
            'scikit-image': '0.21.0', 'SimpleITK': '2.3.1'}
RECIPE = dict(stage='formal', seed=20261022, width=32, patch_size=128, batch_size=2,
    max_steps=10000, lr=.0003, weight_decay=.0001, foreground_probability=.5,
    auxiliary_weight=.1, gradient_clip=12., log_every=25, checkpoint_every=500,
    validate_every=5000, validation_case_limit=0, inference_patch_size=128,
    inference_batch_size=2, inference_overlap=.5, save_validation_predictions=False, cpu_threads=4)
INFERENCE = dict(threshold=.5, patch_size=128, overlap=.5, batch_size=2,
                 condition_mode='predicted', postprocessing='none', grid='original CT')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_ids(repository):
    path = Path(repository) / 'configs/split_seed20261022.json'
    require(file_hash(path) == SPLIT_SHA256, 'The exact released frozen split is required')
    split = read(path)
    prefix = {'D': 'Dongyang', 'K': 'KiTS', 'R': 'Rider'}
    return {name: [prefix[x[0]] + '_' + x for x in split[name]] for name in ('train', 'val', 'test')}


def load_index(path):
    path = Path(path).resolve()
    value = read(path)
    require(set(value) == {'schema', 'role_assignment', 'bundles'} and value['schema'] == SCHEMA
        and value['role_assignment'] == ROLE_ASSIGNMENT, 'Require the explicit portable role-declaration index')
    require(isinstance(value['bundles'], dict) and set(value['bundles']) == set(SLOTS), 'Exactly four named slots are required')
    paths = {}
    for slot, (role, job, method) in SLOTS.items():
        entry = value['bundles'][slot]
        require(isinstance(entry, dict) and set(entry) == {'role', 'job', 'method', 'bundle', 'bundle_sha256'}, 'Unexpected index fields: ' + slot)
        require((entry['role'], entry['job'], entry['method']) == (role, job, method), 'Slot/role/job/method mismatch: ' + slot)
        reference = entry['bundle']
        require(isinstance(reference, str) and reference and '\\' not in reference and ':' not in reference
            and not PurePosixPath(reference).is_absolute() and not PureWindowsPath(reference).drive,
            'Bundle references must be relative POSIX paths')
        hashes = entry['bundle_sha256']
        require(isinstance(hashes, dict) and set(hashes) == set(FILES) and all(sha(x) for x in hashes.values()),
                'Replace every placeholder with the three actual file SHA256 values')
        paths[slot] = (path.parent / reference).resolve()
    require(len(set(paths.values())) == 4, 'Each slot must name a distinct bundle directory')
    return value, paths


def bundle_metadata(directory, entry):
    directory = Path(directory)
    require(directory.is_dir(), 'Missing generic tensor bundle: ' + str(directory))
    hashes = {name: file_hash(directory / name) for name in FILES}
    require(hashes == entry['bundle_sha256'], 'Actual bundle bytes differ from the manually frozen index')
    return read(directory / 'provenance.json'), read(directory / 'export_verified.json'), hashes


def validate_bundle_metadata(slot, provenance, receipt, hashes, source, fixed, runtime):
    """Stdlib metadata admission, never a tensor load or a server attestation.

    Generic export_verified.json records the original checkpoint/signature/
    selected-summary audit. This portable gate checks that evidence and its
    byte binding; it does not recreate private queue or physical-host evidence.
    """
    require(slot in SLOTS, 'Unknown portable slot')
    role, job, method = SLOTS[slot]
    require(set(hashes) == set(FILES) and all(sha(x) for x in hashes.values()), 'Three actual bundle hashes are required')
    require(len(source) == 24 and canonical(source) == SOURCE_SHA256, 'The 24 frozen sources differ')
    require(runtime.get('python_version', [])[:2] == [3, 8] and runtime.get('packages') == PACKAGES, 'Require the declared legacy reference software')
    require(provenance.get('schema') == 'GTA-AVT-reference-weights-v1' and provenance.get('method') == method
        and provenance.get('stage') == 'formal' and provenance.get('weights_file') == FILES[0]
        and provenance.get('weights_sha256') == hashes[FILES[0]] and provenance.get('original_checkpoint_name') == 'best.pt',
        'Require the named genuine formal generic best-checkpoint export')
    require(receipt.get('state') == 'verified' and receipt.get('weights_sha256') == hashes[FILES[0]]
        and receipt.get('provenance_sha256') == hashes[FILES[1]] and receipt.get('tensor_count') == 270
        and receipt.get('all56_cache_and_raw_bytes_verified') is True
        and receipt.get('all56_file_bytes_read_for_identity_only') is True
        and receipt.get('test_predictions_or_metrics_used') is False, 'Generic export verification is missing or byte-mismatched')
    recipe = provenance.get('recipe')
    record = provenance.get('original_run_record', {})
    complete = provenance.get('original_complete_record', {})
    expected = dict(RECIPE, method=method)
    require(canonical(recipe) == canonical(expected) and canonical(record.get('recipe')) == canonical(expected), 'Exact typed 21-field128/10000 recipe required')
    require(provenance.get('original_source_sha256') == source and record.get('source_sha256') == source, 'Training/export source mismatch')
    require(provenance.get('frozen_patient_ids') == fixed and record.get('train_ids') == fixed['train']
        and record.get('validation_ids') == fixed['val'], 'Exact TRAIN36/VAL8/TEST12 case IDs/order required')
    require(provenance.get('final_test_eligible') is True and record.get('selection_valid') is True
        and record.get('test_used') is False and complete.get('test_used') is False, 'Require eligible full-VAL8 selection without TEST use')
    require(complete.get('state') == 'complete' and complete.get('stage') == 'formal'
        and type(complete.get('steps')) is int and complete['steps'] == 10000, 'Partial training is ineligible')
    step, best = provenance.get('selected_step'), provenance.get('selected_validation_dice')
    require(type(step) is int and step in (5000, 10000), 'Selected best must be from the two declared VAL8 timepoints')
    require(type(best) in (int, float) and math.isfinite(best) and 0 <= best <= 1
        and canonical(complete.get('best_val_dice')) == canonical(best), 'Genuine completed best selection is inconsistent')
    final = complete.get('last_validation', {})
    final_dice = final.get('metrics', {}).get('dice', {})
    last = final_dice.get('mean')
    require(final.get('cases') == 8 and final_dice.get('defined_case_count') == 8
        and type(last) in (int, float) and math.isfinite(last) and 0 <= last <= 1,
        'The completed final validation must have all eight defined Dice values')
    require((step == 5000 and best >= last) or (step == 10000 and best == last), 'Selected best is worse than the actual final VAL8')
    for name in ('original_checkpoint_sha256', 'original_run_config_sha256', 'original_complete_sha256',
                 'original_selected_validation_summary_sha256'):
        require(sha(provenance.get(name)), 'Missing original generic checkpoint evidence: ' + name)
    require(sha(record.get('manifest_sha256')), 'Original training manifest SHA is missing')
    require(provenance.get('fixed_reference_inference') == INFERENCE, 'Reference inference recipe changed')
    require(provenance.get('export_tool_sha256') == {'export_reference_weights.py': EXPORTER_SHA256,
        'reference_weights_support.py': SUPPORT_SHA256}, 'Use the released, source-verified generic exporter')
    training = provenance.get('training_environment_receipt', {})
    environment = training.get('value', {})
    exported = provenance.get('export_environment', {})
    require(sha(training.get('sha256')) and environment.get('python') == runtime.get('python')
        and environment.get('packages') == PACKAGES and exported.get('python') == environment.get('python')
        and exported.get('packages') == PACKAGES, 'Original training/export software receipt differs from the common reference')
    data = provenance.get('canonical_data_identity', {})
    payload = data.get('canonical', {})
    require(data.get('schema') == 'GTA-AVT-canonical-data-v1' and sha(data.get('canonical_sha256'))
        and canonical(payload) == data['canonical_sha256'] and payload.get('schema') == data['schema']
        and payload.get('complete') is True and payload.get('split_sha256') == SPLIT_SHA256
        and payload.get('splits') == fixed and canonical(payload.get('spacing_mm')) == canonical(1.0), 'Canonical1mm data/split identity changed')
    ids = [row.get('id') for row in payload.get('cases', [])]
    require(len(ids) == len(set(ids)) == 56 and set(ids) == set(sum(fixed.values(), [])), 'Canonical data does not cover the exact56 cases')
    require(data.get('verification', {}).get('actual_bytes_verified') is True
        and data['verification'].get('manifest_sha256') == record['manifest_sha256'], 'Generic export did not verify the actual original cache bytes')
    inventory = provenance.get('tensor_inventory', {})
    require(isinstance(inventory, dict) and len(inventory) == 270, 'Require the full270-tensor inventory')
    for name, item in inventory.items():
        require(isinstance(name, str) and isinstance(item, dict) and item.get('dtype') in ('torch.float32', 'torch.int64')
            and isinstance(item.get('shape'), list) and all(type(x) is int and x > 0 for x in item['shape'])
            and type(item.get('numel')) is int and item['numel'] == math.prod(item['shape'])
            and item.get('byte_order') == 'little' and sha(item.get('contiguous_cpu_bytes_sha256')), 'Invalid tensor inventory entry')
    for name, shape in {'classifier.generation_head.3.weight': [5, 256], 'classifier.generation_head.3.bias': [5],
        'classifier.structure_head.3.weight': [3, 256], 'classifier.structure_head.3.bias': [3],
        'anatomical_output_fusion.3.weight': [2, 32, 1, 1, 1], 'anatomical_output_fusion.3.bias': [2]}.items():
        require(inventory.get(name, {}).get('shape') == shape, 'Frozen5/3/2 class dimensions differ: ' + name)
    return dict(slot=slot, declared_role=role, declared_job=job, method=method,
        role_identity_basis=ROLE_ASSIGNMENT, bundle_sha256=hashes,
        original_checkpoint_sha256=provenance['original_checkpoint_sha256'], selected_step=step,
        selected_validation_dice=best, training_recipe=recipe, training_recipe_sha256=canonical(recipe),
        training_recipe_without_method_sha256=canonical(RECIPE), frozen_source_sha256=source,
        canonical_data_sha256=data['canonical_sha256'], training_environment_receipt_sha256=training['sha256'],
        training_environment=environment, selection_evidence='Generic exporter verified genuine best.pt/signature/selected VAL summary/completed best; no private queue attestation inferred')


def validate_common(identities):
    require(set(identities) == set(SLOTS), 'All four independently named slots are required')
    first = identities[next(iter(SLOTS))]
    for slot, identity in identities.items():
        for name in ('frozen_source_sha256', 'canonical_data_sha256', 'training_recipe_without_method_sha256'):
            require(canonical(identity[name]) == canonical(first[name]), 'Four models differ in common identity: ' + name)
        # Role names are declared by the index, not recovered from filesystem paths.
        for other, comparison in identities.items():
            if SLOTS[slot][0] == SLOTS[other][0]:
                for name in ('python', 'packages', 'devices'):
                    require(canonical(identity['training_environment'].get(name)) == canonical(comparison['training_environment'].get(name)), 'Within-role training software/device metadata differs')


def load_public_helpers(repository):
    repository = Path(repository).resolve()
    tools = repository / 'tools'
    expected = {'attribute_patch_diagnostics.py': OLD_HELPER_SHA256,
                'reference_weights_support.py': SUPPORT_SHA256, 'export_reference_weights.py': EXPORTER_SHA256}
    for name, digest in expected.items():
        require(file_hash(tools / name) == digest, 'Released public dependency bytes changed: ' + name)
    for name, path in {'reference_weights_support': tools / 'reference_weights_support.py',
        'common': repository / 'code/common.py', 'data': repository / 'code/data.py',
        'geometry': repository / 'code/geometry.py', 'model': repository / 'code/model.py'}.items():
        loaded = sys.modules.get(name)
        require(loaded is None or Path(getattr(loaded, '__file__', '')).resolve() == path.resolve(),
                'A different checkout is already imported under the public module name: ' + name)
    sys.path.insert(0, str(tools))
    # Only reached after the explicit execution flag. Eight functions retain
    # their exact released source bytes; the old MAIN3 admission is never called.
    spec = importlib.util.spec_from_file_location('gta_avt_original_attribute_functions', tools / 'attribute_patch_diagnostics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(all(callable(getattr(module, name, None)) for name in PURE_FUNCTIONS), 'A released diagnostic function is absent')
    return module


def execute(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    old = load_public_helpers(args.repository)
    import torch
    old.require_reference_runtime()
    old.configure_inference_runtime()
    runtime, source = old.runtime_environment(), old.source_hashes()
    index, paths = load_index(args.index)
    index_sha, manifest_sha = file_hash(args.index), file_hash(args.manifest)
    fixed = frozen_ids(args.repository)
    require(fixed == old.frozen_patient_ids(), 'Public split lookup changed')
    identities, provenances = {}, {}
    for slot in SLOTS:
        p, receipt, hashes = bundle_metadata(paths[slot], index['bundles'][slot])
        identities[slot] = validate_bundle_metadata(slot, p, receipt, hashes, source, fixed, runtime)
        provenances[slot] = p
    validate_common(identities)
    # This verifies canonical metadata without reading TRAIN/TEST array files.
    identity = old.canonical_cache_identity(args.manifest, verify_bytes=False)
    for p in provenances.values():
        old.assert_same_data(p, identity)
    manifest = old.load_json(args.manifest)
    require(manifest['splits'] == fixed, 'Manifest cases differ from the exact frozen split')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.set_num_threads(4)
    output = Path(args.output)
    require(not output.exists() and not output.is_symlink(), 'Use a fresh nonexistent output directory; no reuse or overwrite')
    output.mkdir(parents=True, exist_ok=False)
    tool_hashes = {'attribute_patch_diagnostics_minimal4.py': file_hash(__file__),
        'attribute_patch_diagnostics.py': OLD_HELPER_SHA256, 'reference_weights_support.py': SUPPORT_SHA256}
    protocol = dict(schema=DIAGNOSTIC_SCHEMA, created_utc=old.utc(), explicitly_invoked=True,
        split='val', case_ids=fixed['val'], seed=20261022, sampling=old.COUNTS, patch_size=128,
        batch_size=2, maximum_shared_patches=160, index_sha256=index_sha, input_index=index,
        bundles=identities, cache_manifest_sha256=manifest_sha, frozen_source_sha256=source,
        canonical_data_sha256=identity['canonical_sha256'], reference_environment=runtime,
        selected_execution_device=old.selected_execution_device(device), autocast_enabled=device.type == 'cuda',
        test_GT_opened=False, test_inputs_predictions_or_metrics_used=False, cache_modified=False,
        full_segmentation_forward_executed=False, classifier_only_inference=True,
        training_tuning_or_checkpoint_selection_performed=False, tool_sha256=tool_hashes,
        case_unit_metadata=dict(unit='released scan/case ID', patient_independence_verified=False,
            legacy_patient_keys='Case summaries only; not verified unique persons'),
        interpretation='Fixed GT-centred strata are diagnostic samples, not anatomical prevalence. Operational geometry composition and computed conditions are measured; learned GAAM/SAAM gates and segmentation benefit are not. Role assignments are user declarations, not verified physical-server identities.')
    try:
        old.atomic_json(output / 'diagnostic_protocol.json', protocol)
        bank = old.make_bank(manifest, output)
        require(bank['case_ids'] == fixed['val'] and 8 <= bank['unique_patches'] <= 160, 'Shared bank is outside the fixed VAL8 budget')
        aggregates, timings = {}, {}
        for slot in SLOTS:
            p, state = old.load_tensor_bundle(paths[slot])
            require(canonical(p) == canonical(provenances[slot]), 'Bundle provenance changed after admission')
            predictions, seconds = old.predict_bank(p, state, bank, manifest, device)
            scores, result = old.score_bank(predictions, bank, p['method'])
            require(len(predictions) == bank['unique_patches'], 'A classifier omitted shared bank inputs')
            old.atomic_json(output / (slot + '_patch_predictions.json'), dict(slot=slot, method=p['method'], predictions=predictions))
            old.atomic_json(output / (slot + '_case_scores.json'), dict(slot=slot, method=p['method'], case_scores=scores))
            aggregates[slot], timings[slot] = result, seconds
            del state
        require(file_hash(args.index) == index_sha and file_hash(args.manifest) == manifest_sha
            and old.source_hashes() == source and file_hash(__file__) == tool_hashes['attribute_patch_diagnostics_minimal4.py'], 'Inputs/source changed during the diagnostic')
        for name, digest in {'attribute_patch_diagnostics.py': OLD_HELPER_SHA256,
            'reference_weights_support.py': SUPPORT_SHA256, 'export_reference_weights.py': EXPORTER_SHA256}.items():
            require(file_hash(Path(args.repository) / 'tools' / name) == digest, 'Public dependency changed during the diagnostic: ' + name)
        for slot in SLOTS:
            bundle_metadata(paths[slot], index['bundles'][slot])
        old.atomic_json(output / 'summary.json', dict(schema=DIAGNOSTIC_SCHEMA, state='complete', protocol=protocol,
            bank_sha256=file_hash(output / 'validation_attribute_bank.json'), unique_patches=bank['unique_patches'],
            classifier_forward_samples=4 * bank['unique_patches'], inference_seconds=timings,
            case_aggregates=aggregates, test_used=False, no_training_tuning_or_selection_performed_by_helper=True))
        print(json.dumps(dict(state='complete', output=str(output), models=4, unique_shared_patches=bank['unique_patches'], test_used=False)))
    except BaseException as error:
        old.atomic_json(output / 'failed.json', dict(state='failed', type=type(error).__name__, reason=str(error),
            test_used=False, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, help='Manual four-bundle relative-path index with actual three-file hashes')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, help='Fresh nonexistent diagnostic directory')
    parser.add_argument('--repository', default=str(Path(__file__).resolve().parents[1]), help='Public checkout containing existing tools/ and frozen code/')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--execute-frozen-validation', action='store_true')
    args = parser.parse_args()
    if not args.execute_frozen_validation:
        print(json.dumps(dict(state='not_executed', schema=DIAGNOSTIC_SCHEMA,
            required_explicit_flag='--execute-frozen-validation', models=SLOTS,
            maximum_validation_patches=160, role_identity_basis=ROLE_ASSIGNMENT,
            index_bundles_manifest_or_arrays_opened=False, torch_or_numpy_imported=False,
            training_test_selection_or_automatic_next=False), indent=2))
        return
    execute(args)


if __name__ == '__main__':
    main()
