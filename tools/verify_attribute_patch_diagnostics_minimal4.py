"""Temporary metadata/vector fixtures; no real AVT arrays, tensor loads or forward."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reject(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('An intentionally invalid input was accepted')


def source_identity(repository, helper):
    root = repository / 'code'
    files = [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and (p.suffix == '.py' or ('historical' in p.relative_to(root).parts
                  and p.suffix.lower() in ('.json', '.txt', '.md')))]
    return {p.relative_to(root).as_posix(): helper.file_hash(p) for p in sorted(files)}


def fixture(helper, source, fixed, slot):
    sha = lambda word: hashlib.sha256(word.encode()).hexdigest()
    method = helper.SLOTS[slot][2]
    environment = dict(python='synthetic Python3.8 metadata only', python_version=[3, 8, 10],
                       packages=dict(helper.PACKAGES), devices=[dict(name='synthetic device')])
    manifest_sha = sha('synthetic manifest')
    payload = dict(schema='GTA-AVT-canonical-data-v1', complete=True, split_sha256=helper.SPLIT_SHA256,
        splits=fixed, spacing_mm=1.0, cases=[dict(id=x) for x in sorted(sum(fixed.values(), []))])
    shapes = {'classifier.generation_head.3.weight': [5, 256], 'classifier.generation_head.3.bias': [5],
        'classifier.structure_head.3.weight': [3, 256], 'classifier.structure_head.3.bias': [3],
        'anatomical_output_fusion.3.weight': [2, 32, 1, 1, 1], 'anatomical_output_fusion.3.bias': [2]}
    shapes.update({'synthetic_inventory_entry_%03d' % i: [1] for i in range(264)})
    inventory = {name: dict(dtype='torch.float32', shape=shape, numel=math.prod(shape), byte_order='little',
                 contiguous_cpu_bytes_sha256=sha(name)) for name, shape in shapes.items()}
    recipe = dict(helper.RECIPE, method=method)
    provenance = dict(schema='GTA-AVT-reference-weights-v1', method=method, stage='formal',
        weights_file='weights_tensor_only.pt', weights_sha256=sha(slot + ' weights'),
        original_checkpoint_name='best.pt', selected_step=10000, selected_validation_dice=.75,
        original_checkpoint_sha256=sha(slot + ' checkpoint'), original_run_config_sha256=sha(slot + ' config'),
        original_complete_sha256=sha(slot + ' complete'), original_selected_validation_summary_sha256=sha(slot + ' selectedsummary'),
        recipe=recipe, original_source_sha256=source, frozen_patient_ids=fixed, final_test_eligible=True,
        original_run_record=dict(recipe=recipe, source_sha256=source, manifest_sha256=manifest_sha,
            train_ids=fixed['train'], validation_ids=fixed['val'], selection_valid=True, test_used=False),
        original_complete_record=dict(state='complete', stage='formal', steps=10000, test_used=False,
            best_val_dice=.75, last_validation=dict(cases=8, metrics=dict(dice=dict(mean=.75, defined_case_count=8)))),
        fixed_reference_inference=dict(helper.INFERENCE),
        export_tool_sha256={'export_reference_weights.py': helper.EXPORTER_SHA256, 'reference_weights_support.py': helper.SUPPORT_SHA256},
        training_environment_receipt=dict(sha256=sha('synthetic environment receipt'), value=copy.deepcopy(environment)),
        export_environment=copy.deepcopy(environment), tensor_inventory=inventory,
        canonical_data_identity=dict(schema=payload['schema'], canonical=payload, canonical_sha256=helper.canonical(payload),
            verification=dict(actual_bytes_verified=True, manifest_sha256=manifest_sha)))
    hashes = {name: sha(slot + name) for name in helper.FILES}
    hashes['weights_tensor_only.pt'] = provenance['weights_sha256']
    receipt = dict(state='verified', weights_sha256=hashes['weights_tensor_only.pt'],
        provenance_sha256=hashes['provenance.json'], tensor_count=270, all56_cache_and_raw_bytes_verified=True,
        all56_file_bytes_read_for_identity_only=True, test_predictions_or_metrics_used=False)
    return provenance, receipt, hashes, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True, help='Unmodified public checkout; only its source/split are read')
    parser.add_argument('--output')
    args = parser.parse_args()
    repository = Path(args.repository).resolve()
    target = Path(__file__).with_name('attribute_patch_diagnostics_minimal4.py')
    helper = load(target, 'portable_classifier4_fixture_target')
    ast.parse(target.read_text(encoding='utf-8'), feature_version=(3, 8))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    source, fixed = source_identity(repository, helper), helper.frozen_ids(repository)
    assert len(source) == 24 and helper.canonical(source) == helper.SOURCE_SHA256
    assert list(map(len, fixed.values())) == [36, 8, 12]
    assert all(x.startswith(('Dongyang_', 'KiTS_', 'Rider_')) for x in sum(fixed.values(), []))
    passed = ['Python3.8_AST_stdlib_import', 'actual_frozen_split_short_to_full_ID_conversion']
    base = {slot: fixture(helper, source, fixed, slot) for slot in helper.SLOTS}

    def check(slot='main_A_soft', change=None):
        p, r, h, runtime = copy.deepcopy(base[slot])
        src, ids = copy.deepcopy(source), copy.deepcopy(fixed)
        if change:
            change(p, r, h, runtime, src, ids)
        return helper.validate_bundle_metadata(slot, p, r, h, src, ids, runtime)

    identities = {slot: check(slot) for slot in helper.SLOTS}
    helper.validate_common(identities)
    assert identities['main_A_soft']['method'] == identities['ablations_A_soft_platform_bridge']['method']
    assert identities['main_A_soft']['slot'] != identities['ablations_A_soft_platform_bridge']['slot']
    assert all(item['role_identity_basis'] == helper.ROLE_ASSIGNMENT for item in identities.values())
    passed.append('four_generic_exports_no_private_transfer_two_soft_slots_distinct')

    recipe_changes = [lambda p: p['recipe'].__setitem__('patch_size', 96),
        lambda p: p['recipe'].__setitem__('max_steps', 5000),
        lambda p: p['recipe'].__setitem__('batch_size', 2.0),
        lambda p: p['recipe'].__setitem__('seed', 20261023),
        lambda p: p['recipe'].__setitem__('extra_field', True)]
    for action in recipe_changes:
        reject(lambda action=action: check(change=lambda p, *rest: action(p)))
    passed.append('full_typed_recipe_budget_size_seed_extra_field_rejection')
    changes = [lambda p: p.__setitem__('selected_step', 7500),
        lambda p: p.__setitem__('selected_step', True), lambda p: p.__setitem__('final_test_eligible', False),
        lambda p: p['original_complete_record'].__setitem__('steps', 9999),
        lambda p: p['original_complete_record'].__setitem__('test_used', True),
        lambda p: p['original_run_record'].__setitem__('selection_valid', False),
        lambda p: p.__setitem__('selected_validation_dice', .7),
        lambda p: p['original_complete_record']['last_validation']['metrics']['dice'].__setitem__('defined_case_count', 7)]
    for action in changes:
        reject(lambda action=action: check(change=lambda p, *rest: action(p)))
    early = check(change=lambda p, *rest: p.__setitem__('selected_step', 5000))
    assert early['selected_step'] == 5000  # The earlier checkpoint wins a tie.
    passed.append('genuine_complete_eligible_selected_best_support_and_test_use_gates')
    reject(lambda: check(change=lambda p, r, h, runtime, src, ids: src.__setitem__('model.py', '0' * 64)))
    reject(lambda: check(change=lambda p, *rest: p['original_run_record']['train_ids'].__setitem__(0, fixed['test'][0])))
    reject(lambda: check(change=lambda p, *rest: p['canonical_data_identity']['canonical'].__setitem__('spacing_mm', 1.5)))
    reject(lambda: check(change=lambda p, *rest: p['canonical_data_identity']['verification'].__setitem__('actual_bytes_verified', False)))
    passed.append('24_source_exact_split_data_hash_spacing_and_actual_byte_proof_gates')
    reject(lambda: check(change=lambda p, *rest: p['tensor_inventory'].pop(next(iter(p['tensor_inventory'])))))
    reject(lambda: check(change=lambda p, *rest: p['tensor_inventory']['classifier.generation_head.3.weight'].__setitem__('shape', [7, 256])))
    reject(lambda: check(change=lambda p, *rest: p['tensor_inventory']['classifier.structure_head.3.bias'].__setitem__('contiguous_cpu_bytes_sha256', 'missing')))
    reject(lambda: check(change=lambda p, r, *rest: r.__setitem__('tensor_count', 269)))
    passed.append('full270_inventory_descriptor_and_5_3_2_dimensions')
    reject(lambda: check(change=lambda p, r, h, runtime, *rest: runtime.__setitem__('python_version', [3, 11, 1])))
    reject(lambda: check(change=lambda p, r, h, runtime, *rest: runtime['packages'].__setitem__('torch', '2.7.1')))
    reject(lambda: check(change=lambda p, *rest: p['training_environment_receipt']['value'].__setitem__('python', 'different build')))
    reject(lambda: check(change=lambda p, *rest: p['export_tool_sha256'].__setitem__('reference_weights_support.py', '0' * 64)))
    passed.append('reference_training_export_software_and_released_exporter_identity')
    for field in ('canonical_data_sha256', 'training_recipe_without_method_sha256'):
        changed = copy.deepcopy(identities)
        changed['main_A_soft'][field] = '0' * 64
        reject(lambda changed=changed: helper.validate_common(changed))
    changed = copy.deepcopy(identities)
    changed['main_A_soft']['training_environment']['devices'] = [dict(name='different synthetic device')]
    reject(lambda: helper.validate_common(changed))
    passed.append('four_common_identity_and_within_declared_role_device_metadata')

    with tempfile.TemporaryDirectory(prefix='portable_classifier4_metadata_') as tmp:
        temporary = Path(tmp)
        index = dict(schema=helper.SCHEMA, role_assignment=helper.ROLE_ASSIGNMENT, bundles={})
        for slot, (role, job, method) in helper.SLOTS.items():
            folder = temporary / slot
            folder.mkdir()
            p, r, h, runtime = copy.deepcopy(base[slot])
            (folder / helper.FILES[0]).write_bytes(b'Synthetic metadata fixture only; NOT a valid tensor model')
            p['weights_sha256'] = helper.file_hash(folder / helper.FILES[0])
            (folder / helper.FILES[1]).write_text(json.dumps(p), encoding='utf-8')
            r.update(weights_sha256=p['weights_sha256'], provenance_sha256=helper.file_hash(folder / helper.FILES[1]))
            (folder / helper.FILES[2]).write_text(json.dumps(r), encoding='utf-8')
            h = {name: helper.file_hash(folder / name) for name in helper.FILES}
            index['bundles'][slot] = dict(role=role, job=job, method=method, bundle=slot, bundle_sha256=h)
        index_path = temporary / 'index.json'

        def write_index(value):
            index_path.write_text(json.dumps(value), encoding='utf-8')
            return helper.load_index(index_path)

        loaded, paths = write_index(index)
        for slot in helper.SLOTS:
            p, r, h = helper.bundle_metadata(paths[slot], loaded['bundles'][slot])
            helper.validate_bundle_metadata(slot, p, r, h, source, fixed, runtime)
        for reference in ('/absolute/bundle', 'C:/absolute/bundle', '\\\\server\\bundle', 'https://example.invalid/bundle'):
            changed = copy.deepcopy(index)
            changed['bundles']['main_A_soft']['bundle'] = reference
            reject(lambda changed=changed: write_index(changed))
        changed = copy.deepcopy(index)
        changed['bundles']['main_A_soft']['role'] = 'ablations'
        reject(lambda: write_index(changed))
        changed = copy.deepcopy(index)
        changed['bundles'].pop('main_C_concat')
        reject(lambda: write_index(changed))
        changed = copy.deepcopy(index)
        changed['bundles']['main_A_soft']['bundle'] = 'ablations_A_soft_platform_bridge'
        reject(lambda: write_index(changed))
        write_index(index)
        for filename in helper.FILES:
            file = paths['main_A_soft'] / filename
            original = file.read_bytes()
            file.write_bytes(original + b'changed')
            reject(lambda: helper.bundle_metadata(paths['main_A_soft'], index['bundles']['main_A_soft']))
            file.write_bytes(original)
        passed.append('manual_relative_index_role_mapping_distinct_directories_and_all_three_actual_hashes')
        command = [sys.executable, str(target), '--index', str(temporary / 'missing_index.json'),
            '--manifest', str(temporary / 'missing_manifest.json'), '--output', str(temporary / 'must_not_exist')]
        guard = "import importlib.abc,sys,runpy\nclass Guard(importlib.abc.MetaPathFinder):\n def find_spec(self,fullname,path=None,target=None):\n  if fullname.split('.')[0] in ('torch','numpy','scipy','SimpleITK'): raise RuntimeError('heavy import forbidden')\nsys.meta_path.insert(0,Guard())\nsys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n"
        result = subprocess.run([sys.executable, '-c', guard] + command[1:], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        assert plan['state'] == 'not_executed' and not (temporary / 'must_not_exist').exists()
        subprocess.run([sys.executable, '-c', guard, str(target), '--help'], capture_output=True, text=True, check=True)
        passed.append('default_plan_help_no_heavy_import_missing_inputs_unopened_and_no_output_write')

    library = repository / 'tools/attribute_patch_diagnostics.py'
    assert helper.file_hash(library) == helper.OLD_HELPER_SHA256
    raw = library.read_bytes().splitlines(keepends=True)
    tree = ast.parse(library.read_text(encoding='utf-8'))
    function_hashes = {node.name: hashlib.sha256(b''.join(raw[node.lineno - 1:node.end_lineno])).hexdigest()
        for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper.PURE_FUNCTIONS}
    assert set(function_hashes) == set(helper.PURE_FUNCTIONS)
    old = helper.load_public_helpers(repository)
    assert all(callable(getattr(old, name)) for name in helper.PURE_FUNCTIONS)
    assert 'torch' not in sys.modules
    passed.append('eight_original_function_bytes_and_public_dependencies_pinned_no_old_admission_called')
    bank = dict(case_ids=fixed['val'], patches=[])
    predictions = []
    for case_index, case_id in enumerate(fixed['val']):
        for sample_index in range(3):
            q = sample_index / 2.
            scale, structure = [0., q, 1. - q, 0., 0.], [0., 1. - q, q]
            patch = dict(patch_id=case_id + '_' + str(sample_index), case_id=case_id,
                strata=list(old.COUNTS), scale_valid=True, structure_valid=case_index > 0,
                target_scale=scale, target_structure=structure, GT_centerline_support=10)
            bank['patches'].append(patch)
            predictions.append(dict(patch_id=patch['patch_id'], case_id=case_id, strata=patch['strata'],
                classifier_probabilities=dict(scale=scale, structure=structure),
                conditional_compositions_consumed=dict(scale=scale, structure=structure)))
    scores, aggregates = old.score_bank(predictions, bank, 'faithful_soft')
    for item in aggregates:
        assert item['composition_MAE']['mean_patients'] == 0.
        assert item['patients_with_valid_support'] == (8 if item['head'] == 'scale' else 7)
    without = copy.deepcopy(predictions)
    for item in without:
        item['conditional_compositions_consumed'] = None
    _, aggregates = old.score_bank(without, bank, 'faithful_concat')
    assert all(item['patients_with_valid_support'] == 0 and item['composition_MAE']['mean_patients'] is None
               for item in aggregates if item['prediction_kind'] == 'conditional_compositions_consumed')
    constant = copy.deepcopy(predictions)
    for item in constant:
        item['conditional_compositions_consumed'] = dict(scale=[.2] * 5, structure=[1. / 3] * 3)
    _, aggregates = old.score_bank(constant, bank, 'faithful_soft_constant')
    assert any(item['composition_MAE']['mean_patients'] > 0 for item in aggregates
               if item['prediction_kind'] == 'conditional_compositions_consumed')
    assert all(item['composition_MAE']['mean_patients'] == 0 for item in aggregates
               if item['prediction_kind'] == 'classifier_probabilities')
    passed.append('synthetic_independent_head_support_C_null_A_predictions_constant_uniform_separation')
    require_source = target.read_text(encoding='utf-8')
    assert all(text not in require_source for text in ('minimal4_protocol_support', 'formal_transfer_verified',
                                                      'connect.westc', '/root/', 'ssh '))
    assert 'test' not in [option.lstrip('-') for node in ast.walk(ast.parse(require_source))
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                         and node.func.attr == 'add_argument' for value in node.args
                         if isinstance(value, ast.Constant) and isinstance(value.value, str) for option in [value.value]]
    passed.append('no_private_host_gate_dependency_or_TEST_CLI')
    report = dict(schema='GTA-AVT-portable-minimal4-classifier-candidate-QA-v1', state='passed', categories=passed,
        category_count=len(passed), candidate_sha256=helper.file_hash(target),
        original_function_library_sha256=helper.file_hash(library), original_function_sha256=function_hashes,
        frozen_source_canonical_sha256=helper.canonical(source), split_sha256=helper.SPLIT_SHA256,
        real_AVT_arrays_or_models_read=False, tensor_or_checkpoint_load=False, torch_imported='torch' in sys.modules,
        real_forward_or_server_operation=False, synthetic_scores_only=True,
        role_identity_basis=helper.ROLE_ASSIGNMENT)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise ValueError('QA output already exists; choose a new receipt path')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
