"""Temporary metadata-only V2 attribute input fixtures; never load a model."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import attribute_patch_diagnostics as diagnostic


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def fixture(method, source, runtime):
    fixed = diagnostic.frozen_patient_ids()
    recipe = dict(diagnostic.V2_COMMON_RECIPE, method=method)
    record = dict(recipe=recipe, train_ids=fixed['train'], validation_ids=fixed['val'],
                  source_sha256=source, selection_valid=True, test_used=False)
    complete = dict(state='complete', steps=10000, stage='formal', best_val_dice=.7, test_used=False)
    provenance = dict(schema='GTA-AVT-reference-weights-v1', method=method, stage='formal',
        recipe=recipe, original_run_record=record, original_complete_record=complete,
        original_source_sha256=source, frozen_patient_ids=fixed, final_test_eligible=True,
        selected_step=10000, selected_validation_dice=.7, original_checkpoint_name='best.pt',
        original_checkpoint_sha256='a' * 64, original_run_config_sha256='b' * 64,
        original_complete_sha256='c' * 64, original_selected_validation_summary_sha256='d' * 64,
        training_environment_receipt={'sha256': 'e' * 64, 'value': runtime},
        weights_file='weights_tensor_only.pt')
    context = dict(protocol_version=diagnostic.TRAINING_VERSION, server_role='main',
        job=diagnostic.MAIN_JOBS[method], method=method, seed=diagnostic.SEED, recipe=recipe,
        run='/synthetic-project/runs/' + diagnostic.TRAINING_VERSION + '/formal/main/' + diagnostic.MAIN_JOBS[method] + '_seed20261022',
        run_record=record, source_sha256=source, training_environment_sha256='e' * 64,
        v2_authorization_sha256='1' * 64, v2_start_gates_sha256='2' * 64,
        bounded_attestation_sha256='3' * 64,
        attestation=dict(state='verified_complete', protocol_version=diagnostic.TRAINING_VERSION,
            steps=10000, best_sha256='a' * 64, config_sha256='b' * 64, complete_sha256='c' * 64,
            source_sha256=source, test_used=False),
        checkpoint_audits={'latest': {'step': 10000}, 'best': {'step': 10000}},
        validations={'5000': {'summary_sha256': 'f' * 64}, '10000': {'summary_sha256': 'd' * 64}})
    transfer = dict(schema='GTA-AVT-completed-formal-export-v2',
        state='verified_export_and_shared_transfer', context=context, cpu_only=True,
        inference_or_test_run=False, original_training_artifacts_modified=False)
    return dict(provenance=provenance, transfer=transfer)


def seal(directory, value):
    """Create a full file-hash envelope; the weight bytes are explicitly fake."""
    directory.mkdir(exist_ok=True)
    weight_bytes = b'SYNTHETIC_METADATA_FIXTURE_NOT_A_MODEL_OR_TENSOR_FILE\n'
    provenance = value['provenance']
    provenance['weights_sha256'] = hashlib.sha256(weight_bytes).hexdigest()
    provenance_bytes = encode(provenance)
    verified = dict(state='verified', weights_sha256=provenance['weights_sha256'],
        provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
        all56_cache_and_raw_bytes_verified=True, test_predictions_or_metrics_used=False)
    contents = dict(zip(diagnostic.BUNDLE_FILES, (weight_bytes, provenance_bytes, encode(verified))))
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()}
    value['transfer']['bundle_sha256'] = hashes
    for name, raw in contents.items():
        (directory / name).write_bytes(raw)
    (directory / 'formal_transfer_verified.json').write_bytes(encode(value['transfer']))
    return provenance, verified, value['transfer'], hashes


def main():
    source = diagnostic.source_hashes()
    runtime = dict(python='synthetic Python3.8 metadata fixture', python_version=[3, 8, 10],
                   packages=copy.deepcopy(diagnostic.V2_PACKAGES))
    checks = []
    with tempfile.TemporaryDirectory(prefix='avt_attribute_v2_metadata_fixture_') as temp:
        directory = Path(temp)
        for method in diagnostic.MAIN_JOBS:
            for step in (5000, 10000):
                value = fixture(method, source, runtime)
                value['provenance']['selected_step'] = step
                value['provenance']['original_selected_validation_summary_sha256'] = 'f' * 64 if step == 5000 else 'd' * 64
                value['transfer']['context']['checkpoint_audits']['best']['step'] = step
                args = seal(directory, value)
                diagnostic.validate_v2_bundle_metadata(*args, method, runtime, source)
        checks.append('all3_MAIN_methods_genuine_completed10000_best5000_or10000_metadata_accepted')
        original = fixture('faithful_soft', source, runtime)
        failures = ('genuine_shaped_same_recipe_same_runtime_bridge', 'MAIN_labels_with_bridge_runpath', 'wrong_MAIN_job', 'V1_transfer',
            'V1_recipe96_5000', 'recipe_seed', 'recipe_JSON_type', 'extra_recipe_field',
            'missing_TRAIN_patient', 'partial_completion', 'nonvalidation_selected_step',
            'wrong_source', 'wrong_selected_summary', 'missing_authorization',
            'wrong_runtime_package', 'test_use', 'nonboolean_eligibility')
        for name in failures:
            value = copy.deepcopy(original)
            p, t = value['provenance'], value['transfer']
            current = copy.deepcopy(runtime)
            if name == 'genuine_shaped_same_recipe_same_runtime_bridge':
                t['context'].update(server_role='ablations', job='A_soft_platform_bridge')
                t['context']['run'] = '/synthetic-project/runs/' + diagnostic.TRAINING_VERSION + '/formal/ablations/A_soft_platform_bridge_seed20261022'
                # Every other recipe/source/runtime/completion field remains
                # valid and identical to the MAIN A fixture just accepted.
            if name == 'wrong_MAIN_job': t['context']['job'] = 'C_concat'
            if name == 'MAIN_labels_with_bridge_runpath':
                t['context']['run'] = '/synthetic-project/runs/' + diagnostic.TRAINING_VERSION + '/formal/ablations/A_soft_platform_bridge_seed20261022'
            if name == 'V1_transfer': t['schema'] = 'GTA-AVT-completed-formal-export-v1'
            if name == 'V1_recipe96_5000': p['recipe'].update(patch_size=96, max_steps=5000)
            if name == 'recipe_seed': p['recipe']['seed'] += 1
            if name == 'recipe_JSON_type': p['recipe']['gradient_clip'] = 12
            if name == 'extra_recipe_field': p['recipe']['unknown_extra'] = True
            if name == 'missing_TRAIN_patient': p['original_run_record']['train_ids'].pop()
            if name == 'partial_completion': p['original_complete_record']['state'] = 'running'
            if name == 'nonvalidation_selected_step': p['selected_step'] = 9999
            if name == 'wrong_source': p['original_source_sha256']['common.py'] = '0' * 64
            if name == 'wrong_selected_summary': p['original_selected_validation_summary_sha256'] = '0' * 64
            if name == 'missing_authorization': t['context'].pop('v2_authorization_sha256')
            if name == 'wrong_runtime_package': current['packages']['scipy'] = 'wrong-version'
            if name == 'test_use': p['original_run_record']['test_used'] = True
            if name == 'nonboolean_eligibility': p['final_test_eligible'] = 'true'
            args = seal(directory, value)
            try:
                diagnostic.validate_v2_bundle_metadata(*args, 'faithful_soft', current, source)
            except ValueError as error:
                if name == 'genuine_shaped_same_recipe_same_runtime_bridge':
                    assert 'MAIN role/job' in str(error)
            else:
                raise AssertionError('Invalid metadata accepted: ' + name)
        checks.extend(name + '_rejected' for name in failures)
        # Exercise the actual file-hash ingestion path without importing Torch
        # or treating the explicitly fake weight bytes as serialized tensors.
        old_loader = diagnostic.load_tensor_bundle
        diagnostic.load_tensor_bundle = lambda path: (diagnostic.load_json(Path(path) / 'provenance.json'), {})
        try:
            seal(directory, copy.deepcopy(original))
            diagnostic.validate_formal_bundle(directory, 'faithful_soft', runtime)
            for filename in diagnostic.BUNDLE_FILES:
                seal(directory, copy.deepcopy(original))
                with (directory / filename).open('ab') as stream:
                    stream.write(b' \n')  # JSON remains valid; actual bytes differ.
                try:
                    diagnostic.validate_formal_bundle(directory, 'faithful_soft', runtime)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Changed actual bundle bytes accepted: ' + filename)
            checks.append('all3_actual_bundle_file_SHA_corruptions_rejected_with_metadata_only_loader')
        finally:
            diagnostic.load_tensor_bundle = old_loader
    helper = Path(diagnostic.__file__).resolve()
    command = [sys.executable, '-c',
        "import runpy,sys;sys.path.insert(0,%r);sys.argv=['attribute_patch_diagnostics.py','--bundle-c','NOT_OPENED','--bundle-a','NOT_OPENED','--bundle-constant','NOT_OPENED','--manifest','NOT_OPENED','--output','NOT_CREATED'];runpy.run_path(%r,run_name='__main__');assert 'torch' not in sys.modules" % (str(helper.parent), str(helper))]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['state'] == 'not_executed'
    checks.append('default_plan_only_no_Torch_no_model_or_data_paths_opened')
    for path in (Path(__file__), Path(diagnostic.__file__)):
        ast.parse(path.read_text(), feature_version=(3, 8))
    assert 'torch' not in sys.modules
    print(json.dumps(dict(state='passed', checks=checks, Python38_syntax=True,
        fixture_scope='Full-shaped metadata and actual file hashes only; tensor loader mocked; not a serialization/model test.',
        Torch_imported=False, GPU_or_real_data_or_model_run=False, synthetic_values_retained=False)))


if __name__ == '__main__':
    main()
