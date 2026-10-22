"""Export completed owned-run best.pt as CPU tensor-only protocol2 weights.

Run this in the original training software environment. It reads the genuine
checkpoint; it never creates optimizer/RNG fields or a training-checkpoint shim.
"""
import argparse
from collections import OrderedDict
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from reference_weights_support import (
    SCHEMA, atomic_json, load_json, file_hash, source_hashes, canonical_cache_identity,
    runtime_environment, verify_training_environment_receipt, tensor_inventory,
    require_new_directory, load_tensor_bundle, utc, FIXED_INFERENCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--manifest', required=True, help='Exact original training manifest, before any path rebase')
    parser.add_argument('--training-environment-json', required=True, help='Actual training-runtime package/Python receipt')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from common import load_trusted_checkpoint
    run = Path(args.run)
    record = load_json(run / 'config.json')
    recipe = record['recipe']
    if source_hashes() != record['source_sha256']:
        raise ValueError('Current code does not match genuine training source')
    if file_hash(args.manifest) != record['manifest_sha256']:
        raise ValueError('Exporter requires the exact original training manifest SHA256')
    if record.get('test_used') is not False:
        raise ValueError('Training signature must explicitly state no test use')
    complete = load_json(run / 'complete.json')
    if complete.get('state') != 'complete' or complete.get('steps') != recipe['max_steps'] or complete.get('stage') != recipe['stage']:
        raise ValueError('Selected best must come from completed genuine training')
    if complete.get('test_used') is not False:
        raise ValueError('Completed training record indicates test use')
    for name, expected in [('inference_patch_size', 128), ('inference_batch_size', 2), ('inference_overlap', .5)]:
        if recipe.get(name) != expected:
            raise ValueError('Training recipe differs from fixed reference inference: ' + name)
    checkpoint_path = run / 'best.pt'
    checkpoint = load_trusted_checkpoint(checkpoint_path, 'cpu')
    if checkpoint.get('signature') != record:
        raise ValueError('Genuine checkpoint signature differs from its run config')
    step = checkpoint.get('step')
    if not isinstance(step, int) or not 0 < step <= recipe['max_steps']:
        raise ValueError('Invalid selected checkpoint step')
    selected_summary = load_json(run / 'validation' / ('step_%06d' % step) / 'summary.json')
    if checkpoint.get('last_validation') != selected_summary:
        raise ValueError('Best checkpoint does not match its actual validation summary')
    selected_dice = selected_summary['metrics']['dice']['mean']
    if selected_dice is None or selected_dice != checkpoint.get('best_val_dice') or selected_dice != complete.get('best_val_dice'):
        raise ValueError('best.pt is not the completed run\'s genuine selected validation best')
    identity = canonical_cache_identity(args.manifest, verify_bytes=True)
    frozen = identity['canonical']['splits']
    if record['train_ids'] != frozen['train'] or not set(record['validation_ids']).issubset(frozen['val']):
        raise ValueError('Run patient IDs do not match the frozen cache')
    if record.get('selection_valid') and (recipe['stage'] != 'formal' or record['validation_ids'] != frozen['val']):
        raise ValueError('Incorrect final-test eligibility in original run')
    runtime = runtime_environment()
    training_receipt = verify_training_environment_receipt(args.training_environment_json, runtime)
    state = checkpoint['model']
    inventory = tensor_inventory(state)
    cpu_state = OrderedDict((name, tensor.detach().cpu().contiguous().clone()) for name, tensor in state.items())
    if tensor_inventory(cpu_state) != inventory:
        raise ValueError('CPU conversion changed tensor identity')
    output = require_new_directory(args.output)
    weights = output / 'weights_tensor_only.pt'
    temporary = output / 'weights_tensor_only.pt.partial'
    torch.save(cpu_state, str(temporary), pickle_protocol=2, _use_new_zipfile_serialization=False)
    temporary.replace(weights)
    provenance = dict(schema=SCHEMA, created_utc=utc(), method=recipe['method'], stage=recipe['stage'],
        selected_step=step, selected_validation_dice=selected_dice,
        original_checkpoint_sha256=file_hash(checkpoint_path), original_checkpoint_name='best.pt',
        original_run_record=record, original_run_config_sha256=file_hash(run / 'config.json'),
        original_complete_record=complete, original_complete_sha256=file_hash(run / 'complete.json'),
        original_selected_validation_summary_sha256=file_hash(run / 'validation' / ('step_%06d' % step) / 'summary.json'),
        original_source_sha256=record['source_sha256'], recipe=recipe, frozen_patient_ids=frozen,
        final_test_eligible=bool(record['selection_valid'] and recipe['stage'] == 'formal'),
        canonical_data_identity=identity, training_environment_receipt=training_receipt, export_environment=runtime,
        weights_file=weights.name, weights_sha256=file_hash(weights), tensor_inventory=inventory,
        serialization=dict(content='CPU tensor-only OrderedDict', pickle_protocol=2, legacy_serialization=True,
                           optimizer_included=False, rng_included=False, fabricated_training_checkpoint=False),
        fixed_reference_inference=FIXED_INFERENCE,
        export_tool_sha256={Path(__file__).name: file_hash(__file__),
                           'reference_weights_support.py': file_hash(Path(__file__).with_name('reference_weights_support.py'))})
    atomic_json(output / 'provenance.json', provenance)
    loaded, reloaded = load_tensor_bundle(output)
    if tensor_inventory(reloaded) != inventory:
        raise ValueError('Protocol2 serialization roundtrip changed tensor identity')
    atomic_json(output / 'export_verified.json', dict(state='verified', utc=utc(),
        weights_sha256=provenance['weights_sha256'], provenance_sha256=file_hash(output / 'provenance.json'),
        tensor_count=len(inventory), all56_cache_and_raw_bytes_verified=True,
        all56_file_bytes_read_for_identity_only=True, test_predictions_or_metrics_used=False))
    print({'exported': str(output), 'method': recipe['method'], 'selected_step': step,
           'stage': recipe['stage'], 'tensor_count': len(inventory), 'test_predictions_or_metrics_used': False}, flush=True)


if __name__ == '__main__':
    main()
