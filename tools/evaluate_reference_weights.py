"""Evaluate tensor-only exports on Python3.8/torch1.11 using frozen native metrics.

Validation is explicit. Test additionally requires a genuine completed eligible
formal model AND --allow-formal-test. No automatic held-out evaluation exists.
"""
import argparse
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from reference_weights_support import (
    atomic_json, load_json, file_hash, source_hashes, canonical_cache_identity, assert_same_data,
    runtime_environment, require_reference_runtime, configure_inference_runtime,
    load_tensor_bundle, load_strict_model, evaluation_cases, FIXED_INFERENCE, selected_execution_device,
    canonical_digest)


def training_recipe_fields(provenance):
    """Expose the genuine export's full recipe without inferring it from a path."""
    recipe = provenance.get('recipe')
    run_record = provenance.get('original_run_record')
    if (not isinstance(recipe, dict) or not recipe or
            not isinstance(run_record, dict) or provenance.get('stage') not in ('pilot', 'formal') or
            recipe.get('method') != provenance.get('method') or
            recipe.get('stage') != provenance.get('stage') or
            run_record.get('recipe') != recipe):
        raise ValueError('Export training recipe differs from its genuine run/method/stage provenance')
    recipe_sha = canonical_digest(recipe)
    if canonical_digest(run_record['recipe']) != recipe_sha:
        raise ValueError('Export training recipe and genuine run recipe have different canonical JSON identities')
    return dict(training_recipe=recipe, training_recipe_sha256=recipe_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--manifest', required=True, help='Complete reference cache manifest; paths may differ, bytes may not')
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--allow-formal-test', action='store_true')
    parser.add_argument('--intervention', choices=('none', 'availability_only'), default='none')
    parser.add_argument('--reuse', action='store_true', help='Resume only the identical immutable evaluation protocol')
    args = parser.parse_args()
    import torch
    from data import CachedData
    from inference import evaluate_cases
    require_reference_runtime()
    configure_inference_runtime()
    provenance, state = load_tensor_bundle(args.bundle)
    recipe_fields = training_recipe_fields(provenance)
    case_ids = evaluation_cases(provenance, args.split, args.allow_formal_test, args.intervention)
    if args.intervention == 'availability_only' and provenance['method'] != 'faithful_soft':
        raise ValueError('availability_only requires a trained faithful_soft model')
    identity = canonical_cache_identity(args.manifest, verify_bytes=True)
    assert_same_data(provenance, identity)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    torch.set_num_threads(provenance['recipe'].get('cpu_threads', 4))
    model = load_strict_model(provenance, state, device)
    intervention = dict(name='none', separately_trained_model=False, changes_weights=False)
    if args.intervention == 'availability_only':
        from reference_interventions import attach_availability_only
        intervention = attach_availability_only(model, provenance['method'], args.split)
    runtime = runtime_environment()
    signature = dict(schema='GTA-AVT-reference-evaluation-v1', split=args.split, case_ids=case_ids,
        method=provenance['method'], stage=provenance['stage'], selected_step=provenance['selected_step'],
        original_checkpoint_sha256=provenance['original_checkpoint_sha256'],
        weights_sha256=provenance['weights_sha256'], provenance_sha256=file_hash(Path(args.bundle) / 'provenance.json'),
        frozen_source_sha256=source_hashes(), canonical_data_sha256=identity['canonical_sha256'],
        final_test_eligible=provenance['final_test_eligible'], explicitly_authorized_formal_test=bool(args.allow_formal_test),
        inference=FIXED_INFERENCE, intervention=intervention,
        selected_execution_device=selected_execution_device(device), autocast_enabled=device.type == 'cuda',
        precision='Frozen inference.py: CUDA autocast enabled on CUDA, disabled on CPU; TF32 disabled in both backends',
        training_environment_receipt=provenance['training_environment_receipt'], reference_environment=runtime,
        **recipe_fields,
        evaluator_tool_sha256={Path(__file__).name: file_hash(__file__),
                              'reference_weights_support.py': file_hash(Path(__file__).with_name('reference_weights_support.py'))})
    output = Path(args.output)
    protocol = output / 'evaluation_protocol.json'
    if protocol.exists():
        if not args.reuse or load_json(protocol) != signature:
            raise ValueError('Existing output is immutable; --reuse requires an identical evaluation protocol')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty output has no matching evaluation protocol')
    atomic_json(protocol, signature)
    data = CachedData(args.manifest)
    evaluate_cases(model, data, case_ids, device, output, patch=128, batch_size=2, overlap=.5,
                   condition_mode='predicted', reuse=args.reuse)
    print({'evaluated': args.split, 'cases': len(case_ids), 'reference': 'Python3.8/torch1.11',
           'intervention': args.intervention, 'test_used': args.split == 'test'}, flush=True)


if __name__ == '__main__':
    main()
