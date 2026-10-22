"""Original-operator lineage and graph/AMP checks without AVT test access."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import atomic_json, load_json, sha256, source_hashes
from model import GTAAVT
from faithful_model import radius_to_historical_generation, junction_to_historical_structure, foreground_logit
from historical.func.AnatomicalFusionModules import GenerationGuidanceModule, StructureGuidanceModule
from losses import objective
from smoke import native_fixture


def verify_lineage():
    root = Path(__file__).parent / 'historical'
    manifest = load_json(root / 'source_hashes.json')
    matches = {entry['file']: sha256(root / entry['file']) == entry['sha256'] for entry in manifest}
    if not all(matches.values()):
        raise AssertionError('Historical source changed: ' + str(matches))
    return matches


def operator_fixture(device):
    torch.manual_seed(4401)
    generation = GenerationGuidanceModule(8, 8).to(device).eval()
    structure = StructureGuidanceModule(8, 8).to(device).eval()
    feature = torch.randn(2, 8, 6, 6, 6, device=device)
    attribute = torch.randn_like(feature)
    ordinary = torch.ones(2, device=device, dtype=torch.long)
    fine = torch.full_like(ordinary, 5)
    junction = torch.full_like(ordinary, 3)
    confident5 = torch.tensor([[.05, .8, .05, .05, .05]] * 2, device=device)
    confident3 = torch.tensor([[.1, .1, .8]] * 2, device=device)
    with torch.no_grad():
        ordinary_generation = generation(feature, attribute, ordinary, confident5)
        fine_generation = generation(feature, attribute, fine, confident5)
        ordinary_structure = structure(feature, attribute, ordinary, confident3)
        junction_structure = structure(feature, attribute, junction, confident3)
        uncertain_structure = structure(feature, attribute, junction, torch.full_like(confident3, 1. / 3))
    differences = {'GAAM_fine_class_operator_response': float((fine_generation - ordinary_generation).abs().mean()),
                   'SAAM_junction_class_operator_response': float((junction_structure - ordinary_structure).abs().mean()),
                   'SAAM_max_mixture_mass_operator_response': float((junction_structure - uncertain_structure).abs().mean())}
    assert all(value > 0. for value in differences.values()), differences
    classes = torch.arange(5, device=device)
    assert radius_to_historical_generation(classes).tolist() == [0, 5, 1, 2, 3]
    assert junction_to_historical_structure(torch.arange(3, device=device)).tolist() == [0, 1, 3]
    logits = torch.randn(2, 2, 5, 5, 5, device=device)
    assert torch.allclose(foreground_logit(logits).sigmoid(), logits.softmax(1)[:, 1:2], atol=1e-6)
    return differences


def model_fixture(method, args, device):
    torch.manual_seed(20261022)
    model = GTAAVT(method, 32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    p, b = args.patch_size, args.batch_size
    image = torch.randn(b, 1, p, p, p, device=device)
    z = torch.arange(p, device=device) - (p - 1.) / 2.
    tube = (z[None, :, None] ** 2 + z[None, None, :] ** 2 < (p / 8.) ** 2).float()
    mask = tube[None].expand(b, 1, p, p, p)
    target = {'image': image, 'mask': mask,
              'scale': torch.tensor([[0., .1, .3, .4, .2]] * b, device=device),
              'structure': torch.tensor([[0., .7, .3]] * b, device=device),
              'attribute_valid': torch.ones(b, device=device)}
    times = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            output = model(image)
            loss, details = objective(output, target)
        assert torch.isfinite(loss), details
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
        assert torch.isfinite(norm)
        scaler.step(optimizer)
        scaler.update()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - started)
        print(json.dumps({'method': method, 'step': step + 1, 'seconds': times[-1], 'loss': float(loss)}), flush=True)
    active_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.grad is not None)
    result = {'finite_AMP_optimizer': 'passed', 'step_seconds': times,
              'steady_step_seconds': float(np.median(times[1:] or times)),
              'registered_parameters': sum(parameter.numel() for parameter in model.parameters()),
              'parameters_with_grad_on_fixture': active_parameters,
              'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else None}
    if model.use_prior:
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        # Full float32 derivative probe separates graph connectivity from AMP
        # representability; actual AMP optimizer finiteness is checked above.
        output = model(image)
        objective_probe = output['logits'].float().square().mean()
        parameters = (output['scale_logits'], output['structure_logits'], model.classifier.encoder[0].conv.weight)
        derivatives = torch.autograd.grad(objective_probe, parameters, allow_unused=True)
        sizes = [float(x.abs().sum()) if x is not None else None for x in derivatives]
        assert sizes[2] is not None and sizes[2] > 0., sizes
        if method.startswith('faithful_soft'):
            expected = [method not in ('faithful_soft_structure', 'faithful_soft_constant'),
                        method not in ('faithful_soft_scale', 'faithful_soft_constant')]
            for index, active in enumerate(expected):
                assert (sizes[index] is not None and sizes[index] > 0.) if active else derivatives[index] is None, sizes
            result['soft_segmentation_to_attribute_logits'] = {'passed': True, 'active_scale_structure': expected}
        else:
            assert derivatives[0] is None and derivatives[1] is None, sizes
            result['historical_detached_discrete_logits'] = 'passed_expected_no_segmentation_gradient'
        result['segmentation_derivatives_scale_structure_encoder'] = sizes
        with torch.no_grad():
            ordinary = model(image)['logits']
            constant = model(image, condition_mode='constant')['logits']
        result['predicted_vs_constant_mean_logit_difference'] = float((ordinary - constant).abs().mean())
        if method in ('faithful_joint', 'faithful_soft'):
            assert result['predicted_vs_constant_mean_logit_difference'] > 0.
    del output, loss, model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--methods', nargs='+', default=['faithful_baseline', 'faithful_concat', 'faithful_joint', 'faithful_soft', 'faithful_soft_constant'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    results = {'historical_lineage': verify_lineage(), 'native_metric_fixtures': native_fixture(),
               'torch_version': torch.__version__, 'patch_size': args.patch_size,
               'batch_size': args.batch_size, 'source_sha256': source_hashes()}
    reference, common_heads, conditioned_reference = None, None, None
    for method in ('faithful_baseline', 'faithful_concat', 'faithful_joint', 'faithful_soft'):
        torch.manual_seed(20261022)
        model = GTAAVT(method, 32)
        state = model.base_model.state_dict()
        if reference is None:
            reference = {key: value.clone() for key, value in state.items()}
            common_heads = {key: value.clone() for key, value in model.state_dict().items()
                            if key.startswith('ds_heads.') or key.startswith('anatomical_output_fusion.3.')}
        else:
            assert all(torch.equal(value, reference[key]) for key, value in state.items())
            assert all(torch.equal(model.state_dict()[key], value) for key, value in common_heads.items())
        if method != 'faithful_baseline':
            if conditioned_reference is None:
                conditioned_reference = {key: value.clone() for key, value in model.state_dict().items()}
            else:
                assert all(torch.equal(value, conditioned_reference[key]) for key, value in model.state_dict().items())
        del model
    torch.manual_seed(20261022)
    model = GTAAVT('faithful_soft_constant', 32)
    assert all(torch.equal(value, conditioned_reference[key]) for key, value in model.state_dict().items())
    del model, reference, common_heads, conditioned_reference
    results['original_backbone_paired_initialization_B_C_hardA_softA'] = 'passed'
    results['C_A_classifier_common_heads_and_entire_registered_state_paired'] = 'passed'
    results['B_C_A_deep_supervision_and_final_output_projection_paired'] = 'passed'
    results['soft_A_and_soft_constant_entire_state_paired'] = 'passed'
    results['original_conditional_operators'] = operator_fixture(device)
    atomic_json(args.output, results)
    results['models'] = {}
    for method in args.methods:
        results['models'][method] = model_fixture(method, args, device)
        atomic_json(args.output, results)
    results['state'] = 'passed'
    atomic_json(args.output, results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
