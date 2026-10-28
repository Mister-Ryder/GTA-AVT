"""Explicit post-condition airway-style Gaussian revision; never starts on import.

The one common head-only fit is shared by two matched full-S suffixes. A legacy
checkpoint may enter only through the strict, recorded condition migration below.
There is no automatic selection, seed search, TEST use or fallback recipe.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import engine
from .data import load_manifest, PreparedDataset
from .models import AttributePredictor, GTAAVT, MODEL_CONFIG
from .inference import infer, evaluate

PLAN_SCHEMA = 'GTA-AVT-v1.2-gdf-joint-plan-1'
LEGACY_SCHEMA = 'GTA-AVT-v1.2-checkpoint-1'
REVISION = '20261028-gdf-joint-r3'
ARMS = ('AR', 'AR_GDF')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_plan(config):
    value = copy.deepcopy(config)
    if value.get('schema') != PLAN_SCHEMA:
        raise ValueError('Explicit immutable GDF-joint plan schema required')
    for key in ('manifest', 'output_dir', 'condition_checkpoint', 'condition_checkpoint_sha256',
                'selected_base_sha256', 'legacy_source_manifest_sha256', 'evaluation_cases'):
        if not value.get(key):
            raise ValueError('Missing plan field ' + key)
    for key in ('condition_checkpoint_sha256', 'selected_base_sha256', 'legacy_source_manifest_sha256'):
        if len(value[key]) != 64 or any(c not in '0123456789abcdef' for c in value[key]):
            raise ValueError('Expected lowercase SHA256 for ' + key)
    value.setdefault('seed', 20261032)
    value.setdefault('device', 'cuda')
    value.setdefault('num_workers', 4)
    value.setdefault('calibration_batches', 8)
    if value['calibration_batches'] != 8:
        raise ValueError('This fixed revision uses exactly eight TRAIN B2 calibration batches')
    if value.get('budgets', [2000, 20000, 10000]) != [2000, 20000, 10000]:
        raise ValueError('Fixed head/geometry/refine budgets are 2k/20k/10k')
    return value


def _plan_signature(plan):
    return engine._digest(plan)


def _home(plan):
    return Path(plan['output_dir']).resolve()


def _common(plan):
    return dict(manifest=plan['manifest'], seed=plan['seed'], validation_seed=0,
        patch_size=128, batch_size=2, samples_per_epoch=256, val_samples=64,
        num_workers=plan['num_workers'], device=plan['device'], amp=True,
        checkpoint_every=1000, validate_every=1000, deterministic=False,
        dice_weight=1., bce_weight=1., use_gdf=True, bce_class_balance='airway_ratio',
        deep_supervision_weight=0., require_attribute_readiness=True)


def _recipes():
    phases = _read(Path(__file__).resolve().parents[1] / 'configs' / 'phases.json')
    if [phases[name]['max_steps'] for name in ('gdf_head', 'geometry', 'refine')] != [2000, 20000, 10000]:
        raise ValueError('Packaged suffix budget mismatch')
    return phases


def _validated_evaluation_cases(plan):
    cases = _read(plan['evaluation_cases'])
    cases = cases['cases'] if isinstance(cases, dict) else cases
    expected = {row['id'] for row in load_manifest(plan['manifest'])['cases'] if str(row['split']).upper() == 'VAL'}
    actual = [row['id'] for row in cases]
    if (len(cases) != 8 or len(set(actual)) != 8 or set(actual) != expected
            or any(str(row.get('split', '')).upper() != 'VAL' for row in cases)):
        raise ValueError('Native evaluation must match exactly the eight overlay VAL identifiers; never TEST')
    if any(not row.get('image') or not row.get('label') for row in cases):
        raise ValueError('Explicit native CT and label references required for each VAL case')
    return cases


def migrate_condition(plan, destination):
    """Copy all and only compatible P/S tensors; replace incompatible old head.

    Trusted project checkpoints only: torch's legacy format includes Python RNG
    and optimizer objects. No optimizer or old Gaussian weight is imported.
    """
    plan = validate_plan(plan)
    path = Path(plan['condition_checkpoint'])
    if engine.file_digest(path) != plan['condition_checkpoint_sha256']:
        raise ValueError('Condition checkpoint SHA256 does not match the pinned source')
    try:
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location='cpu')
    old, pair = checkpoint.get('config', {}), checkpoint.get('pair', {})
    if (checkpoint.get('schema') != LEGACY_SCHEMA or checkpoint.get('kind') != 'train'
            or checkpoint.get('model_config', {}).get('revision') != '20261027-ratio-r2'
            or old.get('phase') != 'condition' or old.get('stage') != 'local_adapt'
            or old.get('arm') != 'AR' or old.get('seed') != plan['seed']
            or old.get('patch_size') != 128 or old.get('max_steps') != 10000
            or checkpoint.get('step') != 10000
            or pair.get('selected_base_sha256') != plan['selected_base_sha256']
            or not pair.get('attribute_readiness_passed')):
        raise ValueError('Only the complete, ready, clean ratio-r2 AR condition10k parent is accepted')
    if engine._tensor_digest(checkpoint['model'].items()) != pair.get('model_state_sha256'):
        raise ValueError('Legacy whole model tensor digest mismatch')
    stats = load_manifest(plan['manifest'])['statistics']
    if engine._digest(stats) != engine._digest(checkpoint['statistics']):
        raise ValueError('Overlay must preserve exactly the old TRAIN attribute statistics')
    engine._seed(plan['seed'])
    model = GTAAVT(AttributePredictor(stats), arm='AR', use_gdf=True)
    original, current = checkpoint['model'], model.state_dict()
    old_head = {key for key in original if key.startswith('gdf_head.')}
    new_head = {key for key in current if key.startswith('gdf_head.')}
    if old_head != {'gdf_head.0.weight', 'gdf_head.0.bias'} or new_head != old_head:
        raise ValueError('Unexpected Gaussian state names')
    if tuple(original['gdf_head.0.weight'].shape) != (1, 32, 1, 1, 1):
        raise ValueError('Legacy Gaussian head shape differs from the frozen control')
    if set(original) - old_head != set(current) - new_head:
        raise ValueError('P/S architecture keys changed; no partial silent import allowed')
    for key in set(current) - new_head:
        if current[key].shape != original[key].shape or current[key].dtype != original[key].dtype:
            raise ValueError('P/S tensor shape or dtype mismatch: ' + key)
        current[key] = original[key]
    model.load_state_dict(current, strict=True)
    if engine.state_digest(model.predictor) != pair['predictor_state_sha256']:
        raise ValueError('Frozen predictor tensor digest mismatch after migration')
    config = dict(old, manifest=plan['manifest'])
    pair = dict(pair, model_state_sha256=engine.state_digest(model))
    lineage = dict(checkpoint.get('lineage', {}), revision_migration=dict(
        parent_checkpoint_sha256=plan['condition_checkpoint_sha256'],
        legacy_source_manifest_sha256=plan['legacy_source_manifest_sha256'],
        reset_keys=sorted(new_head), imported_ps_tensors=len(current) - len(new_head),
        new_head_parameters=55, optimizer_imported=False,
        gaussian_supervision_in_parent=False, seed=plan['seed']))
    migrated = dict(checkpoint, schema=engine.SCHEMA, model_config=dict(MODEL_CONFIG, arm='AR', use_gdf=True),
        model=model.state_dict(), config=config, pair=pair, lineage=lineage)
    for key in ('optimizer', 'scheduler', 'scaler', 'rng', 'progress', 'config_signature'):
        migrated.pop(key, None)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError('Migration is immutable; refuse to overwrite an existing checkpoint')
    engine._save(destination, migrated)
    report = dict(schema='GTA-AVT-gdf-joint-migration-1', checkpoint_sha256=engine.file_digest(destination),
        source_checkpoint_sha256=plan['condition_checkpoint_sha256'],
        selected_base_sha256=plan['selected_base_sha256'], predictor_state_sha256=pair['predictor_state_sha256'],
        new_model_state_sha256=pair['model_state_sha256'], **lineage['revision_migration'])
    engine._json(destination.with_suffix('.json'), report)
    return report


def _gradient_norm(gradients):
    square = sum(gradient.float().square().sum() for gradient in gradients if gradient is not None)
    norm = float(square.sqrt()) if isinstance(square, torch.Tensor) else 0.
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Calibration requires a finite nonzero shared-S gradient')
    return norm


def choose_lambda(records):
    """Match loss scale first, then guard shared-S gradient ratio within [0.5,2]."""
    if len(records) != 8:
        raise ValueError('Exactly eight predetermined TRAIN batches are required')
    for row in records:
        if any(not math.isfinite(float(row[key])) or row[key] <= 0
               for key in ('seg_loss', 'gdf_loss', 'seg_s_grad_norm', 'gdf_s_grad_norm')):
            raise FloatingPointError('Nonfinite or zero calibration term; no automatic rescue')
    loss_ratio = float(np.median([r['seg_loss'] / r['gdf_loss'] for r in records]))
    gradient_ratio = float(np.median([r['seg_s_grad_norm'] / r['gdf_s_grad_norm'] for r in records]))
    selected = min(max(loss_ratio, .5 * gradient_ratio), 2. * gradient_ratio)
    if not math.isfinite(selected) or not 1e-6 <= selected <= 1e6:
        raise FloatingPointError('Calibration weight outside declared numerical safety interval')
    return dict(lambda_peak=selected, loss_matching_candidate=loss_ratio,
        shared_gradient_matching_candidate=gradient_ratio,
        predicted_median_weighted_shared_gradient_ratio=selected / gradient_ratio,
        rule='median loss ratio, clipped to [0.5,2] times median shared-S gradient-norm ratio',
        scope='eight fixed TRAIN batches only; zero optimizer updates; no VAL or TEST selection')


def calibrate(plan, checkpoint_path):
    checkpoint = engine.load_checkpoint(checkpoint_path)
    device = engine._device(plan)
    engine._seed(plan['seed'])
    model = GTAAVT(AttributePredictor(checkpoint['statistics']), arm='AR', use_gdf=True).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.set_stage('full_joint').train()
    before = engine.state_digest(model)
    recipe = _recipes()['geometry']
    dataset = PreparedDataset(plan['manifest'], split='TRAIN', patch_size=128, samples_per_epoch=16,
        seed=plan['seed'], augment=False, sampling=recipe['sampling'])
    calibration_config = dict(_common(plan), **recipe)
    calibration_config.update(stage='full_joint', phase='geometry',
        output_dir=str(_home(plan) / 'unused_calibration_output'), parent_checkpoint=str(checkpoint_path))
    config = engine._parameters(calibration_config)
    parameters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('gdf_head.')]
    records = []
    for index, batch in enumerate(engine._loader(dataset, dict(config, num_workers=0))):
        batch = engine._to_device(batch, device)
        if len(batch['image']) != 2:
            raise ValueError('Calibration uses eight logical B2 batches')
        # All BN statistics are fixed and the declared losses are case equal.
        # Averaging two FP32 B1 gradients reproduces the mathematical B2 loss
        # gradient without requiring a memory-heavy full FP32 B2 graph.
        seg_gradients, gdf_gradients = [None] * len(parameters), [None] * len(parameters)
        seg_values, gdf_values = [], []
        for micro in range(2):
            one = {key: value[micro:micro + 1] if isinstance(value, torch.Tensor) else value
                   for key, value in batch.items()}
            with torch.autocast(device_type=device.type, enabled=False):
                prediction = model(one['image'].float(), alpha=1.)
                losses = engine._loss(prediction, one, config, 'train', gdf_weight=1.)
            seg, gaussian = losses['primary_segmentation_loss'], losses['gdf_mse']
            sg = torch.autograd.grad(seg, parameters, retain_graph=True, allow_unused=True)
            gg = torch.autograd.grad(gaussian, parameters, retain_graph=False, allow_unused=True)
            for destination, gradients in ((seg_gradients, sg), (gdf_gradients, gg)):
                for parameter_index, gradient in enumerate(gradients):
                    if gradient is not None:
                        part = gradient.detach().float() * .5
                        if destination[parameter_index] is None:
                            destination[parameter_index] = part
                        else:
                            destination[parameter_index].add_(part)
            seg_values.append(float(seg.detach())); gdf_values.append(float(gaussian.detach()))
            del prediction, losses, seg, gaussian, sg, gg
        records.append(dict(batch_index=index, case_ids=list(batch['case_id']), seg_loss=float(np.mean(seg_values)),
            gdf_loss=float(np.mean(gdf_values)), seg_s_grad_norm=_gradient_norm(seg_gradients),
            gdf_s_grad_norm=_gradient_norm(gdf_gradients)))
        del seg_gradients, gdf_gradients
    if engine.state_digest(model) != before:
        raise RuntimeError('Calibration must not alter any parameter or normalization buffer')
    return dict(schema='GTA-AVT-gdf-joint-calibration-1', checkpoint_sha256=engine.file_digest(checkpoint_path),
        precision='FP32; logical B2 gradient is the average of two B1 gradients with fixed BN statistics',
        microbatch_size=1, logical_batch_size=2, optimizer_updates=0,
        batches=records, **choose_lambda(records))


def prepare(config):
    plan = validate_plan(config); home = _home(plan)
    _validated_evaluation_cases(plan)
    home.mkdir(parents=True, exist_ok=True)
    receipt = home / 'prepared.json'
    if receipt.is_file():
        result = _read(receipt)
        if result['plan_sha256'] != _plan_signature(plan) or engine.file_digest(result['shared_checkpoint']) != result['shared_checkpoint_sha256']:
            raise ValueError('Existing prepared result belongs to a changed plan or checkpoint')
        return result
    plan_file = home / 'joint_plan.json'
    if plan_file.is_file() and _read(plan_file) != plan:
        raise ValueError('Cannot change an existing GDF-joint plan')
    engine._json(plan_file, plan)
    migrated = home / 'migration' / 'condition.pt'
    if not migrated.is_file():
        migrate_condition(plan, migrated)
    else:
        report = _read(migrated.with_suffix('.json'))
        if report['source_checkpoint_sha256'] != plan['condition_checkpoint_sha256'] or engine.file_digest(migrated) != report['checkpoint_sha256']:
            raise ValueError('Existing migrated checkpoint provenance mismatch')
    recipe = _recipes()['gdf_head']
    cfg = dict(_common(plan), **recipe, phase='gdf_head', arm='AR',
        output_dir=str(home / 'shared_head'), parent_checkpoint=str(migrated), lambda_gdf=1.)
    if plan.get('tensorboard_logdir'):
        cfg['tensorboard_logdir'] = str(Path(plan['tensorboard_logdir']) / 'shared_head')
    summary = Path(cfg['output_dir']) / 'summary.json'
    if summary.is_file():
        result = _read(summary)
    else:
        last = Path(cfg['output_dir']) / 'last.pt'
        if last.is_file(): cfg['resume'] = str(last)
        result = engine.train(cfg)
    if result['successful_updates'] != 2000:
        raise RuntimeError('Shared Gaussian head did not complete 2k successful updates')
    shared = result['last_checkpoint']
    calibration = home / 'calibration.json'
    if calibration.is_file():
        calibrated = _read(calibration)
        if calibrated['checkpoint_sha256'] != engine.file_digest(shared):
            raise ValueError('Calibration checkpoint mismatch')
    else:
        calibrated = calibrate(plan, shared); engine._json(calibration, calibrated)
    result = dict(schema='GTA-AVT-gdf-joint-prepared-1', plan_sha256=_plan_signature(plan),
        shared_checkpoint=shared, shared_checkpoint_sha256=engine.file_digest(shared),
        common_head_successful_updates=2000, segmentation_updates_in_head_warmup=0,
        condition_checkpoint_sha256=plan['condition_checkpoint_sha256'],
        selected_base_sha256=plan['selected_base_sha256'], lambda_peak=calibrated['lambda_peak'],
        calibration_sha256=engine.file_digest(calibration), completed_unix=time.time())
    engine._json(receipt, result)
    return result


def arm_config(plan, arm, name, parent, lambda_peak):
    if arm not in ARMS or name not in ('geometry', 'refine'):
        raise ValueError('Only two matched full-S arms and two joint phases are declared')
    recipe = dict(_recipes()[name], stage='full_joint')
    if arm == 'AR':
        recipe['gdf_weight_start'] = recipe['gdf_weight_end'] = 0.
    else:
        recipe['gdf_weight_start'] *= lambda_peak
        recipe['gdf_weight_end'] *= lambda_peak
    cfg = dict(_common(plan), **recipe, phase=name, arm='AR',
        output_dir=str(_home(plan) / arm / name), parent_checkpoint=str(parent))
    cfg['lambda_gdf'] = max(cfg['gdf_weight_start'], cfg['gdf_weight_end'])
    if plan.get('tensorboard_logdir'):
        cfg['tensorboard_logdir'] = str(Path(plan['tensorboard_logdir']) / arm / name)
    return cfg


def run_arm(config, arm):
    plan = validate_plan(config)
    cases = _validated_evaluation_cases(plan)
    if arm not in ARMS: raise ValueError('Only AR or AR_GDF may be executed')
    home = _home(plan); prepared = _read(home / 'prepared.json')
    if prepared['plan_sha256'] != _plan_signature(plan) or engine.file_digest(prepared['shared_checkpoint']) != prepared['shared_checkpoint_sha256']:
        raise ValueError('Both arms must inherit the exact same prepared checkpoint and frozen plan')
    output = home / arm; output.mkdir(parents=True, exist_ok=True)
    state_file = output / 'joint_state.json'
    state = _read(state_file) if state_file.is_file() else dict(schema='GTA-AVT-gdf-joint-arm-1',
        status='running', arm=arm, plan_sha256=_plan_signature(plan),
        shared_checkpoint_sha256=prepared['shared_checkpoint_sha256'],
        condition_checkpoint_sha256=plan['condition_checkpoint_sha256'], selected_base_sha256=plan['selected_base_sha256'],
        seed=plan['seed'], completed={}, planned_segmentation_updates=30000, planned_lineage_updates=32000)
    if state['plan_sha256'] != _plan_signature(plan) or state['arm'] != arm:
        raise ValueError('Existing arm state does not match the frozen plan')
    if state['status'] == 'complete': return state
    parent = prepared['shared_checkpoint']
    try:
        for name in ('geometry', 'refine'):
            cfg = arm_config(plan, arm, name, parent, prepared['lambda_peak'])
            if name in state['completed']:
                result = state['completed'][name]
            else:
                state.update(status='running', current=name); engine._json(state_file, state)
                summary = Path(cfg['output_dir']) / 'summary.json'
                if summary.is_file():
                    result = _read(summary)
                else:
                    last = Path(cfg['output_dir']) / 'last.pt'
                    if last.is_file(): cfg['resume'] = str(last)
                    result = engine.train(cfg)
                if result['successful_updates'] != cfg['max_steps']:
                    raise RuntimeError('Incomplete fixed successful-update budget')
                state['completed'][name] = result; engine._json(state_file, state)
            parent = result['last_checkpoint']
        state.update(current='native_VAL'); engine._json(state_file, state)
        predictions = []
        final_sha = engine.file_digest(parent)
        for case in cases:
            destination = output / 'native_VAL' / case['id'] / 'inference.json'
            inferred = _read(destination) if destination.is_file() else infer(dict(checkpoint=parent,
                image=case['image'], case_id=case['id'], output_dir=str(output / 'native_VAL'),
                device=plan['device'], patch_size=128, batch_size=2, overlap=.5, amp=True,
                reviewed_override=case.get('intensity_override')))
            if inferred.get('checkpoint_sha256') != final_sha or inferred.get('case_id') != case['id']:
                raise ValueError('Existing native inference was produced by a different checkpoint or case')
            predictions.append(dict(id=case['id'], image=case['image'], prediction=inferred['prediction'], label=case['label']))
        metric_file = output / 'metrics' / 'metrics.json'
        metrics = _read(metric_file) if metric_file.is_file() else evaluate(dict(cases=predictions,
            output_dir=str(output / 'metrics'), centerline=True, split='VAL'))
        state['completed']['native_VAL'] = metrics
        state.update(status='complete', current=None, finished_unix=time.time(),
            final_checkpoint_sha256=engine.file_digest(parent))
        engine._json(state_file, state)
        return state
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc), failed_unix=time.time())
        engine._json(state_file, state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Explicit GDF-joint revision: prepare once, then two matched arms')
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args(argv)
    config = _read(args.config)
    result = prepare(config) if args.command == 'prepare' else run_arm(config, args.arm)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
