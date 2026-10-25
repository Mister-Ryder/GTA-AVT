"""v1.1 staged optimization with an explicit stable-base/local-adaptation boundary.

Importing this module never starts training. Checkpoints have a new schema; old
v1 weights cannot silently enter this algorithm. Updates count successful AMP
optimizer steps. No dataset or server run is initiated by package installation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PreparedDataset, load_manifest
from .losses import attribute_loss, segmentation_loss
from .metrics import AttributeMetrics, binary_metrics, summarize_cases
from .models import AttributePredictor, GTAAVT

SCHEMA = 'GTA-AVT-v1.1-checkpoint-1'
STAGES = ('base', 'gdf_head', 'local_adapt')
ARMS = ('C0', 'V_only', 'AR')


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _tensor_digest(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_digest(module):
    return _tensor_digest(module.state_dict().items())


def _frozen_digest(module):
    active = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    return _tensor_digest((name, value) for name, value in module.state_dict().items() if name not in active)


def load_checkpoint(path, device='cpu'):
    # This format contains optimizer/RNG objects: load only trusted project files.
    try:
        value = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(str(path), map_location=device)
    if value.get('schema') != SCHEMA:
        raise ValueError('Expected a v1.1 checkpoint; old v1/oracle weights are not compatible')
    return value


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _device(config):
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use device=cpu explicitly for CPU checks')
    return device


def _parameters(config):
    c = dict(config)
    if any(not c.get(key) for key in ('manifest', 'output_dir')):
        raise ValueError('manifest and output_dir are required')
    defaults = dict(seed=31, validation_seed=0, patch_size=128, batch_size=2, max_steps=10000,
        lr=.0003, weight_decay=.00001, num_workers=0, samples_per_epoch=256,
        checkpoint_every=500, validate_every=500, amp=True, availability_weight=1.,
        radius_weight=1., unknown_prior_weight=.01, dice_weight=1., bce_weight=1.,
        lambda_gdf=0., deep_supervision_weight=0., alpha_ramp_steps=250,
        gdf_delay_steps=500, gdf_ramp_steps=500, local_shared_lr=.0001,
        local_gate_lr=.001, local_gdf_lr=.0001, gdf_head_lr=.001,
        arm='AR', stage='base', require_attribute_readiness=True,
        min_support_dice=.5, min_support_recall=.5, use_gdf=True)
    for key, value in defaults.items():
        c.setdefault(key, value)
    for key in ('patch_size', 'batch_size', 'max_steps', 'samples_per_epoch',
                'checkpoint_every', 'validate_every'):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if c['patch_size'] < 32 or c['patch_size'] % 16:
        raise ValueError('patch_size must be >=32 and divisible by16')
    for key in ('num_workers', 'alpha_ramp_steps', 'gdf_delay_steps', 'gdf_ramp_steps'):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(key + ' must be a nonnegative integer')
    for key in ('lr', 'local_shared_lr', 'local_gate_lr', 'local_gdf_lr', 'gdf_head_lr'):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(key + ' must be finite and positive')
    for key in ('weight_decay', 'availability_weight', 'radius_weight', 'unknown_prior_weight',
                'dice_weight', 'bce_weight', 'lambda_gdf', 'deep_supervision_weight'):
        if not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(key + ' must be finite and nonnegative')
    for key in ('min_support_dice', 'min_support_recall'):
        if not math.isfinite(c[key]) or not 0 <= c[key] <= 1:
            raise ValueError(key + ' must be in[0,1]')
    return c


def _output_directory(config):
    output = Path(config['output_dir']).resolve()
    if config.get('resume'):
        if not output.is_dir():
            raise FileNotFoundError('Resume output_dir does not exist')
    else:
        output.mkdir(parents=True, exist_ok=False)
        _json(output / 'config.json', config)
    return output


def _to_device(batch, device):
    return {key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _datasets(config):
    common = dict(manifest=config['manifest'], patch_size=config['patch_size'])
    train = PreparedDataset(split='TRAIN', seed=config['seed'],
        samples_per_epoch=config['samples_per_epoch'], augment=config.get('augment', True), **common)
    val_cases = [case for case in load_manifest(config['manifest'])['cases'] if case['split'].upper() == 'VAL']
    if not val_cases:
        raise ValueError('An explicit VAL split is required; TEST is never used for selection')
    # The validation bank is common across independent model seeds.
    val = PreparedDataset(split='VAL', seed=config['validation_seed'],
        samples_per_epoch=config.get('val_samples', len(val_cases) * 8), augment=False, **common)
    if (config['stage'] == 'gdf_head' or config['lambda_gdf'] > 0) and config['use_gdf']:
        manifest = load_manifest(config['manifest'])
        if manifest.get('heldout_gdf_foreground_prepared') is not True:
            raise ValueError('v1.1 GDF validation requires real held-out foreground targets; prepare data again')
    return train, val


def _worker_init(worker_id):
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def _loader(dataset, config):
    generator = torch.Generator().manual_seed(int(config['seed']))
    workers = config.get('num_workers', 0)
    options = dict(batch_size=config['batch_size'], shuffle=False, num_workers=workers,
                   drop_last=False, generator=generator)
    if workers:
        options.update(worker_init_fn=_worker_init, prefetch_factor=2, persistent_workers=False)
    return DataLoader(dataset, **options)


def _scalar(value):
    if value is None or isinstance(value, (bool, str, int, float)):
        result = value
    elif isinstance(value, torch.Tensor) and value.numel() == 1:
        result = float(value.detach().cpu())
    else:
        raise TypeError('Expected a scalar diagnostic')
    if isinstance(result, float) and not math.isfinite(result):
        raise FloatingPointError('Nonfinite recorded diagnostic')
    return result


def _loss_record(losses):
    return {key: _scalar(value) for key, value in losses.items()}


def _routing_record(value):
    result = {}
    for key, tensor in (value or {}).items():
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().float().cpu()
            if array.numel() > 64:
                result[key + '_mean'] = float(array.mean())
            else:
                result[key] = array.tolist()
        elif value is not None:
            result[key] = _scalar(tensor)
    return result


def _mean_records(rows):
    result = {}; counts = {}
    for key in set(key for row in rows for key in row):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))
                  and not isinstance(row.get(key), bool)]
        counts[key] = len(values)
        if not values:
            result[key] = None
        elif key.endswith(('_voxels', '_cases')):
            result[key] = int(sum(values))
        else:
            result[key] = float(np.mean(values))
    return result, counts


@torch.no_grad()
def _validate(model, dataset, config, device, kind, alpha, gdf_weight=None):
    model.eval(); dataset.set_epoch(0)
    values = []; rows = []; route_rows = []
    prior = dataset.statistics.get('pi_support', dataset.statistics.get('pi', .01))
    attrs = AttributeMetrics(prior, min_support_dice=config['min_support_dice'],
                             min_support_recall=config['min_support_recall']) if kind == 'pretrain' else None
    for batch in _loader(dataset, config):
        batch = _to_device(batch, device)
        output = model(batch['image']) if kind == 'pretrain' else model(batch['image'], alpha=alpha)
        values.append(_loss_record(_loss(output, batch, config, kind, gdf_weight)))
        if attrs is not None:
            attrs.update(output, batch)
        else:
            route_rows.append(_routing_record(output.get('routing')))
            pred = output['logits'].sigmoid() >= .5
            for i, case_id in enumerate(batch['case_id']):
                metric = binary_metrics(pred[i, 0].cpu().numpy(), batch['label'][i, 0].cpu().numpy(),
                                        batch['valid'][i, 0].cpu().numpy(), centerline=False)
                rows.append(dict(case_id=str(case_id), **metric))
    losses, defined = _mean_records(values)
    report = dict(losses=losses, loss_defined_batches=defined,
        scope='fixed sampled VAL patches, common across model seeds; not native whole-case evaluation')
    if attrs is not None:
        report.update(selection_score=float(losses['loss']), selection_key=[float(losses['loss'])],
                      selection_definition='declared support/radius/prior pretraining objective', attributes=attrs.report())
    else:
        pooled = []
        for case in sorted({row['case_id'] for row in rows}):
            subset = [row for row in rows if row['case_id'] == case]
            pooled.append(dict(case_id=case, **{key: float(np.mean([row[key] for row in subset if row[key] is not None]))
                if any(row[key] is not None for row in subset) else None for key in ('dice', 'precision', 'recall')}))
        summary = summarize_cases(pooled)
        dice = float(summary['dice']['mean'])
        report.update(per_case=pooled, summary=summary, selection_score=1. - dice,
            selection_key=[1. - dice, float(losses['segmentation_bce'])],
            selection_definition='maximum case-equal sampled VAL Dice, then lower BCE; GDF excluded',
            routing=route_rows)
    return report


def _loss(output, batch, config, kind, gdf_weight=None):
    if kind == 'pretrain':
        return attribute_loss(output, batch, availability_weight=config['availability_weight'],
            radius_weight=config['radius_weight'], unknown_prior_weight=config['unknown_prior_weight'])
    weight = config['lambda_gdf'] if gdf_weight is None else gdf_weight
    head = config['stage'] == 'gdf_head'
    return segmentation_loss(output, batch, dice_weight=0. if head else config['dice_weight'],
        bce_weight=0. if head else config['bce_weight'], gdf_weight=1. if head else weight,
        deep_supervision_weight=0. if head else config['deep_supervision_weight'])


def _selection_score(losses, config, kind):
    # A scalar fallback; validation uses case-equal Dice and a separate BCE tie key.
    return float(losses['loss'] if kind == 'pretrain' else losses['dice_loss'])


def _alpha(config, step, kind):
    if kind == 'pretrain' or config['stage'] != 'local_adapt':
        return 0.
    return min(1., step / max(1, config['alpha_ramp_steps'])) if config['alpha_ramp_steps'] else 1.


def _gdf_weight(config, step, kind):
    if kind == 'pretrain' or config['stage'] == 'base':
        return 0.
    if config['stage'] == 'gdf_head':
        return 1.
    weight = float(config['lambda_gdf'])
    elapsed = max(0, step - config['gdf_delay_steps'])
    fraction = min(1., elapsed / max(1, config['gdf_ramp_steps'])) if config['gdf_ramp_steps'] else float(elapsed > 0)
    return weight * fraction


def _optimizer(model, config):
    if config['stage'] == 'pretrain':
        return torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    supplied = model.parameter_groups()
    expected = {id(value) for value in model.parameters() if value.requires_grad}
    actual = [id(value) for group in supplied.values() for value in group]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError('Optimizer groups must contain every active parameter exactly once')
    if config['stage'] == 'local_adapt':
        rates = dict(shared=config['local_shared_lr'], gates=config['local_gate_lr'], gdf=config['local_gdf_lr'])
    elif config['stage'] == 'gdf_head':
        rates = dict(shared=config['lr'], gates=config['lr'], gdf=config['gdf_head_lr'])
    else:
        rates = {key: config['lr'] for key in supplied}
    groups = [dict(params=values, lr=rates[name], name=name) for name, values in supplied.items() if values]
    if not groups:
        raise RuntimeError('No active optimization parameters')
    return torch.optim.AdamW(groups, weight_decay=config['weight_decay'])


def _save(path, state):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary); os.replace(temporary, path)


def _load_common(model, predictor, config, stage, expected_predictor_digest, stats):
    if stage == 'base':
        return 'random'
    field, expected_stage = ('base_checkpoint', 'base') if stage == 'gdf_head' else ('gdf_head_checkpoint', 'gdf_head')
    if not config.get(field):
        raise ValueError(stage + ' requires ' + field)
    checkpoint = load_checkpoint(config[field])
    old = checkpoint['config']
    if (checkpoint.get('kind') != 'train' or old.get('stage') != expected_stage or old.get('arm') != 'C0'
            or _digest(checkpoint['statistics']) != _digest(stats)
            or checkpoint['lineage']['predictor_init']['predictor_state_sha256'] != expected_predictor_digest
            or old['seed'] != config['seed'] or old['patch_size'] != config['patch_size']
            or bool(old.get('use_gdf')) != bool(config['use_gdf'])):
        raise ValueError('Common checkpoint stage/predictor/seed/grid/architecture mismatch')
    model.load_state_dict(checkpoint['model'], strict=True)
    if state_digest(predictor) != expected_predictor_digest:
        raise RuntimeError('Common state replaced the declared frozen predictor')
    return dict(common_checkpoint_sha256=file_digest(config[field]), common_stage=expected_stage,
                common_successful_updates=checkpoint['step'], common_state_sha256=state_digest(model),
                optimizer_reset=True)


def _run(config, kind):
    config = dict(config)
    if kind == 'pretrain':
        config.setdefault('stage', 'pretrain')
    config = _parameters(config)
    if kind == 'pretrain' and config['stage'] != 'pretrain':
        raise ValueError('Independent attribute pretraining requires stage=pretrain')
    if kind == 'train' and (config['stage'] not in STAGES or config['arm'] not in ARMS):
        raise ValueError('Expected base/gdf_head/local_adapt and C0/V_only/AR')
    if kind == 'train' and config['stage'] != 'local_adapt' and config['arm'] != 'C0':
        raise ValueError('The common base and GDF-head stages use neutral C0 conditions')
    if config['stage'] == 'base' and any(config.get(key) for key in ('base_checkpoint', 'gdf_head_checkpoint')):
        raise ValueError('The common segmentation base starts from random S initialization')
    if kind == 'train' and config['stage'] in ('gdf_head', 'local_adapt') and not config['use_gdf']:
        raise ValueError('The complete v1.1 local pipeline requires its shared GDF head')
    if config['stage'] == 'base' and config['lambda_gdf']:
        raise ValueError('Base training uses segmentation only; GDF enters later')
    device = _device(config); _seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(config.get('deterministic', False)))
    if config.get('deterministic', False) and device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    dataset, val = _datasets(config)
    stats = dict(dataset.statistics)
    if _digest(stats) != _digest(dict(val.statistics)):
        raise ValueError('TRAIN and VAL normalizer metadata differ')
    predictor = AttributePredictor(stats)
    lineage = dict(predictor_init='random', segmentation_init=None)
    expected_predictor_digest = None
    if kind == 'pretrain':
        model = predictor
        model_config = dict(type='AttributePredictor', version='1.1')
    else:
        if not config.get('predictor_checkpoint'):
            raise ValueError('Independent predictor_checkpoint is required')
        checkpoint = load_checkpoint(config['predictor_checkpoint'])
        if checkpoint.get('kind') != 'pretrain' or _digest(checkpoint['statistics']) != _digest(stats):
            raise ValueError('Attribute stage/TRAIN statistics mismatch')
        ready = checkpoint.get('validation', {}).get('attributes', {}).get('readiness', {})
        if config['require_attribute_readiness'] and not ready.get('passed', False):
            raise ValueError('Attribute consumer prerequisites failed; inspect support overlap/radius/reliability diagnostics')
        predictor.load_state_dict(checkpoint['model'], strict=True)
        expected_predictor_digest = state_digest(predictor)
        model = GTAAVT(predictor, arm=config['arm'], use_gdf=config['use_gdf'])
        model_config = dict(type='GTAAVT', version='1.1', arm=config['arm'], use_gdf=config['use_gdf'])
        lineage['predictor_init'] = dict(checkpoint_sha256=file_digest(config['predictor_checkpoint']),
            successful_updates=checkpoint['step'], predictor_state_sha256=expected_predictor_digest,
            readiness_passed=ready.get('passed', False))
        lineage['segmentation_init'] = _load_common(model, predictor, config, config['stage'], expected_predictor_digest, stats)
        model.set_stage(config['stage'])
    model.to(device)
    frozen_digest = _frozen_digest(model) if kind == 'train' and config['stage'] != 'base' else None
    optimizer = _optimizer(model, config)
    local = config['stage'] in ('gdf_head', 'local_adapt')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=(lambda step: 1.) if local else (lambda step: max(0., 1 - step / config['max_steps']) ** .9))
    use_amp = bool(config['amp'] and device.type == 'cuda')
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    output = _output_directory(config)
    state = dict(step=0, attempts=0, epoch=0, batch_index=0, best_score=math.inf,
                 best_key=[math.inf, math.inf], best_ready_score=math.inf)
    validation = None
    signature = _digest({key: value for key, value in config.items() if key not in ('resume', 'output_dir')})
    if config.get('resume'):
        checkpoint = load_checkpoint(config['resume'])
        if checkpoint['kind'] != kind or checkpoint['config_signature'] != signature:
            raise ValueError('Resume requires identical v1.1 stage/training configuration')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler']); state.update(checkpoint['progress'])
        lineage = checkpoint['lineage']; validation = checkpoint.get('validation'); _restore_rng(checkpoint['rng'])
    started = time.monotonic()

    def snapshot():
        if kind == 'train' and state_digest(predictor) != expected_predictor_digest:
            raise RuntimeError('Frozen predictor weights or BN buffers changed')
        if frozen_digest is not None and _frozen_digest(model) != frozen_digest:
            raise RuntimeError('A frozen base parameter or normalization buffer changed')
        return dict(schema=SCHEMA, kind=kind, config=config, config_signature=signature,
            statistics=stats, model_config=model_config, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=_rng(), progress=dict(state),
            step=state['step'], lineage=lineage, validation=validation,
            trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad))

    while state['step'] < config['max_steps']:
        dataset.set_epoch(state['epoch'])
        loader = _loader(dataset, config)
        model.train()
        for batch_index, batch in enumerate(loader):
            if batch_index < state['batch_index']:
                continue
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            next_step = state['step'] + 1
            alpha, gdf_weight = _alpha(config, next_step, kind), _gdf_weight(config, next_step, kind)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                result = model(batch['image']) if kind == 'pretrain' else model(batch['image'], alpha=alpha)
                losses = _loss(result, batch, config, kind, gdf_weight)
            scaler.scale(losses['loss']).backward()
            scaler.unscale_(optimizer)
            if config.get('max_grad_norm', 0):
                torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad], config['max_grad_norm'])
            previous_scale = scaler.get_scale()
            if not use_amp and any(value.grad is not None and not torch.isfinite(value.grad).all()
                                   for value in model.parameters() if value.requires_grad):
                raise FloatingPointError('Nonfinite gradient; no optimizer update was applied')
            scaler.step(optimizer); scaler.update()
            successful = not use_amp or scaler.get_scale() >= previous_scale
            state['attempts'] += 1; state['batch_index'] = batch_index + 1
            if not successful:
                if state['attempts'] - state['step'] > config.get('max_skipped_updates', 100):
                    raise RuntimeError('Too many AMP-overflow skips')
                continue
            state['step'] += 1; scheduler.step()
            record = dict(successful_update=state['step'], attempt=state['attempts'], epoch=state['epoch'],
                alpha=alpha, gdf_weight=gdf_weight, lr=optimizer.param_groups[0]['lr'],
                learning_rates={group.get('name', 'attribute'): group['lr'] for group in optimizer.param_groups},
                losses=_loss_record(losses))
            if kind == 'train':
                record['routing'] = _routing_record(result.get('routing'))
            is_best = False; is_ready = False
            do_val = state['step'] % config['validate_every'] == 0 or state['step'] == config['max_steps']
            if do_val:
                validation = _validate(model, val, config, device, kind, alpha, gdf_weight)
                record['validation'] = validation
                key = list(validation['selection_key'])
                if tuple(key) < tuple(state['best_key']):
                    state['best_key'] = key; state['best_score'] = validation['selection_score']; is_best = True
                if kind == 'pretrain' and validation['attributes']['readiness']['passed'] and validation['selection_score'] < state['best_ready_score']:
                    state['best_ready_score'] = validation['selection_score']; is_ready = True
                model.train()
            with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            if state['step'] % config['checkpoint_every'] == 0 or do_val:
                checkpoint = snapshot(); _save(output / 'last.pt', checkpoint)
                if is_best:
                    _save(output / 'best.pt', checkpoint)
                if is_ready:
                    _save(output / 'best_ready.pt', checkpoint)
            if state['step'] >= config['max_steps']:
                break
        if state['batch_index'] >= len(loader):
            state['epoch'] += 1; state['batch_index'] = 0
    summary = dict(kind=kind, stage=config['stage'], successful_updates=state['step'], attempts=state['attempts'],
        elapsed_seconds=time.monotonic() - started, best_validation_score=state['best_score'],
        selection_definition='pretraining objective' if kind == 'pretrain' else 'case-equal sampled VAL Dice then BCE',
        last_checkpoint=str(output / 'last.pt'), best_checkpoint=str(output / 'best.pt'), lineage=lineage,
        best_ready_checkpoint=str(output / 'best_ready.pt') if (output / 'best_ready.pt').is_file() else None,
        validation=validation, weights_are_newly_trained=True,
        convergence_claim='A fixed update budget is not convergence or a performance claim.')
    _json(output / 'summary.json', summary)
    return summary


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pretrain(config):
    return _run(config, 'pretrain')


def train(config):
    return _run(config, 'train')
