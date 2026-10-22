"""Matched-recipe train/validation with no automatic final-test access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from common import atomic_json, load_json, sha256, source_hashes, load_trusted_checkpoint
from data import CachedData, batch
from model import GTAAVT, METHODS
from losses import objective
from inference import evaluate_cases


def save_checkpoint(path, state):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def optimizer_update_with_amp_retries(model, optimizer, scaler, forward_loss,
                                      gradient_clip=12., max_attempts=8, on_overflow=None):
    """One successful update, with at most8attempts on the exact same batch.

The caller constructs the batch once. Failed AMP backward attempts roll back
BatchNorm buffers and CPU/CUDA RNG; only GradScaler's reduced scale survives.
An invalid forward loss or invalid FP32 gradient is never silently skipped.
"""
    if max_attempts < 1:
        raise ValueError('At least one optimizer attempt is required')
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    initial_scale = float(scaler.get_scale())
    parameters = list(model.parameters())

    def restore_forward_state():
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    for attempt in range(1, max_attempts + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, details = forward_loss()
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise FloatingPointError('Nonfinite forward objective; AMP retries are not permitted for invalid loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients:
            raise RuntimeError('Objective produced no parameter gradients')
        # One aggregate device->CPU synchronization, not one per parameter.
        finite = bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all().item())
        if not finite:
            if not scaler.is_enabled():
                raise FloatingPointError('Nonfinite FP32 gradients with GradScaler disabled')
            before = float(scaler.get_scale())
            original_step = optimizer.step

            def forbidden_overflow_step(*args, **kwargs):
                raise RuntimeError('GradScaler failed to skip optimizer.step after detected nonfinite gradients')

            # unscale_ recorded actual nonfinite elements. Confirm public
            # GradScaler skip semantics without allowing an accidental update.
            optimizer.step = forbidden_overflow_step
            try:
                scaler.step(optimizer)
            finally:
                optimizer.step = original_step
            scaler.update()
            after = float(scaler.get_scale())
            if not math.isfinite(after) or not 0. < after < before:
                raise RuntimeError('AMP overflow did not reduce the loss scale')
            restore_forward_state()
            record = {'attempt': attempt, 'max_attempts': max_attempts,
                      'loss_scale_before': before, 'loss_scale_after': after,
                      'optimizer_step_skipped': True, 'buffers_and_rng_restored': True,
                      'retry_will_occur': attempt < max_attempts,
                      'reason': 'finite_forward_loss_nonfinite_unscaled_gradient_elements',
                      'finite_forward_loss': float(loss.detach())}
            if on_overflow is not None:
                on_overflow(record)
            if attempt == max_attempts:
                raise FloatingPointError('AMP gradients remained nonfinite after%d attempts (%d retries); no update counted' %
                                         (max_attempts, max_attempts - 1))
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
        if not bool(torch.isfinite(gradient_norm).item()):
            # All elements were finite. A norm-reduction overflow is a distinct
            # failure; it must not take a scaler.step with found_inf==0.
            raise FloatingPointError('Nonfinite gradient norm despite finite gradient elements; optimizer was not stepped')
        scaler.step(optimizer)
        scaler.update()
        return details, {'gradient_norm': float(gradient_norm), 'amp_attempts': attempt,
                         'amp_retries': attempt - 1, 'loss_scale_before': initial_scale,
                         'loss_scale_after': float(scaler.get_scale())}
    raise AssertionError('Unreachable optimizer attempt state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--allow-incomplete-cache', action='store_true', help='Development only')
    args = parser.parse_args()
    config = load_json(args.config)
    if args.method:
        config['method'] = args.method
    if args.seed is not None:
        config['seed'] = args.seed
    data = CachedData(args.manifest)
    if not data.manifest['complete'] and not args.allow_incomplete_cache:
        raise RuntimeError('Preparation incomplete; only an explicitly diagnostic run may proceed')
    all_ids = set(data.cases)
    train_ids, val_ids = data.manifest['splits']['train'], data.manifest['splits']['val']
    if args.allow_incomplete_cache:
        # Never populate missing cases from test and never call this a formal result.
        train_ids = [x for x in train_ids if x in all_ids]
        val_ids = [x for x in val_ids if x in all_ids]
        data.manifest['splits']['train'] = train_ids
    if not train_ids:
        raise ValueError('No prepared train cases')
    if not set(train_ids).isdisjoint(val_ids):
        raise ValueError('Train/validation overlap')
    val_limit = config.get('validation_case_limit', 0)
    selected_validation = config.get('validation_case_ids')
    if selected_validation:
        if len(selected_validation) != len(set(selected_validation)) or not set(selected_validation).issubset(set(val_ids)):
            raise ValueError('Explicit validation cases must be a unique subset of the frozen validation split')
        if config['stage'] == 'formal' and set(selected_validation) != set(val_ids):
            raise ValueError('Formal runs must use all frozen validation cases')
        val_ids = selected_validation
    elif val_limit:
        val_ids = val_ids[:val_limit]
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    record_path = run / 'config.json'
    signature = {'manifest_sha256': sha256(args.manifest), 'source_sha256': source_hashes(), 'recipe': config,
                 'train_ids': train_ids, 'validation_ids': val_ids, 'test_used': False,
                 'selection_valid': bool(data.manifest['complete'] and not val_limit and config['stage'] == 'formal'
                                         and set(val_ids) == set(data.manifest['splits']['val'])),
                 'manifest_path': data.path,
                 'architecture': ('GTA-AVT original-operator migration; no-space/no-Gaussian' if config['method'].startswith('faithful_')
                                  else 'GTA-AVT-v1 experimental soft-FiLM candidate'),
                 'initialization': 'random; paired common backbone; no airway pretrained checkpoint',
                 'device': args.device}
    if record_path.exists():
        if not args.resume:
            raise RuntimeError('Existing run is immutable; choose a fresh output or explicit --resume')
        old = load_json(record_path)
        for key in ('manifest_sha256', 'source_sha256', 'recipe', 'train_ids', 'validation_ids'):
            if old[key] != signature[key]:
                raise ValueError('Resume protocol/source changed: ' + key)
    else:
        if args.resume:
            raise ValueError('Cannot resume a run without its config')
        atomic_json(record_path, signature)
        snapshot = run / 'source'
        snapshot.mkdir(exist_ok=True)
        for relative in signature['source_sha256']:
            file = Path(__file__).parent / relative
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    torch.set_num_threads(config.get('cpu_threads', 4))
    set_seed(config['seed'])
    model = GTAAVT(config['method'], config['width']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    start, best, elapsed_training = 0, -1., 0.
    if args.resume:
        checkpoint = load_trusted_checkpoint(run / 'latest.pt', device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start, best, elapsed_training = checkpoint['step'], checkpoint['best_val_dice'], checkpoint['training_seconds']
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
    max_steps = config['max_steps']
    print(json.dumps({'stage': config['stage'], 'method': config['method'], 'parameters': sum(x.numel() for x in model.parameters()),
                      'train_cases': len(train_ids), 'validation_cases': len(val_ids), 'max_steps': max_steps,
                      'resume_step': start, 'test_used': False}), flush=True)
    last_val = None
    for step in range(start, max_steps):
        started = time.time()
        model.train()
        arrays = batch(data, config['seed'], step, config['batch_size'], config['patch_size'], config['foreground_probability'])
        tensors = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
        lr = config['lr'] * max(0., 1. - step / max_steps) ** .9
        for group in optimizer.param_groups:
            group['lr'] = lr

        def forward_loss():
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                output = model(tensors['image'], tensors['coordinate'])
                return objective(output, tensors, config['auxiliary_weight'])

        def log_overflow(record):
            entry = dict(step=step + 1, sampler_step=step, event='amp_overflow', **record)
            with open(run / 'amp_overflow.jsonl', 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
            print(json.dumps(entry, allow_nan=False), flush=True)

        details, update = optimizer_update_with_amp_retries(
            model, optimizer, scaler, forward_loss, gradient_clip=config['gradient_clip'],
            max_attempts=8, on_overflow=log_overflow)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.time() - started
        elapsed_training += seconds
        log = dict(step=step + 1, lr=lr, seconds=seconds, training_seconds=elapsed_training,
                   **update, **details)
        with open(run / 'training.jsonl', 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(log) + '\n')
        if (step + 1) % config['log_every'] == 0 or step == start:
            print(json.dumps(log), flush=True)
            atomic_json(run / 'status.json', dict(log, stage=config['stage'], method=config['method'],
                        eta_training_hours=seconds * (max_steps - step - 1) / 3600., state='training'))
        is_best = False
        if val_ids and ((step + 1) % config['validate_every'] == 0 or step + 1 == max_steps):
            _, last_val = evaluate_cases(model, data, val_ids, device, run / 'validation' / ('step_%06d' % (step + 1)),
                                         patch=config['inference_patch_size'], batch_size=config['inference_batch_size'],
                                         overlap=config['inference_overlap'], save_predictions=config.get('save_validation_predictions', False))
            dice = last_val['metrics']['dice']['mean']
            if dice is not None and dice > best:
                best, is_best = dice, True
        if (step + 1) % config['checkpoint_every'] == 0 or step + 1 == max_steps or is_best:
            checkpoint = {'step': step + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scaler': scaler.state_dict(), 'best_val_dice': best, 'training_seconds': elapsed_training,
                          'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                          'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                          'signature': signature, 'last_validation': last_val}
            save_checkpoint(run / 'latest.pt', checkpoint)
            if is_best:
                save_checkpoint(run / 'best.pt', checkpoint)
    atomic_json(run / 'complete.json', {'state': 'complete', 'stage': config['stage'], 'steps': max_steps,
                'training_seconds': elapsed_training, 'best_val_dice': best if best >= 0 else None,
                'last_validation': last_val, 'test_used': False})
    print(json.dumps({'complete': str(run), 'training_hours': elapsed_training / 3600., 'test_used': False}), flush=True)


if __name__ == '__main__':
    main()
