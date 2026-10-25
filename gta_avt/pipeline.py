"""Explicit v1.1 sequence: attributes, stable base, Gaussian head, local arms."""
import copy
import hashlib
import json
from pathlib import Path
import time

from .engine import pretrain, train, _json
from .inference import infer, evaluate


def run_pipeline(config):
    """Train only when explicitly invoked; no automatic rescue or server work.

    Proposed update counts are implementation defaults, not tested efficacy.
    Each local arm resets its optimizer and sample stream from one identical
    GDF-head-warmed base. All four carry the same GDF head and capacity.
    """
    plan = copy.deepcopy(config)
    resume = bool(plan.pop('resume_pipeline', False))
    endpoint = plan.get('evaluation_endpoint', 'last')
    if endpoint not in ('last', 'best'):
        raise ValueError('evaluation_endpoint must be explicitly last or best')
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    home = Path(plan['output_dir']).resolve()
    state_path = home / 'pipeline.json'
    if resume:
        state = json.loads(state_path.read_text(encoding='utf8'))
        if state['plan_sha256'] != signature:
            raise ValueError('Pipeline resume requires the identical v1.1 plan')
    else:
        home.mkdir(parents=True, exist_ok=False)
        state = dict(schema='GTA-AVT-v1.1-pipeline-1', status='running', plan_sha256=signature,
                     started_unix=time.time(), completed={}, current=None)
        _json(home / 'plan.json', plan)
    _json(state_path, state)
    common = dict(manifest=plan['manifest'], seed=plan['seed'], validation_seed=plan.get('validation_seed', 0),
        device=plan.get('device', 'cuda'), patch_size=128, batch_size=2, lr=.0003,
        weight_decay=.00001, samples_per_epoch=256, val_samples=64,
        num_workers=plan.get('num_workers', 0), amp=True, max_grad_norm=0.,
        checkpoint_every=500, validate_every=500, deterministic=False,
        dice_weight=1., bce_weight=1., deep_supervision_weight=0., use_gdf=True,
        min_support_dice=float(plan.get('min_support_dice', .5)),
        min_support_recall=float(plan.get('min_support_recall', .5)),
        unknown_prior_weight=float(plan.get('lambda_unknown_prior', .01)))
    common.update(plan.get('common', {}))

    def execute(name, kind, updates, **extra):
        if name in state['completed']:
            return state['completed'][name]
        state.update(status='running', current=name); _json(state_path, state)
        cfg = dict(common, output_dir=str(home / name), max_steps=updates, **extra)
        last = home / name / 'last.pt'
        if resume and last.is_file():
            cfg['resume'] = str(last)
        result = (pretrain if kind == 'pretrain' else train)(cfg)
        if result['successful_updates'] != updates:
            raise RuntimeError('Stage ended without its declared successful-update budget')
        state['completed'][name] = result; _json(state_path, state)
        return result

    try:
        attribute = execute('attribute', 'pretrain', int(plan.get('pretrain_steps', 20000)), stage='pretrain')
        predictor = attribute.get('best_ready_checkpoint')
        if not predictor:
            raise RuntimeError('No attribute checkpoint passed consumer prerequisites; segmentation was not launched')
        base = execute('base', 'train', int(plan.get('base_steps', 10000)), stage='base', arm='C0',
                       predictor_checkpoint=predictor, lambda_gdf=0., require_attribute_readiness=True)
        # The stable common base is selected for segmentation Dice, not GDF loss.
        base_checkpoint = base['best_checkpoint']
        head = execute('gdf_head', 'train', int(plan.get('gdf_head_steps', 1000)), stage='gdf_head',
                       arm='C0', predictor_checkpoint=predictor, base_checkpoint=base_checkpoint,
                       lambda_gdf=1., gdf_head_lr=float(plan.get('gdf_head_lr', .001)), require_attribute_readiness=True)
        shared = head['last_checkpoint']
        definitions = (('C0', 'C0', 0.), ('AR', 'AR', 0.),
                       ('C0_GDF', 'C0', float(plan.get('lambda_gdf', .1))),
                       ('AR_GDF', 'AR', float(plan.get('lambda_gdf', .1))))
        for name, arm, weight in definitions:
            execute(name, 'train', int(plan.get('local_steps', 2000)), stage='local_adapt', arm=arm,
                predictor_checkpoint=predictor, gdf_head_checkpoint=shared, lambda_gdf=weight,
                local_shared_lr=float(plan.get('local_shared_lr', .0001)),
                local_gate_lr=float(plan.get('local_gate_lr', .001)),
                local_gdf_lr=float(plan.get('local_gdf_lr', .0001)),
                alpha_ramp_steps=int(plan.get('alpha_ramp_steps', 250)),
                gdf_delay_steps=int(plan.get('gdf_delay_steps', 500)),
                gdf_ramp_steps=int(plan.get('gdf_ramp_steps', 500)),
                augment=bool(plan.get('local_augment', False)), require_attribute_readiness=True)
        if plan.get('evaluation_cases'):
            value = json.loads(Path(plan['evaluation_cases']).read_text(encoding='utf8'))
            cases = value['cases'] if isinstance(value, dict) else value
            cases = [row for row in cases if str(row.get('split', 'VAL')).upper() == 'VAL']
            if not cases:
                raise ValueError('Native evaluation needs explicit VAL cases')
            for name, _, _ in definitions:
                key = 'native_VAL_' + name
                if key in state['completed']:
                    continue
                state.update(current=key); _json(state_path, state)
                predictions = []
                stage_summary = state['completed'][name]
                checkpoint = stage_summary['last_checkpoint' if endpoint == 'last' else 'best_checkpoint']
                for case in cases:
                    destination = home / 'native_VAL' / name / case['id'] / 'inference.json'
                    if resume and destination.is_file():
                        result = json.loads(destination.read_text(encoding='utf8'))
                    else:
                        result = infer(dict(checkpoint=checkpoint, image=case['image'], case_id=case['id'],
                            output_dir=str(home / 'native_VAL' / name), device=common['device'],
                            patch_size=common['patch_size'], batch_size=common['batch_size'], overlap=.5,
                            reviewed_override=case.get('intensity_override'), amp=common['amp']))
                    predictions.append(dict(id=case['id'], image=case['image'],
                        prediction=result['prediction'], label=case['label']))
                result = evaluate(dict(cases=predictions, output_dir=str(home / 'metrics' / key),
                                       centerline=True, split='VAL'))
                state['completed'][key] = result; _json(state_path, state)
        state.update(status='complete', current=None, finished_unix=time.time()); _json(state_path, state)
        return state
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc), failed_unix=time.time())
        _json(state_path, state)
        raise
