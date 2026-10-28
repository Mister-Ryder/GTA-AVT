"""Declared v1.2 phases: one variant, explicit advancement, immutable lineage."""
import copy
import hashlib
import json
from pathlib import Path
import time

from .engine import pretrain, train, _json, load_checkpoint, PHASE_STAGES
from .inference import infer, evaluate

PHASE_ORDER = ('attribute', 'base_warm', 'base', 'condition', 'gdf_head', 'geometry', 'refine')
VARIANTS = ('C0', 'AR', 'C0_GDF', 'AR_GDF')


def _phase_plan(plan):
    if 'phases' in plan and 'phase_config' in plan:
        raise ValueError('Choose inline phases or phase_config, not both')
    phases = plan.get('phases')
    if phases is None:
        path = Path(plan.pop('phase_config', Path(__file__).resolve().parents[1] / 'configs' / 'phases.json'))
        phases = json.loads(path.read_text(encoding='utf-8-sig'))
    if set(phases) != set(PHASE_ORDER):
        raise ValueError('The declared v1.2 phase set is incomplete or expanded')
    phases = copy.deepcopy(phases)
    for name in PHASE_ORDER:
        cfg = phases[name]
        if cfg.get('stage') != PHASE_STAGES[name] or type(cfg.get('max_steps')) is not int or cfg['max_steps'] <= 0:
            raise ValueError('Invalid stage or successful-update budget for phase ' + name)
        if any(key in cfg for key in ('phase', 'parent_checkpoint', 'predictor_checkpoint', 'output_dir', 'arm')):
            raise ValueError('Phase recipes cannot override lineage or arm fields')
    plan['phases'] = phases
    return phases


def run_pipeline(config):
    """Execute only explicitly requested phases; importing never starts work.

    Stop limits may advance on resume. Variant, sampling, budgets and all scientific
    settings are signed together and cannot change within one output directory.
    Each phase resets its optimizer and loads only the complete declared parent.
    """
    plan = copy.deepcopy(config)
    resume = bool(plan.pop('resume_pipeline', False))
    stop_after = plan.pop('stop_after', 'base')
    if stop_after not in ('attribute', 'base_warm', 'base', 'condition'):
        raise ValueError('The GDF-joint suffix requires its calibrated prepare/run entry point; '
                         'reproduce the prefix with the frozen ratio-r2 source, then use gta_avt.gdf_joint')
    if stop_after not in PHASE_ORDER + ('complete',):
        raise ValueError('stop_after must name one declared phase or complete')
    variant = plan.setdefault('variant', 'AR_GDF')
    if variant not in VARIANTS:
        raise ValueError('Unknown v1.2 variant')
    endpoint = plan.setdefault('evaluation_endpoint', 'last')
    if endpoint not in ('last', 'best'):
        raise ValueError('evaluation_endpoint must be last or best')
    for key in ('minimum_base_dice', 'minimum_condition_dice'):
        value = plan.setdefault(key, None)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1):
            raise ValueError(key + ' must be null or a Dice fraction in [0,1]')
    phases = _phase_plan(plan)
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    home = Path(plan['output_dir']).resolve()
    state_path = home / 'pipeline.json'
    if resume:
        state = json.loads(state_path.read_text(encoding='utf8'))
        if state.get('schema') != 'GTA-AVT-v1.2-pipeline-1' or state['plan_sha256'] != signature:
            raise ValueError('Pipeline resume requires the identical v1.2 scientific plan')
    else:
        home.mkdir(parents=True, exist_ok=False)
        state = dict(schema='GTA-AVT-v1.2-pipeline-1', status='running', plan_sha256=signature,
            variant=variant, started_unix=time.time(), completed={}, current=None,
            planned_successful_updates=sum(phases[name]['max_steps'] for name in PHASE_ORDER))
        _json(home / 'plan.json', plan)
    _json(state_path, state)
    common = dict(manifest=plan['manifest'], seed=plan['seed'], validation_seed=plan.get('validation_seed', 0),
        device=plan.get('device', 'cuda'), patch_size=128, batch_size=2, samples_per_epoch=256,
        val_samples=64, num_workers=plan.get('num_workers', 0), amp=True,
        checkpoint_every=1000, validate_every=1000, deterministic=False,
        dice_weight=1., bce_weight=1., use_gdf=True,
        min_support_dice=float(plan.get('min_support_dice', .5)),
        min_support_recall=float(plan.get('min_support_recall', .5)))
    common.update(plan.get('common', {}))
    predictor = None
    parent = None
    dynamic_arm = 'AR' if variant.startswith('AR') else 'C0'
    geometry_active = variant.endswith('_GDF')

    def quality_check(phase, validation):
        dice = validation.get('summary', {}).get('dice', {}).get('mean')
        minimum = plan['minimum_' + phase + '_dice']
        passed = None if minimum is None else dice is not None and float(dice) >= minimum
        state.setdefault('quality_checks', {})[phase] = dict(
            sampled_val_dice=dice, minimum=minimum, threshold_applied=minimum is not None, passed=passed,
            scope='fixed sampled VAL patches; not native whole-case Dice or convergence')
        _json(state_path, state)
        if passed is False:
            raise RuntimeError(phase + ' Dice did not meet the declared minimum; no later phase was launched')

    try:
        for name in PHASE_ORDER:
            recipe = dict(phases[name])
            kind = 'pretrain' if name == 'attribute' else 'train'
            cfg = dict(common, **recipe)
            cfg.update(phase=name, output_dir=str(home / name),
                arm='C0' if name in ('attribute','base_warm','base') else dynamic_arm,
                lambda_gdf=0.)
            if name == 'base_warm':
                cfg['predictor_checkpoint'] = predictor
            elif name != 'attribute':
                cfg['parent_checkpoint'] = parent
            if name == 'gdf_head':
                cfg['lambda_gdf'] = 1.
            elif name in ('geometry','refine'):
                if not geometry_active:
                    cfg['gdf_weight_start'] = cfg['gdf_weight_end'] = 0.
                cfg['lambda_gdf'] = max(cfg['gdf_weight_start'], cfg['gdf_weight_end'])
            if name in state['completed']:
                result = state['completed'][name]
            else:
                state.update(status='running', current=name); _json(state_path, state)
                last = home / name / 'last.pt'
                if resume and last.is_file():
                    cfg['resume'] = str(last)
                result = (pretrain if kind == 'pretrain' else train)(cfg)
                if result['successful_updates'] != cfg['max_steps']:
                    raise RuntimeError('Phase ended without its declared successful-update budget')
                state['completed'][name] = result
                state['completed_successful_updates'] = sum(v['successful_updates'] for v in state['completed'].values() if 'successful_updates' in v)
                _json(state_path, state)
            if name == 'attribute':
                predictor = result.get('best_ready_checkpoint')
                if not predictor:
                    raise RuntimeError('No attribute checkpoint passed consumer prerequisites; joint base was not launched')
            elif name == 'base':
                parent = result.get('best_ready_checkpoint')
                if not parent:
                    raise RuntimeError('No mature joint base passed consumer prerequisites; local training was not launched')
                selected = load_checkpoint(parent)
                quality_check('base', selected.get('validation', {}))
            else:
                parent = result['last_checkpoint']
                if name == 'condition':
                    quality_check('condition', result.get('validation', {}))
            if stop_after == name:
                state.update(status='staged', current=None, stopped_after=name, staged_unix=time.time())
                _json(state_path, state)
                return state
        if plan.get('evaluation_cases'):
            value = json.loads(Path(plan['evaluation_cases']).read_text(encoding='utf8'))
            cases = value['cases'] if isinstance(value, dict) else value
            cases = [row for row in cases if str(row.get('split', 'VAL')).upper() == 'VAL']
            if not cases:
                raise ValueError('Native evaluation needs explicit VAL cases')
            key = 'native_VAL_' + variant
            if key not in state['completed']:
                state.update(current=key); _json(state_path, state)
                result = state['completed']['refine']
                checkpoint = result['last_checkpoint' if endpoint == 'last' else 'best_checkpoint']
                predictions = []
                for case in cases:
                    destination = home / 'native_VAL' / variant / case['id'] / 'inference.json'
                    if resume and destination.is_file():
                        inferred = json.loads(destination.read_text(encoding='utf8'))
                    else:
                        inferred = infer(dict(checkpoint=checkpoint,image=case['image'],case_id=case['id'],
                            output_dir=str(home/'native_VAL'/variant),device=common['device'],patch_size=common['patch_size'],
                            batch_size=common['batch_size'],overlap=.5,amp=common['amp'],reviewed_override=case.get('intensity_override')))
                    predictions.append(dict(id=case['id'],image=case['image'],prediction=inferred['prediction'],label=case['label']))
                state['completed'][key] = evaluate(dict(cases=predictions,output_dir=str(home/'metrics'/key),centerline=True,split='VAL'))
        state.pop('stopped_after', None)
        state.update(status='complete', current=None, finished_unix=time.time()); _json(state_path, state)
        return state
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc),failed_unix=time.time()); _json(state_path,state)
        raise
