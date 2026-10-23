"""One sparse frozen-readout extraction and CPU64 convex calibration.

No scientific imaging imports; no segmentation trainer. Imported lazily by the
SHA-bound execute mode. The independent probe uses sampled-window estimates.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), .5 * weights.sum())])


def _case_weights(bank, active, indices):
    """Patch-uniform valid-window measure, conditioned on selected stratum."""
    result = np.zeros(len(indices), np.float64)
    case = bank['case_index'][indices]
    for value in np.unique(case):
        chosen = (case == value) & active
        mass = bank['metric_weight'][indices][chosen]
        if len(mass) and mass.sum() > 0:
            result[chosen] = mass / mass.sum()
    return result


def _prior(bank, indices):
    active = bank['foreground'][indices]
    weights = _case_weights(bank, active, indices)
    if not weights.sum():
        raise ValueError('No fit nonempty support')
    y = bank['target'][indices, 1].astype(np.float64)
    mean = float(np.average(y, weights=weights))
    median = weighted_median(y[active], weights[active])
    composition = np.average(bank['target'][indices].astype(np.float64), axis=0,
                             weights=bank['objective_weight'][indices])
    return dict(raw_fine_mean=mean, conditional_fine_mean=mean,
                raw_fine_median=median, conditional_fine_median=median,
                all_valid_window_softCE_mean_composition=composition.tolist(),
                note='fit24 case-equal window measure conditioned on supported nonempty; GT empty mass=0')


def _weighted_stat(prediction, target, weights, prior_mean, prior_median):
    weights = weights / weights.sum()
    pm = float(np.dot(weights, prediction))
    ym = float(np.dot(weights, target))
    pv = float(np.dot(weights, (prediction - pm) ** 2))
    yv = float(np.dot(weights, (target - ym) ** 2))
    corr = float(np.dot(weights, (prediction-pm)*(target-ym)) / np.sqrt(pv*yv)) if pv > 0 and yv > 0 else None
    return dict(model_MSE=float(np.dot(weights, (prediction-target)**2)),
                mean_constant_MSE=float(np.dot(weights, (prior_mean-target)**2)),
                model_MAE=float(np.dot(weights, np.abs(prediction-target))),
                median_constant_MAE=float(np.dot(weights, np.abs(prior_median-target))),
                arithmetic_point2_MSE=float(np.dot(weights, (.2-target)**2)),
                arithmetic_point2_MAE=float(np.dot(weights, np.abs(.2-target))),
                prediction_mean=pm, target_mean=ym, prediction_variance=pv,
                target_variance=yv, weighted_pearson=corr,
                prediction_range=[float(prediction.min()), float(prediction.max())],
                target_range=[float(target.min()), float(target.max())])


def _bins(prediction, target, weights):
    result = []
    for index in range(5):
        active = ((prediction >= index/5.) &
                  ((prediction < (index+1)/5.) if index < 4 else (prediction <= 1.)))
        w = weights[active]
        result.append(dict(lower=index/5., upper=(index+1)/5.,
                           sampled_windows=int(active.sum()),
                           normalized_weight_mass=float(w.sum()),
                           prediction_mean=float(np.average(prediction[active], weights=w)) if w.sum() else None,
                           target_mean=float(np.average(target[active], weights=w)) if w.sum() else None))
    return result


def evaluate(bank, probabilities, indices, plan, prior):
    rows = []
    for c in sorted(np.unique(bank['case_index'][indices])):
        selected = indices[bank['case_index'][indices] == c]
        fg = bank['foreground'][selected]
        case_id = (plan['fit_cases'] + plan['check_cases'])[int(c)]
        source = case_id.rsplit('_', 1)[0]
        row = dict(case_id=case_id, source=source,
                   sampled_nonempty_windows=int(fg.sum()),
                   sampled_trueempty_windows=int((~fg).sum()),
                   sampled_fine_positive_windows=int(np.count_nonzero(bank['target'][selected, 1] > 0)),
                   bank_fulltarget_fine_positive_windows=sum(
                       x['target_audit']['fine_positive_cells'] for x in plan['bank'] if x['case_id'] == case_id))
        all_y = bank['target'][selected].astype(np.float64)
        all_p = probabilities[selected].astype(np.float64)
        all_w = bank['metric_weight'][selected]
        entropy = -(all_y*np.log(np.maximum(all_y,1e-30))).sum(1)
        ce = -(all_y*np.log(np.maximum(all_p,1e-30))).sum(1)
        constant_ce = -(all_y*np.log(np.maximum(np.asarray(prior['all_valid_window_softCE_mean_composition']),1e-30))).sum(1)
        row['all_valid_softCE'] = dict(model=float(np.average(ce,weights=all_w)),
            fit_composition_constant=float(np.average(constant_ce,weights=all_w)),
            target_entropy=float(np.average(entropy,weights=all_w)),
            excess_CE_over_target_entropy=float(np.average(ce-entropy,weights=all_w)))
        if fg.any():
            sel = selected[fg]
            w = bank['metric_weight'][sel]
            y = bank['target'][sel, 1].astype(np.float64)
            p = probabilities[sel].astype(np.float64)
            raw = p[:, 1]
            cond = raw / np.maximum(p[:, 1:].sum(1), 1e-30)
            row['raw'] = _weighted_stat(raw, y, w, prior['raw_fine_mean'], prior['raw_fine_median'])
            row['conditional'] = _weighted_stat(cond, y, w, prior['conditional_fine_mean'], prior['conditional_fine_median'])
            row['raw_bins'] = _bins(raw, y, w/w.sum())
            row['conditional_bins'] = _bins(cond, y, w/w.sum())
            fine = y > 0
            row['fine_positive_raw'] = _weighted_stat(raw[fine], y[fine], w[fine], prior['raw_fine_mean'], prior['raw_fine_median']) if fine.any() else None
        bg = selected[~fg]
        if len(bg):
            w = bank['metric_weight'][bg]
            row['trueempty'] = dict(p_empty_mean=float(np.average(probabilities[bg, 0], weights=w)),
                                    raw_fine_mean=float(np.average(probabilities[bg, 1], weights=w)),
                                    raw_fine_MSE=float(np.average(probabilities[bg, 1]**2, weights=w)),
                                    fine_over_point2_fraction=float(np.average(probabilities[bg, 1] > .2, weights=w)))
        strata = {}
        for name, value in (('GT_centerline_plus_jitter', 0), ('CT_uniform', 1)):
            use = selected[fg & (bank['patch_stratum'][selected] == value)]
            if len(use):
                w = bank['metric_weight'][use]
                y = bank['target'][use, 1].astype(np.float64)
                p = probabilities[use].astype(np.float64)
                strata[name] = dict(sampled_windows=len(use),
                                   raw=_weighted_stat(p[:, 1], y, w, prior['raw_fine_mean'], prior['raw_fine_median']),
                                   conditional=_weighted_stat(p[:, 1]/np.maximum(p[:, 1:].sum(1), 1e-30), y, w, prior['conditional_fine_mean'], prior['conditional_fine_median']))
        row['patch_sampling_strata'] = strata
        rows.append(row)
    defined = [r for r in rows if 'raw' in r]
    keys = ('model_MSE', 'mean_constant_MSE', 'model_MAE', 'median_constant_MAE',
            'arithmetic_point2_MSE', 'arithmetic_point2_MAE')
    macro = {kind: {k: float(np.mean([r[kind][k] for r in defined])) for k in keys}
             for kind in ('raw', 'conditional')} if defined else {}
    sources = {}
    for source in ('Dongyang', 'KiTS', 'Rider'):
        group = [r for r in defined if r['source'] == source]
        sources[source] = dict(cases=len(group), **{
            kind: {k: float(np.mean([r[kind][k] for r in group])) for k in keys}
            for kind in ('raw', 'conditional')}) if group else dict(cases=0)
    return dict(cases=rows, macro=macro, sources=sources,
                valid_nonempty_cases=len(defined),
                conditional_MSE_better_cases=sum(r['conditional']['model_MSE'] < r['conditional']['mean_constant_MSE'] for r in defined),
                metric_measure='case equal; per-case patch-uniform valid-window IPW measure conditioned on each reported stratum; overlapping sampled windows are not independent cases')


def semantic_gate(evaluation, config, optimization_resolved):
    reductions = {}
    for kind in ('raw', 'conditional'):
        values = evaluation.get('macro', {}).get(kind, {})
        baseline = values.get('mean_constant_MSE', 0.)
        reductions[kind] = 1. - values.get('model_MSE', baseline) / baseline if baseline > 0 else None
    supported = sum(r['sampled_fine_positive_windows'] >= config['minimum_sampled_fine_cells_per_case'] for r in evaluation['cases'])
    source_good = all(evaluation['sources'][s]['cases'] == 4 and all(
        evaluation['sources'][s][k]['model_MSE'] <= evaluation['sources'][s][k]['mean_constant_MSE']
        for k in ('raw', 'conditional')) for s in ('Dongyang', 'KiTS', 'Rider'))
    clauses = dict(all12_have_nonempty_support=evaluation['valid_nonempty_cases'] == 12,
                   sampled_fine_support=supported >= config['minimum_fine_supported_cases'],
                   raw_MSE_reduction=reductions['raw'] is not None and reductions['raw'] >= config['minimum_relative_raw_MSE_reduction'],
                   conditional_MSE_reduction=reductions['conditional'] is not None and reductions['conditional'] >= config['minimum_relative_conditional_MSE_reduction'],
                   conditional_case_wins=evaluation['conditional_MSE_better_cases'] >= config['minimum_conditional_MSE_better_cases'],
                   all_three_sources_not_worse=source_good,
                   optimizer_resolved=bool(optimization_resolved))
    return dict(passed=all(clauses.values()), clauses=clauses,
                relative_MSE_reduction=reductions, fine_supported_cases=supported,
                MAE_is_report_only=True, config=config,
                estimate='fixed sampled-window inverse-inclusion estimate, not exact full-field scoring')


def _rng_hash(torch):
    result = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all():
        result.update(state.cpu().numpy().tobytes())
    return result.hexdigest()


def fit_convex(torch, bank, fit_rows, recipe):
    torch.set_num_threads(4)
    x = torch.from_numpy(bank['feature'][fit_rows].astype(np.float64))
    y = torch.from_numpy(bank['target'][fit_rows].astype(np.float64))
    weights = torch.from_numpy(bank['objective_weight'][fit_rows].astype(np.float64))
    weights /= weights.sum()
    weight = torch.nn.Parameter(torch.zeros((5, 64), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([weight, bias], lr=recipe['lr'],
        max_iter=recipe['max_iter'], max_eval=recipe['requested_max_eval'],
        tolerance_grad=recipe['tolerance_grad'], tolerance_change=recipe['tolerance_change'],
        history_size=recipe['history_size'], line_search_fn=recipe['line_search_fn'])
    trace = []
    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = torch.nn.functional.linear(x, weight, bias)
        ce = -(y * logits.log_softmax(1)).sum(1)
        objective = (weights * ce).sum() + .5*recipe['lambda_weight_l2']*weight.square().sum()
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError('Nonfinite convex objective')
        objective.backward()
        grad = max(float(weight.grad.abs().max()), float(bias.grad.abs().max()))
        if not np.isfinite(grad):
            raise FloatingPointError('Nonfinite convex gradient')
        trace.append(dict(function_call=len(trace)+1, objective=float(objective.detach()), gradient_infinity=grad))
        if len(trace) == 1 or len(trace) % 25 == 0:
            print(json.dumps(dict(event='readout_LBFGS', **trace[-1])), flush=True)
        return objective
    optimizer.step(closure)
    # Independent endpoint evaluation is not counted as an optimizer function call.
    state = optimizer.state[weight]
    optimizer_calls = int(state.get('func_evals', 0))
    iterations = int(state.get('n_iter', 0))
    endpoint = float(closure().detach())
    trace[-1]['endpoint_recheck_not_optimizer_call'] = True
    grad = trace[-1]['gradient_infinity']
    budget = iterations >= recipe['max_iter'] or optimizer_calls >= recipe['requested_max_eval']
    previous = float(state['prev_loss']) if 'prev_loss' in state else None
    absolute_change = abs(endpoint - previous) if previous is not None else None
    d = state.get('d')
    t = state.get('t')
    step_inf = float((d*t).abs().max()) if d is not None and t is not None else None
    prev_grad = state.get('prev_flat_grad')
    direction_derivative = float(prev_grad.dot(d)) if prev_grad is not None and d is not None else None
    gradient_done = grad <= recipe['tolerance_grad']
    small_change = ((absolute_change is not None and absolute_change < recipe['tolerance_change']) or
                    (step_inf is not None and step_inf <= recipe['tolerance_change']) or
                    (direction_derivative is not None and direction_derivative > -recipe['tolerance_change']))
    resolved = gradient_done or (not budget and small_change)
    status = ('gradient_tolerance' if gradient_done else
              'solver_absolute_change_or_direction_tolerance' if resolved else
              'budget_with_optimization_unresolved' if budget else 'unresolved_solver_stop')
    report = dict(optimizer='CPU64_fullbatch_LBFGS', resolved=resolved, status=status,
                  objective=endpoint, final_gradient_infinity=grad,
                  n_iter=iterations, actual_optimizer_func_evals=optimizer_calls,
                  closure_calls_including_endpoint_recheck=len(trace),
                  requested_max_eval=recipe['requested_max_eval'],
                  requested_eval_cap_is_not_hard_line_search_cap=True,
                  absolute_objective_change_from_solver_previous=absolute_change,
                  final_step_infinity=step_inf, stored_directional_derivative=direction_derivative,
                  tolerance_change_is_absolute=True, trace=trace)
    return weight.detach().float(), bias.detach().float(), report


def execute_plan(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import frozen_readout as helper
    plan_file = Path(args.plan)
    if helper.sha256(plan_file) != args.plan_sha256.lower():
        raise ValueError('Plan SHA mismatch')
    plan = helper.read_json(plan_file)
    if plan['source_sha256'] != helper._source_map(plan['code_root']):
        raise ValueError('Source changed after plan')
    for key in ('weights', 'manifest'):
        if helper.sha256(plan[key]) != plan[key+'_sha256']:
            raise ValueError(key+' changed after plan')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA feature extraction required; no silent fallback')
    output = plan_file.parent
    if (output/'execution_started.json').exists():
        raise FileExistsError('Immutable execution; no restart')
    torch.set_num_threads(4)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    sys.path.insert(0, plan['code_root'])
    from historical.func.AnatomicalClassifier import AirwayAnatomicalClassifier
    classifier = AirwayAnatomicalClassifier(in_channels=1, generation_classes=5,
        structure_classes=3, location_classes=7)
    bundle = torch.load(plan['weights'], map_location='cpu')
    if isinstance(bundle, dict) and 'model' in bundle:
        bundle = bundle['model']
    state = {k[len('classifier.'):]: v for k, v in bundle.items() if k.startswith('classifier.')}
    classifier.load_state_dict(state, strict=True)
    classifier.eval().cuda()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    frozen_before = helper._state_hash(classifier)
    rng_before = _rng_hash(torch)
    rows = sum(len(s['window_sample']['strata']) for s in plan['bank'])
    estimate = rows*(64*4+5*4+3*2+8*3+4*3)
    if estimate > plan['max_feature_cache_bytes']:
        raise MemoryError('Prespecified sparse feature storage cap exceeded')
    helper.write_json(output/'execution_started.json', dict(plan_sha256=args.plan_sha256,
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
        estimated_sparse_bytes=estimate, old_classifier_state_sha256=frozen_before))
    bank_lists = {k: [] for k in ('feature','target','coord','inclusion_weight','metric_weight',
                                  'objective_weight','foreground','case_index','patch_index','patch_stratum')}
    case_ids = plan['fit_cases']+plan['check_cases']
    by_case = {}
    started = time.perf_counter()
    with torch.no_grad():
        for index, spec in enumerate(plan['bank']):
            identity = plan['inputs'][spec['case_id']]
            if spec['case_id'] not in by_case:
                if not helper._same_file_identity(identity['image']) or not helper._same_file_identity(identity['mask']):
                    raise ValueError('TRAIN image/mask identity changed after plan')
                if helper.sha256(Path(identity['cache_dir'])/'geometry.npz') != identity['geometry_sha256']:
                    raise ValueError('TRAIN geometry changed after plan')
                by_case[spec['case_id']] = helper._load_geometry(identity['cache_dir'])
            local = helper.prepare_example(identity['cache_dir'], by_case[spec['case_id']], spec, True)
            if helper.target_stats(local) != spec['target_audit']:
                raise ValueError('Local target audit changed')
            image = torch.from_numpy(local.pop('image')[None]).cuda()
            with torch.cuda.amp.autocast():
                raw = classifier(image)['encoder_features'][0]
            aligned = helper.align_classifier_features(raw)
            coord = np.asarray(spec['window_sample']['coords_zyx'], np.int64)
            indices = [torch.from_numpy(coord[:, a]).cuda() for a in range(3)]
            feature = aligned[0, :, indices[0], indices[1], indices[2]].T.contiguous().cpu().numpy()
            target = local['target'][:, coord[:,0],coord[:,1],coord[:,2]].T.copy()
            ipw = np.asarray(spec['window_sample']['inverse_inclusion_weights'], np.float64)
            fg = np.asarray([x == 'foreground' for x in spec['window_sample']['strata']])
            if not local['loss_valid'][tuple(coord.T)].all():
                raise ValueError('Plan selected an unknown or unsupported window')
            if not np.array_equal(fg, local['foreground_window'][tuple(coord.T)]):
                raise ValueError('Sample stratum changed')
            metric = ipw / spec['target_audit']['valid_cells']
            values = dict(feature=feature,target=target,coord=coord.astype(np.int16),
                          inclusion_weight=ipw,metric_weight=metric,
                          objective_weight=metric/192.,foreground=fg,
                          case_index=np.full(len(coord),case_ids.index(spec['case_id']),np.int32),
                          patch_index=np.full(len(coord),index,np.int32),
                          patch_stratum=np.full(len(coord),int(spec['stratum']=='CT_uniform'),np.int8))
            for key, value in values.items():
                bank_lists[key].append(value)
            del image, raw, aligned
            if (index+1)%8 == 0:
                print(json.dumps(dict(event='sparse_features_case_complete',patches=index+1,case=spec['case_id'])),flush=True)
    frozen_after = helper._state_hash(classifier)
    rng_after = _rng_hash(torch)
    if frozen_before != frozen_after or rng_before != rng_after:
        raise RuntimeError('Frozen classifier state or RNG changed during deterministic extraction')
    del classifier, by_case
    torch.cuda.empty_cache()
    bank = {k: np.concatenate(v,axis=0) for k,v in bank_lists.items()}
    if any(not np.isfinite(bank[k]).all() for k in ('feature','target','inclusion_weight','objective_weight')):
        raise FloatingPointError('Nonfinite sparse bank')
    total_bytes = sum(x.nbytes for x in bank.values())
    if total_bytes > plan['max_feature_cache_bytes']:
        raise MemoryError('Actual sparse storage cap exceeded')
    np.savez_compressed(output/'fixed_sparse_vectors.npz',**bank)
    fit_rows = np.flatnonzero(bank['case_index'] < 24)
    check_rows = np.flatnonzero(bank['case_index'] >= 24)
    if not np.isclose(bank['objective_weight'][fit_rows].sum(),1.):
        raise AssertionError('IPW case/patch equal fit objective mass invalid')
    weight,bias,optimization = fit_convex(torch,bank,fit_rows,plan['fit_recipe'])
    endpoint = dict(weight=weight.reshape(5,64,1,1,1),bias=bias)
    endpoint_path = output/'readout_endpoint_fp32.pt'
    torch.save(endpoint,endpoint_path)
    with torch.no_grad():
        # Re-evaluate actual FP32 deployment weights on FP32 stored features.
        probability = torch.nn.functional.linear(torch.from_numpy(bank['feature']),weight,bias).softmax(1).numpy()
    prior = _prior(bank,fit_rows)
    evaluation = {name:evaluate(bank,probability,indices,plan,prior)
                  for name,indices in (('fit',fit_rows),('check',check_rows))}
    gate = semantic_gate(evaluation['check'],plan['calibration_gate'],optimization['resolved'])
    report = dict(schema='GTA-AVT-frozen-local-readout-B-calibration-v2',
        plan_sha256=args.plan_sha256,source_sha256=plan['source_sha256'],
        old_classifier_state_sha256=frozen_before,old_classifier_unchanged=True,
        RNG_unchanged_during_extraction=True,shared_encoder_updated=False,
        heldout_data_files_opened=False,fit_case_count=24,check_case_count=12,
        bank_array_bytes=total_bytes,bank_compressed_sha256=helper.sha256(output/'fixed_sparse_vectors.npz'),
        endpoint_head_file_sha256=helper.sha256(endpoint_path),
        endpoint_head_state_sha256=helper._state_hash(_head_from_state(torch,endpoint)),
        optimization=optimization,fit_foreground_prior=prior,evaluations=evaluation,
        calibration_gate=gate,elapsed_seconds=time.perf_counter()-started,
        numpy_version=np.__version__,
        role='frozen eval-feature readability diagnostic and shared head initialization for paired A; this gate alone does not authorize any segmentation continuation',
        limits='TRAIN-internal readout-label probe; all patients previously seen by old backbone; sampled overlapping windows; no segmentation or independent-patient gain established')
    helper.write_json(output/'analysis.json',report)
    print(json.dumps(dict(event='frozen_readout_complete',passed=gate['passed'],
        optimizer_status=optimization['status'],elapsed_seconds=report['elapsed_seconds'])),flush=True)
    return report


def _head_from_state(torch,state):
    # Preserve global RNG even though Conv3d constructor initializes parameters.
    with torch.random.fork_rng(devices=[]):
        head=torch.nn.Conv3d(64,5,1)
    head.load_state_dict(state,strict=True)
    return head
