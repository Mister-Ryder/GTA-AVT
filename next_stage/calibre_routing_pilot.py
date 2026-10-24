"""One bounded three-axis zero-sum calibre-routing candidate; default inert.

Each host runs C0/A_only/AR serially from the same initialization and plan.
The same original four SAAM branches/common projections and576 route gates
learn; only the conditional routing freedom changes from one to three axes. Oracle conditions are GT diagnostic inputs, never CT-only inference.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
HELPER_SHA = '016157cf52f7e9d9b7593435d5557dc55166881e94c59ad70bc68857186c8604'
OLD_PILOT_SHA = '88a764a24bba5db728abefe1142d11d7a7382a780c022e78dd443b6e25632467'
BANK_SHA = '9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
ARMS = ('C0', 'A_only', 'AR')
SEEDS = (20261030, 20261031, 20261032)
RECIPE = dict(updates_per_arm=2000, batch_size=2, patch_size=128, augmentation=False,
    optimizer='AdamW', shared_learning_rate=.0001, gate_learning_rate=.001, weight_decay=.00001,
    gradient_clipping=False, loss='Original main Dice+BCE only; no DS, GDF or attribute loss',
    alpha='min(1, successful_step/250); QA uses1 for both disposable updates',
    trainable_parameters=115458, shared_parameters=114882, gate_parameters=576,
    routing_change='Three bounded zero-sum axes instead of one signed RF contrast; same original learning scope, seeds and plans; all three arms reset',
    source_guard='Each Dongyang/KiTS/Rider case-equal sameweight correct-minus-half-ring Dice must be nonnegative on unchanged primary support',
    CT_uniform_guard='Known-nonempty CT_uniform case-equal clDice/CLrecall/fine must be nonworse versus both controls on identical defined support',
    sampler='FIT24 case-uniform and original16-slot-uniform; same precomputed plan for all arms',
    normalization='FIT24 case-equal first and second log-radius moments over all16slots A1 sites; no epsilon',
    oracle='GT-only partial native-radius oracle at nominal .5+2u SAAM64 grid; unknown neutralization is not a negative target',
    correspondence='Within each same patch, half-ring roll of own A1 flattened radius values; exact ownA/support/radius multiset',
    primary='Known-nonempty original GT-jitter slots0..7, equal patches then equal PROBE12 cases',
    mechanism_gates=dict(minimum_mean_Dice_pp=.1, minimum_positive_cases=8, all12_supported=True),
    scope='New oracle capability candidate; not CT-only GTA validation, not cross-round causal attribution')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(value)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def load_module(path, digest, name):
    require(sha(path) == digest, 'Explicit module SHA differs: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def average(values):
    return st.mean(values) if values and all(v is not None for v in values) else None


def array_sha(np, value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(); h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def radius_permutation(np, A, R):
    """Same patch, same support and complete radius multiset; no donor search."""
    require(A.shape == R.shape == (64, 64, 64), 'Actual SAAM64 oracle grid required')
    eligible = np.flatnonzero(A.reshape(-1) == 1)
    output = R.copy()
    if len(eligible) > 1:
        output.reshape(-1)[eligible] = np.roll(R.reshape(-1)[eligible], len(eligible) // 2)
    require(np.array_equal(np.sort(R.reshape(-1)[eligible]), np.sort(output.reshape(-1)[eligible])) and
        np.array_equal(R[A == 0], output[A == 0]), 'Radius permutation changed multiset or neutral region')
    return output, dict(eligible_sites=int(len(eligible)), shift=int(len(eligible) // 2),
        changed_radius_sites=int(np.count_nonzero(R != output)),
        own_A_sha256=array_sha(np, A), correct_radius_multiset_sha256=array_sha(np, np.sort(R[A == 1])),
        permuted_radius_multiset_sha256=array_sha(np, np.sort(output[A == 1])),
        own_A_and_support_preserved=True)


def oracle_manifest(args, bank):
    require(sha(args.oracle_bank) == args.oracle_bank_sha256, 'Oracle bank SHA differs')
    value = read(args.oracle_bank)
    require(value['bank_sha256'] == BANK_SHA and value['oracle_helper_sha256'] == args.oracle_helper_sha256,
        'Oracle source/bank differs')
    require(sha(args.oracle_helper) == args.oracle_helper_sha256, 'Oracle helper source differs')
    cases = value['cases']; normal = value['normalization']
    require(cases and set(cases) <= set(bank['case_ids']), 'Oracle bank is not TRAIN-only')
    if args.mode == 'run':
        require(value.get('disposable_engineering_only') is False and set(cases) == set(bank['case_ids']) and
            value['fit_cases'] == bank['fit_cases'] and value['probe_cases'] == bank['probe_cases'] and
            normal['case_equal'] is True and normal['fit_cases'] == bank['fit_cases'],
            'Formal run requires complete36-case oracle bank and case-equal FIT24 moments')
    else:
        require(value.get('disposable_engineering_only') is True and normal['mean'] == 0. and
            normal['population_std'] == 1., 'QA requires explicitly disposable local bank with engineering moments0/1')
    require(type(normal['mean']) in (int, float) and math.isfinite(normal['mean']) and
        type(normal['population_std']) in (int, float) and math.isfinite(normal['population_std']) and
        normal['population_std'] > 0., 'Undefined radius moments; no epsilon or fallback')
    return value


def make_plan(np, seed, fit):
    rng = np.random.default_rng(seed)
    return [[(fit[int(rng.integers(24))], int(rng.integers(16))) for _ in range(2)] for _ in range(2000)]


def parameter_groups(model):
    supplied = model.trainable_groups()
    require(set(supplied) == {'shared', 'gates'}, 'Model must expose the exact two optimizer groups')
    shared, gates = supplied['shared'], supplied['gates']
    expected = {id(value) for value in shared + gates}
    active = {name: value for name, value in model.named_parameters() if value.requires_grad}
    require({id(value) for value in active.values()} == expected and
        sum(v.numel() for v in shared) == 114882 and sum(v.numel() for v in gates) == 576,
        'Only exact four original branches plus common projections114882 and gate576 may train')
    return shared, gates, tuple(active)


def check_state(model, active):
    require({name for name, value in model.named_parameters() if value.requires_grad} == set(active), 'Trainable set changed')
    require(all(not module.training for module in model.modules()), 'Frozen/eval mode policy changed')
    require(all(value.grad is None for name, value in model.named_parameters() if name not in active),
        'Frozen parameter received a gradient')


def frozen_digest(old, model, active):
    return old.tensor_map_sha({name: value for name, value in model.state_dict().items() if name not in active})


def trace_stats(trace):
    def stats(value):
        value = value.detach().float()
        return dict(mean=float(value.mean()), SD=float(value.std(unbiased=False)), maxabs=float(value.abs().max()))
    return dict(zA=stats(trace['zA']), zR=stats(trace['zR']),
        legacy_axis1_only=dict(gain_absmean=float(trace['gain_absmean']),
            gain_maxabs=float(trace['gain_maxabs']), D_RMS=float(trace['D_RMS'])),
        routing_axes=[dict(axis=index + 1, gain_absmean=float(trace['axis_gain_absmean'][index]),
            gain_maxabs=float(trace['axis_gain_maxabs'][index]), gain_RMS=float(trace['axis_gain_RMS'][index]),
            direction_RMS=float(trace['axis_direction_RMS'][index])) for index in range(3)],
        route_coeff_min=float(trace['route_coeff_min']), route_coeff_max=float(trace['route_coeff_max']),
        route_coeff_sum_residual_maxabs=float(trace['route_coeff_sum_residual_maxabs']))


def summarize_mechanism(rows, probe):
    cases = {}
    for case in probe:
        selected = [row for row in rows if row['case_id'] == case and row['primary']]
        count = sum(row['radius_moments']['count'] for row in selected)
        total = math.fsum(row['radius_moments']['sum'] for row in selected)
        squares = math.fsum(row['radius_moments']['sum_squares'] for row in selected)
        variance = max(0., squares / count - (total / count)**2) if count else None
        changed = sum(row['zR_changed'] for row in selected)
        supported = len(selected) >= 2 and variance is not None and variance > 0. and changed > 0
        effect = 100. * average([row['normal']['dice'] - row['permuted']['dice'] for row in selected]) if selected else None
        cases[case] = dict(primary_patches=len(selected), supported=supported, radius_sites=count,
            raw_log_radius_variance=variance, changed_primary_zR_slots=changed,
            correct_minus_permuted_Dice_pp=effect if supported else None, descriptive_Dice_pp=effect,
            mean_abs_logit_response=average([row['mean_abs_logit_response'] for row in selected]))
    effects = [cases[c]['correct_minus_permuted_Dice_pp'] for c in probe]
    mean = average(effects)
    parts = dict(all12_supported=len(cases) == 12 and all(cases[c]['supported'] for c in probe),
        mean_at_least_point1pp=mean is not None and mean >= .1,
        at_least8_positive=sum(v is not None and v > 0. for v in effects) >= 8)
    return dict(intervention='own_A_half_ring_radius', per_case=cases, mean_Dice_effect_pp=mean,
        positive_cases=sum(v is not None and v > 0. for v in effects), gate_parts=parts,
        passed=all(parts.values()), own_A_preserved=True, radius_multiset_preserved=True)


def CT_uniform_safety(arm_summaries, probe):
    """Tighter secondary guard only; never rescues the original primary screen."""
    metrics = ('cldice', 'centerline_recall', 'fine_recall')
    output = {}
    for baseline in ('C0', 'A_only'):
        records = {}
        for metric in metrics:
            ar = {case: arm_summaries['AR']['per_case'][case]['uniform_nonempty'][metric] for case in probe}
            control = {case: arm_summaries[baseline]['per_case'][case]['uniform_nonempty'][metric] for case in probe}
            ar_ids = [case for case in probe if ar[case] is not None]
            control_ids = [case for case in probe if control[case] is not None]
            identical = ar_ids == control_ids
            supported = identical and bool(ar_ids)
            differences = {case: ar[case] - control[case] for case in ar_ids} if supported else {}
            effect = average(list(differences.values())) if supported else None
            records[metric] = dict(AR_supported_cases=ar_ids, control_supported_cases=control_ids,
                supports_identical=identical, supported_cases=len(ar_ids) if identical else None,
                AR_mean=average([ar[case] for case in ar_ids]) if ar_ids else None,
                control_mean=average([control[case] for case in control_ids]) if control_ids else None,
                per_case_AR_minus_control=differences, mean_AR_minus_control_pp=100. * effect if effect is not None else None,
                per_case_support={case: dict(AR={key: arm_summaries['AR']['per_case'][case]['uniform_nonempty'][key]
                    for key in ('patches', 'fine_nodes', 'fine_covered')},
                    control={key: arm_summaries[baseline]['per_case'][case]['uniform_nonempty'][key]
                        for key in ('patches', 'fine_nodes', 'fine_covered')}) for case in probe},
                passed=supported and effect >= 0.)
        output[baseline] = dict(metrics=records, passed=all(value['passed'] for value in records.values()))
    return dict(scope='Secondary CT-transfer safety screen, CT_uniform known-nonempty slots8..15 only',
        aggregation='Equal supported cases; within-case fine_covered/fine_nodes pooling unchanged',
        comparisons=output, passed=all(value['passed'] for value in output.values()),
        stricter_only=True, cannot_rescue_original_primary=True, CT_only_efficacy_established=False)


def SOURCE_safety(mechanism, probe):
    """All original cases retained: a source with missing support cannot pass."""
    sources = {}
    for source in ('Dongyang', 'KiTS', 'Rider'):
        cases = [case for case in probe if case.split('_')[0] == source]
        supported = [case for case in cases if mechanism['per_case'][case]['supported'] and
            mechanism['per_case'][case]['correct_minus_permuted_Dice_pp'] is not None]
        complete = len(cases) == 4 and supported == cases
        value = average([mechanism['per_case'][case]['correct_minus_permuted_Dice_pp'] for case in cases]) if complete else None
        sources[source] = dict(expected_case_ids=cases, supported_case_ids=supported,
            complete_unchanged_support=complete, case_equal_correct_minus_half_ring_Dice_pp=value,
            passed=complete and value >= 0.)
    return dict(scope='Prospective SOURCE safety: sameweight original primary patches/cases',
        source_groups=sources, passed=all(value['passed'] for value in sources.values()),
        no_source_reclassification_or_removal=True, stricter_only=True, cannot_rescue_original_primary=True)


def execute(args):
    helper = load_module(args.helper, HELPER_SHA, 'calibre_original_boundary_helper')
    old = load_module(args.old_pilot, OLD_PILOT_SHA, 'calibre_original_pilot_math')
    require(sha(args.request) == args.request_sha256 and sha(args.bank) == args.bank_sha256 == BANK_SHA,
        'Explicit original request/bank differs')
    request = read(args.request); root, source, manifest, bank, records = helper.metadata(request)
    require(sha(request['bank']) == sha(args.bank), 'Request and explicit bank differ')
    oracle = oracle_manifest(args, bank)
    output = Path(args.output); require(not output.exists(), 'Fresh candidate output required')
    import numpy as np
    import torch
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    for path in (root / 'code', root / 'next_stage'):
        sys.path.insert(0, str(path))
    modules = {name: importlib.import_module(name) for name in ('data', 'geometry', 'losses', 'train')}
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == root / 'code' / (name + '.py'), 'Scientific import escaped original source42')
    candidate = load_module(args.model, args.model_sha256, 'calibre_routing_coadapt_actual_model')
    require(torch.cuda.is_available(), 'Actual CUDA QA/run required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    common = torch.load(request['weights'], map_location='cpu')
    plan = make_plan(np, args.seed, bank['fit_cases'])
    if args.mode == 'qa':
        first_case = next(iter(oracle['cases']))
        qa_slots = oracle['cases'][first_case].get('slots', [0, 1])
        require(len(qa_slots) >= 2, 'Disposable QA bank must have two actual patches')
        plan = [[(first_case, int(qa_slots[0])), (first_case, int(qa_slots[1]))]] * 2
    plan_sha = hashlib.sha256(encoded(plan)).hexdigest()
    output.mkdir(parents=True)
    write(output / 'intent.json', dict(schema='GTA-AVT-calibre-routing-coadapt-pilot-intent-v1',
        recipe=RECIPE, mode=args.mode, seed=args.seed, worker_sha256=sha(__file__), model_sha256=args.model_sha256,
        request_sha256=args.request_sha256, bank_sha256=BANK_SHA, oracle_bank_sha256=args.oracle_bank_sha256,
        oracle_helper_sha256=args.oracle_helper_sha256, engineering_only=args.mode == 'qa',
        old_metric_helper_sha256=OLD_PILOT_SHA, no_VAL_or_TEST_arrays=True))
    write(output / 'normalization.json', oracle['normalization'])
    write(output / 'training_plan.json', dict(plan=plan, sha256=plan_sha, common_to_all_three_arms=True))
    specmap = {(r['case_id'], r['slot']): r for r in bank['patches']}
    opened, opened_oracle, inputs_before, oracle_before = {}, {}, {}, {}

    def oracle_arrays(case, slot):
        if case not in opened_oracle:
            item = oracle['cases'][case]
            require(sha(item['path']) == item['sha256'], 'Prepared oracle case bytes differ')
            if 'receipt_path' in item:
                require(sha(item['receipt_path']) == item['receipt_sha256'], 'Oracle sidecar bytes differ')
            with np.load(item['path'], allow_pickle=False) as stream:
                data = {key: stream[key].copy() for key in ('A', 'R', 'state', 'reason', 'slot')}
            slots = data['slot'].tolist()
            require(len(slots) == len(set(slots)) and all(0 <= s < 16 for s in slots), 'Invalid oracle slots')
            require(args.mode != 'run' or slots == list(range(16)), 'Formal oracle bank must retain all16 slots')
            shape = (len(slots), 64, 64, 64)
            require(data['A'].shape == data['R'].shape == data['state'].shape == data['reason'].shape == shape and
                data['A'].dtype == np.uint8 and data['R'].dtype == np.float32 and
                data['state'].dtype == np.uint8 and data['reason'].dtype == np.uint16 and
                np.isfinite(data['R']).all() and np.isin(data['A'], [0, 1]).all() and
                np.array_equal(data['A'] == 1, data['state'] == 1) and (data['R'][data['A'] == 0] == 0.).all(),
                'Oracle finite neutralization/support/dtype differs')
            opened_oracle[case] = (data, {s: i for i, s in enumerate(slots)})
            oracle_before[case] = dict(path=item['path'], sha256=item['sha256'])
        data, positions = opened_oracle[case]; i = positions[slot]
        return data['A'][i], data['R'][i], data['state'][i], data['reason'][i]

    def prepare(case, slot):
        if case not in opened:
            require(case in bank['case_ids'] and records[case]['cache_file_sha256'] == bank['expected_TRAIN_cache_sha256'][case],
                'Non-TRAIN cache access')
            inputs_before.update(helper.cached_inputs(dict(request, case_ids=[case]), records))
            home = Path(records[case]['cache_dir'])
            image = np.load(home / 'image.npy', mmap_mode='r', allow_pickle=False)
            mask = np.load(home / 'mask.npy', mmap_mode='r', allow_pickle=False)
            with np.load(home / 'geometry.npz', allow_pickle=False) as stream:
                geometry = {key: stream[key] for key in stream.files}
            require(image.shape == mask.shape and list(image.shape) == bank['headers'][case]['shape_zyx'], 'Cache shape differs')
            opened[case] = image, mask, geometry
        image, mask, geometry = opened[case]; spec = specmap[(case, slot)]; start = spec['start_zyx']
        slices = tuple(slice(v, min(v + 128, n)) for v, n in zip(start, image.shape))
        raw = np.asarray(image[slices], np.float32); truth = np.asarray(mask[slices], bool)
        pad = [(0, 128 - n) for n in raw.shape]; valid = np.pad(np.ones(raw.shape, bool), pad)
        normalized = np.ascontiguousarray(modules['data'].normalize(np.pad(raw, pad, constant_values=-1024))[None], np.float32)
        scale, structure = modules['geometry'].patch_histograms(geometry, start, [128] * 3)
        attribute_valid = bool(scale.sum() > 0 and not (truth.any() and scale[0] == 1.))
        point = geometry['points']; chosen = ((point >= np.asarray(start)) & (point < np.asarray(start) + 128)).all(1)
        coordinate = 2. * (np.asarray(records[case]['crop_start_zyx']) + np.asarray(start) + 64.) / np.maximum(
            np.asarray(records[case]['full_shape_zyx']) - 1., 1.) - 1.
        A, R, state, reason = oracle_arrays(case, slot)
        return dict(image=normalized, truth=np.pad(truth, pad), valid=valid, points=point[chosen] - np.asarray(start),
            radii=geometry['radius'][chosen], coordinate=coordinate.astype(np.float32), A=A, R=R, state=state, reason=reason,
            case_id=case, source=spec['source'], slot=slot, stratum=spec['stratum'],
            attribute_valid=attribute_valid, known_nonempty=bool(attribute_valid and truth.any()),
            valid_empty=bool(attribute_valid and not truth.any()))

    def batch(pairs):
        prepared = [prepare(case, slot) for case, slot in pairs]
        image = torch.from_numpy(np.stack([r['image'] for r in prepared])).cuda()
        truth = torch.from_numpy(np.stack([r['truth'][None] for r in prepared])).cuda().float()
        coordinate = torch.from_numpy(np.stack([r['coordinate'] for r in prepared])).cuda()
        fields = dict(A=torch.from_numpy(np.stack([r['A'][None] for r in prepared])).cuda().float(),
            logR=torch.from_numpy(np.stack([r['R'][None] for r in prepared])).cuda(),
            valid=torch.from_numpy(np.stack([(r['A'] == 1)[None] for r in prepared])).cuda())
        return prepared, image, truth, coordinate, fields

    arm_summaries, qa, common_init, active_names = {}, {}, None, None
    for arm in ARMS:
        folder = output / arm; folder.mkdir()
        modules['train'].set_seed(args.seed)
        model = candidate.CalibreRoutingStageModel(arm=arm, alpha=0.)
        model.load_common_state(common)
        model.install_moments(oracle['normalization']['mean'], oracle['normalization']['population_std'])
        model.cuda().train()
        shared, gates, active = parameter_groups(model); check_state(model, active)
        require(active_names is None or active_names == active, 'Arm trainable inventories differ')
        active_names = active
        initial = old.tensor_map_sha(model.state_dict())
        require(common_init is None or common_init == initial, 'Three arms do not have exactly identical initial tensors')
        common_init = initial; frozen_before = frozen_digest(old, model, active)
        active_initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if name in active}
        optimizer = torch.optim.AdamW([dict(params=shared, lr=.0001), dict(params=gates, lr=.001)], weight_decay=.00001)
        scaler = torch.cuda.amp.GradScaler(); steps = 2 if args.mode == 'qa' else 2000
        first_grads = None; rows, mechanism_rows = [], []
        with (folder / 'training.jsonl').open('xb') as stream:
            for step in range(steps):
                prepared, image, truth, coordinate, fields = batch(plan[step])
                alpha = 1. if args.mode == 'qa' else min(1., (step + 1) / 250.)
                if step == 0 and args.mode == 'qa':
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        zero = model(image, coordinate=coordinate, oracle=fields, oracle_mode=True, alpha=1.)['logits'].float()
                        parity = model(image, coordinate=coordinate, oracle=fields, oracle_mode=True, alpha=0.)['logits'].float()
                    require(torch.equal(zero, parity), 'Zero residual does not preserve common G0')
                    original = candidate.StageModel('constant', True, 0., .2, .2)
                    incompatible = original.load_state_dict(common, strict=False)
                    require(not incompatible.unexpected_keys and set(incompatible.missing_keys) <= {'gdf_head.weight', 'gdf_head.bias'},
                        'Original StageModel checkpoint loading differs')
                    original.cuda().eval()
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        reference = original(image, coordinate=coordinate, condition_mode='constant', alpha=0.)['logits'].float()
                    require(torch.equal(zero, reference), 'Zero-start candidate differs from actual original constant StageModel')
                    zero_digest = array_sha(np, zero.cpu().numpy())
                    require(not qa or zero_digest == next(iter(qa.values()))['initial_logits_sha256'], 'Initial arm forwards differ')
                    del zero, parity, original, reference
                def forward_loss():
                    with torch.cuda.amp.autocast():
                        prediction = model(image, coordinate=coordinate, oracle=fields, oracle_mode=True, alpha=alpha)['logits']
                        loss = modules['losses'].dice_bce(prediction, truth)
                    return loss, dict(loss=float(loss.detach()))
                details, update = modules['train'].optimizer_update_with_amp_retries(model, optimizer, scaler,
                    forward_loss, gradient_clip=float('inf'), max_attempts=8)
                check_state(model, active)
                gradients = {name: None if value.grad is None else float(value.grad.detach().abs().max())
                    for name, value in model.named_parameters() if name in active}
                coefficient_rows = {name: [None if value.grad is None else float(value.grad[row].detach().abs().max())
                    for row in range(3)] for name, value in model.named_parameters() if name in
                    ('calibre_bias', 'calibre_availability_weight', 'calibre_radius_weight')}
                if step == 0:
                    first_grads = gradients
                    first_row_grads = coefficient_rows
                trace = model.last_calibre_trace
                record = dict(step=step + 1, alpha=alpha, **details, **update,
                    fields=trace_stats(trace), gradient_maxabs=gradients, coefficient_row_gradient_maxabs=coefficient_rows,
                    gate_norms=dict(bias=float(model.calibre_bias.detach().norm()),
                        availability=float(model.calibre_availability_weight.detach().norm()), radius=float(model.calibre_radius_weight.detach().norm())))
                stream.write(encoded(record)); stream.flush()
                del prepared, image, truth, coordinate, fields
        check_state(model, active)
        require(frozen_before == frozen_digest(old, model, active), 'Nontrainable parameters or buffers changed')
        changed = {name: not torch.equal(value.detach().cpu(), active_initial[name])
            for name, value in model.named_parameters() if name in active}
        if args.mode == 'qa':
            require(any(changed[name] for name in active if name.startswith('structure_guidance.morph_fusion.')) and
                any(changed[name] for name in active if name.startswith('anatomical_output_fusion.')),
                'Shared SAAM or final projection did not actually update')
            named = dict(model.named_parameters())
            row_changes = {name: [not torch.equal(named[name][row].detach().cpu(), active_initial[name][row])
                for row in range(3)] for name in ('calibre_bias', 'calibre_availability_weight', 'calibre_radius_weight')}
            require(all(value is not None and value > 0. for value in first_row_grads['calibre_bias']) and
                all(row_changes['calibre_bias']), 'Each route-axis bias needs real gradient/update')
            branch_updates = {}
            for branch in range(4):
                prefix = 'structure_guidance.morphology_convs.%d.' % branch
                names = [prefix + 'weight', prefix + 'bias']
                require(all(name in first_grads and name in changed for name in names),
                    'Original branch parameter missing from optimizer inventory')
                gradients_real = any(first_grads[name] is not None and first_grads[name] > 0. for name in names)
                parameters_changed = any(changed[name] for name in names)
                require(gradients_real and parameters_changed, 'Original branch lacks actual gradient/update: ' + str(branch))
                branch_updates[str(branch)] = dict(first_gradient_maxabs={name: first_grads[name] for name in names},
                    actual_parameter_changes={name: changed[name] for name in names},
                    real_gradient_and_update=True)
            for name, enabled in (('calibre_availability_weight', arm != 'C0'), ('calibre_radius_weight', arm == 'AR')):
                for row, grad in enumerate(first_row_grads[name]):
                    require(grad is not None and (grad > 0. and row_changes[name][row] if enabled else
                        grad == 0. and not row_changes[name][row]),
                        'Condition axis consumed/updated contrary to arm: ' + arm + '/' + name + '/' + str(row))
            qa[arm] = dict(initial_logits_sha256=zero_digest, initial_state_sha256=initial,
                successful_updates=2, alpha=1., first_gradient_maxabs=first_grads, actual_parameters_changed=changed,
                first_coefficient_row_gradient_maxabs=first_row_grads, coefficient_rows_changed=row_changes,
                trainable_parameters=115458, frozen_state_unchanged=True, all_modules_eval=True,
                engineering_moments_only=True, oracle_conditioned_not_CT_only=True,
                four_original_branch_updates=branch_updates)
            del named
            del model, optimizer, scaler, shared, gates, active_initial; torch.cuda.empty_cache()
            continue
        with (folder / 'trainable_final.pt').open('xb') as stream:
            torch.save(dict(schema='GTA-AVT-calibre-routing-coadapt-trainable-state-v1', arm=arm, seed=args.seed,
                successful_updates=2000, state={name: value.detach().cpu() for name, value in model.state_dict().items() if name in active},
                normalization=oracle['normalization'], common_weights_sha256=helper.WEIGHT_SHA,
                model_sha256=args.model_sha256, oracle_bank_sha256=args.oracle_bank_sha256), stream)
        with (folder / 'patches.jsonl').open('xb') as stream:
            for case in bank['probe_cases']:
                for begin in range(0, 16, 2):
                    prepared, image, truth, coordinate, fields = batch([(case, slot) for slot in (begin, begin + 1)])
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        logits = model(image, coordinate=coordinate, oracle=fields, oracle_mode=True, alpha=1.)['logits'].float()
                    binary = (logits.sigmoid() >= .5).cpu().numpy()[:, 0]; cpu = logits.cpu().numpy()[:, 0]
                    normal_zR = model.last_calibre_trace['zR'].detach().clone() if arm == 'AR' else None
                    normals = []
                    for index, item in enumerate(prepared):
                        item['prediction'] = binary[index]
                        row = {name: item[name] for name in ('case_id', 'source', 'slot', 'stratum', 'attribute_valid', 'known_nonempty', 'valid_empty')}
                        row.update(primary=item['slot'] < 8 and item['known_nonempty'],
                            metrics=old.metric(np, ndi, skeletonize, cpu[index], item),
                            oracle=dict(available_radius_sites=int(item['A'].sum()), unknown_sites=int((item['state'] == 255).sum())))
                        rows.append(row); normals.append(row); stream.write(encoded(row))
                    if arm == 'AR':
                        permuted, maps = zip(*(radius_permutation(np, item['A'], item['R']) for item in prepared))
                        override = dict(logR=torch.from_numpy(np.stack([value[None] for value in permuted])).cuda(),
                            valid=fields['valid'])
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            altered = model(image, coordinate=coordinate, oracle=fields, oracle_mode=True,
                                radius_override=override, diagnostic=True, alpha=1.)['logits'].float()
                        changed_zR = model.last_calibre_trace['zR']
                        altered_binary = (altered.sigmoid() >= .5).cpu().numpy()[:, 0]
                        altered_cpu = altered.cpu().numpy()[:, 0]
                        for index, item in enumerate(prepared):
                            item['prediction'] = altered_binary[index]; values = item['R'][item['A'] == 1].astype(np.float64)
                            mechanism_rows.append(dict(case_id=case, source=item['source'], slot=item['slot'],
                                stratum=item['stratum'], primary=normals[index]['primary'], normal=normals[index]['metrics'],
                                permuted=old.metric(np, ndi, skeletonize, altered_cpu[index], item), permutation=maps[index],
                                radius_moments=dict(count=len(values), sum=float(values.sum()), sum_squares=float((values * values).sum())),
                                zR_changed=not torch.equal(normal_zR[index], changed_zR[index]),
                                mean_abs_logit_response=float((altered[index] - logits[index]).abs().mean()),
                                binary_changed_voxels=int(((altered[index].sigmoid() >= .5) != (logits[index].sigmoid() >= .5)).sum())))
                        del altered, normal_zR, override
                    del logits, prepared, image, truth, coordinate, fields
        require(len(rows) == 192 and frozen_before == frozen_digest(old, model, active), 'Final probe incomplete or frozen state changed')
        arm_summaries[arm] = old.case_summary(rows)
        write(folder / 'analysis.json', dict(schema='GTA-AVT-calibre-routing-coadapt-arm-v1', arm=arm,
            successful_updates=2000, summary=arm_summaries[arm], frozen_state_unchanged=True))
        if arm == 'AR':
            require(len(mechanism_rows) == 192, 'Radius correspondence incomplete')
            with (folder / 'mechanism.jsonl').open('xb') as stream:
                for row in mechanism_rows:
                    stream.write(encoded(row))
            mechanism = summarize_mechanism(mechanism_rows, bank['probe_cases'])
            write(folder / 'mechanism_analysis.json', mechanism)
        del model, optimizer, scaler, shared, gates, active_initial; torch.cuda.empty_cache()
    if args.mode == 'qa':
        result = dict(schema='GTA-AVT-calibre-routing-coadapt-engineering-QA-v1', mode='qa', engineering_qa_passed=True,
            arms=qa, actual_patch_size=128, actual_batch_size=2, AMP=True, formal_scientific_outputs=False,
            actual_formal_plan_used=False, no_VAL_or_TEST_arrays=True)
        write(output / 'qa.json', result)
    else:
        comparisons = {}
        for baseline in ('C0', 'A_only'):
            delta = {name: 100. * (arm_summaries['AR']['primary_case_equal'][name] -
                arm_summaries[baseline]['primary_case_equal'][name])
                if arm_summaries['AR']['primary_case_equal'][name] is not None and
                arm_summaries[baseline]['primary_case_equal'][name] is not None else None
                for name in ('dice', 'cldice', 'centerline_recall', 'fine_recall')}
            delta['own_worst2_Dice'] = 100. * (arm_summaries['AR']['own_worst2_Dice'] - arm_summaries[baseline]['own_worst2_Dice'])
            parts = {name: value is not None and (value > 0. if name == 'dice' else value >= 0.) for name, value in delta.items()}
            comparisons[baseline] = dict(AR_minus_baseline_pp=delta, parts=parts, passed=all(parts.values()))
        original_primary = all(value['passed'] for value in comparisons.values()) and mechanism['passed']
        source_guard = SOURCE_safety(mechanism, bank['probe_cases'])
        uniform_guard = CT_uniform_safety(arm_summaries, bank['probe_cases'])
        result = dict(schema='GTA-AVT-calibre-routing-coadapt-pilot-readout-v1', mode='run', arm_summaries=arm_summaries,
            comparisons=comparisons, radius_mechanism=mechanism,
            original_primary_screen_passed=original_primary, CT_uniform_transfer_safety=uniform_guard, SOURCE_safety=source_guard,
            capacity_continue=original_primary and uniform_guard['passed'] and source_guard['passed'],
            oracle_conditioned_not_CT_only=True, goal_completed=False, native_VAL_authorized=False,
            automatic_successor=False, no_VAL_or_TEST_arrays=True,
            cross_round_change='Relative to115074-parameter basis candidate, only route freedom changes1to3 axes (+384 gates); same seeds/plans/attributes/recipe; SOURCE guard is prospectively stricter')
        write(output / 'analysis.json', result)
    after = helper.cached_inputs(dict(request, case_ids=list(inputs_before)), records)
    require(after == inputs_before and sha(request['weights']) == helper.WEIGHT_SHA and sha(args.model) == args.model_sha256 and
        sha(args.oracle_bank) == args.oracle_bank_sha256 and sha(args.oracle_helper) == args.oracle_helper_sha256,
        'Bound source/input/model changed')
    for case, item in oracle_before.items():
        require(sha(item['path']) == item['sha256'], 'Prepared oracle file changed: ' + case)
    for name, digest in source.items():
        require(sha(root / name) == digest, 'Original source42 changed')
    write(output / 'inputs.json', dict(source_manifest_sha256=helper.MAP_SHA, bank_sha256=BANK_SHA,
        common_weights_sha256=helper.WEIGHT_SHA, model_sha256=args.model_sha256, oracle_bank_sha256=args.oracle_bank_sha256,
        oracle_helper_sha256=args.oracle_helper_sha256, request_sha256=args.request_sha256,
        cache_before=inputs_before, cache_after=after, oracle_files=oracle_before,
        training_plan_sha256=plan_sha, initial_all_arm_state_sha256=common_init,
        trainable_parameter_names=list(active_names), no_VAL_or_TEST_arrays=True))
    members = {p.relative_to(output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(output.rglob('*')) if p.is_file()}
    write(output / 'summary.json', dict(schema='GTA-AVT-calibre-routing-coadapt-pilot-v1', status='verified_complete',
        mode=args.mode, seed=args.seed, files=members, engineering_qa_passed=args.mode == 'qa',
        fit_cases=bank['fit_cases'], probe_cases=bank['probe_cases'], worker_sha256=sha(__file__),
        model_sha256=args.model_sha256, oracle_bank_sha256=args.oracle_bank_sha256,
        original_primary_screen_passed=result.get('original_primary_screen_passed'),
        CT_uniform_transfer_safety_passed=result.get('CT_uniform_transfer_safety', {}).get('passed'),
        SOURCE_safety_passed=result.get('SOURCE_safety', {}).get('passed'),
        capacity_continue=result.get('capacity_continue'),
        elapsed_seconds=time.monotonic() - args.started, no_VAL_or_TEST_arrays=True, automatic_successor=False))
    return dict(status='verified_complete', mode=args.mode, engineering_qa_passed=args.mode == 'qa',
        summary_sha256=sha(output / 'summary.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'request-sha256', 'bank', 'bank-sha256', 'oracle-bank', 'oracle-bank-sha256',
        'oracle-helper', 'oracle-helper-sha256', 'model', 'model-sha256', 'output'):
        parser.add_argument('--' + name)
    parser.add_argument('--helper', default=str(Path(__file__).with_name('boundary_attribute_probe.py')))
    parser.add_argument('--old-pilot', default=str(Path(__file__).with_name('boundary_field_pilot.py')))
    parser.add_argument('--mode', choices=('qa', 'run'), default='run')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--port', type=int, choices=(34153, 17552, 38773))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', recipe=RECIPE, inputs_read=False, GPU=False)))
        return
    require(args.seed and all(getattr(args, name) for name in ('request', 'request_sha256', 'bank', 'bank_sha256',
        'oracle_bank', 'oracle_bank_sha256', 'oracle_helper', 'oracle_helper_sha256', 'model', 'model_sha256', 'output')),
        'Explicit source/data/oracle/model digests, seed and fresh output required')
    args.started = time.monotonic()
    print(json.dumps(execute(args), allow_nan=False))


if __name__ == '__main__':
    main()
