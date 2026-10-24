"""One fixed block-mean radius candidate with unchanged three-axis routing; default inert.

Each host newly trains AR from common7406. Fixed prior same-seed C0/A_only
controls are read-only references, not retrained or independent new repeats.
The same original four SAAM branches/common projections and576 route gates
learn; only R granularity changes from point-local to one supported block mean. Oracle conditions are GT diagnostic inputs, never CT-only inference.
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
REUSED_WORKER_SHA = 'a9320edc72b1dcde9fe4486355566bf9c491b78d109bf5315f578e08db32c9ef'
FIXED_MODEL_SHA = '9acd1c9d8b04fdef3e5a7e4cefe9d8f0d234978093eaf09d7d2a03a64288f58e'
RECIPE = dict(updates_per_arm=2000, batch_size=2, patch_size=128, augmentation=False,
    optimizer='AdamW', shared_learning_rate=.0001, gate_learning_rate=.001, weight_decay=.00001,
    gradient_clipping=False, loss='Original main Dice+BCE only; no DS, GDF or attribute loss',
    alpha='min(1, successful_step/250); QA uses1 for both disposable updates',
    trainable_parameters=115458, shared_parameters=114882, gate_parameters=576,
    attribute_change='Each patch ownA1 originalFP32 logR is averaged inF64, roundedF32, broadcast onto the same ownA1; original3-axis model and all recipe factors fixed; newAR resets and C0/A_only are fixed reused references',
    source_guard='Each Dongyang/KiTS/Rider case-equal sameweight correct-minus-donor-blockmean Dice must be nonnegative on unchanged primary support',
    CT_uniform_guard='Known-nonempty CT_uniform case-equal clDice/CLrecall/fine must be nonworse versus both controls on identical defined support',
    sampler='FIT24 case-uniform and original16-slot-uniform; same precomputed plan for all arms',
    normalization='FIT24 case-equal first and second log-radius moments over all16slots A1 sites; no epsilon',
    oracle='GT-only partial native-radius oracle at nominal .5+2u SAAM64 grid; unknown neutralization is not a negative target',
    correspondence='Original primary slots0..7 only: samecase donor_slot=(slot+4)%8 block mean, broadcast onto ownA1; all8 supported, no donor filtering/fallback. Eight block scalars preserved, not site-weighted radius multiset',
    primary='Known-nonempty original GT-jitter slots0..7, equal patches then equal PROBE12 cases',
    mechanism_gates=dict(minimum_mean_Dice_pp=.1, minimum_positive_cases=8, all12_supported=True),
    formal_new_arms=['AR'], control_mode='Read-only original same-seed C0/A_only2000; reused_not_retrained, not new independent replicas',
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


def blockmean_fields(np, A, R):
    """F64 supported mean of each original FP32 grid, rounded once to F32."""
    require(A.shape == R.shape == (16, 64, 64, 64) and R.dtype == np.float32 and
        np.isin(A, [0, 1]).all() and np.isfinite(R).all() and (R[A == 0] == 0.).all(),
        'Original complete16-slot finite neutralized FP32 oracle required')
    output = np.zeros_like(R); counts = []; means64 = []; means32 = []
    for slot in range(16):
        values = R[slot][A[slot] == 1].astype(np.float64)
        count = int(len(values)); counts.append(count)
        mean64 = float(values.mean()) if count else None
        mean32 = np.float32(mean64) if count else None
        means64.append(mean64); means32.append(float(mean32) if count else None)
        if count:
            require(math.isfinite(mean64) and np.isfinite(mean32), 'Nonfinite block radius mean')
            output[slot][A[slot] == 1] = mean32
    require(np.isfinite(output).all() and (output[A == 0] == 0.).all(), 'Block summary changed neutral support')
    return output, dict(supported_sites=counts, mean_logR_FP64=means64, mean_logR_FP32=means32)


def blockmean_donor(np, A, R, stats, slot):
    """Same-case primary shift4; retains ownA, not the site-weighted R multiset."""
    require(0 <= slot < 8 and A.shape == R.shape == (64, 64, 64), 'Primary0..7 and actual64-grid required')
    require(len(stats['supported_sites']) == 16 and all(n > 0 for n in stats['supported_sites'][:8]),
        'All original8 primary blocks require support; no donor selection or fallback')
    donor = (slot + 4) % 8
    means = stats['mean_logR_FP32'][:8]
    require(all(value is not None and math.isfinite(value) for value in means), 'Finite supported primary means required')
    donors = [(index + 4) % 8 for index in range(8)]
    require(sorted(donors) == list(range(8)) and np.array_equal(np.sort(np.asarray(means, np.float32)),
        np.sort(np.asarray([means[index] for index in donors], np.float32))), 'Eight block scalar multiset changed')
    output = np.zeros_like(R); output[A == 1] = np.float32(means[donor])
    require(int((A == 1).sum()) == stats['supported_sites'][slot] and np.isfinite(output).all() and
        (output[A == 0] == 0.).all() and (R[A == 1] == np.float32(means[slot])).all(),
        'Recipient own support/mean differs')
    return output, dict(intervention='same_case_primary_blockmean_shift4', recipient_slot=slot, donor_slot=donor,
        recipient_supported_sites=stats['supported_sites'][slot], donor_supported_sites=stats['supported_sites'][donor],
        recipient_mean_logR_FP64=stats['mean_logR_FP64'][slot], donor_mean_logR_FP64=stats['mean_logR_FP64'][donor],
        recipient_mean_logR_FP32=means[slot], donor_mean_logR_FP32=means[donor],
        scalar_changed=means[slot] != means[donor], changed_radius_sites=int(np.count_nonzero(R != output)),
        own_A_sha256=array_sha(np, A), own_A_and_support_preserved=True,
        block_scalar_multiset_preserved=True, site_weighted_radius_multiset_preserved=False)


def oracle_manifest(args, bank):
    require(sha(args.oracle_bank) == args.oracle_bank_sha256, 'Oracle bank SHA differs')
    value = read(args.oracle_bank)
    require(value['bank_sha256'] == BANK_SHA and value['oracle_helper_sha256'] == args.oracle_helper_sha256,
        'Oracle source/bank differs')
    require(sha(args.oracle_helper) == args.oracle_helper_sha256, 'Oracle helper source differs')
    cases = value['cases']; normal = value['normalization']
    require(cases and set(cases) <= set(bank['case_ids']), 'Oracle bank is not TRAIN-only')
    require(value.get('disposable_engineering_only') is False and set(cases) == set(bank['case_ids']) and
        value['fit_cases'] == bank['fit_cases'] and value['probe_cases'] == bank['probe_cases'] and
        normal['case_equal'] is True and normal['fit_cases'] == bank['fit_cases'],
        'Both actual QA and formal run require complete36-case oracle bank and unchanged point-level FIT24 moments')
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
        len(active) == 15 and sum(v.numel() for v in shared) == 114882 and sum(v.numel() for v in gates) == 576,
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
        scalar_changed = sum(row['permutation']['scalar_changed'] for row in selected)
        changed = sum(row['zR_changed'] for row in selected)
        joint = sum(row['permutation']['scalar_changed'] and row['zR_changed'] and
            row['mean_abs_logit_response'] > 0. for row in selected)
        supported = len(selected) >= 2 and joint > 0
        effect = 100. * average([row['normal']['dice'] - row['permuted']['dice'] for row in selected]) if selected else None
        cases[case] = dict(primary_patches=len(selected), supported=supported,
            different_primary_block_scalars=scalar_changed, changed_primary_zR_slots=changed,
            jointly_informative_primary_slots=joint,
            correct_minus_permuted_Dice_pp=effect if supported else None, descriptive_Dice_pp=effect,
            mean_abs_logit_response=average([row['mean_abs_logit_response'] for row in selected]))
    effects = [cases[c]['correct_minus_permuted_Dice_pp'] for c in probe]
    mean = average(effects)
    parts = dict(all12_supported=len(cases) == 12 and all(cases[c]['supported'] for c in probe),
        mean_at_least_point1pp=mean is not None and mean >= .1,
        at_least8_positive=sum(v is not None and v > 0. for v in effects) >= 8)
    return dict(intervention='own_A_same_case_primary_blockmean_shift4', per_case=cases, mean_Dice_effect_pp=mean,
        positive_cases=sum(v is not None and v > 0. for v in effects), gate_parts=parts,
        passed=all(parts.values()), own_A_preserved=True, block_scalar_multiset_preserved=True,
        site_weighted_radius_multiset_preserved=False,
        support='At least2 original primary rows and at least1 jointly different block scalar, actualFP32 zR change and positive logit response; no within-block radius variance criterion')


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
            complete_unchanged_support=complete, case_equal_correct_minus_donor_blockmean_Dice_pp=value,
            passed=complete and value >= 0.)
    return dict(scope='Prospective SOURCE safety: sameweight original primary patches/cases',
        source_groups=sources, passed=all(value['passed'] for value in sources.values()),
        no_source_reclassification_or_removal=True, stricter_only=True, cannot_rescue_original_primary=True)


def nested_equal(torch, left, right):
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(torch, left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(torch, a, b) for a, b in zip(left, right))
    return left == right


def reused_controls(args, output, helper, oracle, bank, plan, plan_sha):
    """Read only fixed old C0/A_only artifacts; never oldAR weights/optimizer."""
    require(args.reused_controls_root and args.reused_controls_sha256, 'Explicit fixed control root/hash map required')
    home = Path(args.reused_controls_root); proof = json.loads(args.reused_controls_sha256)
    names = ('intent.json', 'inputs.json', 'normalization.json', 'training_plan.json', 'summary.json')
    files = set(names) | {arm + '/' + name for arm in ('C0', 'A_only') for name in
        ('analysis.json', 'patches.jsonl', 'training.jsonl')}
    require(set(proof) == files, 'Only fixed global identity and C0/A_only artifacts permitted')
    for name in sorted(files):
        require(sha(home / name) == proof[name], 'Fixed reused control bytes differ: ' + name)
    intent, inputs, normal, oldplan, summary = [read(home / name) for name in names]
    require(intent['schema'] == 'GTA-AVT-calibre-routing-coadapt-pilot-intent-v1' and intent['mode'] == 'run' and
        intent['seed'] == args.seed and intent['worker_sha256'] == REUSED_WORKER_SHA and
        intent['model_sha256'] == args.model_sha256 == FIXED_MODEL_SHA and
        intent['request_sha256'] == args.request_sha256 and intent['bank_sha256'] == BANK_SHA and
        intent['oracle_bank_sha256'] == args.oracle_bank_sha256 and intent['oracle_helper_sha256'] == args.oracle_helper_sha256,
        'Original control source/model/common input/seed differs')
    require(summary['schema'] == 'GTA-AVT-calibre-routing-coadapt-pilot-v1' and summary['status'] == 'verified_complete' and
        summary['mode'] == 'run' and summary['seed'] == args.seed and summary['worker_sha256'] == REUSED_WORKER_SHA and
        summary['model_sha256'] == FIXED_MODEL_SHA and summary['fit_cases'] == bank['fit_cases'] and
        summary['probe_cases'] == bank['probe_cases'], 'Original completed-control summary differs')
    require(inputs['source_manifest_sha256'] == helper.MAP_SHA and inputs['bank_sha256'] == BANK_SHA and
        inputs['common_weights_sha256'] == helper.WEIGHT_SHA and inputs['model_sha256'] == FIXED_MODEL_SHA and
        inputs['oracle_bank_sha256'] == args.oracle_bank_sha256 and inputs['request_sha256'] == args.request_sha256 and
        inputs['training_plan_sha256'] == plan_sha and encoded(oldplan['plan']) == encoded(plan) and oldplan['sha256'] == plan_sha and
        normal == oracle['normalization'], 'Original control plan/moments/source/common metadata differs')
    for name in ('updates_per_arm', 'batch_size', 'patch_size', 'augmentation', 'optimizer', 'shared_learning_rate',
            'gate_learning_rate', 'weight_decay', 'gradient_clipping', 'loss', 'alpha', 'trainable_parameters',
            'shared_parameters', 'gate_parameters', 'sampler', 'normalization'):
        require(intent['recipe'][name] == RECIPE[name], 'Reused control recipe differs: ' + name)
    expected = [(case, slot) for case in bank['probe_cases'] for slot in range(16)]
    result = {}; copied = {}
    for arm in ('C0', 'A_only'):
        analysis = read(home / arm / 'analysis.json')
        rows = [json.loads(line) for line in (home / arm / 'patches.jsonl').read_bytes().splitlines()]
        updates = [json.loads(line) for line in (home / arm / 'training.jsonl').read_bytes().splitlines()]
        require(analysis['schema'] == 'GTA-AVT-calibre-routing-coadapt-arm-v1' and analysis['arm'] == arm and
            analysis['successful_updates'] == 2000 and analysis['frozen_state_unchanged'] is True and
            [(r['case_id'], r['slot']) for r in rows] == expected and
            [r['step'] for r in updates] == list(range(1, 2001)), 'Original control endpoint/budget/population differs')
        destination = output / 'reused_controls' / arm; destination.mkdir(parents=True)
        for name in ('analysis.json', 'patches.jsonl', 'training.jsonl'):
            content = (home / arm / name).read_bytes()
            with (destination / name).open('xb') as stream:
                stream.write(content)
            copied[arm + '/' + name] = proof[arm + '/' + name]
        result[arm] = analysis['summary']
    write(output / 'reused_controls.json', dict(reused_not_retrained=True, independent_new_replicas=False,
        old_worker_sha256=REUSED_WORKER_SHA, unchanged_model_sha256=FIXED_MODEL_SHA,
        seed=args.seed, original_successful_updates_per_control=2000, new_updates_per_control=0,
        proof_sha256=proof, copied_control_file_sha256=copied, no_old_AR_weights_or_optimizer_read=True,
        limitation='Source-level R masking plus finite same-current-model no-grad forward QA; no bitwise backward, statistical equivalence or2000step trajectory/replay claim'))
    return result, inputs['initial_all_arm_state_sha256']


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
        first_case = bank['probe_cases'][0]
        plan = [[(first_case, 0), (first_case, 1)]] * 2
    plan_sha = hashlib.sha256(encoded(plan)).hexdigest()
    output.mkdir(parents=True)
    write(output / 'intent.json', dict(schema='GTA-AVT-calibre-blockmean-coadapt-pilot-intent-v1',
        recipe=RECIPE, mode=args.mode, seed=args.seed, worker_sha256=sha(__file__), model_sha256=args.model_sha256,
        request_sha256=args.request_sha256, bank_sha256=BANK_SHA, oracle_bank_sha256=args.oracle_bank_sha256,
        oracle_helper_sha256=args.oracle_helper_sha256, engineering_only=args.mode == 'qa',
        formal_point_level_FIT_moments_used_in_both_modes=True, original_model_unchanged=True,
        mechanism_population='Samecase primary slots0..7 only; uniform8..15 mechanism not evaluated',
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
            require(slots == list(range(16)), 'Both QA/run oracle bank must retain all16 slots')
            shape = (len(slots), 64, 64, 64)
            require(data['A'].shape == data['R'].shape == data['state'].shape == data['reason'].shape == shape and
                data['A'].dtype == np.uint8 and data['R'].dtype == np.float32 and
                data['state'].dtype == np.uint8 and data['reason'].dtype == np.uint16 and
                np.isfinite(data['R']).all() and np.isin(data['A'], [0, 1]).all() and
                np.array_equal(data['A'] == 1, data['state'] == 1) and (data['R'][data['A'] == 0] == 0.).all(),
                'Oracle finite neutralization/support/dtype differs')
            if args.mode == 'qa':
                data['point_R'] = data['R'].copy()
            data['R'], block_stats = blockmean_fields(np, data['A'], data['R'])
            opened_oracle[case] = (data, {s: i for i, s in enumerate(slots)}, block_stats)
            oracle_before[case] = dict(path=item['path'], sha256=item['sha256'])
        data, positions, block_stats = opened_oracle[case]; i = positions[slot]
        return data['A'][i], data['R'][i], data['state'][i], data['reason'][i]

    def donor_field(case, slot, A, R):
        return blockmean_donor(np, A, R, opened_oracle[case][2], slot)

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
    if args.mode == 'run':
        arm_summaries, common_init = reused_controls(args, output, helper, oracle, bank, plan, plan_sha)
    for arm in (ARMS if args.mode == 'qa' else ('AR',)):
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
        reuse_qa = []
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
                if args.mode == 'qa' and arm in ('C0', 'A_only'):
                    point_fields = dict(fields, logR=torch.from_numpy(np.stack([
                        opened_oracle[item['case_id']][0]['point_R'][item['slot']][None] for item in prepared])).cuda())
                    require(point_fields['logR'].shape == fields['logR'].shape and point_fields['logR'].dtype == fields['logR'].dtype,
                        'Point/mean payload shape/dtype differ')
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        point_logits = model(image, coordinate=coordinate, oracle=point_fields, oracle_mode=True, alpha=alpha)['logits']
                        point_loss = modules['losses'].dice_bce(point_logits, truth)
                    trace = model.last_calibre_trace
                    require(bool((trace['zR'] == 0).all()) and torch.equal(trace['own_A'], fields['A']) and
                        torch.equal(trace['own_valid'], fields['valid']) and torch.equal(trace['consumed_valid'], fields['valid']),
                        'Point payload changed disabled R axis or ownA/valid')
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        mean_logits = model(image, coordinate=coordinate, oracle=fields, oracle_mode=True, alpha=alpha)['logits']
                        mean_loss = modules['losses'].dice_bce(mean_logits, truth)
                    trace = model.last_calibre_trace
                    require(bool((trace['zR'] == 0).all()) and torch.equal(trace['own_A'], fields['A']) and
                        torch.equal(trace['own_valid'], fields['valid']) and torch.equal(trace['consumed_valid'], fields['valid']),
                        'Mean payload changed disabled R axis or ownA/valid')
                    require(torch.equal(point_logits, mean_logits) and torch.equal(point_loss, mean_loss),
                        'Same-current-model point/mean no-grad logits/loss differ')
                    reuse_qa.append(dict(step=step+1, same_current_model=True, both_forward_no_grad=True,
                        logits_elementwise_equal=True, loss_elementwise_equal=True, zR_exact_zero=True,
                        own_A_and_valid_unchanged=True, no_shadow_optimizer=True,
                        no_bitwise_backward_or_trajectory_claim=True,
                        known_CUDA_warning='trilinear3d_backward_cuda is nondeterministic; no cross-backward equality requirement or empirical tolerance'))
                    del point_fields, point_logits, point_loss, mean_logits, mean_loss
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
                if args.mode == 'qa':
                    gradient_semantics = {name: 'None' if value.grad is None else 'zero' if bool((value.grad == 0).all()) else 'nonzero'
                        for name, value in model.named_parameters() if name in active}
                    require(len(gradient_semantics) == 15 and all(value.grad is None or bool(torch.isfinite(value.grad).all())
                        for name, value in model.named_parameters() if name in active), 'All15 actual QA gradients must be finite with explicit None/zero semantics')
                    disabled = ('calibre_availability_weight', 'calibre_radius_weight') if arm == 'C0' else (
                        ('calibre_radius_weight',) if arm == 'A_only' else ())
                    require(all(gradient_semantics[name] == 'zero' for name in disabled), 'Disabled condition axes must have exact zero gradients')
                    if arm in ('C0', 'A_only'):
                        reuse_qa[-1].update(actual_successful_mean_update=True,
                            all15_gradient_values_finite=True, gradient_None_zero_semantics=gradient_semantics,
                            disabled_condition_axes_exact_zero=list(disabled))
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
            donor_qa = None
            if arm == 'AR':
                prepared, image, truth, coordinate, fields = batch(plan[0])
                before = old.tensor_map_sha(model.state_dict())
                with torch.no_grad(), torch.cuda.amp.autocast():
                    correct = model(image, coordinate=coordinate, oracle=fields, oracle_mode=True, alpha=1.)['logits'].float()
                correct_zR = model.last_calibre_trace['zR'].detach().clone()
                altered_fields, mappings = zip(*(donor_field(item['case_id'], item['slot'], item['A'], item['R']) for item in prepared))
                override = dict(logR=torch.from_numpy(np.stack([value[None] for value in altered_fields])).cuda(), valid=fields['valid'])
                with torch.no_grad(), torch.cuda.amp.autocast():
                    altered = model(image, coordinate=coordinate, oracle=fields, oracle_mode=True,
                        radius_override=override, diagnostic=True, alpha=1.)['logits'].float()
                trace = model.last_calibre_trace
                require(torch.equal(trace['own_A'], fields['A']) and torch.equal(trace['own_valid'], fields['valid']) and
                    torch.equal(trace['consumed_valid'], fields['valid']), 'QA donor changed ownA/support')
                donor_qa = [dict(permutation=mappings[i], zR_changed=not torch.equal(correct_zR[i], trace['zR'][i]),
                    mean_abs_logit_response=float((correct[i] - altered[i]).abs().mean())) for i in range(2)]
                require(any(row['permutation']['scalar_changed'] and row['zR_changed'] and row['mean_abs_logit_response'] > 0.
                    for row in donor_qa), 'QA block donor must measurably change consumed condition and response')
                require(before == old.tensor_map_sha(model.state_dict()), 'QA donor intervention changed model state')
                del prepared, image, truth, coordinate, fields, correct, correct_zR, altered_fields, override, altered, trace
            qa[arm] = dict(initial_logits_sha256=zero_digest, initial_state_sha256=initial,
                successful_updates=2, alpha=1., first_gradient_maxabs=first_grads, actual_parameters_changed=changed,
                first_coefficient_row_gradient_maxabs=first_row_grads, coefficient_rows_changed=row_changes,
                trainable_parameters=115458, frozen_state_unchanged=True, all_modules_eval=True,
                engineering_moments_only=False, actual_formal_bank_and_point_moments=True, oracle_conditioned_not_CT_only=True,
                four_original_branch_updates=branch_updates, primary_block_donor_QA=donor_qa,
                point_vs_mean_control_equivalence_QA=reuse_qa,
                no_bitwise_backward_or_trajectory_claim=True,
                known_CUDA_warning='trilinear3d_backward_cuda nondeterministic; source R-masking is reuse basis; no measured-error tolerance')
            del named
            del model, optimizer, scaler, shared, gates, active_initial; torch.cuda.empty_cache()
            continue
        with (folder / 'trainable_final.pt').open('xb') as stream:
            torch.save(dict(schema='GTA-AVT-calibre-blockmean-coadapt-trainable-state-v1', arm=arm, seed=args.seed,
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
                    normal_zR = model.last_calibre_trace['zR'].detach().clone() if arm == 'AR' and begin < 8 else None
                    normals = []
                    for index, item in enumerate(prepared):
                        item['prediction'] = binary[index]
                        row = {name: item[name] for name in ('case_id', 'source', 'slot', 'stratum', 'attribute_valid', 'known_nonempty', 'valid_empty')}
                        row.update(primary=item['slot'] < 8 and item['known_nonempty'],
                            metrics=old.metric(np, ndi, skeletonize, cpu[index], item),
                            oracle=dict(available_radius_sites=int(item['A'].sum()), unknown_sites=int((item['state'] == 255).sum()),
                                supplied_logR_is_block_mean=True,
                                mean_logR_FP64=opened_oracle[case][2]['mean_logR_FP64'][item['slot']],
                                mean_logR_FP32=opened_oracle[case][2]['mean_logR_FP32'][item['slot']]))
                        rows.append(row); normals.append(row); stream.write(encoded(row))
                    if arm == 'AR' and begin < 8:
                        permuted, maps = zip(*(donor_field(case, item['slot'], item['A'], item['R']) for item in prepared))
                        override = dict(logR=torch.from_numpy(np.stack([value[None] for value in permuted])).cuda(),
                            valid=fields['valid'])
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            altered = model(image, coordinate=coordinate, oracle=fields, oracle_mode=True,
                                radius_override=override, diagnostic=True, alpha=1.)['logits'].float()
                        changed_zR = model.last_calibre_trace['zR']
                        require(torch.equal(model.last_calibre_trace['own_A'], fields['A']) and
                            torch.equal(model.last_calibre_trace['own_valid'], fields['valid']) and
                            torch.equal(model.last_calibre_trace['consumed_valid'], fields['valid']), 'Donor intervention changed ownA/support')
                        altered_binary = (altered.sigmoid() >= .5).cpu().numpy()[:, 0]
                        altered_cpu = altered.cpu().numpy()[:, 0]
                        for index, item in enumerate(prepared):
                            item['prediction'] = altered_binary[index]
                            mechanism_rows.append(dict(case_id=case, source=item['source'], slot=item['slot'],
                                stratum=item['stratum'], primary=normals[index]['primary'], normal=normals[index]['metrics'],
                                permuted=old.metric(np, ndi, skeletonize, altered_cpu[index], item), permutation=maps[index],
                                zR_changed=not torch.equal(normal_zR[index], changed_zR[index]),
                                mean_abs_logit_response=float((altered[index] - logits[index]).abs().mean()),
                                binary_changed_voxels=int(((altered[index].sigmoid() >= .5) != (logits[index].sigmoid() >= .5)).sum())))
                        del altered, normal_zR, override
                    del logits, prepared, image, truth, coordinate, fields
        require(len(rows) == 192 and frozen_before == frozen_digest(old, model, active), 'Final probe incomplete or frozen state changed')
        arm_summaries[arm] = old.case_summary(rows)
        write(folder / 'analysis.json', dict(schema='GTA-AVT-calibre-blockmean-coadapt-arm-v1', arm=arm,
            successful_updates=2000, summary=arm_summaries[arm], frozen_state_unchanged=True))
        if arm == 'AR':
            require(len(mechanism_rows) == 96 and [(r['case_id'], r['slot']) for r in mechanism_rows] ==
                [(case, slot) for case in bank['probe_cases'] for slot in range(8)], 'Primary block correspondence must retain all12x8 rows')
            with (folder / 'mechanism.jsonl').open('xb') as stream:
                for row in mechanism_rows:
                    stream.write(encoded(row))
            mechanism = summarize_mechanism(mechanism_rows, bank['probe_cases'])
            write(folder / 'mechanism_analysis.json', mechanism)
        del model, optimizer, scaler, shared, gates, active_initial; torch.cuda.empty_cache()
    if args.mode == 'qa':
        result = dict(schema='GTA-AVT-calibre-blockmean-coadapt-engineering-QA-v1', mode='qa', engineering_qa_passed=True,
            arms=qa, actual_patch_size=128, actual_batch_size=2, AMP=True, formal_scientific_outputs=False,
            actual_formal_plan_used=False, actual_formal_bank_and_point_moments=True, saved_checkpoints=False,
            no_bitwise_backward_or_trajectory_claim=True, r1_r2_failures_unchanged=True,
            known_CUDA_warning='trilinear3d_backward_cuda nondeterministic; no empirical tolerance or trajectory claim',
            no_VAL_or_TEST_arrays=True)
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
        result = dict(schema='GTA-AVT-calibre-blockmean-coadapt-pilot-readout-v1', mode='run', arm_summaries=arm_summaries,
            comparisons=comparisons, radius_mechanism=mechanism,
            mechanism_evaluated_rows=96, uniform_mechanism='not_evaluated',
            controls_reused_not_retrained=True, original_control_updates=2000, new_control_updates=0,
            new_AR_successful_updates=2000, shared_controls_not_new_independent_replicas=True,
            original_primary_screen_passed=original_primary, CT_uniform_transfer_safety=uniform_guard, SOURCE_safety=source_guard,
            capacity_continue=original_primary and uniform_guard['passed'] and source_guard['passed'],
            oracle_conditioned_not_CT_only=True, goal_completed=False, native_VAL_authorized=False,
            automatic_successor=False, no_VAL_or_TEST_arrays=True,
            cross_round_change='Only supplied R granularity changes: supported pointwiseFP32 logR to per-patch F64 mean roundedF32 and broadcast on sameA. Original model115458/seeds/plans/point-level FITmoments/recipe/gates fixed; correspondence estimand changes to samecase primary block matching')
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
    write(output / 'summary.json', dict(schema='GTA-AVT-calibre-blockmean-coadapt-pilot-v1', status='verified_complete',
        mode=args.mode, seed=args.seed, files=members, engineering_qa_passed=args.mode == 'qa',
        fit_cases=bank['fit_cases'], probe_cases=bank['probe_cases'], worker_sha256=sha(__file__),
        model_sha256=args.model_sha256, oracle_bank_sha256=args.oracle_bank_sha256,
        controls_reused_not_retrained=args.mode == 'run',
        original_control_updates=2000 if args.mode == 'run' else None, new_control_updates=0 if args.mode == 'run' else None,
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
        'oracle-helper', 'oracle-helper-sha256', 'model', 'model-sha256', 'output',
        'reused-controls-root', 'reused-controls-sha256'):
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
