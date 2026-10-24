"""Short TRAIN-only capacity pilot: C0, availability-only, and boundary field.

The common7406 G0 and independent heads stay frozen/eval. Only256 adapter
parameters train, for250 successful updates per arm, on one common fixed plan.
No native VAL/TEST, checkpoint selection, automatic extension or goal completion.
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
FIELD_MODEL_SHA = '949a1efd1713a661c73762d7cef387be01205a93e425fba60d2515af4e798cf2'
HEAD_SHA = 'b61885b06fd6413efc0cf86bb664f69aa1a4e9f7fb2a26a32c373a71ce5e11bb'
FULL_FEATURE_WORKER_SHA = 'd6b001285da745090bb34afce0209367003d9ff52faad358b711c27c36a74b42'
ARMS = ('C0', 'A_only', 'AB')
INTERVENTIONS = ('whole_field_donor', 'own_A_donor_B')
RECIPE = dict(updates_per_arm=250, batch_size=2, patch_size=128, augmentation=False,
    optimizer='AdamW', learning_rate=.001, weight_decay=.00001, gradient_clipping=False,
    loss='Original main-output Dice+BCE only; no deep or attribute auxiliary loss',
    alpha=1., parameters='Only128 interaction weights and128 biases, zero initialized',
    frozen='All G0/classifier/independent heads and buffers eval/frozen',
    sampling='FIT24 case-uniform and original16-slot-uniform; one shared precomputed plan',
    normalization='FIT24 CT-predicted AB raw field, all16x512 cells; same mean/std for all arms',
    primary='Known-nonempty original GT-jitter slots0..7, equal patches then equal PROBE12 cases',
    fine_recall='Within-case pooled cached fine-node coverage across primary patches; cases equal',
    donor='Fixed+1 cyclic slot inside same case and original8-slot stratum, no GT donor filtering',
    mechanism_gates=dict(minimum_mean_Dice_pp=.1, minimum_positive_cases=8, all12_supported=True),
    QA_head_reference_max_abs_tolerance=.01,
    scope='Short frozen-G0 TRAIN capacity diagnostic; no full joint or native validation efficacy')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def load_module(path, expected, name):
    require(sha(path) == expected, 'Pinned module differs: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def average(values):
    return st.mean(values) if values and all(v is not None for v in values) else None


def donor_slot(slot):
    base = (slot // 8) * 8
    return base + (slot - base + 1) % 8


def tensor_map_sha(state):
    value = hashlib.sha256()
    for name, tensor in state.items():
        tensor = tensor.detach().contiguous().cpu()
        value.update(name.encode()); value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode()); value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def frozen_sha(model):
    return tensor_map_sha({k: v for k, v in model.state_dict().items()
                           if k not in ('interaction_weight', 'interaction_bias')})


def check_frozen(model):
    active = [name for name, value in model.named_parameters() if value.requires_grad]
    require(set(active) == {'interaction_weight', 'interaction_bias'} and
            sum(value.numel() for value in model.parameters() if value.requires_grad) == 256,
            'Only the existing256 interaction parameters may train')
    require(all(not module.training for module in model.modules()), 'A frozen module escaped eval mode')
    require(all(value.grad is None for name, value in model.named_parameters() if name not in active),
            'Frozen parameter received a gradient')


def field_moments(np, torch, shared, fit, checkpoint):
    a, b = checkpoint['availability'], checkpoint['B']; values = []
    mean, scale = a['feature_mean'].double(), a['feature_std'].double()
    weight, bias = a['standardized_weights'].double(), a['bias'].double()
    coefficients = b['coefficients'].double()
    for case in fit:
        with np.load(shared[case]['path'], allow_pickle=False) as stream:
            features = stream['features']  # ALL512 CT features; no GT mask is opened or consulted.
        require(features.shape == (16, 512, 64) and np.isfinite(features).all(), 'Fixed all-cell feature shape differs')
        f = torch.from_numpy(features.astype(np.float64))
        with torch.no_grad():
            availability = torch.sigmoid(((f - mean) / scale) @ weight + bias)
            boundary = (f @ coefficients[1:] + coefficients[0]).clamp(0., 1.)
            values.extend((availability * boundary).reshape(-1).tolist())
    require(len(values) == 24 * 16 * 512 and all(math.isfinite(v) for v in values), 'Incomplete CT-only moments')
    mu = math.fsum(values) / len(values)
    variance = math.fsum((v - mu)**2 for v in values) / len(values); sigma = math.sqrt(variance)
    require(sigma > 0. and math.isfinite(sigma), 'Zero/undefined CT-predicted AB variance: stop pilot')
    return dict(mean=mu, population_std=sigma, population_variance=variance,
        cells=len(values), fit_cases=fit, no_GT_mask_used=True, common_to_all_arms=True,
        computation='CPU FP64 prepared heads on immutable FP32 features; all512 cells of all16 FIT24 patches')


def metric(np, ndi, skeletonize, logits, prepared):
    truth, valid, points, radii = prepared['truth'], prepared['valid'], prepared['points'], prepared['radii']
    require(np.isfinite(logits).all(), 'Nonfinite segmentation output')
    # The caller supplies the actual FP32 sigmoid>=.5 prediction, not a logit shortcut.
    prediction = prepared['prediction'] & valid; gt = truth & valid
    TP, FP, FN = [int(x.sum()) for x in (prediction & gt, prediction & ~gt, ~prediction & gt)]
    skp = skeletonize(prediction, method='lee').astype(bool) if prediction.any() else np.zeros_like(prediction)
    covered = prediction[tuple(points.T)] if len(points) else np.asarray([], bool)
    recall = float(covered.mean()) if len(points) else None
    precision = float(gt[skp].mean()) if skp.any() else None
    cldice = (0. if not skp.any() else 2. * recall * precision / max(recall + precision, 1e-12)) if recall is not None else None
    fine = radii <= 2.; fine_nodes = int(fine.sum()); fine_covered = int(covered[fine].sum())
    active_logits = logits[valid].astype(np.float64); target = gt[valid].astype(np.float64)
    return dict(dice=(2. * TP + 1e-6) / (2. * TP + FP + FN + 1e-6), cldice=cldice,
        centerline_recall=recall, centerline_precision=precision,
        fine_recall=fine_covered / fine_nodes if fine_nodes else None,
        fine_nodes=fine_nodes, fine_covered=fine_covered, GT_nodes=len(points), GT_covered=int(covered.sum()),
        TP=TP, FP=FP, FN=FN, BCE=float((np.logaddexp(0., active_logits) - target * active_logits).mean()),
        real_CT_voxels=int(valid.sum()), truth_foreground=int(gt.sum()), prediction_skeleton_points=int(skp.sum()))


def field_stats(trace):
    raw = trace['own_raw_c8'].detach().double(); z = trace['centered_c'].detach().float(); gain = trace['gain'].detach().float()
    return dict(raw_mean=float(raw.mean()), raw_std=float(raw.std(unbiased=False)),
        z_mean=float(z.mean()), z_std=float(z.std(unbiased=False)),
        gain_absmean=float(gain.abs().mean()), gain_maxabs=float(gain.abs().max()))


def case_summary(rows):
    cases = list(dict.fromkeys(r['case_id'] for r in rows)); result = {}
    for case in cases:
        group = [r for r in rows if r['case_id'] == case]
        primary = [r for r in group if r['primary']]
        def summarize(items):
            fine_n = sum(r['metrics']['fine_nodes'] for r in items)
            fine_k = sum(r['metrics']['fine_covered'] for r in items)
            return dict(patches=len(items), **{name: average([r['metrics'][name] for r in items])
                for name in ('dice', 'cldice', 'centerline_recall')},
                fine_recall=fine_k / fine_n if fine_n else None, fine_nodes=fine_n, fine_covered=fine_k,
                fine_patch_mean=st.mean([r['metrics']['fine_recall'] for r in items if r['metrics']['fine_recall'] is not None])
                    if any(r['metrics']['fine_recall'] is not None for r in items) else None,
                FP=sum(r['metrics']['FP'] for r in items), FN=sum(r['metrics']['FN'] for r in items))
        result[case] = dict(primary=summarize(primary), all16=summarize(group),
            uniform_nonempty=summarize([r for r in group if r['slot'] >= 8 and r['known_nonempty']]),
            uniform_empty_FP=sum(r['metrics']['FP'] for r in group if r['slot'] >= 8 and r['valid_empty']),
            uniform_empty_patches=sum(r['slot'] >= 8 and r['valid_empty'] for r in group),
            unknown_slots=[r['slot'] for r in group if not r['attribute_valid']])
    means = {name: average([result[c]['primary'][name] for c in cases])
             for name in ('dice', 'cldice', 'centerline_recall', 'fine_recall')}
    ranked = sorted(cases, key=lambda c: result[c]['primary']['dice'])
    return dict(per_case=result, primary_case_equal=means, own_worst2_cases=ranked[:2],
        own_worst2_Dice=average([result[c]['primary']['dice'] for c in ranked[:2]]),
        Dice_case_SD=st.stdev([result[c]['primary']['dice'] for c in cases]) if len(cases) > 1 else None)


def mechanism_summary(rows, probe):
    output = {}
    for name in INTERVENTIONS:
        cases = {}
        for case in probe:
            selected = [r for r in rows if r['case_id'] == case and r['primary']]
            raw_variance = st.pvariance([v for r in selected for v in r['normal_raw_field']]) if selected else None
            changed = sum(r['interventions'][name]['z_changed'] for r in selected)
            supported = len(selected) >= 2 and raw_variance is not None and raw_variance > 0. and changed > 0
            effect = 100. * average([r['normal']['dice'] - r['interventions'][name]['metrics']['dice'] for r in selected]) if selected else None
            cases[case] = dict(primary_patches=len(selected), supported=supported,
                support_reasons=[] if supported else ['insufficient_primary_or_zero_raw_variance_or_no_actual_z_change'],
                raw_field_variance=raw_variance, changed_primary_z_slots=changed,
                correct_minus_permuted_Dice_pp=effect if supported else None,
                descriptive_Dice_pp=effect,
                mean_abs_logit_response=average([r['interventions'][name]['mean_abs_logit_response'] for r in selected]))
        effects = [cases[c]['correct_minus_permuted_Dice_pp'] for c in probe]; value = average(effects)
        parts = dict(all12_supported=all(cases[c]['supported'] for c in probe),
            mean_at_least_point1pp=value is not None and value >= .1,
            at_least8_positive=sum(v is not None and v > 0 for v in effects) >= 8)
        output[name] = dict(per_case=cases, mean_Dice_effect_pp=value,
            positive_cases=sum(v is not None and v > 0 for v in effects), gate_parts=parts, passed=all(parts.values()))
    return dict(interventions=output, both_passed=all(v['passed'] for v in output.values()))


def execute(args):
    helper = load_module(args.helper, HELPER_SHA, 'pilot_boundary_helper')
    request = read(args.request); root, source, manifest, bank, records = helper.metadata(request)
    require(sha(args.heads) == args.heads_sha256 == HEAD_SHA and
            sha(args.field_model) == args.field_model_sha256 == FIELD_MODEL_SHA, 'Prepared model or heads differ')
    require(sha(args.shared_manifest) == args.shared_manifest_sha256, 'Shared manifest differs')
    shared = read(args.shared_manifest); require(set(shared) == set(bank['case_ids']), 'Full TRAIN36 features required')
    for case, item in shared.items():
        r = item['receipt']
        require(r['case_id'] == case and r['worker_sha256'] == FULL_FEATURE_WORKER_SHA and
            r['weights_sha256'] == helper.WEIGHT_SHA and r['bank_sha256'] == helper.BANK_SHA and
            r['source_manifest_sha256'] == helper.MAP_SHA and sha(item['path']) == r['sha256'], 'Shared feature source differs')
    output = Path(args.output); require(not output.exists(), 'Fresh pilot output required')
    import numpy as np
    import torch
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    for path in (root / 'code', root / 'next_stage'):
        sys.path.insert(0, str(path))
    modules = {name: importlib.import_module(name) for name in ('data', 'geometry', 'losses', 'train')}
    for name in modules:
        require(Path(modules[name].__file__).resolve() == root / 'code' / (name + '.py'), 'Scientific import escaped source42')
    field = load_module(args.field_model, FIELD_MODEL_SHA, 'pilot_boundary_field_model')
    require(torch.cuda.is_available(), 'Actual CUDA pilot required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    heads = torch.load(args.heads, map_location='cpu'); common = torch.load(request['weights'], map_location='cpu')
    require(heads.get('optimization_qualified') is True, 'Unqualified independent heads cannot enter the pilot')
    normalization = field_moments(np, torch, shared, bank['fit_cases'], heads)
    rng = np.random.default_rng(args.seed)
    plan = [[(bank['fit_cases'][int(rng.integers(24))], int(rng.integers(16))) for _ in range(2)] for step in range(250)]
    plan_sha = hashlib.sha256(encoded(plan)).hexdigest()
    output.mkdir(); write(output / 'intent.json', dict(schema='GTA-AVT-boundary-field-pilot-intent-v1',
        recipe=RECIPE, seed=args.seed, mode=args.mode, worker_sha256=sha(__file__), heads_sha256=HEAD_SHA,
        field_model_sha256=FIELD_MODEL_SHA, request_sha256=sha(args.request), shared_manifest_sha256=args.shared_manifest_sha256))
    write(output / 'normalization.json', normalization); write(output / 'training_plan.json', dict(plan=plan, sha256=plan_sha))
    specmap = {(r['case_id'], r['slot']): r for r in bank['patches']}; opened, inputs_before = {}, {}
    def prepare(case, slot):
        if case not in opened:
            require(case in bank['case_ids'] and records[case]['cache_file_sha256'] == bank['expected_TRAIN_cache_sha256'][case], 'Non-TRAIN cache access')
            one = dict(request, case_ids=[case]); inputs_before.update(helper.cached_inputs(one, records))
            home = Path(records[case]['cache_dir'])
            image = np.load(home / 'image.npy', mmap_mode='r', allow_pickle=False)
            mask = np.load(home / 'mask.npy', mmap_mode='r', allow_pickle=False)
            with np.load(home / 'geometry.npz', allow_pickle=False) as g:
                geometry = {k: g[k] for k in g.files}
            require(image.shape == mask.shape and list(image.shape) == bank['headers'][case]['shape_zyx'], 'Cache shape differs')
            opened[case] = (image, mask, geometry)
        image, mask, geometry = opened[case]; spec = specmap[(case, slot)]; start = spec['start_zyx']
        sl = tuple(slice(v, min(v + 128, n)) for v, n in zip(start, image.shape))
        raw = np.asarray(image[sl], np.float32); truth = np.asarray(mask[sl], bool)
        pad = [(0, 128 - n) for n in raw.shape]; valid = np.pad(np.ones(raw.shape, bool), pad)
        normalized = np.ascontiguousarray(modules['data'].normalize(np.pad(raw, pad, constant_values=-1024))[None], np.float32)
        scale, structure = modules['geometry'].patch_histograms(geometry, start, [128] * 3)
        attr_valid = bool(scale.sum() > 0 and not (truth.any() and scale[0] == 1.))
        point = geometry['points']; chosen = ((point >= np.asarray(start)) & (point < np.asarray(start) + 128)).all(1)
        points = point[chosen] - np.asarray(start)
        coordinate = 2. * (np.asarray(records[case]['crop_start_zyx']) + np.asarray(start) + 64.) / np.maximum(
            np.asarray(records[case]['full_shape_zyx']) - 1., 1.) - 1.
        return dict(image=normalized, truth=np.pad(truth, pad), valid=valid, points=points,
            radii=geometry['radius'][chosen], coordinate=coordinate.astype(np.float32),
            case_id=case, source=spec['source'], slot=slot, stratum=spec['stratum'],
            attribute_valid=attr_valid, known_nonempty=bool(attr_valid and truth.any()),
            valid_empty=bool(attr_valid and not truth.any()))
    def batch(pairs):
        prepared = [prepare(c, s) for c, s in pairs]
        image = torch.from_numpy(np.stack([r['image'] for r in prepared])).cuda()
        truth = torch.from_numpy(np.stack([r['truth'][None] for r in prepared])).cuda().float()
        coordinate = torch.from_numpy(np.stack([r['coordinate'] for r in prepared])).cuda()
        return prepared, image, truth, coordinate
    arm_summaries, qa, common_init = {}, {}, None
    for arm in ARMS:
        folder = output / arm; folder.mkdir()
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        model = field.BoundaryFieldStageModel(arm, alpha=1.)
        model.load_common_state(common); model.install_heads(heads, normalization['mean'], normalization['population_std'])
        model.cuda().train(); check_frozen(model)
        initial = tensor_map_sha(model.state_dict())
        require(common_init is None or common_init == initial, 'Three arms do not share exact initial tensors')
        common_init = initial; frozen_before = frozen_sha(model)
        optimizer = torch.optim.AdamW([model.interaction_weight, model.interaction_bias], lr=.001, weight_decay=.00001)
        scaler = torch.cuda.amp.GradScaler(); steps = 2 if args.mode == 'qa' else 250
        rows = []; first_w_grad = first_b_grad = None
        with (folder / 'training.jsonl').open('xb') as stream:
            for step in range(steps):
                prepared, image, truth, coordinate = batch(plan[step])
                if step == 0 and args.mode == 'qa':
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        zero = model(image, coordinate=coordinate, alpha=1.)['logits'].float()
                        parity = model(image, coordinate=coordinate, alpha=0.)['logits'].float()
                    require(torch.equal(zero, parity), 'Zero interaction does not preserve G0')
                    zero_digest = hashlib.sha256(zero.cpu().numpy().tobytes()).hexdigest()
                    require(not qa or zero_digest == next(iter(qa.values()))['initial_logits_sha256'], 'Arm zero-start output differs')
                    references_A, references_B = [], []
                    a, b = heads['availability'], heads['B']
                    for case, slot in plan[0]:
                        with np.load(shared[case]['path'], allow_pickle=False) as source_features:
                            f = torch.from_numpy(source_features['features'][slot].astype(np.float64))
                        reference_A = torch.sigmoid(((f - a['feature_mean']) / a['feature_std']) @ a['standardized_weights'] + a['bias'])
                        reference_B = (f @ b['coefficients'][1:] + b['coefficients'][0]).clamp(0., 1.)
                        references_A.append(reference_A.reshape(1, 8, 8, 8)); references_B.append(reference_B.reshape(1, 8, 8, 8))
                    live = model.last_interaction_trace
                    A_difference = float((live['own_A'].cpu() - torch.stack(references_A)).abs().max())
                    B_difference = float((live['own_B'].cpu() - torch.stack(references_B)).abs().max())
                    require(max(A_difference, B_difference) <= .01, 'Live CT heads disagree grossly with immutable feature reference')
                    del zero, parity
                def forward_loss():
                    with torch.cuda.amp.autocast():
                        logits = model(image, coordinate=coordinate, alpha=1.)['logits']
                        loss = modules['losses'].dice_bce(logits, truth)
                    return loss, dict(loss=float(loss.detach()))
                details, update = modules['train'].optimizer_update_with_amp_retries(model, optimizer, scaler,
                    forward_loss, gradient_clip=float('inf'), max_attempts=8)
                check_frozen(model)
                if step == 0:
                    first_w_grad = float(model.interaction_weight.grad.detach().abs().max())
                    first_b_grad = float(model.interaction_bias.grad.detach().abs().max())
                record = dict(step=step + 1, **details, **update, **field_stats(model.last_interaction_trace),
                    weight_norm=float(model.interaction_weight.detach().norm()), bias_norm=float(model.interaction_bias.detach().norm()))
                stream.write(encoded(record)); stream.flush()
                del image, truth, coordinate, prepared
        require(frozen_before == frozen_sha(model), 'Frozen G0/head/buffer state changed')
        if args.mode == 'qa':
            require(first_b_grad > 0. and bool((model.interaction_bias != 0).any()), 'Bias lacks actual gradient/update')
            if arm == 'C0':
                require(first_w_grad == 0. and bool((model.interaction_weight == 0).all()), 'C0 exact null slope escaped')
            else:
                require(first_w_grad > 0. and bool((model.interaction_weight != 0).any()), 'Dynamic slope lacks actual gradient/update')
            qa[arm] = dict(initial_logits_sha256=zero_digest, frozen_state_unchanged=True,
                first_weight_gradient_maxabs=first_w_grad, first_bias_gradient_maxabs=first_b_grad,
                live_reference_A_max_abs_difference=A_difference, live_reference_B_max_abs_difference=B_difference,
                successful_updates=2, trainable_parameters=256, all_modules_eval=True)
            del model; torch.cuda.empty_cache(); continue
        with (folder / 'adapter_final.pt').open('xb') as stream:
            torch.save(dict(arm=arm, seed=args.seed, steps=250, weight=model.interaction_weight.detach().cpu(),
                bias=model.interaction_bias.detach().cpu(), normalization=normalization, heads_sha256=HEAD_SHA,
                common_weights_sha256=helper.WEIGHT_SHA), stream)
        mechanism_rows = []
        with (folder / 'patches.jsonl').open('xb') as stream:
            for case in bank['probe_cases']:
                case_rows, normal_logits, own_A, own_B = [], [], [], []
                for begin in range(0, 16, 2):
                    prepared, image, truth, coordinate = batch([(case, s) for s in range(begin, begin + 2)])
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        logits = model(image, coordinate=coordinate, alpha=1.)['logits'].float()
                    binary = (logits.sigmoid() >= .5).cpu().numpy()[:, 0]; logits_cpu = logits.cpu().numpy()[:, 0]
                    trace = model.last_interaction_trace
                    for index, item in enumerate(prepared):
                        item['prediction'] = binary[index]
                        row = {key: item[key] for key in ('case_id', 'source', 'slot', 'stratum', 'attribute_valid', 'known_nonempty', 'valid_empty')}
                        row.update(primary=item['slot'] < 8 and item['known_nonempty'],
                            metrics=metric(np, ndi, skeletonize, logits_cpu[index], item),
                            field=field_stats({k: v[index:index + 1] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == 2 else v for k, v in trace.items()}))
                        rows.append(row); case_rows.append(row); stream.write(encoded(row))
                        if arm == 'AB':
                            normal_logits.append(logits_cpu[index].copy())
                            own_A.append(trace['own_A'][index:index + 1].cpu()); own_B.append(trace['own_B'][index:index + 1].cpu())
                    del logits, image, truth, coordinate, prepared
                if arm == 'AB':
                    A = torch.cat(own_A, 0); B = torch.cat(own_B, 0)
                    for begin in range(0, 16, 2):
                        slots = list(range(begin, begin + 2)); donors = [donor_slot(s) for s in slots]
                        prepared, image, truth, coordinate = batch([(case, s) for s in slots])
                        base = torch.from_numpy(np.stack([normal_logits[s][None] for s in slots])).cuda()
                        pending = [dict(case_id=case, source=case.split('_')[0], slot=s,
                            stratum=case_rows[s]['stratum'], primary=case_rows[s]['primary'],
                            normal=case_rows[s]['metrics'], normal_raw_field=(A[s] * B[s]).reshape(-1).tolist(),
                            interventions={}) for s in slots]
                        for name in INTERVENTIONS:
                            override = dict(A=A[donors].cuda(), B=B[donors].cuda()) if name == INTERVENTIONS[0] else dict(B=B[donors].cuda())
                            with torch.no_grad(), torch.cuda.amp.autocast():
                                changed = model(image, coordinate=coordinate, alpha=1., boundary_override=override, diagnostic=True)['logits'].float()
                            tr = model.last_interaction_trace
                            binary = (changed.sigmoid() >= .5).cpu().numpy()[:, 0]; altered = changed.cpu().numpy()[:, 0]
                            # Compare exact consumed FP32z with a reconstruction of the normal same-model field.
                            normal_raw = (A[slots] * B[slots]).cuda()
                            with torch.cuda.amp.autocast(enabled=False):
                                normalized = (torch.nn.functional.interpolate(normal_raw.double(), size=tr['centered_c'].shape[2:],
                                    mode='trilinear', align_corners=False) - normalization['mean']) / normalization['population_std']
                            for index, item in enumerate(prepared):
                                item['prediction'] = binary[index]
                                response = (changed[index:index + 1] - base[index:index + 1]).abs()
                                pending[index]['interventions'][name] = dict(donor_slot=donors[index],
                                    metrics=metric(np, ndi, skeletonize, altered[index], item),
                                    raw8_changed=not torch.equal(tr['consumed_raw_c8'][index].cpu(), (A[slots[index]] * B[slots[index]])),
                                    z_changed=not torch.equal(tr['centered_c'][index], normalized.float()[index]),
                                    mean_abs_logit_response=float(response.mean()),
                                    binary_changed_voxels=int(((changed[index].sigmoid() >= .5) != (base[index].sigmoid() >= .5)).sum()))
                            del changed, normal_raw, normalized
                        mechanism_rows.extend(pending); del image, truth, coordinate, base, prepared
        require(len(rows) == 192 and frozen_before == frozen_sha(model), 'Probe incomplete or frozen state changed')
        arm_summaries[arm] = case_summary(rows)
        write(folder / 'analysis.json', dict(schema='GTA-AVT-boundary-field-arm-pilot-v1', arm=arm,
            summary=arm_summaries[arm], frozen_state_unchanged=True, steps=250, primary='known-nonempty slots0..7'))
        if arm == 'AB':
            require(len(mechanism_rows) == 192, 'Mechanism probe incomplete')
            with (folder / 'mechanism.jsonl').open('xb') as stream:
                for row in mechanism_rows:
                    stream.write(encoded(row))
            mechanisms = mechanism_summary(mechanism_rows, bank['probe_cases'])
            write(folder / 'mechanism_analysis.json', mechanisms)
        del model; torch.cuda.empty_cache()
    if args.mode == 'qa':
        write(output / 'qa.json', dict(schema='GTA-AVT-boundary-field-engineering-QA-v1', engineering_qa_passed=True,
            arms=qa, actual_patch_size=128, actual_batch_size=2, AMP=True, normalization=normalization))
        result = dict(engineering_qa_passed=True, mode='qa')
    else:
        comparisons = {}
        for baseline in ('C0', 'A_only'):
            delta = {name: 100. * (arm_summaries['AB']['primary_case_equal'][name] - arm_summaries[baseline]['primary_case_equal'][name])
                if arm_summaries['AB']['primary_case_equal'][name] is not None and arm_summaries[baseline]['primary_case_equal'][name] is not None else None
                for name in ('dice', 'cldice', 'centerline_recall', 'fine_recall')}
            delta['own_worst2_Dice'] = 100. * (arm_summaries['AB']['own_worst2_Dice'] - arm_summaries[baseline]['own_worst2_Dice'])
            parts = {name: value is not None and (value > 0. if name == 'dice' else value >= 0.) for name, value in delta.items()}
            comparisons[baseline] = dict(AB_minus_baseline_pp=delta, parts=parts, passed=all(parts.values()))
        result = dict(schema='GTA-AVT-boundary-field-pilot-readout-v1', mode='run', arm_summaries=arm_summaries,
            comparisons=comparisons, mechanisms=mechanisms,
            capacity_continue=all(v['passed'] for v in comparisons.values()) and mechanisms['both_passed'],
            preliminary_TRAIN_capacity_only=True, goal_completed=False, native_VAL_authorized=False,
            automatic_successor=False, no_VAL_or_TEST_arrays=True)
        write(output / 'analysis.json', result)
    after_inputs = helper.cached_inputs(dict(request, case_ids=list(inputs_before)), records)
    require(after_inputs == inputs_before and sha(args.heads) == HEAD_SHA and sha(args.field_model) == FIELD_MODEL_SHA and
            sha(request['weights']) == helper.WEIGHT_SHA,
            'Frozen inputs changed')
    for name, digest in source.items():
        require(sha(root / name) == digest, 'Scientific source changed')
    write(output / 'inputs.json', dict(source_manifest_sha256=helper.MAP_SHA, bank_sha256=helper.BANK_SHA,
        common_weights_sha256=helper.WEIGHT_SHA, heads_sha256=HEAD_SHA, field_model_sha256=FIELD_MODEL_SHA,
        shared_manifest_sha256=args.shared_manifest_sha256, request_sha256=sha(args.request),
        cache_before=inputs_before, cache_after=after_inputs, training_plan_sha256=plan_sha,
        initial_all_arm_state_digest=common_init, no_VAL_or_TEST_arrays=True))
    files = {p.relative_to(output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(output.rglob('*')) if p.is_file()}
    write(output / 'summary.json', dict(schema='GTA-AVT-boundary-field-pilot-v1', status='verified_complete',
        mode=args.mode, seed=args.seed, files=files, engineering_qa_passed=args.mode == 'qa',
        fit_cases=bank['fit_cases'], probe_cases=bank['probe_cases'],
        elapsed_seconds=time.monotonic() - args.started, no_VAL_or_TEST_arrays=True, automatic_successor=False))
    return dict(status='verified_complete', mode=args.mode, engineering_qa_passed=args.mode == 'qa',
                summary_sha256=sha(output / 'summary.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'heads', 'heads-sha256', 'shared-manifest', 'shared-manifest-sha256',
                 'field-model', 'field-model-sha256', 'output'):
        parser.add_argument('--' + name)
    parser.add_argument('--helper', default=str(Path(__file__).with_name('boundary_attribute_probe.py')))
    parser.add_argument('--seed', type=int, choices=(20261030, 20261031, 20261032))
    parser.add_argument('--mode', choices=('qa', 'run'), default='run'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', recipe=RECIPE))); return
    require(args.seed and all(getattr(args, name) for name in ('request', 'heads', 'heads_sha256', 'shared_manifest',
        'shared_manifest_sha256', 'field_model', 'field_model_sha256', 'output')), 'Explicit fixed inputs, seed and fresh output required')
    args.started = time.monotonic(); print(json.dumps(execute(args), allow_nan=False))


if __name__ == '__main__':
    main()
