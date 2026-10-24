"""Four fixed-radius interventions on completed AR weights; default inert.

No optimizer, weight update, image export, candidate rescue or success gate.
Whole-patch metrics retain the original88 implementation. Regional statistics
are voxel confusion/BCE only, never a new regional skeleton or Dice analysis.
"""
import argparse
import importlib
import json
from pathlib import Path
import struct
import sys
import time

BASIS_SHA = '68840bce80d3576c353fe6c43c49c5ea7bfc7db44e772e9bd687283af1198f7c'
MODEL_SHA = '6f5e6fcf84600c737fc8844c0fdce6839f980d19d734512686f374cce1a0c294'
VARIANTS = ('correct', 'half_ring', 'patch_mean_FP32', 'FIT_mean_FP64')


def load_basis(path, digest):
    import hashlib
    import importlib.util
    if digest != BASIS_SHA or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError('Original basis worker bytes differ')
    spec = importlib.util.spec_from_file_location('basis_frozen_original_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_parity(actual, expected):
    """All mask/count-derived metrics exact; only nonnegative BCE allows2 F64 ULP."""
    if set(actual) != set(expected):
        raise ValueError('Reference metric keys differ')
    bce_ulps = 0
    for key, value in actual.items():
        other = expected[key]
        if type(value) is not type(other):
            raise ValueError('Reference metric type differs: ' + key)
        if key != 'BCE':
            if value != other:
                raise ValueError('Discrete/count-derived metric mismatch: ' + key)
        else:
            import math
            if not (math.isfinite(value) and math.isfinite(other) and value >= 0 and other >= 0):
                raise ValueError('Nonfinite/negative BCE')
            bits = lambda x: struct.unpack('>Q', struct.pack('>d', x))[0]
            bce_ulps = abs(bits(value) - bits(other))
            if bce_ulps > 2 or ((value == 0 or other == 0) and bits(value) != bits(other)):
                raise ValueError('BCE differs beyond2 binary64 ULP')
    return dict(all_count_and_derived_metrics_exact=True, BCE_binary64_ULP=bce_ulps,
        old_voxelwise_mask_bitwise_parity_not_observable=True)


def regional_metrics(np, logits, prediction, item):
    real_shape = item['real_shape']; axes = np.ogrid[:128, :128, :128]
    distance = np.minimum(np.minimum(np.minimum(axes[0], real_shape[0] - 1 - axes[0]),
        np.minimum(axes[1], real_shape[1] - 1 - axes[1])), np.minimum(axes[2], real_shape[2] - 1 - axes[2]))
    result = {}
    for name, zone in (('0_to15', distance < 16), ('16_to31', (distance >= 16) & (distance < 32)),
            ('at_least32', distance >= 32)):
        zone = zone & item['valid']; count = int(zone.sum())
        gt, pred = item['truth'][zone], prediction[zone]
        values = logits[zone].astype(np.float64)
        bce_sum = float((np.logaddexp(0., values) - gt.astype(np.float64) * values).sum())
        result[name] = dict(real_CT_voxels=count, truth_foreground=int(gt.sum()),
            TP=int((pred & gt).sum()), FP=int((pred & ~gt).sum()), FN=int((~pred & gt).sum()),
            BCE_sum=bce_sum, BCE=bce_sum / count if count else None)
    return result


def execute(args):
    basis = load_basis(args.basis_worker, args.basis_worker_sha256)
    require, sha, read, write = basis.require, basis.sha, basis.read, basis.write
    for name in ('request', 'trained_state', 'reference_patches', 'reference_mechanism'):
        require(sha(getattr(args, name)) == getattr(args, name + '_sha256'), name + ' bytes differ')
    require(args.model_sha256 == MODEL_SHA and sha(args.bank) == args.bank_sha256 == basis.BANK_SHA, 'Fixed model/bank differs')
    helper = basis.load_module(args.helper, basis.HELPER_SHA, 'frozen_diag_original_boundary')
    old = basis.load_module(args.old_pilot, basis.OLD_PILOT_SHA, 'frozen_diag_original_metric')
    request = read(args.request); root, source, manifest, bank, records = helper.metadata(request)
    require(sha(request['bank']) == args.bank_sha256, 'Request/bank differs')
    oracle = basis.oracle_manifest(args, bank)
    normal = oracle['normalization']
    references = [json.loads(line) for line in Path(args.reference_patches).read_bytes().splitlines() if line]
    mechanisms = [json.loads(line) for line in Path(args.reference_mechanism).read_bytes().splitlines() if line]
    order = [(c, s) for c in bank['probe_cases'] for s in range(16)]
    require([(r['case_id'], r['slot']) for r in references] == [(r['case_id'], r['slot']) for r in mechanisms] == order,
        'Reference must retain all192 original PROBE rows')
    refs = dict(zip(order, references)); perms = dict(zip(order, mechanisms))
    output = Path(args.output); require(not output.exists(), 'Fresh diagnostic output required')
    import numpy as np
    import torch
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    for path in (root / 'code', root / 'next_stage'):
        sys.path.insert(0, str(path))
    modules = {name: importlib.import_module(name) for name in ('data', 'geometry', 'train')}
    require(all(Path(m.__file__).resolve() == root / 'code' / (n + '.py') for n, m in modules.items()), 'Source import escaped')
    candidate = basis.load_module(args.model, MODEL_SHA, 'frozen_diag_basis_model')
    require(torch.cuda.is_available(), 'Actual CUDA required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    modules['train'].set_seed(args.seed)
    model = candidate.CalibreBasisStageModel(arm='AR', alpha=1.)
    model.load_common_state(torch.load(request['weights'], map_location='cpu'))
    model.install_moments(normal['mean'], normal['population_std'])
    checkpoint = torch.load(args.trained_state, map_location='cpu')
    active = set(model.trainable_parameter_names())
    require(checkpoint['schema'] == 'GTA-AVT-calibre-basis-coadapt-trainable-state-v1' and checkpoint['arm'] == 'AR' and
        checkpoint['seed'] == args.seed and checkpoint['successful_updates'] == 2000 and checkpoint['normalization'] == normal and
        checkpoint['common_weights_sha256'] == helper.WEIGHT_SHA and checkpoint['model_sha256'] == MODEL_SHA and
        checkpoint['oracle_bank_sha256'] == args.oracle_bank_sha256 and set(checkpoint['state']) == active,
        'Completed AR checkpoint identity differs')
    # Merge before strict loading. Loading only parameter fragments invokes
    # BatchNorm's old-checkpoint fallback for missing num_batches_tracked.
    # Every frozen common parameter and buffer must retain its actual value.
    merged = model.state_dict()
    merged.update(checkpoint['state'])
    loaded = model.load_state_dict(merged, strict=True)
    require(not loaded.unexpected_keys and not loaded.missing_keys, 'Trained-state keys differ')
    del merged
    del checkpoint
    model.requires_grad_(False).cuda().eval()
    state_before = old.tensor_map_sha(model.state_dict())
    require(all(not m.training for m in model.modules()), 'All modules must remain eval')
    specs = {(r['case_id'], r['slot']): r for r in bank['patches']}
    output.mkdir(parents=True)
    write(output / 'intent.json', dict(schema='GTA-AVT-calibre-basis-frozen-diagnostic-intent-v1', seed=args.seed,
        worker_sha256=sha(__file__), basis_worker_sha256=BASIS_SHA, model_sha256=MODEL_SHA,
        trained_state_sha256=args.trained_state_sha256, reference_patches_sha256=args.reference_patches_sha256,
        reference_mechanism_sha256=args.reference_mechanism_sha256, oracle_bank_sha256=args.oracle_bank_sha256,
        variants=list(VARIANTS), model_eval=True, no_grad=True, optimizer_created=False,
        candidate_STOP_unchanged=True, new_success_gate=False, region_distance='Real observed patch/cache intersection six faces; not padded128 boundary'))
    rows = 0; maximum_bce_ulps = 0; started = time.monotonic()
    with (output / 'diagnostics.jsonl').open('xb') as stream:
        for case in bank['probe_cases']:
            helper.cached_inputs(dict(request, case_ids=[case]), records)
            home = Path(records[case]['cache_dir'])
            image = np.load(home / 'image.npy', mmap_mode='r', allow_pickle=False)
            mask = np.load(home / 'mask.npy', mmap_mode='r', allow_pickle=False)
            with np.load(home / 'geometry.npz', allow_pickle=False) as f:
                geometry = {k: f[k] for k in f.files}
            item = oracle['cases'][case]; require(sha(item['path']) == item['sha256'], 'Oracle case bytes differ')
            with np.load(item['path'], allow_pickle=False) as f:
                A, R, slots = f['A'].copy(), f['R'].copy(), f['slot'].tolist()
            require(slots == list(range(16)) and A.shape == R.shape == (16, 64, 64, 64) and np.isin(A, [0, 1]).all() and
                np.isfinite(R).all() and (R[A == 0] == 0).all(), 'Oracle fields differ')
            for begin in range(0, 16, 2):
                prepared = []
                for slot in (begin, begin + 1):
                    spec = specs[(case, slot)]; start = spec['start_zyx']
                    slices = tuple(slice(v, min(v + 128, n)) for v, n in zip(start, image.shape))
                    raw, truth = np.asarray(image[slices], np.float32), np.asarray(mask[slices], bool)
                    pad = [(0, 128 - n) for n in raw.shape]
                    scale, structure = modules['geometry'].patch_histograms(geometry, start, [128] * 3)
                    known = bool(scale.sum() > 0 and not (truth.any() and scale[0] == 1.))
                    points = geometry['points']; chosen = ((points >= np.asarray(start)) & (points < np.asarray(start) + 128)).all(1)
                    coordinate = 2. * (np.asarray(records[case]['crop_start_zyx']) + np.asarray(start) + 64.) / np.maximum(
                        np.asarray(records[case]['full_shape_zyx']) - 1., 1.) - 1.
                    prepared.append(dict(image=np.ascontiguousarray(modules['data'].normalize(np.pad(raw, pad,
                        constant_values=-1024))[None], np.float32), truth=np.pad(truth, pad), valid=np.pad(np.ones(raw.shape, bool), pad),
                        real_shape=tuple(raw.shape), points=points[chosen] - np.asarray(start), radii=geometry['radius'][chosen],
                        coordinate=coordinate.astype(np.float32), A=A[slot], R=R[slot], case_id=case, source=spec['source'], slot=slot,
                        stratum=spec['stratum'], attribute_valid=known, known_nonempty=bool(known and truth.any()), valid_empty=bool(known and not truth.any())))
                CT = torch.from_numpy(np.stack([p['image'] for p in prepared])).cuda()
                coordinates = torch.from_numpy(np.stack([p['coordinate'] for p in prepared])).cuda()
                fields = dict(A=torch.from_numpy(A[begin:begin+2, None]).cuda().float(),
                    logR=torch.from_numpy(R[begin:begin+2, None]).cuda(), valid=torch.from_numpy((A[begin:begin+2, None] == 1)).cuda())
                outputs = [dict(case_id=case, source=p['source'], slot=p['slot'], stratum=p['stratum'],
                    primary=p['slot'] < 8 and p['known_nonempty'], known_nonempty=p['known_nonempty'], real_shape=list(p['real_shape']),
                    radius_support=int(p['A'].sum()), variants={}, parity={}) for p in prepared]
                normal_zR = None
                for variant in VARIANTS:
                    overrides = []; maps = []
                    for p in prepared:
                        values = p['R'][p['A'] == 1].astype(np.float64)
                        if variant == 'half_ring':
                            override, mapping = basis.radius_permutation(np, p['A'], p['R']); maps.append(mapping)
                        elif variant == 'patch_mean_FP32':
                            override = np.zeros_like(p['R']); override[p['A'] == 1] = np.float32(values.mean()) if len(values) else 0.
                        elif variant == 'FIT_mean_FP64':
                            override = np.zeros(p['R'].shape, np.float64); override[p['A'] == 1] = normal['mean']
                        else:
                            override = p['R']
                        overrides.append(override)
                    kwargs = {} if variant == 'correct' else dict(radius_override=dict(
                        logR=torch.from_numpy(np.stack([v[None] for v in overrides])).cuda(), valid=fields['valid']), diagnostic=True)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        logits = model(CT, coordinate=coordinates, oracle=fields, oracle_mode=True, alpha=1., **kwargs)['logits'].float()
                    prediction = (logits.sigmoid() >= .5).cpu().numpy()[:, 0]; cpu = logits.cpu().numpy()[:, 0]
                    zR = model.last_calibre_trace['zR'].detach()
                    if variant == 'correct': normal_zR = zR.clone()
                    if variant == 'FIT_mean_FP64': require(bool((zR == 0).all()), 'FIT mean must produce exact zero R axis')
                    for i, p in enumerate(prepared):
                        p['prediction'] = prediction[i]; metrics = old.metric(np, ndi, skeletonize, cpu[i], p)
                        outputs[i]['variants'][variant] = dict(metrics=metrics, regions=regional_metrics(np, cpu[i], prediction[i], p),
                            actual_zR_changed_from_correct=not torch.equal(zR[i], normal_zR[i]))
                        key = case, p['slot']
                        if variant in ('correct', 'half_ring'):
                            expected = refs[key]['metrics'] if variant == 'correct' else perms[key]['permuted']
                            parity = metric_parity(metrics, expected); maximum_bce_ulps = max(maximum_bce_ulps, parity['BCE_binary64_ULP'])
                            outputs[i]['parity'][variant] = parity
                            if variant == 'half_ring':
                                require(maps[i] == perms[key]['permutation'] and
                                    outputs[i]['variants'][variant]['actual_zR_changed_from_correct'] == perms[key]['zR_changed'],
                                    'Original half-ring map/actual consumption differs')
                    del logits, cpu, prediction, zR, kwargs
                for row in outputs:
                    stream.write(basis.encoded(row)); rows += 1
                stream.flush()
                del CT, coordinates, fields, normal_zR, prepared, outputs
            require(sha(item['path']) == item['sha256'], 'Oracle file changed')
            helper.cached_inputs(dict(request, case_ids=[case]), records)
    require(rows == 192 and state_before == old.tensor_map_sha(model.state_dict()), 'Frozen model changed/incomplete diagnostic')
    for name in ('trained_state', 'reference_patches', 'reference_mechanism'):
        require(sha(getattr(args, name)) == getattr(args, name + '_sha256'), 'Completed input changed')
    write(output / 'summary.json', dict(schema='GTA-AVT-calibre-basis-frozen-diagnostic-v1', status='verified_complete', seed=args.seed,
        patches=rows, variants=list(VARIANTS), whole_metric_reference_parity_passed=True, maximum_BCE_ULP=maximum_bce_ulps,
        model_state_unchanged=True, optimizer_created=False, weights_updated=False, candidate_STOP_unchanged=True,
        no_new_success_gate=True, CT_predictor_training_authorized=False, elapsed_seconds=time.monotonic()-started))
    return dict(status='verified_complete', patches=rows, weights_updated=False, candidate_STOP_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'bank', 'oracle-bank', 'oracle-helper', 'model', 'trained-state', 'reference-patches', 'reference-mechanism'):
        parser.add_argument('--' + name); parser.add_argument('--' + name + '-sha256')
    for name, filename in (('basis-worker', 'calibre_basis_pilot.py'), ('helper', 'boundary_attribute_probe.py'), ('old-pilot', 'boundary_field_pilot.py')):
        parser.add_argument('--' + name, default=str(Path(__file__).with_name(filename)))
    parser.add_argument('--basis-worker-sha256', default=BASIS_SHA)
    parser.add_argument('--seed', type=int, choices=(20261030,20261031,20261032)); parser.add_argument('--port', type=int)
    parser.add_argument('--mode', choices=('run',), default='run'); parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', variants=list(VARIANTS), data_or_weights_read=False, GPU_or_optimizer_created=False)))
        return
    required = ('request','bank','oracle_bank','oracle_helper','model','trained_state','reference_patches','reference_mechanism')
    if args.seed is None or not args.output or not all(getattr(args,k) and getattr(args,k+'_sha256') for k in required):
        parser.error('Explicit original inputs/model, completed AR/reference paths+SHA, seed and fresh output required')
    print(json.dumps(execute(args), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
