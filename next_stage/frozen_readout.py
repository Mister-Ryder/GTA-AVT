"""Frozen local-scale readout B: NumPy targets and Torch-only calibration.

This file imports no scipy/skimage/data/geometry module. Normal inference
helpers accept only CT features and CT input-validity, never GT validity.
The CLI separates an immutable plan from an explicitly bound execute action.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

SIZE = 128
GRID = 32
EDGES = np.asarray([2., 4., 8.], np.float64)
CORE_RULE = "GAAM centres=1.5+4u; integer window=[4u-6,4u+10); 16mm at1mm"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to replace an existing audit artifact: '+str(path))
    part = path.with_name(path.name + '.tmp')
    with part.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(str(part), str(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalize(image):
    """Byte-for-byte expression of the frozen data.py normalize formula."""
    return (np.clip(image, -200., 1000.) - 200.) / 400.


def _volume(value, name):
    value = np.asarray(value)
    if value.shape != (SIZE, SIZE, SIZE):
        raise ValueError(name + ' must have shape128^3')
    return value


def _box_sum(volume):
    """Sum fixed16^3 windows on the GAAM grid, clipping only for querying."""
    volume = _volume(volume, 'volume')
    # The largest total is128^3; int32 is exact and keeps CPU workspace small.
    prefix = np.pad(volume.astype(np.int32, copy=False), ((1, 0),) * 3)
    for axis in range(3):
        prefix = np.cumsum(prefix, axis=axis, dtype=np.int32)
    lower = np.clip(4 * np.arange(GRID) - 6, 0, SIZE)
    upper = np.clip(4 * np.arange(GRID) + 10, 0, SIZE)
    result = np.zeros((GRID, GRID, GRID), np.int32)
    for iz in (0, 1):
        for iy in (0, 1):
            for ix in (0, 1):
                zz = upper if iz else lower
                yy = upper if iy else lower
                xx = upper if ix else lower
                sign = (-1) ** (3 - iz - iy - ix)
                result += sign * prefix[np.ix_(zz, yy, xx)]
    return result


def ct_window_support(input_valid):
    """CT-only support, independent of mask/geometry: full4096 visible voxels."""
    return _box_sum(_volume(input_valid, 'input_valid').astype(bool)) == 16 ** 3


def local_targets(mask, points_local, radius_mm, input_valid, geometry_valid=True):
    mask = _volume(mask, 'mask').astype(bool)
    ct_support = ct_window_support(input_valid)
    points = np.asarray(points_local)
    radii = np.asarray(radius_mm, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or radii.shape != (len(points),):
        raise ValueError('Expected aligned point/radius arrays')
    if not np.isfinite(points).all() or not np.isfinite(radii).all() or (radii <= 0.).any():
        raise ValueError('Nonfinite/nonpositive geometry')
    if not np.equal(points, np.round(points)).all():
        raise ValueError('Cache skeleton coordinates must be integer voxels')
    points = points.astype(np.int64)
    inside = ((points >= 0) & (points < SIZE)).all(1)
    points, radii = points[inside], radii[inside]
    if len(points) != len(np.unique(points, axis=0)):
        raise ValueError('Duplicate centreline seeds')
    if len(points) and not mask[tuple(points.T)].all():
        raise ValueError('Centreline point outside TRAIN foreground')
    hist = np.zeros((4, GRID, GRID, GRID), np.int32)
    if geometry_valid:
        bins = np.searchsorted(EDGES, radii, side='left')
        for group in range(4):
            occupancy = np.zeros(mask.shape, np.uint8)
            selected = points[bins == group]
            if len(selected):
                occupancy[tuple(selected.T)] = 1
            hist[group] = _box_sum(occupancy)
    counts = hist.sum(0)
    foreground = _box_sum(mask) > 0
    true_empty = ct_support & ~foreground
    supported = ct_support & (counts > 0)
    loss_valid = true_empty | supported
    target = np.zeros((5, GRID, GRID, GRID), np.float32)
    target[0, true_empty] = 1.
    target[1:, supported] = hist[:, supported] / counts[supported][None, :]
    if not np.allclose(target.sum(0)[loss_valid], 1., rtol=1e-6, atol=1e-6):
        raise AssertionError('Local target mass invalid')
    return dict(target=target, loss_valid=loss_valid, ct_support=ct_support,
                foreground_window=foreground, support_nodes=counts)


def transform_spatial(array, flips, rotation, axes, spatial_start=0):
    value = array
    for axis, flip in enumerate(flips):
        if flip:
            value = np.flip(value, axis=axis + spatial_start)
    value = np.rot90(value, int(rotation),
                     tuple(int(a) + spatial_start for a in axes))
    return np.ascontiguousarray(value)


def align_classifier_features(feature):
    """Classifier4u grid -> GAAM(1.5+4u) nominal grid, explicit sampling."""
    import torch
    from torch.nn import functional as F
    if feature.ndim != 5 or tuple(feature.shape[1:]) != (64, GRID, GRID, GRID):
        raise ValueError('Expected B,64,32,32,32 classifier feature')
    coord = 2. * (torch.arange(GRID, device=feature.device, dtype=torch.float32) + .375) / (GRID - 1.) - 1.
    zz, yy, xx = torch.meshgrid(coord, coord, coord)
    sampling = torch.stack((xx, yy, zz), dim=-1)[None].expand(feature.shape[0], -1, -1, -1, -1)
    # FP32 resampling and storage; do not introduce an extra half quantization.
    with torch.cuda.amp.autocast(enabled=False):
        return F.grid_sample(feature.float(), sampling, mode='bilinear',
                             padding_mode='border', align_corners=True)


def ct_support_tensor(input_valid):
    import torch
    from torch.nn import functional as F
    if input_valid.ndim != 5 or tuple(input_valid.shape[1:]) != (1, SIZE, SIZE, SIZE):
        raise ValueError('Expected CT input validity B,1,128,128,128')
    return F.avg_pool3d(input_valid.float(), kernel_size=16, stride=4,
                       padding=6, count_include_pad=True) == 1.


def make_readout_head(seed=20261023):
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        head = torch.nn.Conv3d(64, 5, 1)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        return head


def readout_probabilities(head, feature):
    import torch
    aligned = align_classifier_features(feature)
    with torch.cuda.amp.autocast(enabled=False):
        return head(aligned).float().softmax(1)


def normal_local_score(head, feature, input_valid, s0=.2):
    """The only normal consumption gate is CT-known full-window support."""
    import torch
    probability = readout_probabilities(head, feature)
    valid = ct_support_tensor(input_valid)
    return torch.where(valid, probability[:, 1:2],
                       torch.full_like(probability[:, 1:2], float(s0)))


def _sid(case_id):
    short = str(case_id).rsplit('_', 1)[-1]
    if len(short) < 2 or short[0] not in 'DKR' or not short[1:].isdigit():
        raise ValueError('Unexpected case identity')
    return short


def _source(case_id):
    return {'D': 'Dongyang', 'K': 'KiTS', 'R': 'Rider'}[_sid(case_id)[0]]


def split_train(manifest, seed):
    train = list(manifest['splits']['train'])
    if len(train) != 36 or len(set(map(_sid, train))) != 36:
        raise ValueError('Expected frozen TRAIN36')
    for role in ('val', 'test'):
        if set(map(_sid, train)) & set(map(_sid, manifest['splits'].get(role, []))):
            raise ValueError('TRAIN/heldout identity overlap')
    fit, check = [], []
    for index, source in enumerate(('Dongyang', 'KiTS', 'Rider')):
        members = sorted(x for x in train if _source(x) == source)
        if len(members) != 12:
            raise ValueError('Expected12TRAIN cases per source')
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 16]))
        shuffled = [members[int(i)] for i in rng.permutation(len(members))]
        fit.extend(shuffled[:8])
        check.extend(shuffled[8:])
    return fit, check


def _load_geometry(directory):
    with np.load(Path(directory) / 'geometry.npz', allow_pickle=False) as stream:
        result = {k: stream[k] for k in stream.files}
    return result


def _file_identity(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _same_file_identity(record):
    return _file_identity(record['path']) == record


def _patch_spec(case_id, directory, geometry, role, ordinal, slot, seed):
    image = np.load(Path(directory) / 'image.npy', mmap_mode='r', allow_pickle=False)
    shape = np.asarray(image.shape)
    del image
    rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal, slot, 731]))
    if slot < 4:
        points = geometry['points']
        if not len(points):
            raise ValueError('Requested fixed FG bank but no centreline: ' + case_id)
        centre = points[rng.integers(len(points))] + rng.integers(-32, 33, size=3)
        start = np.clip(centre - 64, 0, np.maximum(shape - SIZE, 0))
        stratum = 'GT_centerline_plus_jitter'
    else:
        start = np.asarray([rng.integers(max(1, int(n) - SIZE + 1)) for n in shape])
        stratum = 'CT_uniform'
    return dict(case_id=case_id, role=role, source=_source(case_id), slot=slot,
                stratum=stratum, start_zyx=start.tolist(),
                flips=[bool(rng.random() < .5) for _ in range(3)],
                rotation=int(rng.integers(4)),
                rotation_axes=list(((0, 1), (0, 2), (1, 2))[int(rng.integers(3))]),
                intensity_seed=[int(seed), ordinal, slot, 932])


def prepare_example(directory, geometry, spec, include_image=True):
    directory = Path(directory)
    mask_mm = np.load(directory / 'mask.npy', mmap_mode='r', allow_pickle=False)
    shape = np.asarray(mask_mm.shape)
    start = np.asarray(spec['start_zyx'])
    slices = tuple(slice(int(x), min(int(x) + SIZE, int(n))) for x, n in zip(start, shape))
    mask = np.asarray(mask_mm[slices], dtype=bool)
    del mask_mm
    actual_shape = mask.shape
    pad = [(0, SIZE - n) for n in actual_shape]
    input_valid = np.pad(np.ones(actual_shape, bool), pad)
    mask = np.pad(mask, pad)
    local = local_targets(mask, geometry['points'] - start, geometry['radius'],
                          input_valid, bool(np.asarray(geometry.get('geometry_valid', True)).item()))
    transform = (spec['flips'], spec['rotation'], spec['rotation_axes'])
    for key in ('loss_valid', 'ct_support', 'foreground_window', 'support_nodes'):
        local[key] = transform_spatial(local[key], *transform)
    local['target'] = transform_spatial(local['target'], *transform, spatial_start=1)
    local['input_valid'] = transform_spatial(input_valid, *transform)
    if include_image:
        image_mm = np.load(directory / 'image.npy', mmap_mode='r', allow_pickle=False)
        image = np.asarray(image_mm[slices], dtype=np.float32)
        del image_mm
        image = np.pad(image, pad, constant_values=-1024)
        image = transform_spatial(image, *transform)
        image = normalize(image)
        rng = np.random.default_rng(np.random.SeedSequence(spec['intensity_seed']))
        image = image * rng.uniform(.9, 1.1) + rng.uniform(-.1, .1)
        if rng.random() < .15:
            image += rng.normal(0., .03, image.shape).astype(np.float32)
        local['image'] = np.ascontiguousarray(image[None], dtype=np.float32)
    return local


def target_stats(local):
    supported = local['loss_valid'] & local['foreground_window']
    unknown = local['ct_support'] & local['foreground_window'] & ~local['loss_valid']
    fine = local['target'][1]
    return dict(ct_supported_cells=int(local['ct_support'].sum()),
                valid_cells=int(local['loss_valid'].sum()),
                foreground_valid_cells=int(supported.sum()),
                true_empty_cells=int((local['loss_valid'] & ~local['foreground_window']).sum()),
                foreground_unknown_cells=int(unknown.sum()),
                fine_positive_cells=int((supported & (fine > 0.)).sum()),
                fine_fraction_fg_mean=float(fine[supported].mean()) if supported.any() else None)


def select_windows(local, seed, ordinal, slot, cap=64):
    """Fixed stratified inclusion; weights estimate uniform valid-window CE."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal, slot, 617]))
    selected, weights, strata = [], [], []
    counts = {}
    for label, active in (
            ('foreground', local['loss_valid'] & local['foreground_window']),
            ('empty', local['loss_valid'] & ~local['foreground_window'])):
        indices = np.flatnonzero(active)
        count = len(indices)
        counts[label] = int(count)
        if not count:
            continue
        chosen = np.sort(rng.choice(indices, min(cap, count), replace=False))
        selected.extend(chosen.tolist())
        weights.extend([float(count) / len(chosen)] * len(chosen))
        strata.extend([label] * len(chosen))
    if not selected:
        raise ValueError('No CT-supported supervised window in fixed patch')
    coords = np.stack(np.unravel_index(selected, (GRID,) * 3), axis=1)
    weight = np.asarray(weights, np.float64)
    if not np.isclose(weight.sum(), int(local['loss_valid'].sum()), rtol=1e-12):
        raise AssertionError('Inclusion weights do not reconstruct valid population size')
    return dict(coords_zyx=coords.tolist(), inverse_inclusion_weights=weights,
                strata=strata, population_counts=counts,
                selected_fine_cells=int(np.count_nonzero(
                    local['target'][1].ravel()[np.asarray(selected)] > 0.)),
                sampling='fixed up to64 supported nonempty and64 trueempty; no error-based selection')


def _source_map(code_root):
    return {'frozen_readout.py': sha256(__file__),
            'frozen_readout_fit.py': sha256(Path(__file__).with_name('frozen_readout_fit.py')),
            'AnatomicalClassifier.py': sha256(Path(code_root) / 'historical/func/AnatomicalClassifier.py'),
            'frozen_data.py_reference': sha256(Path(code_root) / 'data.py')}


def build_plan(args):
    manifest = read_json(args.manifest)
    if float(manifest['spacing_mm']) != 1.:
        raise ValueError('Local readout fixes1mm cache')
    fit, check = split_train(manifest, args.seed)
    records = {x['id']: x for x in manifest['cases']}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Immutable new plan output required')
    if sha256(args.weights) != args.weights_sha256.lower():
        raise ValueError('Common checkpoint SHA mismatch')
    gate = read_json(args.gate_config)
    if not isinstance(gate, dict) or not gate.get('locked_before_fit'):
        raise ValueError('A prespecified gate config locked_before_fit is required')
    required_gate = dict(minimum_relative_raw_MSE_reduction=.1,
                         minimum_relative_conditional_MSE_reduction=.1,
                         minimum_conditional_MSE_better_cases=8,
                         minimum_fine_supported_cases=8,
                         minimum_sampled_fine_cells_per_case=16,
                         all_three_source_raw_and_conditional_MSE_not_worse=True,
                         optimizer_resolved_required=True)
    if any(gate.get(k) != v for k, v in required_gate.items()):
        raise ValueError('Gate differs from the single prespecified B recipe')
    bank, inputs = [], {}
    for ordinal, case_id in enumerate(fit + check):
        record = records[case_id]
        directory = Path(args.cache_root) / case_id
        if not record.get('geometry_training_only', True):
            raise ValueError('Not TRAIN geometry: ' + case_id)
        geometry = _load_geometry(directory)
        inputs[case_id] = dict(
            cache_dir=str(directory.resolve()),
            geometry_sha256=sha256(directory / 'geometry.npz'),
            image=_file_identity(directory / 'image.npy'),
            mask=_file_identity(directory / 'mask.npy'),
            image_mask_verification='size+mtime; no fresh full byte SHA claimed')
        for slot in range(8):
            spec = _patch_spec(case_id, directory, geometry,
                               'fit' if case_id in fit else 'check', ordinal, slot, args.seed)
            local = prepare_example(directory, geometry, spec, False)
            spec['target_audit'] = target_stats(local)
            spec['window_sample'] = select_windows(local, args.seed, ordinal, slot)
            bank.append(spec)
        print(json.dumps({'event': 'planned_case', 'id': case_id,
                          'role': 'fit' if case_id in fit else 'check'}), flush=True)
    plan = dict(schema='GTA-AVT-frozen-local-readout-B-plan-v1', mode='plan_only_no_GPU_fit',
                numpy_version=np.__version__,
                seed=args.seed, head_seed=args.head_seed,
                manifest=str(Path(args.manifest).resolve()), manifest_sha256=sha256(args.manifest),
                code_root=str(Path(args.code_root).resolve()), source_sha256=_source_map(args.code_root),
                weights=str(Path(args.weights).resolve()), weights_sha256=args.weights_sha256.lower(),
                fit_cases=fit, check_cases=check, inputs=inputs, bank=bank,
                target_definition=CORE_RULE, thresholds_mm=EDGES.tolist(),
                normal_inference_support='CT full-window/padding only; never GT geometry mask',
                frozen_feature_precision='classifier AMP then aligned FP32 selected64dim vectors',
                readout_precision='FP32 64-to5 1x1 conv plus softmax',
                fit_recipe=dict(optimizer='CPU64_fullbatch_LBFGS', lr=1.,
                                initialization='zero_weight_zero_bias',
                                lambda_weight_l2=.0001, bias_penalized=False,
                                max_iter=500, requested_max_eval=650,
                                tolerance_grad=1e-6, tolerance_change=1e-9,
                                history_size=100, line_search_fn='strong_wolfe',
                                objective='case_equal_patch_equal_uniform_valid_window_CE_IPW_estimator',
                                local_loss_normalization='inverse_inclusion_uniform_window'),
                max_feature_cache_bytes=args.max_feature_cache_bytes,
                split_rule='lexicographic_ID_then_SeedSequence(seed,source_index,16)',
                calibration_gate=gate, calibration_gate_sha256=sha256(args.gate_config),
                heldout_data_files_opened=False, parameters_updated=False)
    output.mkdir(parents=True)
    write_json(output / 'plan.json', plan)
    print(json.dumps({'event': 'plan_complete', 'file': str(output / 'plan.json'),
                      'sha256': sha256(output / 'plan.json')}), flush=True)
    return plan


def _state_hash(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode())
        result.update(str(array.dtype).encode())
        result.update(np.asarray(array.shape, np.int64).tobytes())
        result.update(array.tobytes())
    return result.hexdigest()


def execute_plan(args):
    from frozen_readout_fit import execute_plan as run
    return run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest')
    parser.add_argument('--cache-root')
    parser.add_argument('--code-root')
    parser.add_argument('--weights')
    parser.add_argument('--weights-sha256')
    parser.add_argument('--output')
    parser.add_argument('--gate-config')
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--head-seed', type=int, default=20261023)
    parser.add_argument('--max-feature-cache-bytes', type=int, default=64 * 1024 ** 2)
    parser.add_argument('--plan')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.plan_only:
        required = ('manifest', 'cache_root', 'code_root', 'weights', 'weights_sha256',
                    'output', 'gate_config')
        if any(getattr(args, key) is None for key in required):
            parser.error('plan requires ' + ','.join(required))
        build_plan(args)
    else:
        if not args.plan or not args.plan_sha256:
            parser.error('execute requires --plan and --plan-sha256')
        execute_plan(args)


if __name__ == '__main__':
    main()
