"""TRAIN-only audit of AVT annotation-derived geometric surrogate attributes.

No model is loaded. No data, cache, target or old result is modified.
The radius perturbation intervals below are sensitivity probes, not confidence
intervals or measured annotation error. Legacy 1.5-mm results are explicitly
separate from the formal 1-mm training cache.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

EDGES = np.asarray([2., 4., 8.])
MARGINS = (.25, .5)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def short_id(value):
    value = str(value).rsplit('_', 1)[-1]
    if len(value) < 2 or value[0] not in 'DKR' or not value[1:].isdigit():
        raise ValueError('Unexpected released case identity: ' + value)
    return value


def cohort(case_id):
    return {'D': 'Dongyang', 'K': 'KiTS', 'R': 'Rider'}[short_id(case_id)[0]]


def assert_train_split(split):
    train = list(split['train'])
    if len(set(map(short_id, train))) != len(train):
        raise ValueError('Duplicate TRAIN identity')
    for key in ('val', 'test'):
        if set(map(short_id, train)) & set(map(short_id, split.get(key, []))):
            raise ValueError('TRAIN intersects ' + key)
    if not train:
        raise ValueError('Empty TRAIN population')
    return train


def scalar(record, key, default=None):
    if key not in record:
        return default
    return np.asarray(record[key]).item()


def stats(values):
    values = np.asarray([x for x in values if x is not None], np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'defined': 0, 'mean': None, 'median': None, 'min': None,
                'max': None, 'q10': None, 'q90': None}
    return {'defined': int(len(values)), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'min': float(values.min()),
            'max': float(values.max()), 'q10': float(np.quantile(values, .1)),
            'q90': float(np.quantile(values, .9))}


def fraction(values):
    return float(np.mean(values)) if len(values) else None


def local_windows(points_mm, radius, seed, count, window_mm, grid_mm, shape_mm=None):
    """Sparse GT-enriched geometry probe, not natural CT sampling prevalence."""
    if not len(radius):
        return {'sampled_centres': 0, 'reason': 'no_centerline'}
    centres = np.unique(np.round(points_mm / grid_mm) * grid_mm, axis=0)
    half = window_mm / 2.
    if shape_mm is not None:
        centres = centres[((centres >= half) &
                           (centres <= np.asarray(shape_mm) - half)).all(1)]
    if not len(centres):
        return {'sampled_centres': 0, 'reason': 'no_full_window_support'}
    rng = np.random.default_rng(seed)
    if len(centres) > count:
        centres = centres[rng.choice(len(centres), count, replace=False)]
    tree = cKDTree(points_mm)
    groups = tree.query_ball_point(centres, half, p=np.inf)
    supports = np.asarray([len(x) for x in groups], np.int32)
    fractions = np.asarray([np.mean(radius[x] <= 2.) if len(x) else np.nan
                            for x in groups], np.float64)
    valid = np.isfinite(fractions)
    f = fractions[valid]
    return {'sampling': 'GT-centerline-enriched unique grid centres; descriptive only',
            'window_shape': 'axis-aligned physical cube',
            'window_side_mm': window_mm, 'grid_mm': grid_mm,
            'sampled_centres': int(len(centres)),
            'full_input_support_enforced': shape_mm is not None,
            'positive_support_centres': int(valid.sum()),
            'support_nodes': stats(supports.tolist()),
            'fine_fraction': stats(f.tolist()),
            'fine_fraction_sample_SD': float(f.std(ddof=1)) if len(f) > 1 else None,
            'mixed_fine_and_nonfine_fraction': fraction((f > 0.) & (f < 1.)),
            'any_fine_fraction': fraction(f > 0.),
            'all_fine_fraction': fraction(f == 1.),
            'global_case_fine_fraction': float(np.mean(radius <= 2.))}


def patch_support(mask_path, points, structure, seed, count, patch):
    if mask_path is None:
        return {'available': False, 'reason': 'legacy sparse audit has no formal cache mask'}
    mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
    shape = np.asarray(mask.shape)
    if mask.ndim != 3:
        raise ValueError('Expected 3D TRAIN mask: ' + str(mask_path))
    rng = np.random.default_rng(seed)
    records = []
    for index in range(count):
        enriched = index >= count // 2 and len(points)
        if enriched:
            centre = points[rng.integers(len(points))] + rng.integers(
                -patch // 4, patch // 4 + 1, size=3)
            start = np.clip(centre - patch // 2, 0, np.maximum(shape - patch, 0))
        else:
            start = np.asarray([rng.integers(max(1, int(n) - patch + 1)) for n in shape])
        inside = ((points >= start) & (points < start + patch)).all(1)
        slices = tuple(slice(int(s), min(int(s) + patch, int(n)))
                       for s, n in zip(start, shape))
        foreground = bool(np.any(mask[slices]))
        n = int(inside.sum())
        valid_structure = int(np.count_nonzero(structure[inside] != 255)) if n else 0
        status = ('true_empty' if not foreground else
                  'foreground_without_centerline_unknown' if n == 0 else
                  'scale_supported_structure_unknown' if valid_structure == 0 else
                  'scale_and_structure_supported')
        records.append({'stratum': 'GT_centerline_enriched' if enriched else 'CT_uniform',
                        'start_zyx': start.tolist(), 'status': status,
                        'centerline_nodes': n,
                        'structure_valid_nodes': valid_structure})
    summary = {}
    for stratum in ('CT_uniform', 'GT_centerline_enriched'):
        selected = [x for x in records if x['stratum'] == stratum]
        summary[stratum] = {
            'patches': len(selected),
            'status_counts': {k: sum(x['status'] == k for x in selected)
                              for k in sorted(set(x['status'] for x in selected))},
            'centerline_support': stats([x['centerline_nodes'] for x in selected])}
    del mask
    return {'available': True, 'patch_voxels_per_axis': patch,
            'sampling_note': 'fixed per-case diagnostic bank, not actual global sampler sequence',
            'strata': summary, 'patch_records': records}


def audit_case(case_id, record, points_mm, radius, structure, geometry_path,
               mask_path, points_grid, args, ordinal):
    radius = np.asarray(radius, np.float64)
    points_mm = np.asarray(points_mm, np.float64)
    if len(points_mm) != len(radius):
        raise ValueError('Point/radius length mismatch: ' + case_id)
    original_nodes = len(radius)
    nonfinite_nodes = int(np.count_nonzero(~np.isfinite(radius)))
    nonpositive_nodes = int(np.count_nonzero(np.isfinite(radius) & (radius <= 0.)))
    valid_radius = np.isfinite(radius) & (radius > 0.)
    radius = radius[valid_radius]
    points_mm = points_mm[valid_radius]
    if structure is not None:
        structure = np.asarray(structure)[valid_radius]
    if points_grid is not None:
        points_grid = np.asarray(points_grid)[valid_radius]
    bins = np.searchsorted(EDGES, radius, side='left')
    counts = np.bincount(bins, minlength=4)
    result = {'id': case_id, 'short_id': short_id(case_id), 'cohort': cohort(case_id),
              'geometry_path': str(geometry_path), 'geometry_sha256': sha256(geometry_path),
              'original_geometry_nodes': int(original_nodes),
              'excluded_nonfinite_radius_nodes': nonfinite_nodes,
              'excluded_nonpositive_radius_nodes': nonpositive_nodes,
              'node_denominator': 'finite positive-radius nodes; excluded invalid nodes reported above',
              'points': int(len(radius)), 'radius_mm': stats(radius.tolist()),
              'radius_quantiles_05_25_50_75_95': np.quantile(radius, [.05, .25, .5, .75, .95]).tolist()
              if len(radius) else None,
              'scale_bin_node_counts': counts.tolist(),
              'fine_fraction': fraction(radius <= 2.),
              'scale_bin_fractions': (counts / len(radius)).tolist() if len(radius) else None,
              'geometry_valid': bool(scalar(record, 'geometry_valid', True)),
              'threshold_interval_probes': {}}
    for margin in MARGINS:
        ambiguous = np.min(np.abs(radius[:, None] - EDGES), axis=1) <= margin if len(radius) else []
        result['threshold_interval_probes'][str(margin)] = {
            'meaning': 'hypothetical radius interval; not calibrated annotation uncertainty',
            'any_bin_boundary_vulnerable_fraction': fraction(ambiguous),
            'at_boundaries_mm': {str(t): {'nodes': int(np.count_nonzero(np.abs(radius - t) <= margin)),
                                         'fraction': fraction(np.abs(radius - t) <= margin)}
                                 for t in EDGES},
            'fine_fraction_if_radius_plus_margin': fraction(radius + margin <= 2.),
            'fine_fraction_if_radius_minus_margin': fraction(radius - margin <= 2.)}
    if structure is not None:
        structure = np.asarray(structure)
        if len(structure) != len(radius) or not np.isin(structure, [1, 2, 255]).all():
            raise ValueError('Unexpected formal structure labels: ' + case_id)
        result['structure'] = {
            'available': True, 'unknown_nodes': int(np.count_nonzero(structure == 255)),
            'unknown_fraction': fraction(structure == 255),
            'junction_nodes': int(np.count_nonzero(structure == 2)),
            'junction_fraction_all_nodes': fraction(structure == 2),
            'candidate_junctions': scalar(record, 'candidate_junction_count'),
            'supported_junctions': scalar(record, 'supported_junction_count'),
            'rejected_junctions': scalar(record, 'unknown_junction_count')}
    else:
        result['structure'] = {'available': False,
                               'reason': 'legacy geometry does not contain supported/unknown structure labels'}
    result['local_window_probe'] = local_windows(
        points_mm, radius, args.seed + ordinal, args.window_centres, args.window_mm,
        args.grid_mm, record.get('shape_mm'))
    result['patch_supervision_probe'] = patch_support(
        mask_path, points_grid, structure, args.seed + ordinal + 10000,
        args.patch_samples_per_case, args.patch_size) if structure is not None else {
            'available': False, 'reason': 'formal cache required'}
    return result


def aggregate(cases):
    return {'cases': len(cases), 'nodes': sum(x['points'] for x in cases),
            'cases_with_fine_support': sum(x['fine_fraction'] is not None and x['fine_fraction'] > 0. for x in cases),
            'original_geometry_nodes': sum(x['original_geometry_nodes'] for x in cases),
            'excluded_nonfinite_radius_nodes': sum(x['excluded_nonfinite_radius_nodes'] for x in cases),
            'excluded_nonpositive_radius_nodes': sum(x['excluded_nonpositive_radius_nodes'] for x in cases),
            'scale_bin_node_counts': np.sum([x['scale_bin_node_counts'] for x in cases], axis=0).tolist(),
            'fine_fraction_case_stats': stats([x['fine_fraction'] for x in cases]),
            'radius_median_case_stats': stats([x['radius_mm']['median'] for x in cases]),
            'near_2mm_within_025_case_stats': stats([
                x['threshold_interval_probes']['0.25']['at_boundaries_mm']['2.0']['fraction'] for x in cases]),
            'near_2mm_within_05_case_stats': stats([
                x['threshold_interval_probes']['0.5']['at_boundaries_mm']['2.0']['fraction'] for x in cases]),
            'unknown_structure_case_stats': stats([
                x['structure'].get('unknown_fraction') for x in cases]),
            'local_mixed_case_stats': stats([
                x['local_window_probe'].get('mixed_fine_and_nonfine_fraction') for x in cases])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest', type=Path)
    mode.add_argument('--legacy-dir', type=Path)
    p.add_argument('--split', type=Path, help='Required for legacy-dir; same frozen original split')
    p.add_argument('--output', required=True, type=Path, help='Must not exist')
    p.add_argument('--seed', type=int, default=20261023)
    p.add_argument('--window-mm', type=float, default=16.)
    p.add_argument('--grid-mm', type=float, default=4.)
    p.add_argument('--window-centres', type=int, default=256)
    p.add_argument('--patch-samples-per-case', type=int, default=16)
    p.add_argument('--patch-size', type=int, default=128)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite output: ' + str(args.output))
    if min(args.window_mm, args.grid_mm, args.window_centres, args.patch_samples_per_case, args.patch_size) <= 0:
        raise ValueError('Positive audit dimensions required')
    if args.manifest:
        manifest = load_json(args.manifest)
        train = assert_train_split(manifest['splits'])
        records = {x['id']: x for x in manifest['cases']}
        source = {'mode': 'formal_training_cache', 'manifest_sha256': sha256(args.manifest),
                  'cache_spacing_mm': manifest['spacing_mm']}
        cases = []
        for ordinal, case_id in enumerate(train):
            record = records[case_id]
            if not record.get('geometry_training_only', True):
                raise ValueError('Geometry not designated TRAIN-only: ' + case_id)
            directory = Path(record['cache_dir'])
            path = directory / 'geometry.npz'
            with np.load(path, allow_pickle=False) as z:
                geometry = {k: z[k] for k in z.files}
            spacing = float(record.get('spacing_mm', manifest['spacing_mm']))
            mask_path = directory / 'mask.npy'
            mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
            geometry['shape_mm'] = np.asarray(mask.shape) * spacing
            del mask
            cases.append(audit_case(case_id, geometry, geometry['points'] * spacing,
                                    geometry['radius'], geometry['structure'], path,
                                    mask_path, geometry['points'], args, ordinal))
            print(json.dumps({'audited_train_case': case_id, 'nodes': cases[-1]['points']}), flush=True)
    else:
        if args.split is None:
            p.error('--split required with --legacy-dir')
        split = load_json(args.split)
        train = assert_train_split(split)
        source = {'mode': 'legacy_independent_sparse_geometry_not_formal_training_cache',
                  'split_sha256': sha256(args.split),
                  'warning': 'Do not treat legacy 1.5-mm skeleton as formal 1-mm cache'}
        cases = []
        for ordinal, case_id in enumerate(train):
            sid = short_id(case_id)
            path = args.legacy_dir / (sid + '_centerline.npz')
            record = load_json(args.legacy_dir / (sid + '.json'))
            if record.get('population') != 'train_only':
                raise ValueError('Legacy record not TRAIN-only: ' + sid)
            with np.load(path, allow_pickle=False) as z:
                geometry = {k: z[k] for k in z.files}
            cases.append(audit_case(case_id, {}, geometry['lps_mm'],
                                    geometry['radius_mm'], None, path, None, None, args, ordinal))
            cases[-1]['legacy_geometry_iso_mm'] = record['geometry_iso_mm']
            print(json.dumps({'audited_train_case': case_id, 'nodes': cases[-1]['points']}), flush=True)
    result = {'schema': 'GTA-AVT-TRAIN-geometric-surrogate-audit-v1',
              'source': source,
              'scope': {'train_cases_only': True, 'heldout_images_masks_geometry_opened': False,
                        'models_loaded': False, 'targets_changed': False,
                        'old_artifacts_changed': False},
              'definitions': {
                  'radius': 'annotation-derived physical EDT at skeleton points; not clinical diameter',
                  'radius_thresholds_mm': EDGES.tolist(),
                  'thresholds_status': 'operational, not validated clinical categories',
                  'vulnerability_margins_mm': list(MARGINS),
                  'vulnerability_status': 'hypothetical sensitivity intervals, not measured annotation error',
                  'points': 'correlated skeleton nodes, not independent vessels or branch length',
                  'local_probe': 'GT-enriched centres; not natural CT occupancy; no model inference',
                  'structure': 'supported skeleton-neighbourhood proxy; unknown excluded, not false background',
                  'legacy_separation': 'legacy and formal outputs must not be pooled'},
              'parameters': {'seed': args.seed, 'window_mm': args.window_mm, 'grid_mm': args.grid_mm,
                             'window_centres': args.window_centres,
                             'patch_samples_per_case': args.patch_samples_per_case,
                             'patch_size': args.patch_size},
              'code_sha256': sha256(__file__), 'all_train': aggregate(cases),
              'cohorts': {name: aggregate([x for x in cases if x['cohort'] == name])
                          for name in sorted(set(x['cohort'] for x in cases))},
              'cases': cases}
    args.output.mkdir(parents=True)
    (args.output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                        allow_nan=False) + '\n', encoding='utf-8')
    rows = []
    for case in cases:
        rows.append({'id': case['id'], 'cohort': case['cohort'], 'nodes': case['points'],
                     'fine_fraction': case['fine_fraction'],
                     'near2_within025_fraction': case['threshold_interval_probes']['0.25']['at_boundaries_mm']['2.0']['fraction'],
                     'near2_within05_fraction': case['threshold_interval_probes']['0.5']['at_boundaries_mm']['2.0']['fraction'],
                     'unknown_structure_fraction': case['structure'].get('unknown_fraction'),
                     'local_mixed_fraction': case['local_window_probe'].get('mixed_fine_and_nonfine_fraction')})
    with (args.output / 'casewise.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'complete': True, 'output': str(args.output),
                      'all_train': result['all_train']}), flush=True)


if __name__ == '__main__':
    main()
