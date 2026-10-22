"""Read-only TRAIN36 cache QC and ALL56 CT-only intensity summary to stdout.

Opens the completed manifest, case records and TRAIN geometry.npz files only.
No image/mask array, held-out geometry, prediction, checkpoint or model is read.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import sys
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def scalar(geometry, name, default=None):
    return np.asarray(geometry[name]).item() if name in geometry else default


def divide(a, b):
    return float(a / b) if b else None


def describe(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {'defined_cases': 0, 'mean_cases': None, 'median_cases': None, 'min': None, 'max': None}
    return {'defined_cases': len(values), 'mean_cases': statistics.mean(values),
            'median_cases': statistics.median(values), 'min': min(values), 'max': max(values)}


def aggregate(rows):
    fields = ['native_roundtrip_dice', 'native_centerline_coverage', 'native_thin_le2mm_coverage',
              'fine_radius_le2mm_fraction', 'structure_valid_fraction', 'structure_unknown_fraction',
              'supported_junction_node_fraction', 'unknown_candidate_fraction']
    result = {'cases': len(rows), 'geometry_valid_cases': sum(row['geometry_valid'] for row in rows),
              'empty_skeleton_nonempty_mask_cases': sum(row['skeleton_empty_nonempty_mask'] for row in rows),
              'casewise': {field: describe([row.get(field) for row in rows]) for field in fields}}
    counts = ['resampled_skeleton_points', 'fine_radius_le2mm_points', 'structure_valid_points',
              'structure_unknown_points', 'supported_junction_points', 'candidate_junctions',
              'supported_junctions', 'unknown_junctions', 'nonfinite_radius_points', 'nonpositive_radius_points']
    result['totals'] = {field: sum(row[field] for row in rows) for field in counts}
    denominator = result['totals']['resampled_skeleton_points']
    result['point_weighted_descriptive_fractions'] = {
        'fine_radius_le2mm': divide(result['totals']['fine_radius_le2mm_points'], denominator),
        'structure_valid': divide(result['totals']['structure_valid_points'], denominator),
        'structure_unknown': divide(result['totals']['structure_unknown_points'], denominator),
        'supported_junction': divide(result['totals']['supported_junction_points'], denominator)}
    supported = [row for row in rows if row['native_thin_support'] and row['native_thin_le2mm_coverage'] is not None]
    support = sum(row['native_thin_support'] for row in supported)
    result['native_thin_le2mm'] = {'cases_with_support': len(supported), 'native_skeleton_support': support,
        'support_weighted_roundtrip_coverage': divide(sum(row['native_thin_support'] * row['native_thin_le2mm_coverage']
                                                          for row in supported), support),
        'definition': 'descriptive node-weighted preprocessing retention, not independent samples or model performance'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--cache-root', required=True)
    args = parser.parse_args()
    project, cache = Path(args.project_root).resolve(), Path(args.cache_root).resolve()
    sys.path.insert(0, str(project / 'code'))
    from common import source_hashes
    manifest_path = cache / 'manifest.json'
    manifest = read(manifest_path)
    if not manifest.get('complete') or len(manifest['cases']) != 56 or len(manifest['splits']['train']) != 36:
        raise ValueError('This readout requires the completed56-case cache and36train patients')
    source = source_hashes()
    expected_preprocess = {name: value for name, value in source.items()
                           if name in ('common.py', 'prepare.py', 'geometry.py', 'intensity.py')}
    train_ids = set(manifest['splits']['train'])
    train, intensity, concerns = [], [], []
    for case in manifest['cases']:
        directory = Path(case['cache_dir']).resolve()
        if directory.parent != cache or directory.name != case['id']:
            raise ValueError('Cache directory escaped expected root')
        if read(directory / 'record.json') != case:
            raise ValueError('Case record does not match completed manifest:' + case['id'])
        audit = case['intensity_audit']
        spatial = audit.get('spatial_exterior_support') or {}
        ct_row = {'case_id': case['id'], 'cohort': case['cohort'],
                  'source_image_sha256': case['image_sha256'],
                  'offset_subtracted': audit['offset_subtract'], 'rule': audit['rule'],
                  'reviewed_exception': 'reviewed_override' in audit,
                  'requires_review': bool(audit['requires_review']),
                  'negative_air_fraction': audit.get('negative_air_fraction'),
                  'positive_air_fraction': audit.get('positive_air_fraction'),
                  'negative_air_peak': (audit.get('negative_air_peak') or {}).get('center'),
                  'positive_air_peak': (audit.get('positive_air_peak') or {}).get('center'),
                  'exterior_peak_fraction': spatial.get('exterior_peak_fraction'),
                  'dense_fraction_above_peak_plus250': spatial.get('dense_fraction_above_peak_plus250'),
                  'offset_manifest_sha256': case.get('offset_overrides_sha256'),
                  'used_GT_or_predictions_for_offset': False}
        intensity.append(ct_row)
        if ct_row['requires_review']:
            concerns.append({'case_id': case['id'], 'kind': 'unresolved_CT_intensity_convention'})
        # The following branch is the ONLY geometry/QC access path.
        if case['id'] not in train_ids:
            continue
        if case['preprocessing_source_sha256'] != expected_preprocess:
            raise ValueError('TRAIN cache preprocessing source mismatch:' + case['id'])
        path = directory / 'geometry.npz'
        actual_hash = digest(path)
        if actual_hash != case['cache_file_sha256']['geometry.npz']:
            raise ValueError('TRAIN geometry actual bytes mismatch:' + case['id'])
        with np.load(path, allow_pickle=False) as stream:
            geometry = {name: stream[name] for name in stream.files}
        radius = geometry['radius']
        structure = geometry['structure']
        n = len(geometry['points'])
        if len(radius) != n or len(structure) != n or geometry['points'].shape != (n, 3):
            raise ValueError('TRAIN geometry vector lengths disagree:' + case['id'])
        if scalar(geometry, 'skeleton_point_count') != n:
            raise ValueError('TRAIN skeleton metadata/vector count mismatch:' + case['id'])
        finite = np.isfinite(radius)
        positive = finite & (radius > 0)
        fine = finite & (radius <= 2.)
        valid_structure = (structure == 1) | (structure == 2)
        unknown = structure == 255
        junction = structure == 2
        qc = case.get('native_roundtrip_qc') or {}
        row = {'case_id': case['id'], 'cohort': case['cohort'], 'cache_spacing_mm': case['spacing_mm'],
               'shape_zyx': case['shape_zyx'], 'geometry_valid': bool(scalar(geometry, 'geometry_valid', False)),
               'mask_nonempty': bool(scalar(geometry, 'mask_nonempty', False)),
               'skeleton_empty_nonempty_mask': bool(scalar(geometry, 'skeleton_empty_nonempty_mask', False)),
               'resampled_skeleton_points': n, 'native_roundtrip_dice': qc.get('native_roundtrip_dice'),
               'native_centerline_coverage': qc.get('native_centerline_coverage'),
               'native_all_centerline_support': None,
               'native_all_centerline_support_note': 'not stored by frozen preprocessor; resampled node count is a different population',
               'native_thin_le2mm_coverage': qc.get('native_centerline_radius_le2mm_coverage'),
               'native_thin_support': qc.get('native_thin_centerline_support', 0),
               'native_foreground_voxels': qc.get('native_foreground_voxels'),
               'resampled_foreground_voxels': qc.get('resampled_foreground_voxels'),
               'fine_radius_le2mm_points': int(fine.sum()), 'fine_radius_le2mm_fraction': divide(int(fine.sum()), n),
               'nonfinite_radius_points': int((~finite).sum()),
               'nonpositive_radius_points': int((finite & (radius <= 0)).sum()),
               'radius_mm_quantiles_min_p10_p25_median_p75_p90_max': np.percentile(radius[finite], [0, 10, 25, 50, 75, 90, 100]).tolist() if finite.any() else None,
               'structure_valid_points': int(valid_structure.sum()), 'structure_valid_fraction': divide(int(valid_structure.sum()), n),
               'structure_unknown_points': int(unknown.sum()), 'structure_unknown_fraction': divide(int(unknown.sum()), n),
               'supported_junction_points': int(junction.sum()), 'supported_junction_node_fraction': divide(int(junction.sum()), n),
               'candidate_junctions': int(scalar(geometry, 'candidate_junction_count', 0)),
               'supported_junctions': int(scalar(geometry, 'supported_junction_count', 0)),
               'unknown_junctions': int(scalar(geometry, 'unknown_junction_count', 0)),
               'geometry_npz_actual_sha256': actual_hash, 'record_cache_file_sha256': case['cache_file_sha256'],
               'source_image_sha256': case['image_sha256'], 'source_label_sha256': case['label_sha256']}
        row['unknown_candidate_fraction'] = divide(row['unknown_junctions'], row['candidate_junctions'])
        row['scale_bin_point_counts_le2_2to4_4to8_gt8'] = np.bincount(np.searchsorted([2., 4., 8.], radius[finite], side='left'), minlength=4).tolist()
        if not row['geometry_valid']:
            concerns.append({'case_id': case['id'], 'kind': 'TRAIN_geometry_invalid_auxiliary_targets_masked'})
        if row['nonfinite_radius_points'] or row['nonpositive_radius_points']:
            concerns.append({'case_id': case['id'], 'kind': 'TRAIN_radius_nonfinite_or_nonpositive',
                             'nonfinite': row['nonfinite_radius_points'], 'nonpositive': row['nonpositive_radius_points']})
        if not np.isin(structure, [1, 2, 255]).all():
            concerns.append({'case_id': case['id'], 'kind': 'unexpected_skeleton_structure_code'})
        train.append(row)
    train.sort(key=lambda row: row['case_id'])
    intensity.sort(key=lambda row: row['case_id'])
    if len(train) != 36 or len({row['case_id'] for row in train}) != 36:
        raise ValueError('Expected exactly36unique TRAIN geometry rows')
    offsets = {str(value): sum(row['offset_subtracted'] == value for row in intensity)
               for value in sorted({row['offset_subtracted'] for row in intensity})}
    report = {'protocol': 'frozen_preprocessing_TRAIN36_QC_v1',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'host': socket.gethostname(),
              'scope': {'geometry_population': 'TRAIN36 only', 'intensity_population': 'ALL56 CT-only audit',
                        'heldout_masks_opened': False, 'heldout_geometry_opened': False,
                        'predictions_or_checkpoints_opened': False, 'models_run': False,
                        'adaptation_changed': False, 'full_native_centerline_support_recomputed': False},
              'provenance': {'cache_manifest_sha256': digest(manifest_path), 'split_sha256': manifest['split_sha256'],
                             'preprocessing_source_sha256': expected_preprocess, 'full_frozen_code_sha256': source,
                             'collector_sha256': digest(Path(__file__)),
                             'TRAIN_geometry_npz_actual_bytes_rehashed': True,
                             'other_cache_file_hashes': 'recorded provenance only; no image/mask arrays opened'},
              'definitions': {'radius': 'native-spacing label EDT sampled at resampled centreline nodes, mm',
                              'fractions': 'centreline-node weighted within each case; not vessel-volume or anatomical branch counts',
                              'junction': 'supported local skeleton graph junction neighbourhood, not expert anatomical bifurcation',
                              'unknown_structure': 'rejected-candidate neighbourhood255; excluded from structure auxiliary target',
                              'roundtrip': 'native GT->1mm NN label cache->native NN reconstruction, not segmentation accuracy',
                              'native_thin': 'native skeleton radius<=2mm; support comes from frozen per-case record'},
              'TRAIN36_summary': aggregate(train),
              'TRAIN_cohort_summaries': {cohort: aggregate([row for row in train if row['cohort'] == cohort])
                                         for cohort in sorted({row['cohort'] for row in train})},
              'ALL56_CT_intensity_summary': {'cases': len(intensity), 'offset_counts': offsets,
                    'reviewed_exception_cases': [row['case_id'] for row in intensity if row['reviewed_exception']],
                    'unresolved_cases': [row['case_id'] for row in intensity if row['requires_review']]},
              'TRAIN36_cases': train, 'ALL56_CT_intensity_cases': intensity, 'quality_concerns': concerns,
              'limitations': ['Native all-centreline denominator was not stored; it is not replaced by a resampled denominator.',
                             'Lee skeleton misses some grid-phase-dependent branches; valid geometry is not complete anatomy.',
                             'No held-out segmentation or geometry outcome was inspected.',
                             'This diagnostic does not select or change frozen adaptation, model, or hyperparameters.']}
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
