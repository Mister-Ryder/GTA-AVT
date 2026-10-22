"""Package small read-only preprocessing evidence; no cache/model operations."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def pct(value):
    return 'undefined' if value is None else '%.2f%%' % (100. * value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--comparison')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.input)
    report = read(source)
    if report['protocol'] != 'frozen_preprocessing_TRAIN36_QC_v1':
        raise ValueError('Unexpected remote QC protocol')
    if len(report['TRAIN36_cases']) != 36 or len(report['ALL56_CT_intensity_cases']) != 56:
        raise ValueError('Unexpected population size')
    for field in ('heldout_masks_opened', 'heldout_geometry_opened', 'predictions_or_checkpoints_opened', 'models_run', 'adaptation_changed'):
        if report['scope'][field]:
            raise ValueError('Read-only TRAIN36/CT-only scope changed:' + field)
    agreement = None
    if args.comparison:
        comparison = read(args.comparison)
        agreement = {
            'TRAIN36_all_case_values_and_geometry_hashes_identical': report['TRAIN36_cases'] == comparison['TRAIN36_cases'],
            'ALL56_CT_only_values_identical': report['ALL56_CT_intensity_cases'] == comparison['ALL56_CT_intensity_cases'],
            'full_frozen_code_hashes_identical': report['provenance']['full_frozen_code_sha256'] == comparison['provenance']['full_frozen_code_sha256'],
            'preprocessing_source_hashes_identical': report['provenance']['preprocessing_source_sha256'] == comparison['provenance']['preprocessing_source_sha256'],
            'comparison_input_sha256': hashlib.sha256(Path(args.comparison).read_bytes()).hexdigest()}
        if not all(value for key, value in agreement.items() if key.endswith('identical')):
            raise ValueError('Cross-server data/code QC disagreement')
    rows = report['TRAIN36_cases']
    supported = [row for row in rows if row['native_thin_support'] and row['native_thin_le2mm_coverage'] is not None]
    lowest_thin = sorted(supported, key=lambda row: row['native_thin_le2mm_coverage'])[:5]
    lowest_dice = sorted(rows, key=lambda row: row['native_roundtrip_dice'])[:5]
    highest_unknown = sorted(rows, key=lambda row: row['structure_unknown_fraction'], reverse=True)[:5]
    cautions = {
        'lowest_supported_native_thin_roundtrip_coverage': [{key: row[key] for key in
             ('case_id', 'native_thin_le2mm_coverage', 'native_thin_support', 'structure_unknown_fraction')} for row in lowest_thin],
        'lowest_native_roundtrip_dice': [{key: row[key] for key in ('case_id', 'native_roundtrip_dice')} for row in lowest_dice],
        'highest_unknown_structure_fraction': [{key: row[key] for key in
             ('case_id', 'structure_unknown_fraction', 'structure_unknown_points', 'resampled_skeleton_points')} for row in highest_unknown],
        'interpretation': 'Descriptive post-freeze preprocessing limitations only; not segmentation results or a reason to tune the frozen adaptation.'}
    summary = {'protocol': report['protocol'], 'created_utc': report['created_utc'],
        'scope': report['scope'], 'definitions': report['definitions'], 'TRAIN36_summary': report['TRAIN36_summary'],
        'TRAIN_cohort_summaries': report['TRAIN_cohort_summaries'],
        'ALL56_CT_intensity_summary': report['ALL56_CT_intensity_summary'], 'cross_server_agreement': agreement,
        'preprocessing_cautions': cautions, 'invalid_data_concerns': report['quality_concerns'],
        'provenance': report['provenance'], 'collector_input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'limitations': report['limitations']}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'summary.json', summary)
    write(output / 'TRAIN36_casewise.json', {'scope': 'TRAIN36 only', 'cases': rows, 'provenance': report['provenance']})
    write(output / 'ALL56_CT_only_intensity.json', {'scope': 'ALL56 CT-only; no held-out GT geometry',
         'cases': report['ALL56_CT_intensity_cases'], 'summary': report['ALL56_CT_intensity_summary']})
    qc = report['TRAIN36_summary']
    casewise = qc['casewise']
    points = qc['point_weighted_descriptive_fractions']
    lines = [
        'Frozen1mm preprocessing QC: TRAIN36 geometry, ALL56 CT-only intensity',
        'This is preprocessing retention, not segmentation performance. No adaptation changed.',
        '',
        'Both-server exact data/code agreement: ' + ('passed' if agreement else 'not requested'),
        'Train geometries valid:%d/36; nonempty-mask/empty-skeleton cases:%d.' %
             (qc['geometry_valid_cases'], qc['empty_skeleton_nonempty_mask_cases']),
        'Native NN roundtrip Dice: case mean%s; median%s; minimum%s.' % tuple(pct(casewise['native_roundtrip_dice'][key]) for key in ('mean_cases', 'median_cases', 'min')),
        'Native centreline retention: case mean%s; minimum%s. Native all-centreline denominator was not logged.' %
             (pct(casewise['native_centerline_coverage']['mean_cases']), pct(casewise['native_centerline_coverage']['min'])),
        'Native radius<=2mm retention: case mean%s across%d supported cases; node-weighted%s over%d native skeleton points.' %
             (pct(casewise['native_thin_le2mm_coverage']['mean_cases']), qc['native_thin_le2mm']['cases_with_support'],
              pct(qc['native_thin_le2mm']['support_weighted_roundtrip_coverage']), qc['native_thin_le2mm']['native_skeleton_support']),
        'Resampled skeleton points:%d; fine-radius points:%d(%s).' %
             (qc['totals']['resampled_skeleton_points'], qc['totals']['fine_radius_le2mm_points'], pct(points['fine_radius_le2mm'])),
        'Structure supervision: valid%s; unknown%s; supported-junction neighbourhood%s of resampled skeleton nodes.' %
             (pct(points['structure_valid']), pct(points['structure_unknown']), pct(points['supported_junction'])),
        'Candidates:%d; arm-supported:%d; unknown:%d. Unknown nodes are excluded from structure auxiliary supervision.' %
             (qc['totals']['candidate_junctions'], qc['totals']['supported_junctions'], qc['totals']['unknown_junctions']),
        '', 'Largest observed preprocessing limitations (training patients only):']
    for row in lowest_thin[:2]:
        lines.append('- %s: native<=2mm retention%s, support%d; unknown structure%s.' %
             (row['case_id'], pct(row['native_thin_le2mm_coverage']), row['native_thin_support'], pct(row['structure_unknown_fraction'])))
    for row in lowest_dice[:2]:
        lines.append('- %s: native roundtrip Dice%s.' % (row['case_id'], pct(row['native_roundtrip_dice'])))
    intensity = report['ALL56_CT_intensity_summary']
    lines.extend(['', 'CT-only intensity: offset0 in19cases; subtract1024 in37cases. Three source-hash-locked reviewed exceptions: ' +
                  ','.join(intensity['reviewed_exception_cases']) + '. Unresolved:' + str(len(intensity['unresolved_cases'])) + '.',
                  'All case identities, actual train geometry hashes and frozen preprocessing/code hashes accompany the JSON evidence.',
                  'Do not substitute resampled skeleton-node count for the unlogged native all-centreline support.',
                  'No held-out mask/geometry array, prediction or checkpoint was opened. No model was run.'])
    (output / 'readout.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'TRAIN36_cases': len(rows),
                      'ALL56_CT_cases': len(report['ALL56_CT_intensity_cases']), 'cross_server_agreement': agreement}, indent=2))


if __name__ == '__main__':
    main()
