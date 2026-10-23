"""Recompute released diagnostic summaries using Python's standard library only.

Run `python recompute.py --check` to verify the allowlist, hashes, safety patterns,
recorded count/Dice consistency and the released expected numerical summaries.
This validates derived tables; it does not recreate model outputs or patient data.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

FILES = {
    'readout.txt', 'metadata.json', 'readout_casewise.csv',
    'oracle_patchwise.csv', 'gdf_casewise.csv',
    'gdf_reported_summaries.json', 'expected_summary.json',
    'recompute.py', 'manifest.json',
}
MODES = ('constant', 'correct', 'permuted')
MSE_KEYS = ('model_MSE', 'mean_constant_MSE', 'model_MAE',
            'median_constant_MAE', 'arithmetic_point2_MSE', 'arithmetic_point2_MAE')
TOL = 2e-12
DICE_TOL = 2e-7


def load(root, name):
    return json.loads((root / name).read_text(encoding='utf-8'))


def table(root, name):
    with (root / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    return None if value == '' else float(value)


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def close(left, right, path='summary'):
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            close(left[key], right[key], path + '.' + key)
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            close(a, b, path + '[' + str(i) + ']')
    elif isinstance(left, bool) or left is None or isinstance(left, str):
        assert left == right, path
    else:
        assert isinstance(right, (int, float)) and math.isfinite(left) and math.isfinite(right), path
        assert abs(left - right) <= TOL * max(1., abs(left), abs(right)), path


def check_package(root):
    assert {p.name for p in root.iterdir()} == FILES, 'Unexpected or missing package file'
    assert all((root / name).is_file() and not (root / name).is_symlink() for name in FILES)
    manifest = load(root, 'manifest.json')
    assert set(manifest['files']) == FILES - {'manifest.json'}
    forbidden = (
        r'[A-Za-z]:[\\/]', r'/(?:root|home|dev/shm|mnt|tmp|Users)/',
        r'\\\\[A-Za-z0-9]', r'\b(?:root|ubuntu|admin)\x40[A-Za-z0-9]',
        r'\b(?:ssh|sftp)\s+-p\s+\d+', r'\b(?:ssh|sftp)://[A-Za-z0-9]',
        r'\bconnect[.][A-Za-z0-9.-]+', r'\b[A-Za-z0-9.-]+[.]seetacloud[.]com\b',
        r'\b(?:\d{1,3}[.]){3}\d{1,3}\b', r'\bGPU-[0-9a-fA-F-]{8,}',
    )
    for name in FILES:
        data = (root / name).read_bytes()
        text = data.decode('utf-8')
        assert not any(re.search(pattern, text) for pattern in forbidden), 'Forbidden content: ' + name
        if name != 'manifest.json':
            expected = manifest['files'][name]
            assert len(data) == expected['bytes'], 'Byte count: ' + name
            assert hashlib.sha256(data).hexdigest() == expected['sha256'], 'SHA256: ' + name


def readout_summary(root, metadata):
    rows = table(root, 'readout_casewise.csv')
    result = {}
    for role, n in (('fit', 24), ('check', 12)):
        rr = [r for r in rows if r['role'] == role]
        assert len(rr) == n * 2
        assert len({r['case_id'] for r in rr}) == n
        macro = {kind: {key: mean(float(r[key]) for r in rr if r['condition'] == kind)
                        for key in MSE_KEYS} for kind in ('raw', 'conditional')}
        sources = {source: {kind: {key: mean(float(r[key]) for r in rr
                    if r['condition'] == kind and r['source'] == source) for key in MSE_KEYS}
                    for kind in ('raw', 'conditional')} for source in ('Dongyang', 'KiTS', 'Rider')}
        conditional = [r for r in rr if r['condition'] == 'conditional']
        wins = sum(float(r['model_MSE']) < float(r['mean_constant_MSE']) for r in conditional)
        support = sum(int(r['sampled_fine_positive_windows']) >= metadata['readout_gate']['minimum_sampled_fine_cells_per_case']
                      for r in conditional)
        result[role] = dict(cases=n, macro=macro, sources=sources, conditional_MSE_better_cases=wins,
            fine_supported_cases=support,
            raw_prediction_mean=mean(float(r['prediction_mean']) for r in rr if r['condition'] == 'raw'),
            nonempty_target_mean=mean(float(r['target_mean']) for r in rr if r['condition'] == 'raw'))
    check = result['check']; gate = metadata['readout_gate']
    reduction = {kind: 1 - check['macro'][kind]['model_MSE'] / check['macro'][kind]['mean_constant_MSE']
                 for kind in ('raw', 'conditional')}
    clauses = dict(all12_have_nonempty_support=all(int(r['sampled_nonempty_windows']) > 0 for r in rows if r['role'] == 'check'),
        sampled_fine_support=check['fine_supported_cases'] >= gate['minimum_fine_supported_cases'],
        raw_MSE_reduction=reduction['raw'] >= gate['minimum_relative_raw_MSE_reduction'],
        conditional_MSE_reduction=reduction['conditional'] >= gate['minimum_relative_conditional_MSE_reduction'],
        conditional_case_wins=check['conditional_MSE_better_cases'] >= gate['minimum_conditional_MSE_better_cases'],
        all_three_sources_not_worse=all(s[kind]['model_MSE'] <= s[kind]['mean_constant_MSE']
            for s in check['sources'].values() for kind in ('raw', 'conditional')),
        optimizer_resolved=metadata['readout_optimization']['resolved'])
    result['gate'] = dict(passed=all(clauses.values()), clauses=clauses, relative_MSE_reduction=reduction)
    return result


def oracle_summary(root, metadata):
    rows = table(root, 'oracle_patchwise.csv')
    assert len(rows) == 96
    assert len({(r['case_id'], int(r['slot'])) for r in rows}) == 96
    cases = []
    for r in rows:
        for mode in MODES:
            for region in ('full128', 'real_CT'):
                p = mode + '_' + region + '_'
                tp, fp, fn = (int(r[p + k]) for k in ('TP', 'FP', 'FN'))
                assert int(r[p + 'truth']) == tp + fn
                assert int(r[p + 'prediction']) == tp + fp
                stored = number(r[p + 'Dice'])
                if tp + fn:
                    # Stored science score used native-precision sigmoid then
                    # FP32 scalar arithmetic. Python float64 is a tolerance check,
                    # not a claim to reproduce native sigmoid or be bit-exact.
                    ratio = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
                    assert stored is not None and abs(stored - ratio) <= DICE_TOL
                else:
                    assert stored is None
    for cid in metadata['oracle_check_case_ids']:
        rr = [r for r in rows if r['case_id'] == cid]
        assert sorted(int(r['slot']) for r in rr) == list(range(8))
        pp = [r for r in rr if int(r['slot']) < 4]
        assert all(r['true_empty'] == 'False' for r in pp)
        scores = {m: mean(float(r[m + '_full128_Dice']) for r in pp) for m in MODES}
        cases.append(dict(case_id=cid, source=rr[0]['source'], Dice_ratio=scores,
            correct_minus_permuted_Dice_pp=100 * (scores['correct'] - scores['permuted']),
            correct_minus_constant_Dice_pp=100 * (scores['correct'] - scores['constant']),
            informative=any(int(r['eligible_windows']) >= 2 and r['nonconstant_fine'] == 'True' for r in pp),
            logit_response=mean(float(r['mean_abs_logit_correct_vs_permuted']) for r in pp)))
    primary = dict(complete_cases=len(cases), informative_cases=sum(c['informative'] for c in cases),
        mean_Dice_ratio={m: mean(c['Dice_ratio'][m] for c in cases) for m in MODES},
        case_Dice_sample_SD_ratio={m: statistics.stdev(c['Dice_ratio'][m] for c in cases) for m in MODES},
        worst2_mean_Dice_ratio={m: mean(sorted(c['Dice_ratio'][m] for c in cases)[:2]) for m in MODES},
        correct_minus_permuted_mean_Dice_pp=mean(c['correct_minus_permuted_Dice_pp'] for c in cases),
        correct_minus_constant_mean_Dice_pp=mean(c['correct_minus_constant_Dice_pp'] for c in cases),
        positive_cases=sum(c['correct_minus_permuted_Dice_pp'] > 0 for c in cases),
        mean_abs_logit_response=mean(c['logit_response'] for c in cases))
    sources = {s: {key: mean(c[key] for c in cases if c['source'] == s)
                   for key in ('correct_minus_permuted_Dice_pp', 'correct_minus_constant_Dice_pp')}
               for s in ('Dongyang', 'KiTS', 'Rider')}
    empty = [r for r in rows if r['true_empty'] == 'True']
    nonempty_uniform = [r for r in rows if int(r['slot']) >= 4 and r['true_empty'] == 'False']
    for r in empty:
        assert len({r[m + '_prediction_sha256'] for m in MODES}) == 1
    uniform = dict(trueempty_patches=len(empty), nonempty_patches=len(nonempty_uniform),
        trueempty_FP_full128={m: sum(int(r[m + '_full128_FP']) for r in empty) for m in MODES},
        trueempty_FP_real_CT={m: sum(int(r[m + '_real_CT_FP']) for r in empty) for m in MODES},
        nonempty_patch_mean_Dice_ratio={m: mean(float(r[m + '_full128_Dice']) for r in nonempty_uniform) for m in MODES})
    gate = metadata['oracle_gate']
    clauses = dict(complete_cases=primary['complete_cases'] >= gate['minimum_complete_cases'],
        informative_cases=primary['informative_cases'] >= gate['minimum_informative_cases'],
        correct_minus_permuted_Dice=primary['correct_minus_permuted_mean_Dice_pp'] >= gate['minimum_correct_minus_permuted_mean_Dice_pp'],
        positive_cases=primary['positive_cases'] >= gate['minimum_strictly_positive_case_Dice_differences'],
        mean_absolute_logit_response=primary['mean_abs_logit_response'] >= gate['minimum_mean_absolute_logit_response'],
        correct_minus_constant_Dice=primary['correct_minus_constant_mean_Dice_pp'] >= gate['minimum_correct_minus_constant_mean_Dice_pp'],
        required_invariants=all(metadata['oracle_recorded_invariants'].values()) and all(r['invariants_passed'] == 'True' for r in rows))
    return dict(primary=primary, cases=cases, sources=sources, CT_uniform=uniform,
                gate=dict(passed=all(clauses.values()), clauses=clauses))


def gdf_summary(root):
    rows = table(root, 'gdf_casewise.csv')
    assert len(rows) == 36 * 5 and len({(r['case_id'], r['radius_bin']) for r in rows}) == len(rows)
    summaries = load(root, 'gdf_reported_summaries.json')
    result = {}
    for source in ('all_TRAIN', 'Dongyang', 'KiTS', 'Rider'):
        rr = rows if source == 'all_TRAIN' else [r for r in rows if r['source'] == source]
        bins = {}
        for label in ('all_nodes', 'radius_le_2mm', 'radius_2_to_4mm', 'radius_4_to_8mm', 'radius_gt_8mm'):
            all_cases = [r for r in rr if r['radius_bin'] == label]
            supported = [r for r in all_cases if int(r['node_support']) > 0]
            for r in all_cases:
                if not int(r['node_support']):
                    assert r['target_mean'] == r['target_mean_square'] == r['target_sd'] == ''
                    assert r['Pearson_null_reason'] == 'no_supported_nodes'
            target = summaries[source]['all_nodes'] if label == 'all_nodes' else summaries[source]['bins'][label]
            group = dict(cases_in_scope=len(all_cases), cases_with_support=len(supported), cases_without_support=len(all_cases)-len(supported))
            for weighting in ('case_equal', 'node_weighted'):
                weights = [1. if weighting == 'case_equal' else int(r['node_support']) for r in supported]
                total = sum(weights)
                moments = {k: sum(w*float(r[k]) for w, r in zip(weights, supported))/total if total else None
                           for k in ('radius_mean_mm', 'target_mean', 'target_mean_square')}
                moments['target_sd'] = math.sqrt(max(0., moments['target_mean_square']-moments['target_mean']**2)) if total else None
                moments['node_support'] = sum(int(r['node_support']) for r in supported)
                for key, value in moments.items():
                    close(value, target[weighting][key], 'GDF.'+source+'.'+label+'.'+weighting+'.'+key)
                group[weighting] = moments
            for key in ('cases_in_scope', 'cases_with_support', 'cases_without_support'):
                assert group[key] == target[key]
            bins[label] = group
        result[source] = bins
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    check_package(root)
    metadata = load(root, 'metadata.json')
    result = dict(readout=readout_summary(root, metadata), oracle=oracle_summary(root, metadata), gdf=gdf_summary(root))
    if args.check:
        close(result, load(root, 'expected_summary.json'))
        print(json.dumps(dict(status='passed',files=len(FILES),patient_data_required=False,
            private_collection_required=False,stdlib_only=True,oracle_gate=result['oracle']['gate'],
            readout_gate=result['readout']['gate'],absolute_summary_tolerance=TOL,
            count_to_stored_Dice_tolerance=DICE_TOL), indent=2))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
