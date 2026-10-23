"""Standard-library recomputation of the released fixed-endpoint GDF VAL tables.

Run python recompute.py --check. This reads only the release's derived tables;
it does not load models, predictions, original images or study label arrays.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

FILES = {'readout.txt', 'metadata.json', 'casewise.csv',
         'geometry_casewise.csv', 'expected_summary.json',
         'recompute.py', 'manifest.json'}
METRICS = ('dice', 'cldice', 'centerline_recall', 'centerline_precision',
           'precision', 'recall', 'fp_ml', 'fn_ml', 'assd_mm', 'hd95_mm')
PROPORTIONS = set(METRICS[:6])
GROUPS = ('radius_le_2mm', 'radius_2_to_4mm', 'radius_4_to_8mm',
          'radius_gt_8mm', 'nonjunction', 'junction_neighbourhood')
TOL = 2e-11


def read(root, name):
    return json.loads((root/name).read_text(encoding='utf-8'))


def table(root, name):
    with (root/name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def optional(value):
    if value == '':
        return None
    value = float(value)
    assert math.isfinite(value)
    return value


def close(left, right, path='summary'):
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            close(left[key], right[key], path+'.'+key)
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            close(a, b, path+'['+str(i)+']')
    elif isinstance(left, bool) or left is None or isinstance(left, str):
        assert left == right, path
    else:
        assert math.isfinite(left) and math.isfinite(right), path
        assert abs(left-right) <= TOL*max(1., abs(left), abs(right)), path


def distribution(values):
    defined = [x for x in values if x is not None]
    return dict(cases=len(values), defined_cases=len(defined),
        mean=statistics.mean(defined) if defined and len(defined) == len(values) else None,
        defined_case_mean=statistics.mean(defined) if defined else None,
        sample_SD=statistics.stdev(defined) if len(defined)>1 else None)


def paired(after, before):
    delta = [a-b if a is not None and b is not None else None for a,b in zip(after,before)]
    defined = [x for x in delta if x is not None]
    return dict(after=distribution(after), before=distribution(before),
        difference=distribution(delta), positive=sum(x>1e-10 for x in defined),
        ties=sum(abs(x)<=1e-10 for x in defined), negative=sum(x< -1e-10 for x in defined))


def check_files(root):
    assert {p.name for p in root.iterdir()} == FILES, 'Unexpected or missing file'
    manifest = read(root, 'manifest.json')
    assert set(manifest['files']) == FILES-{'manifest.json'}
    patterns = (r'(?<![A-Za-z])[A-Za-z]:[\\/]',
        r'/(?:root|home|dev/shm|mnt|tmp|Users)/', r'\\\\[A-Za-z0-9]',
        r'\b(?:root|ubuntu|admin)\x40[A-Za-z0-9]',
        r'\b(?:ssh|sftp)\s+-p\s+\d+', r'\bconnect[.][A-Za-z0-9.-]+',
        r'\b[A-Za-z0-9.-]+[.]seetacloud[.]com\b',
        r'\b(?:\d{1,3}[.]){3}\d{1,3}\b', r'\bGPU-[0-9a-fA-F-]{8,}')
    for name in FILES:
        path = root/name
        assert path.is_file() and not path.is_symlink()
        data = path.read_bytes()
        assert not any(re.search(p, data.decode('utf-8')) for p in patterns), name
        if name != 'manifest.json':
            assert len(data) == manifest['files'][name]['bytes'], name
            assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['sha256'], name


def recompute(root):
    meta = read(root, 'metadata.json')
    ids = meta['case_ids']
    rows = table(root, 'casewise.csv')
    groups = table(root, 'geometry_casewise.csv')
    assert len(rows) == 16 and len(groups) == 96
    indexed = {(r['arm'],r['case_id']):r for r in rows}
    assert len(indexed) == 16 and set(indexed) == {(a,c) for a in ('Kp','Gp') for c in ids}
    geometry_rows = {(r['arm'],r['case_id'],r['group']):r for r in groups}
    assert len(geometry_rows) == 96
    assert set(geometry_rows) == {(a,c,g) for a in ('Kp','Gp') for c in ids for g in GROUPS}
    metric_rows = {}
    for key, row in indexed.items():
        values = {m:optional(row[m]) for m in METRICS}
        assert all(v is None or (0<=v<=1 if m in PROPORTIONS else v>=0) for m,v in values.items())
        metric_rows[key] = values
        assert len(row['original_metric_sha256']) == 64
        assert row['original_metric_sha256'] == meta['identity']['metric_files_sha256'][key[0]][key[1]]
        assert int(row['fp_voxels'])>=0 and int(row['fn_voxels'])>=0
        # Both error volumes must imply the same native voxel volume. This is
        # a consistency check, not a new independent native-grid evaluation.
        fp,fn = int(row['fp_voxels']),int(row['fn_voxels'])
        if fp and fn:
            close(values['fp_ml']/fp, values['fn_ml']/fn, 'native-voxel-volume')
    metrics = {}
    for m in METRICS:
        scale = 100 if m in PROPORTIONS else 1
        metrics[m] = paired([None if metric_rows['Gp',c][m] is None else metric_rows['Gp',c][m]*scale for c in ids],
                            [None if metric_rows['Kp',c][m] is None else metric_rows['Kp',c][m]*scale for c in ids])
    cases = [dict(case_id=c, source=indexed['Kp',c]['source'], delta={m:
        None if metric_rows['Gp',c][m] is None or metric_rows['Kp',c][m] is None else
        metric_rows['Gp',c][m]*(100 if m in PROPORTIONS else 1)-metric_rows['Kp',c][m]*(100 if m in PROPORTIONS else 1) for m in METRICS}) for c in ids]
    tails = {}
    for arm in ('Kp','Gp'):
        ordered = sorted(ids, key=lambda c:(metric_rows[arm,c]['dice'],ids.index(c)))[:2]
        tails[arm] = dict(case_ids=ordered, mean_Dice_pp=statistics.mean(metric_rows[arm,c]['dice']*100 for c in ordered))
    tail_delta = tails['Gp']['mean_Dice_pp']-tails['Kp']['mean_Dice_pp']
    sd_reduction = metrics['dice']['before']['sample_SD']-metrics['dice']['after']['sample_SD']
    worst = min(c['delta']['dice'] for c in cases)
    geometry = {}
    for group in GROUPS:
        values = {}
        supports = []
        for arm in ('Kp','Gp'):
            values[arm] = []
            for c in ids:
                r = geometry_rows[arm,c,group]
                value = optional(r['centerline_recall'])
                support = int(r['support_nodes'])
                assert support>=0 and (value is None if support==0 else value is not None and 0<=value<=1)
                values[arm].append(None if value is None else 100*value)
        for c in ids:
            assert geometry_rows['Kp',c,group]['support_nodes'] == geometry_rows['Gp',c,group]['support_nodes']
            supports.append(dict(case_id=c,support_nodes=int(geometry_rows['Kp',c,group]['support_nodes'])))
        geometry[group] = dict(paired=paired(values['Gp'],values['Kp']),case_support=supports)
    gate = meta['gate_thresholds']
    dd,cd = metrics['dice']['difference']['mean'],metrics['cldice']['difference']['mean']
    parts = dict(mean_Dice_nonnegative=dd is not None and dd>=0,
        mean_clDice_defined_and_nonnegative=cd is not None and cd>=0,
        worst_two_Dice_nonnegative=tail_delta>=0,
        no_case_Dice_decline_gt_3pp=worst>=gate['minimum_single_case_Dice_delta_pp'],
        substantive_signal=(dd is not None and dd>=gate['substantive_Dice_delta_pp'] or
            cd is not None and cd>=gate['substantive_clDice_delta_pp'] or
            tail_delta>=gate['substantive_worst_two_delta_pp'] or sd_reduction>=gate['substantive_Dice_SD_reduction_pp']))
    return dict(metrics=metrics,cases=cases,geometry=geometry,worst_two=tails,
        worst_two_mean_Dice_difference_pp=tail_delta,Dice_SD_reduction_pp=sd_reduction,
        worst_paired_Dice_difference_pp=worst,
        FP_FN_totals={arm:{k:sum(int(indexed[arm,c][k]) for c in ids) for k in ('fp_voxels','fn_voxels')} for arm in ('Kp','Gp')},
        continue_gate=dict(parts=parts,passed=all(parts.values())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    check_files(root)
    result=recompute(root)
    if args.check:
        close(result,read(root,'expected_summary.json'))
        print(json.dumps(dict(status='passed',stdlib_only=True,files=len(FILES),
            source_data_or_private_collection_required=False,continue_gate=result['continue_gate'],
            relative_tolerance=TOL),indent=2))
    else:
        print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
