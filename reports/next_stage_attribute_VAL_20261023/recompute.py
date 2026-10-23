"""Recompute the released U/K derived tables using only Python's standard library.

Run python recompute.py --check. No model, image, checkpoint, remote connection,
or private collection is needed. Proportions in CSV are fractions; summary
proportion differences are percentage points. Missing values remain null.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ARMS = ('PRIMARY_U', 'AMENDED_K')
METRICS = ('dice','cldice','centerline_recall','centerline_precision','precision','recall',
           'fp_ml','fn_ml','assd_mm','hd95_mm')
RATIOS = set(METRICS[:6])
GROUPS = ('radius_le_2mm','radius_2_to_4mm','radius_4_to_8mm','radius_gt_8mm',
          'nonjunction','junction_neighbourhood')
FILES = {'readout.txt','metadata.json','casewise.csv','geometry_casewise.csv',
         'expected_summary.json','recompute.py','manifest.json'}
TOL = 2e-11


def read(root, name):
    return json.loads((root/name).read_text(encoding='utf-8'))


def table(root, name):
    with (root/name).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def optional(value):
    if value == '':
        return None
    value = float(value)
    assert math.isfinite(value)
    return value


def equal(a,b,path='summary'):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for key in a:
            equal(a[key],b[key],path+'.'+key)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):
            equal(x,y,path+'.'+str(i))
    elif a is None or isinstance(a,(bool,str)):
        assert a==b,path
    else:
        assert b is not None and math.isfinite(a) and math.isfinite(b),path
        assert abs(a-b)<=TOL*max(1,abs(a),abs(b)),(path,a,b)


def distribution(values):
    good=[v for v in values if v is not None]
    return dict(cases=len(values),defined_cases=len(good),
        mean=st.mean(good) if good and len(good)==len(values) else None,
        defined_case_mean=st.mean(good) if good else None,
        sample_SD=st.stdev(good) if len(good)>1 else None)


def paired(after,before):
    delta=[a-b if a is not None and b is not None else None for a,b in zip(after,before)]
    valid=[v for v in delta if v is not None]
    return dict(after=distribution(after),before=distribution(before),difference=distribution(delta),
        positive=sum(v>1e-10 for v in valid),ties=sum(abs(v)<=1e-10 for v in valid),
        negative=sum(v< -1e-10 for v in valid))


def check_files(root):
    assert {p.name for p in root.iterdir()}==FILES,'Only the seven declared release files are allowed'
    manifest=read(root,'manifest.json')
    assert set(manifest['files'])==FILES-{'manifest.json'}
    for name,row in manifest['files'].items():
        p=root/name
        assert p.is_file() and not p.is_symlink()
        raw=p.read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],name


def recompute(root):
    meta=read(root,'metadata.json');ids=meta['case_ids']
    assert len(ids)==len(set(ids))==8
    rows=table(root,'casewise.csv');geometry_rows=table(root,'geometry_casewise.csv')
    index={(r['arm'],r['case_id']):r for r in rows}
    geom={(r['arm'],r['case_id'],r['group']):r for r in geometry_rows}
    assert len(rows)==len(index)==16 and set(index)=={(a,c) for a in ARMS for c in ids}
    assert len(geometry_rows)==len(geom)==96 and set(geom)=={(a,c,g) for a in ARMS for c in ids for g in GROUPS}
    vals={}
    for key,row in index.items():
        vals[key]={m:optional(row[m]) for m in METRICS}
        assert vals[key]['dice'] is not None,'All original eight Dice measurements are mandatory'
        assert all(v is None or (0<=v<=1 if m in RATIOS else v>=0) for m,v in vals[key].items())
        assert row['original_metric_sha256']==meta['identity']['metric_files_sha256'][key[0]][key[1]]
        fp,fn=int(row['fp_voxels']),int(row['fn_voxels']);assert min(fp,fn)>=0
        if fp and fn:
            equal(vals[key]['fp_ml']/fp,vals[key]['fn_ml']/fn,'native voxel-volume consistency')
    before,after=ARMS
    metrics={m:paired([None if vals[after,c][m] is None else vals[after,c][m]*(100 if m in RATIOS else 1) for c in ids],
                       [None if vals[before,c][m] is None else vals[before,c][m]*(100 if m in RATIOS else 1) for c in ids]) for m in METRICS}
    cases=[dict(case_id=c,source=index[before,c]['source'],
        before={m:None if vals[before,c][m] is None else vals[before,c][m]*(100 if m in RATIOS else 1) for m in METRICS},
        after={m:None if vals[after,c][m] is None else vals[after,c][m]*(100 if m in RATIOS else 1) for m in METRICS},
        delta={m:None if vals[after,c][m] is None or vals[before,c][m] is None else
               (vals[after,c][m]-vals[before,c][m])*(100 if m in RATIOS else 1) for m in METRICS}) for c in ids]
    tails={}
    for arm in ARMS:
        chosen=sorted(ids,key=lambda c:(vals[arm,c]['dice'],ids.index(c)))[:2]
        tails[arm]=dict(case_ids=chosen,mean_Dice_pp=st.mean(vals[arm,c]['dice']*100 for c in chosen))
    tail=tails[after]['mean_Dice_pp']-tails[before]['mean_Dice_pp']
    sd=metrics['dice']['before']['sample_SD']-metrics['dice']['after']['sample_SD']
    worst=min(r['delta']['dice'] for r in cases)
    groups={}
    for name in GROUPS:
        numbers={a:[] for a in ARMS};support=[]
        for c in ids:
            item={a:geom[a,c,name] for a in ARMS}
            n={a:None if item[a]['support_nodes']=='' else int(item[a]['support_nodes']) for a in ARMS}
            for a in ARMS:
                v=optional(item[a]['centerline_recall']);reason=item[a]['missing_reason'] or None
                assert n[a] is None or n[a]>=0
                assert v is None or 0<=v<=1
                if n[a] is None:
                    assert v is None and reason=='binary_metrics_early_empty_return'
                elif n[a]==0:
                    assert v is None and reason is None
                else:
                    assert v is not None and reason is None
                numbers[a].append(None if v is None else 100*v)
            if all(n[a] is not None for a in ARMS):
                assert n[before]==n[after]
            support.append(dict(case_id=c,before_support_nodes=n[before],after_support_nodes=n[after],
                before_missing_reason=item[before]['missing_reason'] or None,
                after_missing_reason=item[after]['missing_reason'] or None))
        groups[name]=dict(paired=paired(numbers[after],numbers[before]),case_support=support)
    dd,cd=metrics['dice']['difference']['mean'],metrics['cldice']['difference']['mean'];g=meta['gate_thresholds']
    parts=dict(mean_Dice_nonnegative=dd is not None and dd>=0,
        mean_clDice_defined_and_nonnegative=cd is not None and cd>=0,
        worst_two_Dice_nonnegative=tail>=0,no_case_Dice_decline_gt_3pp=worst>=g['minimum_single_case_Dice_delta_pp'],
        substantive_signal=dd is not None and dd>=g['substantive_Dice_delta_pp'] or
            cd is not None and cd>=g['substantive_clDice_delta_pp'] or tail>=g['substantive_worst_two_delta_pp'] or
            sd>=g['substantive_Dice_SD_reduction_pp'])
    return dict(metrics=metrics,cases=cases,geometry=groups,worst_two=tails,
        worst_two_mean_Dice_difference_pp=tail,Dice_SD_reduction_pp=sd,worst_paired_Dice_difference_pp=worst,
        leave_one_case_out_Dice_delta_pp={c:st.mean(r['delta']['dice'] for r in cases if r['case_id']!=c) for c in ids},
        FP_FN_totals={a:{k:sum(int(index[a,c][k]) for c in ids) for k in ('fp_voxels','fn_voxels')} for a in ARMS},
        continue_gate=dict(parts=parts,passed=all(parts.values())))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;check_files(root);value=recompute(root)
    if args.check:
        equal(value,read(root,'expected_summary.json'))
        print(json.dumps(dict(status='passed',stdlib_only=True,private_collection_required=False,
            actual_model_or_array_execution=False,continue_gate=value['continue_gate']),indent=2))
    else:
        print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
