"""Recompute released derived metrics using stdlib, without model/data access.

This verifies numerical consistency, not remote execution or checkpoint replay.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics as st
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('released_arithmetic',HERE/'arithmetic.py')
arithmetic=importlib.util.module_from_spec(spec);spec.loader.exec_module(arithmetic)


def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(a,b,label='value'):
    if a is None or b is None:
        assert a is b,label
    elif type(a) is bool or type(b) is bool:
        assert a is b,label
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(label,a,b)
    elif isinstance(a,dict) and isinstance(b,dict):
        assert set(a)==set(b),(label,'keys')
        for key in a:equal(a[key],b[key],label+'.'+key)
    elif isinstance(a,list) and isinstance(b,list):
        assert len(a)==len(b),(label,'length')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,label+'.'+str(i))
    else:assert a==b,(label,a,b)


def table(name,schemas):
    rows=[]
    with (HERE/name).open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        assert set(reader.fieldnames)==set(schemas[name])
        for row in reader:
            converted={}
            for key,value in row.items():
                kind=schemas[name][key]
                if value=='':converted[key]=None
                elif kind=='bool':
                    assert value in ('True','False');converted[key]=value=='True'
                elif kind=='int':converted[key]=int(value)
                elif kind=='float':converted[key]=float(value)
                else:
                    assert kind=='str';converted[key]=value
            rows.append(converted)
    return rows


def unflatten(row):
    out={}
    for key,value in row.items():
        if key=='platform':continue
        where=out;parts=key.split('.')
        for item in parts[:-1]:where=where.setdefault(item,{})
        where[parts[-1]]=value
    return out


def val_check(platform,summary,rows,geometry,counts,definitions):
    ids=definitions['VAL_case_ids'];metrics=definitions['metrics']
    chosen=[r for r in rows if r['platform']==platform]
    assert len(chosen)==8*10
    lookup={(r['case_id'],r['metric']):r for r in chosen};assert len(lookup)==80
    cases=[]
    for cid in ids:
        case=dict(case_id=cid,after={},before={},delta={})
        for metric in metrics:
            row=lookup[(cid,metric)]
            delta=None if row['Dlearn'] is None or row['Clearn'] is None else row['Dlearn']-row['Clearn']
            equal(delta,row['delta'],'CSV_delta')
            case['after'][metric]=row['Dlearn'];case['before'][metric]=row['Clearn'];case['delta'][metric]=delta
        cases.append(case)
    equal(cases,summary['cases'],'VAL_cases')
    computed={metric:arithmetic.paired([c['after'][metric] for c in cases],[c['before'][metric] for c in cases]) for metric in metrics}
    equal(computed,summary['metrics'],'VAL_metrics')
    tails={}
    for label in ('after','before'):
        ordered=sorted(cases,key=lambda c:(c[label]['dice'],ids.index(c['case_id'])))[:2]
        tails[label]=dict(case_ids=[c['case_id'] for c in ordered],mean_Dice_pp=st.mean(c[label]['dice'] for c in ordered))
    equal(tails,summary['worst_two'],'own_worst_two')
    tail=tails['after']['mean_Dice_pp']-tails['before']['mean_Dice_pp']
    sd=computed['dice']['before']['sample_SD']-computed['dice']['after']['sample_SD']
    worst=min(c['delta']['dice'] for c in cases);dd=computed['dice']['difference']['mean'];cd=computed['cldice']['difference']['mean']
    equal(tail,summary['worst_two_mean_Dice_difference_pp']);equal(sd,summary['Dice_SD_reduction_pp']);equal(worst,summary['worst_paired_Dice_difference_pp'])
    equal({cid:st.mean(c['delta']['dice'] for c in cases if c['case_id']!=cid) for cid in ids},summary['leave_one_case_out_Dice_delta_pp'],'LOO')
    parts=dict(mean_Dice_nonnegative=dd>=0,mean_clDice_defined_and_nonnegative=cd is not None and cd>=0,
        worst_two_Dice_nonnegative=tail>=0,no_case_Dice_decline_gt_3pp=worst>=-3,
        substantive_signal=(dd>=.3 or cd is not None and cd>=.5 or tail>=.5 or sd>=1))
    equal(parts,summary['continue_gate']['parts'],'gate_parts');assert all(parts.values())==summary['continue_gate']['passed']
    for group in definitions['geometry_groups']:
        selected=[{k:v for k,v in r.items() if k not in ('platform','group')} for r in geometry if r['platform']==platform and r['group']==group]
        assert len(selected)==8 and [r['case_id'] for r in selected]==ids
        expected=summary['geometry'][group];equal(selected,expected['case_support'],'geometry_support')
        equal(arithmetic.paired([r['after_recall_pp'] for r in selected],[r['before_recall_pp'] for r in selected]),expected['paired'],'geometry_paired')
        for row in selected:
            if row['support_nodes']==0:assert row['after_recall_pp'] is row['before_recall_pp'] is None
    for metric in ('fp_voxels','fn_voxels'):
        selected=[{k:v for k,v in r.items() if k not in ('platform','metric')} for r in counts if r['platform']==platform and r['metric']==metric]
        expected=summary['FP_FN_voxel_counts'][metric];equal(selected,expected['cases'],'FPFN_cases')
        for row in selected:assert row['after']-row['before']==row['difference']
        equal(arithmetic.paired([r['after'] for r in selected],[r['before'] for r in selected]),expected['paired'],'FPFN_statistics')
        for key,column in (('after_total','after'),('before_total','before'),('difference_total','difference')):assert expected[key]==sum(r[column] for r in selected)
    return dict(Dice_delta_pp=dd,clDice_delta_pp=cd,continuation_gate_passed=all(parts.values()),worst_case_Dice_delta_pp=worst)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');parser.parse_args()
    manifest=read('file_manifest.json')
    actual={p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual==set(manifest['files'])|{'file_manifest.json'}
    for name,value in manifest['files'].items():
        path=HERE/name;assert path.stat().st_size==value['bytes'] and sha(path)==value['sha256'],name
    definitions=read('definitions.json');schemas=read('table_schemas.json')
    tables={name:table(name,schemas) for name in schemas}
    summary=read('VAL_summary.json');mechanism=read('mechanism_summary.json');semantics=read('semantics_summary.json')
    results={}
    for platform in definitions['platform_order']:
        results[platform]=val_check(platform,summary[platform],tables['VAL_casewise.csv'],tables['VAL_geometry.csv'],tables['VAL_FP_FN_counts.csv'],definitions)
        patches=[unflatten(row) for row in tables['TRAIN_intervention_patchwise.csv'] if row['platform']==platform]
        assert len(patches)==96
        computed=arithmetic.summarize_metrics(patches,definitions['TRAIN_case_ids'])
        equal(computed,mechanism[platform]['metric_readout'],'mechanism_all_cases_support_and_null')
        for row in patches:
            for condition in ('normal','repaired'):
                for view in ('full','real_CT'):
                    item=row[condition][view]
                    # The original patch metric has epsilon in numerator/denominator.
                    expected=(2.*item['TP']+1e-6)/(2.*item['TP']+item['FP']+item['FN']+1e-6)
                    equal(expected,item['dice'],'patch_Dice_formula')
                    assert item['TP']+item['FN']==item['truth_foreground'] and item['TP']+item['FP']==item['predicted_foreground']
            assert row['condition_changed']==(row['normal_c']!=row['repaired_c'])
        results[platform]['TRAIN_correct_minus_repaired_Dice_pp']=computed['primary_case_equal']['normal_minus_repaired_Dice_pp']
        results[platform]['mechanism_supported_cases']=computed['mechanism_supported_case_count']
    pred=tables['TRAIN_attribute_predictions.csv'];assert len(pred)==288
    for stage in ('INITIAL_SHARED','PRIMARY_Dlearn','SECONDARY_Dlearn'):
        selected=[r for r in pred if r['stage']==stage];assert len(selected)==96
        assert sum(r['known_nonempty'] for r in selected)==74 and sum(r['attribute_valid'] for r in selected)==91
        for row in selected:
            if not row['attribute_valid']:assert row['GT_q'] is None and all(row['GT_r'+str(i)] is None for i in range(1,5))
            elif row['known_nonempty']:assert row['GT_q']==0 and row['GT_r1']==row['GT_fine']
            else:assert row['GT_q']==1 and all(row['GT_r'+str(i)] is None for i in range(1,5))
            assert abs(sum(row['p'+str(i)] for i in range(5))-1.)<2e-7
        expected=semantics['INITIAL_SHARED' if stage=='INITIAL_SHARED' else stage.split('_')[0]]
        computed=arithmetic.semantic_summary(selected,dict(case_ids=definitions['TRAIN_case_ids']),definitions['support_gate'])
        for key in set(computed)&set(expected):equal(computed[key],expected[key],'semantics.'+stage+'.'+key)
        if stage!='INITIAL_SHARED':
            platform=stage.split('_')[0];patch={(r['case_id'],r['slot']):r for r in tables['TRAIN_intervention_patchwise.csv'] if r['platform']==platform}
            for row in selected:equal(row['raw_fine'],patch[(row['case_id'],row['slot'])]['normal_c'],'normal_consumed_prediction')
    training=read('training_summary.json')
    for endpoint,value in training.items():
        assert value['steps']==value['successful_updates']==2000 and value['gdf_weight']==0
        assert value['AMP_attempts']==value['successful_updates']+value['AMP_retries']
        equal(value['training_seconds']/2000,value['mean_seconds_per_successful_update'],'cost_per_step')
        for window,record in value['windows'].items():
            selected=[r for r in tables['training_windows.csv'] if r['endpoint']==endpoint and r['window']==window]
            assert selected and all(r['first_step']==record['first_step'] and r['last_step']==record['last_step'] and r['count']==record['successful_updates'] for r in selected)
            assert record['successful_updates']==record['last_step']-record['first_step']+1
            for r in selected:assert r['minimum']<=r['mean']<=r['maximum'] and r['minimum']<=r['median']<=r['maximum'] and r['population_SD']>=0
        quarters=[v for k,v in value['windows'].items() if k.startswith('quarter')]
        assert sum(q['successful_updates'] for q in quarters)==2000
        assert sum(q['AMP_retries'] for q in quarters)==value['AMP_retries']
        assert sum(q['preclip_gradient_norm_above_clip_count'] for q in quarters)==value['preclip_gradient_norm_above_clip_count']
    assert all(Path(name).suffix in ('.py','.csv','.json','.txt') and not Path(name).is_absolute() and '..' not in Path(name).parts for name in actual)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(json.dumps(dict(status='passed',scope='derived_statistics_and_safety_only_no_model_or_remote_execution',
        platforms=results,platforms_pooled=False,seed_replicates=False,
        source_file_hashes_verified=len(manifest['files']),VAL_cases_each=8,TRAIN_patches_each=96,
        initial_and_final_prediction_rows=288,training_windows='fixed first/last100 and quarters; copied moments, not replayed logs',
        training_update_seconds_total=sum(v['training_seconds'] for v in training.values()),
        model_updates_in_this_process=0,Torch_imported=False,GPU_used=False)))


if __name__=='__main__':main()
