"""Temporary synthetic JSON only; no genuine four-model result or model use."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=True),encoding='utf-8')
def reject(fn):
    try:fn()
    except (ValueError,KeyError,TypeError,OSError):return
    raise AssertionError('Invalid declared source accepted')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def fixture(tool,repository,directory,stage):
    fixed,source=tool.fixed_sources(repository);ids=fixed[stage];h=lambda v:hashlib.sha256(('TEMP_SYNTHETIC_'+v).encode()).hexdigest()
    ref=dict(python_version=[3,8,10],packages=tool.PACKAGES,torch_cuda='11.3',cudnn_version=8302,backend_settings=tool.BACKEND,
        selected_execution_device=dict(type='cuda',index=0,name='TEMP_SYNTHETIC_GPU_NOT_QUERIED',capability=[8,9],total_memory_bytes=16*1024**3),
        autocast_enabled=True,original_runtime_canonical_sha256=h('runtime'))
    declaration=dict(schema=tool.SCHEMA,role_assignment=tool.ROLE_BASIS,stage=stage,case_ids=ids,frozen_case_ids=fixed,frozen_source_sha256=source,
        canonical_data_sha256=tool.DATA_SHA,split_sha256=tool.SPLIT_SHA,reference=ref,models={},all_models_declared_complete=True,no_model_reselection_declared=True)
    for slot,(role,job,method) in tool.SLOTS.items():
        rows=[];hashes={};folder=directory/'evaluations'/slot
        for i,cid in enumerate(ids):
            value=.5+i*.005;row=dict(case_id=cid,cohort=cid.split('_',1)[0],fixed_threshold=.5,condition_mode='predicted',evaluation_grid='original CT',postprocessing='none',
                dice=value,precision=value,recall=value,fp_ml=10.+i,fn_ml=15.+i,assd_mm=None if i==0 else 2.,hd95_mm=4.,
                centerline_recall=value,centerline_precision=value,cldice=value,truth_skeleton_support=100,
                geometry_groups={g:dict(support_nodes=0 if i==0 else 10,centerline_recall=None if i==0 else .4) for g in ('radius_le_2mm','radius_2_to_4mm','radius_4_to_8mm','radius_gt_8mm','nonjunction','junction_neighbourhood')})
            if slot=='main_A_soft':row.update(dice=value+(-.08 if i==0 else .03),fp_ml=row['fp_ml']+(3 if i==0 else -1),cldice=value+(-.04 if i==1 else 0))
            if slot=='ablations_A_soft_platform_bridge':row['dice']+=.02
            if i==2:row['geometry_groups'].pop('junction_neighbourhood')
            p=folder/cid/'metrics.json';write(p,row);hashes[cid]=tool.digest(p);rows.append(row)
        fields=('dice','precision','recall','fp_ml','fn_ml','assd_mm','hd95_mm','centerline_recall','centerline_precision','cldice')
        summary=dict(cases=len(ids),statistical_unit='case',metrics={})
        for key in fields:
            good=[r[key] for r in rows if r[key] is not None]
            summary['metrics'][key]=dict(mean=statistics.mean(good) if len(good)==len(ids) else None,defined_case_count=len(good),std_cases=statistics.stdev(good) if len(good)>1 else None)
        write(folder/'summary.json',summary)
        declaration['models'][slot]=dict(role=role,job=job,method=method,training_recipe=dict(tool.RECIPE,method=method),selected_step=10000,
            weights_sha256=h(slot+'weights'),provenance_sha256=h(slot+'provenance'),original_checkpoint_sha256=h(slot+'checkpoint'),
            original_evaluation_protocol_sha256=h(slot+'protocol'),original_runtime_canonical_sha256=ref['original_runtime_canonical_sha256'],
            evaluation_directory='evaluations/'+slot,summary_sha256=tool.digest(folder/'summary.json'),case_metrics_sha256=hashes)
    path=directory/'evaluation_declaration.json';write(path,declaration)
    return declaration,path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',required=True);p.add_argument('--output');args=p.parse_args()
    target=Path(__file__).with_name('analyze_minimal4_portable.py');tool=load(target,'portable_minimal4_reader_fixture')
    stat_path=target.with_name('minimal4_statistics.py');ast.parse(target.read_text(encoding='utf-8'),feature_version=(3,8));ast.parse(stat_path.read_text(encoding='utf-8'),feature_version=(3,8))
    assert 'numpy' not in sys.modules and 'torch' not in sys.modules
    functions=tool.load_statistics(args.repository);math4=functions.math4;methods=copy.deepcopy(math4.METHODS)
    raw=stat_path.read_bytes().splitlines(keepends=True);tree=ast.parse(stat_path.read_text(encoding='utf-8'))
    actual={n.name:hashlib.sha256(b''.join(raw[n.lineno-1:n.end_lineno])).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert actual==functions.ORIGINAL_FUNCTION_SHA256 and tool.digest(stat_path)==tool.STATISTICS_SHA
    checks=['Python3.8/stdlb reader; five exact original function-byte hashes and released31f mathematics pin']
    with tempfile.TemporaryDirectory(prefix='GTA_AVT_TEMP_PORTABLE4_SYNTHETIC_') as tmp:
        root=Path(tmp);declaration,path=fixture(tool,args.repository,root/'val','val')
        loaded=tool.load_verified_inputs(path,tool.digest(path),args.repository)
        assert set(loaded)=={'numeric_index','raw_rows','declaration','functions','result_provenance','bound_file_sha256'}
        assert set(loaded['raw_rows'])==set(tool.SLOTS) and len(loaded['bound_file_sha256'])==37
        tree=ast.parse(target.read_text(encoding='utf-8'))
        loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_verified_inputs')
        assert not any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Attribute) and n.func.attr in ('analyze_case_rows','paired'))
            or (isinstance(n.func,ast.Name) and n.func.id=='analyze_files')) for n in ast.walk(loader))
        result,_=tool.analyze_files(path,tool.digest(path),args.repository)
        assert all(result[k]==v for k,v in loaded['result_provenance'].items())
        tool.recheck_files(loaded['bound_file_sha256'])
        assert len(result['casewise'])==32 and set(result['comparisons'])=={'MAIN_A_minus_C','SECOND_A_minus_constant'}
        assert result['source_marker_FILE_sha256'] is None and result['declared_selection_file_sha256'] is None
        assert result['comparisons']['MAIN_A_minus_C']['scopes']['all']['metrics']['assd_mm']['paired_cases']==7
        assert result['comparisons']['MAIN_A_minus_C']['scopes']['all']['geometry']['radius_le_2mm']['no_GT_support_cases']==1
        assert result['comparisons']['MAIN_A_minus_C']['scopes']['all']['geometry']['junction_neighbourhood']['unrecorded_cases']==1
        assert abs(result['comparisons']['MAIN_A_minus_C']['scopes']['all']['metrics']['dice']['delta_a_minus_b']-1.625)<1e-10
        assert abs(result['comparisons']['SECOND_A_minus_constant']['scopes']['all']['metrics']['dice']['delta_a_minus_b']-2.)<1e-10
        checks.append('complete synthetic4VAL8/two pairs, legal distance null and geometry0/absent support; never private marker or author attestation')
        def wrong(change):
            value=copy.deepcopy(declaration);change(value);write(path,value);reject(lambda:tool.analyze_files(path,tool.digest(path),args.repository));write(path,declaration)
        wrong(lambda d:d['models'].pop('main_A_soft'))
        wrong(lambda d:d['models']['main_A_soft'].__setitem__('role','ablations'))
        wrong(lambda d:d['models']['main_A_soft']['training_recipe'].__setitem__('batch_size',2.))
        wrong(lambda d:d['models']['main_A_soft'].__setitem__('original_runtime_canonical_sha256','f'*64))
        wrong(lambda d:d['reference'].__setitem__('hostname','TEMP_PRIVATE_FIELD_MUST_NOT_PUBLISH'))
        wrong(lambda d:d['reference']['backend_settings'].__setitem__('matmul_tf32',True))
        wrong(lambda d:d['models']['main_A_soft']['case_metrics_sha256'].pop(d['case_ids'][-1]))
        wrong(lambda d:d.__setitem__('canonical_data_sha256','f'*64))
        reject(lambda:tool.analyze_files(path,'0'*64,args.repository))
        checks.append('wrong slot/role/typed recipe/runtime/unsafe private key/precision/partial cases/data or explicit approval SHA rejected')
        cid=declaration['case_ids'][1];metric=root/'val/evaluations/main_A_soft'/cid/'metrics.json';original=metric.read_bytes();metric.write_bytes(original+b'\n')
        reject(lambda:tool.recheck_files(loaded['bound_file_sha256']))
        reject(lambda:tool.analyze_files(path,tool.digest(path),args.repository));metric.write_bytes(original)
        value=json.loads(original);value['dice']=True;write(metric,value);bad=copy.deepcopy(declaration);bad['models']['main_A_soft']['case_metrics_sha256'][cid]=tool.digest(metric);write(path,bad)
        reject(lambda:tool.analyze_files(path,tool.digest(path),args.repository));metric.write_bytes(original);write(path,declaration)
        summary=root/'val/evaluations/main_A_soft/summary.json';original=summary.read_bytes();value=json.loads(original);value['metrics']['dice']['mean']=.99;write(summary,value)
        bad=copy.deepcopy(declaration);bad['models']['main_A_soft']['summary_sha256']=tool.digest(summary);write(path,bad)
        reject(lambda:tool.analyze_files(path,tool.digest(path),args.repository));summary.write_bytes(original);write(path,declaration)
        for slot in ('ablations_A_soft_platform_bridge','ablations_A_soft_constant'):
            metric=root/'val/evaluations'/slot/cid/'metrics.json';value=tool.read(metric);value['geometry_groups']['radius_le_2mm']['support_nodes']=20;write(metric,value)
            declaration['models'][slot]['case_metrics_sha256'][cid]=tool.digest(metric)
        write(path,declaration);reject(lambda:tool.analyze_files(path,tool.digest(path),args.repository))
        checks.append('actual file-byte drift and fully rebound bad probability/summary/GT support contradiction rejected')
        checks.append('shared metadata-only loader returns all gates/raw rows/provenance without paired/bootstrap calls; recheck detects drift')
        test,test_path=fixture(tool,args.repository,root/'test','test')
        reject(lambda:tool.analyze_files(test_path,tool.digest(test_path),args.repository))
        selection=tool.selection_identity(test);record=dict(schema=tool.SELECTION_SCHEMA,state='declared_before_TEST',declared_created_utc='TEMP_SYNTHETIC_NOT_ACTUAL_TIME',
            timing_basis='reproducer_declaration_not_independent_timing_proof',selection=selection,selection_identity_sha256=tool.canonical(selection))
        selection_path=root/'test/declared_selection.json';write(selection_path,record)
        output,_=tool.analyze_files(test_path,tool.digest(test_path),args.repository,selection_path,tool.digest(selection_path))
        assert len(output['casewise'])==48 and output['source_marker_FILE_sha256'] is None and output['declared_selection_file_sha256']==tool.digest(selection_path)
        assert output['selection_time_independently_verified'] is False and output['author_private_marker_used'] is False
        bad=copy.deepcopy(record);bad['selection']['models']['main_A_soft']['weights_sha256']='f'*64;bad['selection_identity_sha256']=tool.canonical(bad['selection']);write(selection_path,bad)
        reject(lambda:tool.analyze_files(test_path,tool.digest(test_path),args.repository,selection_path,tool.digest(selection_path)))
        checks.append('TEST12 existing-file analysis requires separate explicit declaration FILEsha and exact4 selected identities; timing nonproof and private marker null preserved')
        guard="import importlib.abc,sys,runpy\nclass G(importlib.abc.MetaPathFinder):\n def find_spec(self,fullname,path=None,target=None):\n  if fullname.split('.')[0] in ('numpy','torch','SimpleITK'):raise RuntimeError('heavy import forbidden')\nsys.meta_path.insert(0,G());sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
        response=subprocess.run([sys.executable,'-c',guard,str(target)],capture_output=True,text=True);assert response.returncode==0 and json.loads(response.stdout)['state']=='plan_only'
        response=subprocess.run([sys.executable,str(target),'--allow-partial'],capture_output=True,text=True);assert response.returncode!=0
    assert math4.METHODS==methods and 'torch' not in sys.modules and 'SimpleITK' not in sys.modules
    checks.append('no old METHODS mutation/partial path, default plan no files/heavy imports, no inference/training or TEST launch')
    report=dict(schema='GTA-AVT-minimal4-portable-paired-candidate-QA-v1',state='passed',checks=checks,
        temporary_synthetic_metadata_and_metrics_only=True,genuine_four_model_results_generated=False,real_protocols_or_weights_or_arrays_or_GPU_or_server_used=False,
        files_sha256={p.name:tool.digest(p) for p in (target,stat_path,Path(__file__))},pure_function_sha256=actual)
    if args.output:
        out=Path(args.output)
        with out.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
