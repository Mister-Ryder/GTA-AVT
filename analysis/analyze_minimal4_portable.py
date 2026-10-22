"""Read four safely declared, file-hash-locked evaluations; never run a model.

Default plan uses only stdlib and opens no input. Declarations authenticate no
physical server, exporter, checkpoint selection time or private author freeze.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import statistics
import sys

SCHEMA='GTA-AVT-minimal4-published-evaluation-declaration-v1'
SELECTION_SCHEMA='GTA-AVT-minimal4-published-declared-selection-v1'
ROLE_BASIS='declared_for_this_reproduction_not_proved_server_identity'
SOURCE_LEVEL='published_safe_declaration_and_original_summary_case_metric_file_hashes_only_not_author_attestation'
MATH_SHA='31f355f735f6744589e464bbfb50cef276bc4db650f5edc70c5f6938d834f3c4'
STATISTICS_SHA='9e8b1ba7f05e02b27435a7a9596494d7b0938c0f1e2b469489eebc0fc105820b'
RULES_SHA='347d0850f9e54150f63c3cdee7a69216c22344eadc2b023a25e8fb92287ab565'
SOURCE_SHA='7c5b29ff069f5b33fe2212d721945c6b2e468bc5d15a9f87f6f12e1661f7fe96'
SPLIT_SHA='df9647da8b331260fb9f66b294eced01180c1eaf812de27b5148b6800a344a3b'
DATA_SHA='7f091053f2633f18f1f6e63fd1b206a4c28ee052cef38ff2a63dd512fe14d71c'
SLOTS={'main_C_concat':('main','C_concat','faithful_concat'),'main_A_soft':('main','A_soft','faithful_soft'),
    'ablations_A_soft_platform_bridge':('ablations','A_soft_platform_bridge','faithful_soft'),
    'ablations_A_soft_constant':('ablations','A_soft_constant','faithful_soft_constant')}
PACKAGES={'torch':'1.11.0+cu113','numpy':'1.22.4','scipy':'1.10.1','scikit-image':'0.21.0','SimpleITK':'2.3.1'}
RECIPE=dict(stage='formal',seed=20261022,width=32,patch_size=128,batch_size=2,max_steps=10000,lr=.0003,weight_decay=.0001,
    foreground_probability=.5,auxiliary_weight=.1,gradient_clip=12.,log_every=25,checkpoint_every=500,validate_every=5000,
    validation_case_limit=0,inference_patch_size=128,inference_batch_size=2,inference_overlap=.5,save_validation_predictions=False,cpu_threads=4)
BACKEND=dict(matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,deterministic_algorithms=True,cublas_workspace_config=':4096:8')
DECLARATION_KEYS={'schema','role_assignment','stage','case_ids','frozen_case_ids','frozen_source_sha256','canonical_data_sha256','split_sha256',
    'reference','models','all_models_declared_complete','no_model_reselection_declared'}
MODEL_KEYS={'role','job','method','training_recipe','selected_step','weights_sha256','provenance_sha256','original_checkpoint_sha256',
    'original_evaluation_protocol_sha256','original_runtime_canonical_sha256','evaluation_directory','summary_sha256','case_metrics_sha256'}
IDENTITY_KEYS=('role','job','method','selected_step','weights_sha256','provenance_sha256','original_checkpoint_sha256')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def require(ok,message):
    if not ok:raise ValueError(message)
def equal(value,expected,label):require(canonical(value)==canonical(expected),label+' differs')
def keys(value,expected,label):require(isinstance(value,dict) and set(value)==set(expected),'Exact safe '+label+' fields required')


def relative(path,root):
    require(isinstance(path,str) and path and '\\' not in path and ':' not in path
        and not PurePosixPath(path).is_absolute() and not PureWindowsPath(path).drive and str(PurePosixPath(path))==path,
        'Evaluation directories must be relative POSIX paths')
    return (Path(root)/path).resolve()


def load_statistics(repository):
    repository=Path(repository).resolve();math_path=repository/'analysis/analyze_key_ablations.py'
    equal(digest(math_path),MATH_SHA,'Released31f mathematics file')
    equal(digest(Path(__file__).resolve().parents[1]/'docs/minimal4_final_readout_rules_20261023.txt'),RULES_SHA,'Original final readout rules bytes')
    path=Path(__file__).with_name('minimal4_statistics.py');equal(digest(path),STATISTICS_SHA,'Five-function statistics file')
    loaded=sys.modules.get('analyze_key_ablations')
    require(loaded is None or Path(loaded.__file__).resolve()==math_path,'A different math checkout is already imported')
    sys.path.insert(0,str(repository/'analysis'))
    spec=importlib.util.spec_from_file_location('gta_avt_portable_minimal4_statistics',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def fixed_sources(repository):
    split_file=Path(repository)/'configs/split_seed20261022.json';equal(digest(split_file),SPLIT_SHA,'Released frozen split bytes')
    short=read(split_file);prefix={'D':'Dongyang','K':'KiTS','R':'Rider'}
    fixed={s:[prefix[x[0]]+'_'+x for x in short[s]] for s in ('train','val','test')}
    root=Path(repository)/'code';files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and (p.suffix=='.py' or ('historical' in p.relative_to(root).parts and p.suffix.lower() in ('.json','.txt','.md')))]
    source={p.relative_to(root).as_posix():digest(p) for p in sorted(files)}
    require(len(source)==24 and canonical(source)==SOURCE_SHA,'The released24 code files differ')
    return fixed,source


def validate_reference(value):
    keys(value,{'python_version','packages','torch_cuda','cudnn_version','backend_settings','selected_execution_device','autocast_enabled','original_runtime_canonical_sha256'},'reference declaration')
    p=value['python_version'];require(isinstance(p,list) and len(p)==3 and all(type(x) is int for x in p) and p[:2]==[3,8],'Declared Python3.8 version required')
    equal(value['packages'],PACKAGES,'Declared core software');equal(value['backend_settings'],BACKEND,'Declared reference precision')
    require(value['torch_cuda']=='11.3' and type(value['cudnn_version']) is int and value['cudnn_version']>0 and value['autocast_enabled'] is True,'Declared reference CUDA/autocast metadata required')
    require(valid_sha(value['original_runtime_canonical_sha256']),'Original runtime identity must be a SHA,never its private plaintext')
    d=value['selected_execution_device'];keys(d,{'type','index','name','capability','total_memory_bytes'},'device')
    require(d['type']=='cuda' and type(d['index']) is int and d['index']==0 and isinstance(d['name'],str) and d['name']
        and isinstance(d['capability'],list) and len(d['capability'])==2 and all(type(x) is int and x>=0 for x in d['capability'])
        and type(d['total_memory_bytes']) is int and d['total_memory_bytes']>0,'Declared CUDA0 device metadata required')


def identity(value):return dict({k:value[k] for k in IDENTITY_KEYS},training_recipe_sha256=canonical(value['training_recipe']))


def selection_identity(declaration):
    return dict(models={s:identity(declaration['models'][s]) for s in SLOTS},frozen_case_ids=declaration['frozen_case_ids'],
        frozen_source_sha256=declaration['frozen_source_sha256'],canonical_data_sha256=declaration['canonical_data_sha256'],
        split_sha256=SPLIT_SHA,reference=declaration['reference'],TEST_intervention='none',checkpoint_reselection_declared=False)


def validate_selection(path,approved_sha,declaration):
    require(path is not None and valid_sha(approved_sha),'TEST analysis needs a separate explicitly hash-locked local selection declaration')
    equal(digest(path),approved_sha,'Explicit selection declaration FILE SHA');record=read(path)
    keys(record,{'schema','state','declared_created_utc','timing_basis','selection','selection_identity_sha256'},'selection record')
    require(record['schema']==SELECTION_SCHEMA and record['state']=='declared_before_TEST'
        and record['timing_basis']=='reproducer_declaration_not_independent_timing_proof'
        and isinstance(record['declared_created_utc'],str) and record['declared_created_utc'],'Explicit nonattested selection timing declaration required')
    expected=selection_identity(declaration);equal(record['selection'],expected,'Four declared selected model identities')
    equal(record['selection_identity_sha256'],canonical(expected),'Canonical declared selection identity')
    return record,approved_sha


def validate_summary(summary,rows,ids,math4):
    require(type(summary.get('cases')) is int and summary['cases']==len(ids) and summary.get('statistical_unit')=='case'
        and set(summary.get('metrics',{}))==set(math4.METRICS),'Full fixed-case ten-metric summary required')
    for key in math4.METRICS:
        values=[rows[c][key] for c in ids];defined=[v for v in values if v is not None]
        expected=statistics.mean(defined) if defined and len(defined)==len(values) else None
        sd=statistics.stdev(defined) if len(defined)>1 else None;actual=summary['metrics'][key]
        require(type(actual.get('defined_case_count')) is int and actual['defined_case_count']==len(defined),'Summary support differs')
        for field,value in (('mean',expected),('std_cases',sd)):
            require(field in actual,'Summary missing explicit value/null')
            observed=actual[field]
            require(observed is None if value is None else type(observed) in (int,float) and math.isfinite(observed)
                and math.isclose(observed,value,rel_tol=1e-9,abs_tol=1e-12),'Summary mean/sampleSD differs')


def recheck_files(bound_file_sha256):
    """Rehash already admitted inputs; filenames stay internal, not in output."""
    for name,h in bound_file_sha256.items():equal(digest(name),h,'Input unchanged after admission/analysis')


def validate_shared_support(rows,ids,math4):
    """The same explicit-support gate as the unchanged four-slot pure math."""
    for cid in ids:
        truth=set();groups={g:set() for g in math4.GROUPS}
        for slot in SLOTS:
            value=rows[slot][cid].get('truth_skeleton_support')
            if value is not None:
                require(type(value) is int and value>=0,'GT skeleton support requires a nonnegative integer')
                truth.add(value)
            for group in math4.GROUPS:
                support,_,_=math4.group_record(rows[slot][cid],group)
                if support is not None:groups[group].add(support)
        require(len(truth)<=1 and not any(len(values)>1 for values in groups.values()),'Explicit GT skeleton/group support differs across four slots')


def load_verified_inputs(declaration_path,approved_sha,repository,selection_path=None,selection_sha=None):
    """Admit declared sources and raw rows without paired estimates/bootstrap."""
    require(valid_sha(approved_sha),'Explicit evaluation declaration FILE SHA required')
    path=Path(declaration_path).resolve();equal(digest(path),approved_sha,'Evaluation declaration FILE SHA');declaration=read(path)
    keys(declaration,DECLARATION_KEYS,'evaluation declaration')
    require(declaration['schema']==SCHEMA and declaration['role_assignment']==ROLE_BASIS and declaration['stage'] in ('val','test')
        and declaration['all_models_declared_complete'] is True and declaration['no_model_reselection_declared'] is True,'Fixed explicit declared-source scope required')
    fixed,source=fixed_sources(repository);equal(declaration['frozen_case_ids'],fixed,'Exact released36/8/12 case split')
    equal(declaration['frozen_source_sha256'],source,'Released code map');equal(declaration['canonical_data_sha256'],DATA_SHA,'Declared original canonical data')
    equal(declaration['split_sha256'],SPLIT_SHA,'Declared split bytes');ids=fixed[declaration['stage']];equal(declaration['case_ids'],ids,'Complete stage IDs/order')
    validate_reference(declaration['reference']);require(set(declaration['models'])==set(SLOTS),'All four independently named slots required')
    functions=load_statistics(repository);math4=functions.math4;rows={};numeric_slots={};paths=[];evidence={};bound={str(path):approved_sha}
    for slot,(role,job,method) in SLOTS.items():
        item=declaration['models'][slot];keys(item,MODEL_KEYS,'model '+slot)
        equal([item['role'],item['job'],item['method']],[role,job,method],'Fixed role declaration')
        equal(item['training_recipe'],dict(RECIPE,method=method),'Exact typed21-field training recipe')
        require(type(item['selected_step']) is int and item['selected_step'] in (5000,10000),'Declared formal best step must be5000 or10000')
        for key in ('weights_sha256','provenance_sha256','original_checkpoint_sha256','original_evaluation_protocol_sha256'):
            require(valid_sha(item[key]),'Actual declared identity SHA required: '+key)
        equal(item['original_runtime_canonical_sha256'],declaration['reference']['original_runtime_canonical_sha256'],'All four declared original reference runtimes')
        directory=relative(item['evaluation_directory'],path.parent);paths.append(directory)
        require(set(item['case_metrics_sha256'])==set(ids) and all(valid_sha(x) for x in item['case_metrics_sha256'].values()) and valid_sha(item['summary_sha256']),'Exact full stage summary/case FILE hashes required')
        summary=directory/'summary.json';equal(digest(summary),item['summary_sha256'],'Original summary FILE SHA');bound[str(summary)]=item['summary_sha256']
        rows[slot]={}
        for cid in ids:
            metric=directory/cid/'metrics.json';equal(digest(metric),item['case_metrics_sha256'][cid],'Original case metrics FILE SHA');bound[str(metric)]=item['case_metrics_sha256'][cid]
            value=read(metric)
            require(value.get('case_id')==cid and value.get('cohort')==cid.split('_',1)[0] and value.get('fixed_threshold')==.5
                and value.get('condition_mode')=='predicted' and value.get('evaluation_grid')=='original CT' and value.get('postprocessing')=='none','Original native fixed-threshold case protocol differs')
            for key in math4.METRICS:math4.number(value[key],key)
            require(all(value[key] is not None for key in ('dice','precision','recall','fp_ml','fn_ml')),'Whole-case overlap/FP/FN need defined values')
            for group in math4.GROUPS:math4.group_record(value,group)
            rows[slot][cid]=value
        validate_summary(read(summary),rows[slot],ids,math4)
        numeric_slots[slot]=dict(bundle_identity=dict(identity(item),slot=slot,role_identity_basis=ROLE_BASIS))
        evidence[slot]=dict(evaluation_directory=item['evaluation_directory'],summary_sha256=item['summary_sha256'],case_metrics_sha256=item['case_metrics_sha256'],
            declared_identity=identity(item),original_evaluation_protocol_sha256=item['original_evaluation_protocol_sha256'])
    require(len(set(paths))==4,'Four distinct evaluation directories required; method alone never identifies the two soft models')
    if declaration['stage']=='test':
        selection,selection_file_sha=validate_selection(selection_path,selection_sha,declaration);bound[str(Path(selection_path).resolve())]=selection_file_sha
    else:
        require(selection_path is None and selection_sha is None,'VAL never needs or consumes a future TEST selection record');selection=None;selection_file_sha=None
    numeric=dict(schema=functions.NUMERIC_INPUT_SCHEMA,stage=declaration['stage'],case_ids=ids,slots=numeric_slots,source_verification_level=SOURCE_LEVEL,
        marker_file_sha256=None,selection_payload_sha256=selection['selection_identity_sha256'] if selection else None,
        common_reference_runtime=dict(safe_reference_declaration=declaration['reference'],verification_basis='Declared metadata,not independent hardware attestation'),availability_only_VAL=None)
    validate_shared_support(rows,ids,math4)
    provenance=dict(source_index_sha256=approved_sha,source_evaluation_declaration_sha256=approved_sha,
        declared_selection_file_sha256=selection_file_sha,declared_selection_identity_sha256=selection['selection_identity_sha256'] if selection else None,
        author_private_marker_used=False,author_server_or_generic_export_verified=False,selection_time_independently_verified=False,
        declared_role_assignment=ROLE_BASIS,verified_original_files=evidence,
        verification_scope='Safe declaration and original metric/summary bytes are self-consistent. Model/export authenticity,physical host,data bytes and pre-TEST timing are not independently established.')
    recheck_files(bound)
    return dict(numeric_index=numeric,raw_rows=rows,declaration=declaration,functions=functions,
        result_provenance=provenance,bound_file_sha256=bound)


def analyze_files(declaration_path,approved_sha,repository,selection_path=None,selection_sha=None):
    loaded=load_verified_inputs(declaration_path,approved_sha,repository,selection_path,selection_sha)
    functions=loaded['functions'];result=functions.analyze_case_rows(loaded['numeric_index'],loaded['raw_rows'])
    result.update(loaded['result_provenance']);recheck_files(loaded['bound_file_sha256'])
    return result,functions


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('declaration','declaration-sha256','selection-record','selection-sha256','output'):parser.add_argument('--'+name)
    parser.add_argument('--repository',default=str(Path(__file__).resolve().parents[1]));parser.add_argument('--execute-analysis',action='store_true')
    args=parser.parse_args()
    if not args.execute_analysis:
        print(json.dumps(dict(state='plan_only',explicit_flag='--execute-analysis',fixed_slots=list(SLOTS),sources_opened=False,
            numpy_or_Torch_imported=False,no_training_inference_TEST_launch_or_selection=True)));return
    require(all((args.declaration,args.declaration_sha256,args.output)),'Declaration,explicit actual FILE SHA and fresh output required')
    output=Path(args.output).absolute();require(not output.exists() and output.resolve()==output,'Fresh ordinary output directory required')
    result,functions=analyze_files(args.declaration,args.declaration_sha256,args.repository,args.selection_record,args.selection_sha256)
    output.mkdir(parents=True,exist_ok=False)
    (output/'paired.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'figure_selection.json').write_text(json.dumps(result['figure_selection'],indent=2,allow_nan=False)+'\n',encoding='utf-8')
    prefix='来源层级：安全声明与原指标/summary文件哈希自洽；不独证作者服务器、export、数据字节或TEST前冻结时间。\n'
    (output/'readout.txt').write_text(prefix+functions.text(result),encoding='utf-8')
    with (output/'casewise.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['casewise'][0]));writer.writeheader();writer.writerows(result['casewise'])
    print(json.dumps(dict(state='complete_declared_source_analysis',stage=result['stage'],models=4,actual_inference_or_TEST_run=False)))


if __name__=='__main__':main()
