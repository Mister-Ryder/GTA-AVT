"""Export public-safe objective GTA-AVT v1 results from locally collected runs.

This script does not contact servers, load model weights, run inference, choose
new settings or interpret performance. Missing/failed results remain null.
Only fixed numeric fields and restricted identifiers enter public outputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import tempfile

ARMS = ('C0', 'A_only', 'AR', 'AR_GDF')
STAGES = ('attribute', 'warmup') + ARMS
CONTRASTS = (('C0','A_only'),('A_only','AR'),('C0','AR'),('AR','AR_GDF'))
SCORES = ('dice','precision','recall','cldice','centerline_precision','centerline_recall',
          'fine_centerline_recall')
COUNTS = ('tp_voxels','fp_voxels','fn_voxels','predicted_skeleton_voxels',
          'target_skeleton_voxels','fine_centerline_supported_voxels')
VOLUMES = ('predicted_volume_ml','target_volume_ml','fp_volume_ml','fn_volume_ml')
METRICS = SCORES + COUNTS + VOLUMES
LOWER_BETTER = {'fp_voxels','fn_voxels','fp_volume_ml','fn_volume_ml'}
PLAN_KEYS = ('seed','device','num_workers','pretrain_steps','warmup_steps','adapt_steps',
             'alpha_ramp_steps','adapt_lr','lambda_gdf')
COMMON_KEYS = ('patch_size','batch_size','lr','weight_decay','samples_per_epoch',
               'val_samples','num_workers','amp','max_grad_norm','checkpoint_every',
               'validate_every','deterministic','dice_weight','bce_weight',
               'deep_supervision_weight','availability_weight','radius_weight')
ATTR_KEYS = ('availability_bce','constant_availability_bce','emission_precision','emission_recall',
             'radius_mse','constant_radius_mse','case_centered_radius_ratio',
             'predicted_supported_radius_variance','emitted_target_sites',
             'predicted_supported_sites','predicted_support_fraction')
LOSS_KEYS = ('loss','availability_bce','radius_mse','dice_loss','segmentation_bce','gdf_mse','auxiliary_bce')
SAFE_ID = re.compile(r'^[A-Za-z0-9_.-]{1,160}$')
SAFE_TEXT = re.compile(r'^[A-Za-z0-9 _().+:-]{1,160}$')
SHA256 = re.compile(r'^[a-f0-9]{64}$')
FROZEN_SOURCE_COMMIT = 'e17e91465022fd38f8ca1fb82a73b3c504f050d0'
FROZEN_SOURCE_MANIFEST_SHA256 = 'c709eafcb43d8a1fce02d3678aeb9f0785d0c32d5e56d03672733e0cf211b722'
FROZEN_DATA_MANIFEST_SHA256 = 'cdf74b47c3cde32dda039ea1f2072e57c6b7362829380b1f97805106dd8c33fa'
EVALUATION_ERRATUM_CODE_SHA256 = '93900d714dc3927aa0f416509ba7309d353cc0a86f2761e6b27e6e4d03cbf9ea'
FROZEN_VAL_CASE_IDS = {'Dongyang_D2','Dongyang_D5','KiTS_K13','KiTS_K15','KiTS_K18',
                       'KiTS_K8','Rider_R10','Rider_R9'}
READINESS_TESTS = {'all_cases_have_radius_support_and_variation','availability_beats_train_prior',
                   'radius_beats_train_mean','case_centered_radius_beats_constant',
                   'consumer_has_nonconstant_supported_radius'}


def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        return None
    return value


def identifier(value):
    return value if isinstance(value,str) and SAFE_ID.fullmatch(value) and value not in ('.','..') else None


def safe_text(value):
    return value if isinstance(value,str) and SAFE_TEXT.fullmatch(value) else None


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:return json.load(stream)


def scalar_config(obj,keys):
    result={}
    for key in keys:
        value=obj.get(key)
        if isinstance(value,bool):result[key]=value
        elif number(value) is not None:result[key]=value
        elif isinstance(value,str) and key=='device':result[key]=safe_text(value)
    return result


def summary(values,*,lower_better=False):
    values=[float(value) for value in values if number(value) is not None]
    if not values:
        return dict(mean=None,std_population=None,minimum=None,maximum=None,worst20_mean=None,defined_cases=0)
    tail=sorted(values,reverse=lower_better)[:max(1,math.ceil(.2*len(values)))]
    return dict(mean=statistics.fmean(values),std_population=statistics.pstdev(values),
                minimum=min(values),maximum=max(values),worst20_mean=statistics.fmean(tail),defined_cases=len(values))


def _attribute(report):
    if not isinstance(report,dict):return None
    rows=[]
    for original in report.get('per_case',[]):
        case=identifier(original.get('case_id'))
        if case is None:continue
        rows.append(dict(case_id=case,**{key:number(original.get(key)) for key in ATTR_KEYS}))
    sums={}
    for key in ATTR_KEYS:
        original=report.get('summary',{}).get(key,{})
        sums[key]={k:number(original.get(k)) for k in ('mean','std_population','minimum','defined_cases')}
    ready=report.get('readiness',{})
    tests={identifier(key):value for key,value in ready.get('tests',{}).items()
           if identifier(key) is not None and isinstance(value,bool)}
    return dict(scope='sampled VAL patches; case-equal attribute diagnostics',per_case=rows,summary=sums,
                readiness_passed=ready.get('passed') if isinstance(ready.get('passed'),bool) else None,
                readiness_tests=tests)


def _validation(report):
    if not isinstance(report,dict):return None
    return dict(losses={key:number(report.get('losses',{}).get(key)) for key in LOSS_KEYS
                        if key in report.get('losses',{})},
                selection_score=number(report.get('selection_score')),
                attributes=_attribute(report.get('attributes')))


def _platform(root,seed,metadata):
    data={}
    for candidate in ('platform.json','runtime.json','environment.json'):
        path=root/candidate
        if path.is_file():
            try:
                value=read_json(path)
                if isinstance(value,dict):data.update(value)
            except (ValueError,OSError):pass
    if isinstance(metadata,dict):
        candidates=(str(seed),root.parent.name,root.name)
        for key in candidates:
            value=metadata.get(key)
            if isinstance(value,dict):data.update(value)
    result={key:safe_text(data.get(key)) for key in
            ('platform_scope','device_name','torch','cuda','driver')}
    result['status']='recorded' if any(value is not None for value in result.values()) else 'not_recorded'
    return result


def _find_root(directory):
    directory=Path(directory)
    if not directory.is_dir():return None,'missing_input_directory'
    if (directory/'pipeline.json').is_file():return directory,None
    found=list(directory.rglob('pipeline.json'))
    if len(found)!=1:return None,'missing_pipeline' if not found else 'ambiguous_pipeline_roots'
    return found[0].parent,None


def load_run(directory,metadata=None):
    """Read one seed. Problems are fixed reason codes, never raw exceptions."""
    root,problem=_find_root(directory)
    record=dict(seed=None,status='invalid',pipeline_status=None,problems=[],stages={},native={},
                attribute_history=[],provenance_sha256={})
    if problem:
        record['problems'].append(problem);return record
    for name in ('plan.json','pipeline.json'):
        try:
            value=read_json(root/name)
            if not isinstance(value,dict):raise ValueError()
            record[name[:-5]]=value
            record['provenance_sha256'][name]=digest(root/name)
        except (ValueError,OSError):
            record['problems'].append('invalid_or_missing_'+name[:-5]);return record
    plan,pipeline=record.pop('plan'),record.pop('pipeline')
    seed=plan.get('seed')
    if type(seed) is not int:
        record['problems'].append('invalid_seed');return record
    record['seed']=seed;record['recipe']=scalar_config(plan,PLAN_KEYS)
    record['recipe']['common']=scalar_config(plan.get('common',{}),COMMON_KEYS)
    record['platform']=_platform(root,seed,metadata)
    actual_plan_hash=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    record['plan_sha256']=actual_plan_hash
    if pipeline.get('plan_sha256')!=actual_plan_hash:record['problems'].append('pipeline_plan_hash_mismatch')
    status=pipeline.get('status')
    record['pipeline_status']=status if status in ('complete','running','failed') else 'unknown'
    known_stages=STAGES+tuple('native_VAL_'+arm for arm in ARMS)
    record['failure']={'error_type':identifier(pipeline.get('error_type')),
                       'stage':pipeline.get('current') if pipeline.get('current') in known_stages else None}
    completed=pipeline.get('completed',{})
    if not isinstance(completed,dict):completed={};record['problems'].append('invalid_completed_map')
    expected_steps=dict(attribute=plan.get('pretrain_steps',20000),warmup=plan.get('warmup_steps',10000),
                        **{arm:plan.get('adapt_steps',5000) for arm in ARMS})
    for stage in STAGES:
        path=root/stage/'summary.json'
        result=completed.get(stage)
        if path.is_file():
            try:result=read_json(path);record['provenance_sha256'][stage+'/summary.json']=digest(path)
            except (ValueError,OSError):record['problems'].append('invalid_'+stage+'_summary');result=None
        stage_record=dict(status='missing',expected_successful_updates=number(expected_steps[stage]),
                          successful_updates=None,last_validation=None)
        if isinstance(result,dict):
            updates=number(result.get('successful_updates'))
            stage_record.update(successful_updates=updates,
                status='complete' if stage in completed and updates==expected_steps[stage] else 'incomplete',
                last_validation=_validation(result.get('validation')))
            lineage=result.get('lineage',{})
            source=lineage.get('segmentation_init')
            stage_record['warmup_successful_updates']=number(source.get('warmup_successful_updates')) if isinstance(source,dict) else None
            if isinstance(source,dict):
                value=source.get('warmup_state_sha256')
                stage_record['warmup_state_sha256']=value if isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) else None
            predictor=lineage.get('predictor_init')
            stage_record['predictor_successful_updates']=number(predictor.get('successful_updates')) if isinstance(predictor,dict) else None
            if isinstance(predictor,dict):
                value=predictor.get('predictor_state_sha256')
                stage_record['predictor_state_sha256']=value if isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) else None
        cfg_path=root/stage/'config.json'
        if cfg_path.is_file():
            try:
                cfg=read_json(cfg_path)
                stage_record['recipe']=scalar_config(cfg,COMMON_KEYS+('max_steps','arm','stage','lambda_gdf','alpha_ramp_steps','use_gdf'))
                for key in ('arm','stage'):
                    if identifier(cfg.get(key)):stage_record['recipe'][key]=cfg[key]
                if cfg.get('seed')!=seed or cfg.get('max_steps')!=expected_steps[stage]:
                    stage_record['status']='metadata_mismatch';record['problems'].append(stage+'_config_identity_mismatch')
                if stage!='attribute':
                    expected_arm='AR' if stage=='AR_GDF' else 'C0' if stage=='warmup' else stage
                    expected_stage='warmup' if stage=='warmup' else 'adapt'
                    if cfg.get('arm')!=expected_arm or cfg.get('stage')!=expected_stage:
                        stage_record['status']='metadata_mismatch';record['problems'].append(stage+'_arm_stage_mismatch')
            except (ValueError,OSError):
                stage_record['status']='metadata_mismatch';record['problems'].append(stage+'_config_invalid')
        elif stage_record['status']=='complete':
            stage_record['status']='metadata_missing';record['problems'].append(stage+'_config_missing')
        record['stages'][stage]=stage_record
    for key,stages in (('predictor_state_sha256',('warmup',)+ARMS),('warmup_state_sha256',ARMS)):
        identities={record['stages'][stage].get(key) for stage in stages}
        identities.discard(None)
        if len(identities)>1:record['problems'].append('matched_'+key+'_mismatch')
    recipes=[{key:value for key,value in record['stages'][arm].get('recipe',{}).items()
              if key not in ('arm','lambda_gdf')} for arm in ARMS
              if record['stages'][arm].get('recipe')]
    if recipes and any(recipe!=recipes[0] for recipe in recipes[1:]):
        record['problems'].append('matched_arm_recipe_mismatch')
    history=root/'attribute'/'history.jsonl'
    if history.is_file():
        try:
            observations={};replayed=0
            with history.open(encoding='utf-8') as stream:
                for line in stream:
                    item=json.loads(line);step=item.get('successful_update')
                    if type(step) is not int or step<=0:raise ValueError()
                    if step in observations:replayed+=1
                    observations[step]=item
            # A resume may append repeated steps after the last saved cursor.
            # Keep the latest observation at each step, and disclose replay.
            for step,item in sorted(observations.items()):
                    if item.get('validation') is not None:
                        record['attribute_history'].append(dict(successful_update=step,
                            validation=_validation(item['validation'])))
            record['attribute_history_replayed_updates']=replayed
            record['provenance_sha256']['attribute/history.jsonl']=digest(history)
        except (ValueError,OSError):record['problems'].append('attribute_history_invalid')
    ready=[item for item in record['attribute_history']
           if item['validation'] and item['validation'].get('attributes') and
           item['validation']['attributes']['readiness_passed'] is True and
           number(item['validation']['selection_score']) is not None]
    record['selected_attribute_ready']=min(ready,key=lambda x:x['validation']['selection_score']) if ready else None
    for arm in ARMS:
        key='native_VAL_'+arm;path=root/'metrics'/key/'metrics.json'
        value=completed.get(key)
        if path.is_file():
            try:value=read_json(path);record['provenance_sha256']['metrics/'+key+'/metrics.json']=digest(path)
            except (ValueError,OSError):record['problems'].append(key+'_invalid');value=None
        rows={};native_status='missing'
        if isinstance(value,dict) and isinstance(value.get('per_case'),list):
            native_status='recorded'
            for original in value['per_case']:
                case=identifier(original.get('case_id'))
                if case is None or case in rows:
                    record['problems'].append(key+'_invalid_or_duplicate_case');native_status='invalid';continue
                row=dict(case_id=case,source=identifier(original.get('source')),
                         **{metric:number(original.get(metric)) for metric in METRICS})
                if any(row[metric] is not None and not 0<=row[metric]<=1 for metric in SCORES):
                    native_status='invalid';record['problems'].append(key+'_out_of_range_score')
                    for metric in SCORES:
                        if row[metric] is not None and not 0<=row[metric]<=1:row[metric]=None
                for metric in COUNTS+VOLUMES:
                    if row[metric] is not None and (row[metric]<0 or (metric in COUNTS and row[metric]!=int(row[metric]))):
                        row[metric]=None;native_status='invalid';record['problems'].append(key+'_invalid_count_or_volume')
                if (row['fine_centerline_supported_voxels'] is None or row['target_skeleton_voxels'] is None
                    or row['cldice'] is None or (row['fine_centerline_supported_voxels']>0 and
                    row['fine_centerline_recall'] is None)):
                    native_status='invalid';record['problems'].append(key+'_missing_centerline_metric_support')
                rows[case]=row
        record['native'][arm]=dict(status=native_status,pipeline_completed=key in completed,per_case=rows)
    record['status']='complete' if (not record['problems'] and record['pipeline_status']=='complete'
        and all(stage['status']=='complete' for stage in record['stages'].values())) else 'partial'
    return record


def _write_csv(path,rows,keys):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def _erratum_require(condition,reason):
    """An explicitly supplied invalid sidecar never becomes usable measurements."""
    if not condition:raise ValueError('evaluation_erratum_'+reason)


def _erratum_json(root,relative):
    # Sidecar paths are fixed by this reader, never supplied from JSON content.
    path=Path(root)/relative
    _erratum_require(path.is_file(),'missing_artifact')
    try:value=read_json(path)
    except (OSError,ValueError):raise ValueError('evaluation_erratum_invalid_artifact') from None
    _erratum_require(isinstance(value,dict),'invalid_artifact')
    return value


def _erratum_native(value):
    rows={}
    _erratum_require(isinstance(value.get('per_case'),list),'missing_metrics')
    for original in value['per_case']:
        _erratum_require(isinstance(original,dict),'invalid_metric_case')
        case=identifier(original.get('case_id'))
        _erratum_require(case is not None and case not in rows,'invalid_metric_case')
        row=dict(case_id=case,source=identifier(original.get('source')),
                 **{metric:number(original.get(metric)) for metric in METRICS})
        _erratum_require(all(row[metric] is not None and 0<=row[metric]<=1
                            for metric in ('dice','cldice')),'invalid_score')
        _erratum_require(all(row[metric] is not None and row[metric]>=0 and
                            (metric not in COUNTS or row[metric]==int(row[metric]))
                            for metric in COUNTS+VOLUMES),'invalid_count_or_volume')
        denominators=dict(precision=row['tp_voxels']+row['fp_voxels'],
                          recall=row['tp_voxels']+row['fn_voxels'],
                          centerline_precision=row['predicted_skeleton_voxels'],
                          centerline_recall=row['target_skeleton_voxels'])
        for metric,support in denominators.items():
            score=row[metric]
            _erratum_require((support==0 and score is None) or
                             (support>0 and score is not None and 0<=score<=1),'invalid_score_support')
        fine=row['fine_centerline_recall'];support=row['fine_centerline_supported_voxels']
        _erratum_require((support==0 and fine is None) or
                         (support>0 and fine is not None and 0<=fine<=1),'invalid_fine_support')
        rows[case]=row
    return rows


def _erratum_header(value):
    _erratum_require(isinstance(value,dict) and value.get('dimension')==3 and value.get('components')==1,
                     'invalid_grid_header')
    size=value.get('size')
    _erratum_require(isinstance(size,list) and len(size)==3 and all(type(n) is int and n>0 for n in size),
                     'invalid_grid_header')
    for key,length in (('spacing',3),('origin',3),('direction',9)):
        values=value.get(key)
        _erratum_require(isinstance(values,list) and len(values)==length and
                         all(number(n) is not None for n in values),'invalid_grid_header')
    _erratum_require(all(n>0 for n in value['spacing']),'invalid_grid_header')
    return value


def apply_evaluation_erratum(run,directory,original_directory):
    """Consume certified evaluation-only measurements, preserving failed provenance.

    All file hashes bind the certificate to the collected original run and exact
    final checkpoint inference metadata. This reader does not change any original
    artifact, physical header, voxel array, model or training result.
    """
    root=Path(directory)
    original,problem=_find_root(original_directory)
    _erratum_require(problem is None,'missing_original_run')
    value=_erratum_json(root,'evaluation.json')
    _erratum_require(value.get('schema')=='gta-avt-v1-evaluation-erratum-1' and
                     value.get('status')=='complete','incomplete')
    _erratum_require(type(value.get('seed')) is int and value['seed']==run['seed'],'seed_mismatch')
    _erratum_require(value.get('source_commit')==FROZEN_SOURCE_COMMIT and
                     value.get('source_manifest_sha256')==FROZEN_SOURCE_MANIFEST_SHA256,
                     'source_identity_mismatch')
    code_sha=value.get('erratum_code_sha256')
    _erratum_require(code_sha==EVALUATION_ERRATUM_CODE_SHA256,'code_identity_mismatch')
    plan=_erratum_json(original,'plan.json');pipeline=_erratum_json(original,'pipeline.json')
    _erratum_require(plan.get('source_commit')==FROZEN_SOURCE_COMMIT and
                     plan.get('data_manifest_sha256')==FROZEN_DATA_MANIFEST_SHA256,'original_source_mismatch')
    _erratum_require(value.get('original_plan_sha256')==digest(original/'plan.json') and
                     value.get('original_plan_signature')==run['plan_sha256'] and
                     value.get('original_pipeline_sha256')==digest(original/'pipeline.json'),
                     'original_identity_mismatch')
    process=_erratum_json(original,'control/process.json')
    _erratum_require(value.get('original_process_sha256')==digest(original/'control/process.json') and
                     type(process.get('exit_code')) is int and process['exit_code']!=0,
                     'original_terminal_failure_missing')
    _erratum_require(process.get('status')=='failed' and process.get('seed')==run['seed'] and
                     process.get('source_commit')==FROZEN_SOURCE_COMMIT and
                     process.get('data_manifest_sha256')==FROZEN_DATA_MANIFEST_SHA256,
                     'original_process_identity_mismatch')
    _erratum_require(run['pipeline_status']=='failed' and value.get('original_pipeline_status')=='failed'
                     and pipeline.get('current')=='native_VAL_C0' and pipeline.get('error_type')=='ValueError'
                     and value.get('original_error_stage')=='native_VAL_C0'
                     and value.get('original_error_type')=='ValueError'
                     and value.get('original_failure_reason')=='native_header_precision_check',
                     'original_failure_mismatch')
    _erratum_require(value.get('training_untouched') is True and
                     value.get('original_pipeline_untouched') is True and
                     value.get('prediction_voxels_modified') is False and
                     value.get('evaluation_only') is True,'scope_mismatch')
    _erratum_require(not run['problems'] and all(stage['status']=='complete' for stage in run['stages'].values()),
                     'original_training_invalid')
    _erratum_require(plan.get('pretrain_steps')==20000 and plan.get('warmup_steps')==10000 and
                     plan.get('adapt_steps')==5000,'training_budget_mismatch')
    artifacts=value.get('original_stage_artifacts_sha256')
    required={stage+'/'+name for stage in STAGES for name in ('config.json','summary.json')}
    _erratum_require(isinstance(artifacts,dict) and set(artifacts)==required,'stage_artifact_set_mismatch')
    for relative in required:
        _erratum_require((original/relative).is_file() and artifacts[relative]==digest(original/relative),
                         'stage_artifact_identity_mismatch')
    ready=run.get('selected_attribute_ready')
    _erratum_require(ready is not None and
                     set(ready['validation']['attributes']['readiness_tests'])==READINESS_TESTS and
                     all(ready['validation']['attributes']['readiness_tests'].values()),
                     'attribute_readiness_missing')
    cases=value.get('case_ids')
    _erratum_require(isinstance(cases,list) and len(cases)==8 and len(set(cases))==8 and
                     all(identifier(case)==case for case in cases) and set(cases)==FROZEN_VAL_CASE_IDS,
                     'case_set_invalid')
    _erratum_require({item['case_id'] for item in ready['validation']['attributes']['per_case']}==set(cases),
                     'attribute_case_set_mismatch')
    predictor_identity=None;warmup_identity=None
    for stage in ('warmup',)+ARMS:
        report=_erratum_json(original,stage+'/summary.json')
        predictor=report.get('lineage',{}).get('predictor_init',{})
        _erratum_require(isinstance(predictor,dict) and predictor.get('readiness_passed') is True and
                         predictor.get('successful_updates')==ready['successful_update'] and
                         all(isinstance(predictor.get(key),str) and SHA256.fullmatch(predictor[key])
                             for key in ('checkpoint_sha256','predictor_state_sha256')),
                         'predictor_lineage_invalid')
        identity=(predictor['checkpoint_sha256'],predictor['predictor_state_sha256'])
        if predictor_identity is None:predictor_identity=identity
        _erratum_require(identity==predictor_identity,'predictor_lineage_mismatch')
        if stage in ARMS:
            parent=report.get('lineage',{}).get('segmentation_init',{})
            _erratum_require(isinstance(parent,dict) and parent.get('warmup_successful_updates')==10000
                             and parent.get('optimizer_reset') is True and
                             all(isinstance(parent.get(key),str) and SHA256.fullmatch(parent[key])
                                 for key in ('warmup_checkpoint_sha256','warmup_state_sha256')),
                             'warmup_lineage_invalid')
            identity=(parent['warmup_checkpoint_sha256'],parent['warmup_state_sha256'])
            if warmup_identity is None:warmup_identity=identity
            _erratum_require(identity==warmup_identity,'warmup_lineage_mismatch')
    certificate=_erratum_json(root,'grid_certificate.json')
    _erratum_require(value.get('grid_certificate_sha256')==digest(root/'grid_certificate.json') and
                     certificate.get('schema')=='gta-avt-v1-grid-certificate-1' and
                     certificate.get('status')=='certified' and certificate.get('seed')==run['seed'] and
                     certificate.get('source_commit')==FROZEN_SOURCE_COMMIT and
                     certificate.get('erratum_code_sha256')==code_sha,'certificate_identity_mismatch')
    source_pairs=certificate.get('source_pairs')
    _erratum_require(isinstance(source_pairs,list) and len(source_pairs)==8 and
                     {row.get('case_id') for row in source_pairs if isinstance(row,dict)}==set(cases),
                     'source_certificate_case_set_mismatch')
    source_headers={}
    for row in source_pairs:
        differences=row.get('max_abs_header_difference')
        _erratum_require(row.get('status')=='certified' and row.get('size_equal') is True and
                         isinstance(differences,dict) and set(differences)=={'spacing','origin','direction'} and
                         all(number(x) is not None and 0<=x<=1e-10 for x in differences.values()) and
                         number(row.get('corner_mm')) is not None and 0<=row['corner_mm']<=1e-8,
                         'source_grid_not_certified')
        image=_erratum_header(row.get('image'));label=_erratum_header(row.get('label'))
        _erratum_require(image['size']==label['size'] and all(
                         abs(a-b)<=1e-10 for field in ('spacing','origin','direction')
                         for a,b in zip(image[field],label[field])),'source_grid_not_certified')
        source_headers[row['case_id']]=image
    arms=value.get('arms');cert_arms=certificate.get('arms')
    _erratum_require(isinstance(arms,dict) and set(arms)==set(ARMS) and
                     isinstance(cert_arms,dict) and set(cert_arms)==set(ARMS),'arm_set_mismatch')
    certified_native={};public_arms={}
    for arm in ARMS:
        info=arms[arm];rows=cert_arms[arm]
        _erratum_require(isinstance(info,dict) and info.get('checkpoint_successful_updates')==5000 and
                         info.get('checkpoint_arm')==('AR' if arm=='AR_GDF' else arm) and
                         isinstance(info.get('checkpoint_sha256'),str) and SHA256.fullmatch(info['checkpoint_sha256']),
                         'checkpoint_identity_invalid')
        _erratum_require(isinstance(rows,list) and len(rows)==8 and
                         {row.get('case_id') for row in rows if isinstance(row,dict)}==set(cases),
                         'prediction_certificate_case_set_mismatch')
        for row in rows:
            case=row['case_id'];modes=row.get('field_modes')
            _erratum_require(row.get('status')=='certified' and row.get('size_equal') is True and
                             row.get('prediction_voxels_modified') is False and
                             isinstance(modes,dict) and set(modes)=={'spacing','origin','direction'} and
                             all(mode in ('exact_CT','float32_CT') for mode in modes.values()) and
                             all(isinstance(row.get(key),str) and SHA256.fullmatch(row[key]) for key in
                                 ('prediction_file_sha256','prediction_voxel_sha256','inference_metadata_sha256')),
                             'prediction_grid_not_certified')
            prediction=_erratum_header(row.get('prediction_header'));image=source_headers[case]
            _erratum_require(prediction['size']==image['size'],'prediction_grid_not_certified')
            all_exact=all_quantized=True
            for field in ('spacing','origin','direction'):
                try:quantized=[struct.unpack('<f',struct.pack('<f',float(n)))[0] for n in image[field]]
                except (OverflowError,struct.error):raise ValueError('evaluation_erratum_invalid_grid_header') from None
                exact=prediction[field]==image[field];q_equal=prediction[field]==quantized
                _erratum_require((q_equal and modes[field]=='float32_CT') or
                                 (exact and not q_equal and modes[field]=='exact_CT'),
                                 'prediction_grid_not_certified')
                all_exact=all_exact and exact;all_quantized=all_quantized and q_equal
            _erratum_require(row.get('ct_grid_exact_equal') is all_exact and
                             row.get('float32_ctgrid_equal') is all_quantized,'prediction_grid_not_certified')
            relative='native_VAL/'+arm+'/'+case+'/inference.json'
            inference=_erratum_json(root,relative)
            _erratum_require(digest(root/relative)==row['inference_metadata_sha256'] and
                             inference.get('case_id')==case and inference.get('arm')==info['checkpoint_arm'] and
                             inference.get('checkpoint_sha256')==info['checkpoint_sha256'] and
                             inference.get('checkpoint_successful_updates')==5000 and
                             inference.get('condition_mode')=='predicted' and inference.get('patch_size')==128 and
                             inference.get('threshold')==.5 and inference.get('overlap')==.5,
                             'inference_identity_mismatch')
            if row.get('reused_original_prediction') is True:
                _erratum_require((original/relative).is_file() and
                                 digest(original/relative)==row['inference_metadata_sha256'],
                                 'reused_prediction_identity_mismatch')
        relative='metrics/native_VAL_'+arm+'/metrics.json'
        metrics=_erratum_json(root,relative)
        _erratum_require(info.get('metrics_sha256')==digest(root/relative),'metrics_identity_mismatch')
        native_rows=_erratum_native(metrics)
        _erratum_require(set(native_rows)==set(cases),'metrics_case_set_mismatch')
        certified_native[arm]=dict(status='recorded',pipeline_completed=False,
                                   external_evaluation_complete=True,per_case=native_rows)
        public_arms[arm]=dict(checkpoint_sha256=info['checkpoint_sha256'],checkpoint_successful_updates=5000,
                              metrics_sha256=info['metrics_sha256'])
    # This assignment happens only after the entire supplied sidecar has passed.
    run['native']=certified_native
    run['failure']=dict(error_type='ValueError',stage='native_VAL_C0',reason='header_precision_check')
    run['external_evaluation_status']='complete'
    run['status']='complete_with_evaluation_erratum'
    run['evaluation_erratum']=dict(schema=value['schema'],status='complete',source_commit=FROZEN_SOURCE_COMMIT,
        source_manifest_sha256=FROZEN_SOURCE_MANIFEST_SHA256,erratum_code_sha256=code_sha,
        evaluation_manifest_sha256=digest(root/'evaluation.json'),grid_certificate_sha256=value['grid_certificate_sha256'],
        original_pipeline_status='failed',original_failure_reason='header_precision_check',
        original_pipeline_sha256=value['original_pipeline_sha256'],original_process_sha256=value['original_process_sha256'],
        original_plan_sha256=value['original_plan_sha256'],original_stage_artifacts_sha256=artifacts,
        training_untouched=True,original_pipeline_untouched=True,prediction_voxels_modified=False,
        evaluation_only=True,arms=public_arms)
    return run


def export(inputs,output,*,expected_inputs=3,expected_case_count=8,expected_seeds=None,platform_metadata=None,
           evaluation_inputs=None):
    output=Path(output)
    runs=[load_run(directory,platform_metadata) for directory in inputs]
    if evaluation_inputs is not None:
        by_seed={}
        for directory in evaluation_inputs:
            value=_erratum_json(Path(directory),'evaluation.json');seed=value.get('seed')
            _erratum_require(type(seed) is int and seed not in by_seed,'duplicate_or_invalid_seed')
            by_seed[seed]=directory
        _erratum_require(set(by_seed).issubset({run['seed'] for run in runs}),'unmatched_seed')
        for directory,run in zip(inputs,runs):
            if run['seed'] in by_seed:apply_evaluation_erratum(run,by_seed[run['seed']],directory)
    output.mkdir(parents=True,exist_ok=False)
    seen=set()
    for run in runs:
        seed=run['seed']
        if seed is not None and seed in seen:run['problems'].append('duplicate_seed');run['status']='invalid'
        seen.add(seed)
    case_ids=sorted({case for run in runs for value in run['native'].values() for case in value['per_case']})
    problems=[]
    if len(runs)!=expected_inputs:problems.append('input_count_mismatch')
    if expected_seeds is not None and seen!=set(expected_seeds):problems.append('seed_set_mismatch')
    if len(case_ids)!=expected_case_count:problems.append('native_case_count_mismatch')
    for case in case_ids:
        for metric in ('target_volume_ml','target_skeleton_voxels','fine_centerline_supported_voxels'):
            values=[native['per_case'][case][metric] for run in runs for native in run['native'].values()
                    if case in native['per_case'] and native['per_case'][case][metric] is not None]
            if values and not all(math.isclose(value,values[0],rel_tol=0,abs_tol=1e-7) for value in values):
                problems.append('native_target_metadata_mismatch');break
    per_case=[];summaries=[];deltas=[]
    for run in runs:
        scope=run.get('platform',{}).get('platform_scope')
        for arm in ARMS:
            native=run['native'].get(arm,dict(status='missing',per_case={}))
            rows=native['per_case']
            arm_status='recorded' if (native['status']=='recorded' and
                       (native.get('pipeline_completed',False) or native.get('external_evaluation_complete',False)) and set(rows)==set(case_ids)
                       and len(rows)==expected_case_count and all(r.get('dice') is not None for r in rows.values())) else 'missing_or_incomplete'
            if arm_status!='recorded':
                run['problems'].append('native_VAL_'+arm+'_incomplete');run['status']='partial'
            for case in case_ids or [None]:
                original=rows.get(case,{})
                per_case.append(dict(seed=run['seed'],platform_scope=scope,arm=arm,
                    status='recorded' if original else 'missing',case_id=case,
                    source=original.get('source'),**{metric:original.get(metric) for metric in METRICS}))
            for metric in METRICS:
                values=[row[metric] for row in rows.values() if row[metric] is not None]
                summaries.append(dict(seed=run['seed'],platform_scope=scope,arm=arm,metric=metric,
                    status='undefined' if arm_status=='recorded' and not values else arm_status,expected_cases=expected_case_count,
                    worst_direction='largest' if metric in LOWER_BETTER else 'smallest',
                    **summary(values,lower_better=metric in LOWER_BETTER)))
        for before,after in CONTRASTS:
            left=run['native'].get(before,{}).get('per_case',{})
            right=run['native'].get(after,{}).get('per_case',{})
            contrast=before+'_to_'+after
            for metric in METRICS:
                pairs=[]
                for case in case_ids or [None]:
                    a=left.get(case,{}).get(metric);b=right.get(case,{}).get(metric)
                    delta=b-a if a is not None and b is not None else None
                    if delta is not None:pairs.append((a,b,delta))
                    deltas.append(dict(seed=run['seed'],platform_scope=scope,contrast=contrast,
                        aggregation='case',case_id=case,metric=metric,from_value=a,to_value=b,delta=delta,
                        paired_cases=1 if delta is not None else 0,status='paired' if delta is not None
                            else 'undefined_metric' if case in left and case in right else 'missing_pair'))
                deltas.append(dict(seed=run['seed'],platform_scope=scope,contrast=contrast,
                    aggregation='seed_mean',case_id=None,metric=metric,
                    from_value=statistics.fmean(p[0] for p in pairs) if pairs else None,
                    to_value=statistics.fmean(p[1] for p in pairs) if pairs else None,
                    delta=statistics.fmean(p[2] for p in pairs) if pairs else None,paired_cases=len(pairs),
                    status='paired_complete' if len(pairs)==expected_case_count else 'paired_partial'))
    complete=(not problems and len(runs)==expected_inputs and all(run['status'] in ('complete','complete_with_evaluation_erratum') and
              all(stage['status']=='complete' for stage in run['stages'].values()) for run in runs))
    manifest=dict(schema='GTA-AVT-v1-objective-export-1',status='complete' if complete else 'partial',
        generated_utc=datetime.now(timezone.utc).isoformat(),expected_seed_repeats=expected_inputs,
        expected_native_val_cases=expected_case_count,expected_seeds=expected_seeds,
        input_runs=len(runs),case_ids=case_ids,problems=problems,runs=runs,
        definitions=dict(scope='Three independently initialized seed repeats; native VAL at common final update; no new TEST evaluation',
            score_units='Fractions; percentage-point deltas equal raw score delta times100',
            delta='to minus from, paired on the same seed and case; undefined metrics stay null',
            variance='Population SD across defined cases; no claim about patient independence',
            worst20='ceil(20% of defined cases), smallest for scores and largest for FP/FN; other count/volume tails descriptive',
            fine='GT native Lee skeleton with physical native EDT radius<=2mm; no supported sites gives null recall',
            partial='Partial exports are explicitly incomplete; failed/missing measurements are never imputed as zero'))
    if evaluation_inputs is not None:
        manifest['definitions']['evaluation_erratum']='Explicit certified external evaluation; original failed pipeline status and artifacts preserved. Training, model and prediction voxel arrays unchanged.'
    _write_csv(output/'native_per_case.csv',per_case,
        ['seed','platform_scope','arm','status','case_id','source']+list(METRICS))
    _write_csv(output/'native_summary.csv',summaries,
        ['seed','platform_scope','arm','metric','status','expected_cases','defined_cases','mean',
         'std_population','minimum','maximum','worst20_mean','worst_direction'])
    _write_csv(output/'paired_deltas.csv',deltas,
        ['seed','platform_scope','contrast','aggregation','case_id','metric','from_value','to_value',
         'delta','paired_cases','status'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    return manifest


def self_test():
    """Synthetic complete/partial exports, unsafe-field exclusion and null pairs."""
    with tempfile.TemporaryDirectory() as temporary:
        home=Path(temporary);inputs=[]
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
        for seed in (31,32,33):
            root=home/str(seed);inputs.append(root)
            plan=dict(seed=seed,manifest='/private/data',output_dir='/private/run',pretrain_steps=2,warmup_steps=3,adapt_steps=4)
            ph=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
            state=dict(status='complete',plan_sha256=ph,completed={})
            write(root/'plan.json',plan)
            for stage in STAGES:
                updates=2 if stage=='attribute' else 3 if stage=='warmup' else 4
                value=dict(successful_updates=updates,validation=None,last_checkpoint='/private/weights.pt')
                state['completed'][stage]=value
                write(root/stage/'summary.json',value)
                cfg=dict(seed=seed,max_steps=updates)
                if stage!='attribute':
                    cfg.update(arm='AR' if stage=='AR_GDF' else 'C0' if stage=='warmup' else stage,
                               stage='warmup' if stage=='warmup' else 'adapt')
                write(root/stage/'config.json',cfg)
            for index,arm in enumerate(ARMS):
                rows=[dict(case_id=case,dice=.7+.01*index,cldice=.8+.01*index,
                           fine_centerline_recall=None,fine_centerline_supported_voxels=0,
                           target_skeleton_voxels=5,fp_voxels=3-index) for case in ('a','b')]
                value=dict(per_case=rows);state['completed']['native_VAL_'+arm]=value
                write(root/'metrics'/('native_VAL_'+arm)/'metrics.json',value)
            write(root/'pipeline.json',state)
        result=export(inputs,home/'complete',expected_case_count=2)
        assert result['status']=='complete'
        assert '/private/' not in (home/'complete'/'manifest.json').read_text()
        with (home/'complete'/'paired_deltas.csv').open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
        target=next(r for r in rows if r['contrast']=='A_only_to_AR' and r['metric']=='dice' and r['aggregation']=='seed_mean')
        assert abs(float(target['delta'])-.01)<1e-12
        null=next(r for r in rows if r['metric']=='fine_centerline_recall')
        assert null['delta']=='' and null['status']=='undefined_metric'
        # A failed run's raw error, which may contain paths/secrets, is excluded.
        state=read_json(inputs[0]/'pipeline.json');state.update(status='failed',error='secret /private/password',error_type='RuntimeError')
        state['completed'].pop('native_VAL_AR_GDF')
        write(inputs[0]/'pipeline.json',state)
        (inputs[0]/'metrics'/'native_VAL_AR_GDF'/'metrics.json').unlink()
        partial=export(inputs,home/'partial',expected_case_count=2)
        assert partial['status']=='partial'
        text=(home/'partial'/'manifest.json').read_text()
        assert 'secret' not in text and '/private/' not in text
        with (home/'partial'/'native_per_case.csv').open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
        missing=next(row for row in rows if row['seed']=='31' and row['arm']=='AR_GDF')
        assert missing['status']=='missing' and missing['dice']==''
    return dict(status='passed',scope='synthetic local objective-export self-test; no experiment results read')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+');parser.add_argument('--output')
    parser.add_argument('--expected-inputs',type=int,default=3)
    parser.add_argument('--expected-case-count',type=int,default=8)
    parser.add_argument('--expected-seeds',nargs='+',type=int)
    parser.add_argument('--evaluation-inputs',nargs='+',help='Optional certified evaluation-only sidecar directories; invalid sidecars are rejected')
    parser.add_argument('--platform-metadata');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    if not args.inputs or not args.output:parser.error('--inputs and --output are required')
    metadata=read_json(args.platform_metadata) if args.platform_metadata else None
    result=export(args.inputs,args.output,expected_inputs=args.expected_inputs,
        expected_case_count=args.expected_case_count,expected_seeds=args.expected_seeds,platform_metadata=metadata,
        evaluation_inputs=args.evaluation_inputs)
    print(json.dumps(dict(status=result['status'],input_runs=result['input_runs'],cases=len(result['case_ids']))))


if __name__=='__main__':main()
