"""Local standard-library integrity checks for the external evaluation reader."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import struct

import v1_objective_readout as reader


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture(home):
    original=home/'original';sidecar=home/'sidecar';seed=20261031
    cases=sorted(reader.FROZEN_VAL_CASE_IDS)
    plan=dict(seed=seed,source_commit=reader.FROZEN_SOURCE_COMMIT,pretrain_steps=20000,
              warmup_steps=10000,adapt_steps=5000,data_manifest_sha256=reader.FROZEN_DATA_MANIFEST_SHA256)
    signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    pipeline=dict(status='failed',plan_sha256=signature,current='native_VAL_C0',error_type='ValueError',
                  error='private credential /root/private',completed={})
    write(original/'plan.json',plan)
    ready=dict(readiness=dict(passed=True,tests={key:True for key in reader.READINESS_TESTS}),
               per_case=[dict(case_id=case) for case in cases])
    for stage in reader.STAGES:
        updates=20000 if stage=='attribute' else 10000 if stage=='warmup' else 5000
        lineage=dict(predictor_init=dict(checkpoint_sha256='a'*64,predictor_state_sha256='b'*64,
                                        successful_updates=17000,readiness_passed=True))
        if stage in reader.ARMS:
            lineage['segmentation_init']=dict(warmup_successful_updates=10000,optimizer_reset=True,
                warmup_checkpoint_sha256='c'*64,warmup_state_sha256='d'*64)
        report=dict(successful_updates=updates,lineage=lineage)
        pipeline['completed'][stage]=report
        write(original/stage/'summary.json',report)
        config=dict(seed=seed,max_steps=updates)
        if stage!='attribute':config.update(arm='AR' if stage=='AR_GDF' else 'C0' if stage=='warmup' else stage,
                                            stage='warmup' if stage=='warmup' else 'adapt')
        write(original/stage/'config.json',config)
    history=original/'attribute/history.jsonl'
    history.write_text(json.dumps(dict(successful_update=17000,validation=dict(selection_score=.1,attributes=ready)))+'\n',encoding='utf-8')
    write(original/'pipeline.json',pipeline)
    write(original/'control/process.json',dict(status='failed',exit_code=1,seed=seed,
        source_commit=reader.FROZEN_SOURCE_COMMIT,data_manifest_sha256=reader.FROZEN_DATA_MANIFEST_SHA256))
    certificate=dict(schema='gta-avt-v1-grid-certificate-1',status='certified',seed=seed,
        source_commit=reader.FROZEN_SOURCE_COMMIT,erratum_code_sha256=reader.EVALUATION_ERRATUM_CODE_SHA256,source_pairs=[],arms={})
    image=dict(size=[10,10,10],spacing=[1.1,1.,1.],origin=[0.,0.,0.],
               direction=[1.,0.,0.,0.,1.,0.,0.,0.,1.],dimension=3,components=1)
    prediction=copy.deepcopy(image)
    prediction['spacing'][0]=struct.unpack('<f',struct.pack('<f',1.1))[0]
    for case in cases:
        certificate['source_pairs'].append(dict(case_id=case,status='certified',size_equal=True,
            image=copy.deepcopy(image),label=copy.deepcopy(image),
            max_abs_header_difference=dict(spacing=0,origin=0,direction=0),corner_mm=0))
    arm_metadata={}
    for index,arm in enumerate(reader.ARMS):
        checkpoint_arm='AR' if arm=='AR_GDF' else arm
        checkpoint_sha=hashlib.sha256(arm.encode()).hexdigest()
        cert_rows=[];metrics=[]
        for case in cases:
            inference=dict(case_id=case,arm=checkpoint_arm,checkpoint_sha256=checkpoint_sha,
                checkpoint_successful_updates=5000,condition_mode='predicted',patch_size=128,threshold=.5,overlap=.5,
                prediction='/private/prediction')
            relative='native_VAL/'+arm+'/'+case+'/inference.json'
            write(sidecar/relative,inference)
            if arm=='C0':write(original/relative,inference)
            cert_rows.append(dict(case_id=case,status='certified',size_equal=True,
                field_modes=dict(spacing='float32_CT',origin='float32_CT',direction='float32_CT'),
                prediction_header=copy.deepcopy(prediction),ct_grid_exact_equal=False,float32_ctgrid_equal=True,
                prediction_voxels_modified=False,prediction_file_sha256='f'*64,prediction_voxel_sha256='9'*64,
                inference_metadata_sha256=reader.digest(sidecar/relative),reused_original_prediction=arm=='C0'))
            row=dict(case_id=case,source='synthetic')
            row.update({key:.7+.01*index for key in reader.SCORES})
            row.update({key:5 for key in reader.COUNTS})
            row.update({key:3.0 for key in reader.VOLUMES})
            metrics.append(row)
        certificate['arms'][arm]=cert_rows
        relative='metrics/native_VAL_'+arm+'/metrics.json'
        write(sidecar/relative,dict(per_case=metrics))
        arm_metadata[arm]=dict(checkpoint_arm=checkpoint_arm,checkpoint_successful_updates=5000,
                               checkpoint_sha256=checkpoint_sha,metrics_sha256=reader.digest(sidecar/relative))
    write(sidecar/'grid_certificate.json',certificate)
    value=dict(schema='gta-avt-v1-evaluation-erratum-1',status='complete',seed=seed,
        source_commit=reader.FROZEN_SOURCE_COMMIT,source_manifest_sha256=reader.FROZEN_SOURCE_MANIFEST_SHA256,
        erratum_code_sha256=reader.EVALUATION_ERRATUM_CODE_SHA256,original_pipeline_sha256=reader.digest(original/'pipeline.json'),
        original_plan_sha256=reader.digest(original/'plan.json'),original_plan_signature=signature,
        original_process_sha256=reader.digest(original/'control/process.json'),original_pipeline_status='failed',
        original_error_type='ValueError',original_error_stage='native_VAL_C0',
        original_failure_reason='native_header_precision_check',training_untouched=True,
        original_pipeline_untouched=True,prediction_voxels_modified=False,evaluation_only=True,
        case_ids=cases,arms=arm_metadata,grid_certificate_sha256=reader.digest(sidecar/'grid_certificate.json'),
        original_stage_artifacts_sha256={stage+'/'+name:reader.digest(original/stage/name)
            for stage in reader.STAGES for name in ('config.json','summary.json')})
    write(sidecar/'evaluation.json',value)
    return original,sidecar,value,certificate


class ErratumIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.home=Path(self.temporary.name)
        self.original,self.sidecar,self.value,self.certificate=fixture(self.home)

    def tearDown(self):self.temporary.cleanup()

    def export(self,**kwargs):
        return reader.export([self.original],self.home/'output',expected_inputs=1,expected_seeds=[20261031],
                             evaluation_inputs=[self.sidecar],**kwargs)

    def reject(self):
        with self.assertRaisesRegex(ValueError,'^evaluation_erratum_'):self.export()
        self.assertFalse((self.home/'output').exists())

    def save(self):write(self.sidecar/'evaluation.json',self.value)

    def cert_save(self):
        write(self.sidecar/'grid_certificate.json',self.certificate)
        self.value['grid_certificate_sha256']=reader.digest(self.sidecar/'grid_certificate.json');self.save()

    def test_complete_keeps_original_failure(self):
        before=reader.digest(self.original/'pipeline.json')
        result=self.export()
        self.assertEqual(result['status'],'complete')
        run=result['runs'][0]
        self.assertEqual(run['pipeline_status'],'failed')
        self.assertEqual(run['status'],'complete_with_evaluation_erratum')
        self.assertEqual(run['failure']['reason'],'header_precision_check')
        self.assertTrue(all(not item['pipeline_completed'] for item in run['native'].values()))
        self.assertEqual(reader.digest(self.original/'pipeline.json'),before)
        text=(self.home/'output/manifest.json').read_text()
        self.assertNotIn('/private/',text);self.assertNotIn('credential',text)

    def test_default_failed_stays_partial(self):
        result=reader.export([self.original],self.home/'output',expected_inputs=1)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['runs'][0]['pipeline_status'],'failed')

    def test_failed_sidecar(self):
        self.value['status']='failed';self.save();self.reject()

    def test_seed_mismatch(self):
        self.value['seed']=20261032;self.save();self.reject()

    def test_wrong_source(self):
        self.value['source_commit']='0'*40;self.save();self.reject()

    def test_wrong_erratum_code(self):
        self.value['erratum_code_sha256']='0'*64;self.save();self.reject()

    def test_missing_arm(self):
        self.value['arms'].pop('AR_GDF');self.save();self.reject()

    def test_wrong_checkpoint(self):
        self.value['arms']['AR']['checkpoint_sha256']='0'*64;self.save();self.reject()

    def test_wrong_budget(self):
        self.value['arms']['AR']['checkpoint_successful_updates']=4000;self.save();self.reject()

    def test_modified_original_stage(self):
        path=self.original/'AR/summary.json';value=reader.read_json(path)
        value['successful_updates']=4999;write(path,value);self.reject()

    def test_changed_pipeline(self):
        path=self.original/'pipeline.json';value=reader.read_json(path)
        value['status']='complete';write(path,value);self.reject()

    def test_missing_case_certificate(self):
        self.certificate['arms']['AR'].pop();self.cert_save();self.reject()

    def test_unapproved_grid_mode(self):
        self.certificate['arms']['AR'][0]['field_modes']['origin']='resampled_CT';self.cert_save();self.reject()

    def test_falsely_certified_grid(self):
        self.certificate['arms']['AR'][0]['prediction_header']['origin'][0]=.01
        self.cert_save();self.reject()

    def test_wrong_component_count(self):
        self.certificate['arms']['AR'][0]['prediction_header']['components']=2
        self.cert_save();self.reject()

    def test_exact_ct_spacing_accepted(self):
        row=self.certificate['arms']['AR'][0]
        row['prediction_header']['spacing'][0]=1.1;row['field_modes']['spacing']='exact_CT'
        row['ct_grid_exact_equal']=True;row['float32_ctgrid_equal']=False
        self.cert_save();self.assertEqual(self.export()['status'],'complete')

    def test_pred_voxels_modified(self):
        self.certificate['arms']['AR'][0]['prediction_voxels_modified']=True;self.cert_save();self.reject()

    def test_source_grid_misaligned(self):
        self.certificate['source_pairs'][0]['max_abs_header_difference']['origin']=.001
        self.cert_save();self.reject()

    def test_metric_file_modified(self):
        path=self.sidecar/'metrics/native_VAL_AR/metrics.json';value=reader.read_json(path)
        value['per_case'][0]['dice']=.99;write(path,value);self.reject()

    def test_missing_fine_support(self):
        path=self.sidecar/'metrics/native_VAL_AR/metrics.json';value=reader.read_json(path)
        value['per_case'][0]['fine_centerline_recall']=None;write(path,value)
        self.value['arms']['AR']['metrics_sha256']=reader.digest(path);self.save();self.reject()

    def test_empty_prediction_preserves_undefined_precision(self):
        path=self.sidecar/'metrics/native_VAL_AR/metrics.json';value=reader.read_json(path)
        row=value['per_case'][0]
        row.update(tp_voxels=0,fp_voxels=0,predicted_skeleton_voxels=0,dice=0,cldice=0,
                   precision=None,centerline_precision=None,recall=0,centerline_recall=0,
                   fine_centerline_recall=0,predicted_volume_ml=0,fp_volume_ml=0)
        write(path,value);self.value['arms']['AR']['metrics_sha256']=reader.digest(path);self.save()
        result=self.export();self.assertEqual(result['status'],'complete')
        self.assertIsNone(result['runs'][0]['native']['AR']['per_case'][row['case_id']]['precision'])


if __name__=='__main__':unittest.main()
