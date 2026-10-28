import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('trajectory',Path(__file__).with_name('v12_training_trajectory_export.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    validation=dict(scope='private arbitrary source string is not copied',
        summary=dict(dice=dict(mean=.9,std_population=.1)),
        per_case=[dict(case_id='case_A',dice=.8,precision=None,recall=.9)],
        attributes=dict(summary=dict(radius_mse=dict(mean=.04)),
            per_case=[dict(case_id='case_A',support_dice=.7)],readiness=dict(passed=True)))
    history=[dict(successful_update=1,attempt=1,epoch=0,lr=.001,losses=dict(loss=.3),
                 sampling=dict(foreground_fraction=[.01,.03],requested_counts=dict(vessel=2))),
             dict(successful_update=5,attempt=6,lr=.001,losses=dict(loss=.2)),
             dict(successful_update=100,attempt=101,lr=.001,losses=dict(loss=.1),validation=validation)]
    return dict(window=200,plan=dict(source_commit='a'*40,output_dir='E:/PRIVATE/experiment'),
        phases=dict(condition=dict(config=dict(max_steps=100,seed=20261032,device='cuda',stage='local_adapt',
            output_dir='E:/PRIVATE/experiment',manifest='/root/private/data.json',sampling=dict(weights=dict(vessel=.5,uniform=.5))),
            summary=dict(successful_updates=100,attempts=101,elapsed_seconds=50.),
            rows=100,raw_history_rows=100,history=history,
            amp_skip_events=[dict(successful_update=5,previous_attempt=4,next_attempt=6,skipped_attempts=[5])],
            complete_line_sha256='b'*64,nonfinite_losses=[],gaps=[])))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'output'
    def tearDown(self):self.tmp.cleanup()
    def export(self,snapshot=None):
        return module.export_snapshots([dict(run_label='AR',snapshot=snapshot or fixture())],self.path)
    def csv(self,name):
        with (self.path/(name+'.csv')).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

    def test_raw_units_coordinates_and_skip_boundary_retained(self):
        r=self.export();rows=self.csv('training_scalars')
        self.assertEqual([int(x['successful_update']) for x in rows],[1,5,100])
        self.assertEqual(float(rows[0]['losses.loss']),.3)
        self.assertEqual(float(rows[0]['sampling.foreground_fraction.mean']),.02)
        self.assertEqual(float(rows[0]['sampling.requested_counts.vessel']),2.)
        self.assertNotIn('elapsed_seconds',rows[0])
        self.assertEqual(r['runs'][0]['phases']['condition']['represented_successful_rows'],100)

    def test_validation_and_attribute_metrics_keep_original_fractions(self):
        self.export();v=self.csv('validation_summary')[0]
        self.assertEqual(float(v['summary.dice.mean']),.9)
        self.assertEqual(float(v['attributes.summary.radius_mse.mean']),.04)
        cases=self.csv('validation_case_metrics')
        self.assertEqual({v['metric_family'] for v in cases},{'attributes','segmentation'})
        self.assertEqual(cases[0]['precision'],'')

    def test_attempt_differences_are_explicitly_inferred_not_failed_batch_losses(self):
        self.export();rows=self.csv('amp_skips')
        self.assertEqual(rows[0]['event_scope'],'phase_total')
        self.assertEqual(int(rows[0]['skipped_attempts']),1)
        self.assertEqual(json.loads(rows[1]['skipped_attempt_indices']),[5])
        self.assertIn('inferred',rows[1]['inference'])

    def test_private_strings_paths_and_narrative_are_not_published(self):
        self.export();text=''.join(p.read_text() for p in self.path.iterdir())
        self.assertNotIn('E:/PRIVATE',text);self.assertNotIn('/root/private',text)
        self.assertNotIn('arbitrary source string',text)
        self.assertIn('a'*40,text)

    def test_windows_remain_aggregated_separate_from_raw_points(self):
        s=fixture();p=s['phases']['condition'];p.pop('history');p.pop('amp_skip_events')
        p['first']=dict(step=1,attempt=1,values={'losses.loss':.3})
        p['last']=dict(step=100,attempt=101,values={'losses.loss':.1})
        p['windows']=[dict(first=1,last=100,n=100,stats={'losses.loss':dict(mean=.2,p10=.1,p90=.3,n=100)})]
        self.export(s)
        self.assertEqual(len(self.csv('training_scalars')),2)
        self.assertEqual(self.csv('training_scalars')[0]['sample_kind'],'recorded_boundary_scalar')
        self.assertEqual(float(self.csv('training_windows')[0]['mean']),.2)

    def test_partial_stage_is_not_marked_complete(self):
        s=fixture();s['phases']['condition']['summary']=None
        self.export(s);self.assertEqual(self.csv('phase_endpoints')[0]['status'],'partial')

    def test_nan_and_unsafe_labels_are_rejected(self):
        s=fixture();s['phases']['condition']['history'][0]['losses']['loss']=float('nan')
        with self.assertRaises(ValueError):self.export(s)
        with self.assertRaises(ValueError):module.export_snapshots([dict(run_label='../unsafe',snapshot=fixture())],Path(self.tmp.name)/'other')

    def test_bad_amp_counter_claim_is_rejected(self):
        s=fixture();s['phases']['condition']['amp_skip_events'][0]['skipped_attempts']=[999]
        with self.assertRaises(ValueError):self.export(s)

    def test_duplicate_validation_case_is_rejected(self):
        s=fixture();v=s['phases']['condition']['history'][-1]['validation'];v['per_case']*=2
        with self.assertRaises(ValueError):self.export(s)

    def test_existing_export_is_immutable(self):
        self.export()
        with self.assertRaises(FileExistsError):self.export()

    def test_saved_checkpoint_marker_uses_downstream_lineage_not_curve_extremum(self):
        s=fixture();p=s['phases'].pop('condition');s['phases']['attribute']=p
        s['phases']['base_warm']=dict(config=dict(max_steps=20,stage='base'),rows=20,
            summary=dict(successful_updates=20,attempts=20,lineage=dict(predictor_init=dict(
                successful_updates=29,checkpoint_sha256='c'*64))))
        self.export(s);marker=self.csv('checkpoint_markers')[0]
        self.assertEqual(marker['successful_update'],'29')
        self.assertEqual(marker['checkpoint_sha256'],'c'*64)

    def test_distinct_old_control_commit_and_expected_boundary_error_are_preserved(self):
        s=fixture();s['plan']['source_commit']='d'*40
        s['pipeline']=dict(status='failed',current='base_warm',error_type='FileExistsError',error='/root/private/sentinel')
        r=self.export(s)
        self.assertEqual(r['runs'][0]['identities']['plan.source_commit'],'d'*40)
        self.assertEqual(r['runs'][0]['pipeline_error_type_at_snapshot'],'FileExistsError')
        self.assertNotIn('/root/private/sentinel',(self.path/'trajectory_manifest.json').read_text())


if __name__=='__main__':unittest.main()
