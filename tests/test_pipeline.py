import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gta_avt.pipeline import run_pipeline, PHASE_ORDER


class PipelineTests(unittest.TestCase):
    def plan(self, root, **changes):
        phases=json.loads((Path(__file__).resolve().parents[1]/'configs'/'phases.json').read_text())
        for i,name in enumerate(PHASE_ORDER):phases[name]['max_steps']=i+1
        cfg=dict(manifest='not-read',output_dir=str(root/'run'),seed=30,phases=phases,stop_after='complete')
        cfg.update(changes);return cfg

    def stage(self, calls):
        def execute(config):
            calls.append(copy.deepcopy(config));home=Path(config['output_dir']);home.mkdir()
            return dict(successful_updates=config['max_steps'],last_checkpoint=str(home/'last.pt'),
                best_checkpoint=str(home/'best.pt'),best_ready_checkpoint=str(home/'best_ready.pt'),
                validation={'summary':{'dice':{'mean':.75}}})
        return execute

    def loaded(self, *args):return {'validation':{'summary':{'dice':{'mean':.8}}}}

    def test_declared_single_variant_sequence_and_pair_handoffs(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls=[];cfg=self.plan(Path(temporary));execute=self.stage(calls)
            with patch('gta_avt.pipeline.pretrain',execute),patch('gta_avt.pipeline.train',execute),patch('gta_avt.pipeline.load_checkpoint',self.loaded):
                result=run_pipeline(cfg)
                self.assertEqual(result['status'],'complete')
                self.assertEqual([c['phase'] for c in calls],list(PHASE_ORDER))
                self.assertEqual(result['planned_successful_updates'],28)
                self.assertEqual(result['completed_successful_updates'],28)
                self.assertEqual(Path(calls[1]['predictor_checkpoint']).parts[-2:],('attribute','best_ready.pt'))
                self.assertEqual(Path(calls[2]['parent_checkpoint']).parts[-2:],('base_warm','last.pt'))
                self.assertEqual(Path(calls[3]['parent_checkpoint']).parts[-2:],('base','best_ready.pt'))
                self.assertEqual(Path(calls[4]['parent_checkpoint']).parts[-2:],('condition','last.pt'))
                self.assertTrue(all('predictor_checkpoint' not in c for c in calls[2:]))
                self.assertEqual([c['arm'] for c in calls],['C0','C0','C0','AR','AR','AR','AR'])
                self.assertEqual(calls[5]['gdf_weight_start'],.1)
                self.assertEqual(calls[5]['gdf_weight_end'],1.)
                self.assertEqual(calls[6]['gdf_weight_end'],.2)
                self.assertEqual(calls[6]['local_gate_lr'],.0005)
                self.assertEqual(result['quality_checks']['base']['sampled_val_dice'],.8)
                self.assertEqual(result['quality_checks']['condition']['sampled_val_dice'],.75)
                before=len(calls)
                run_pipeline(dict(cfg,resume_pipeline=True))
                self.assertEqual(len(calls),before)

    def test_default_base_stop_can_advance_without_recipe_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls=[];cfg=self.plan(Path(temporary));cfg.pop('stop_after');execute=self.stage(calls)
            with patch('gta_avt.pipeline.pretrain',execute),patch('gta_avt.pipeline.train',execute),patch('gta_avt.pipeline.load_checkpoint',self.loaded):
                result=run_pipeline(cfg)
                self.assertEqual(result['status'],'staged');self.assertEqual(result['stopped_after'],'base')
                self.assertEqual(len(calls),3)
                result=run_pipeline(dict(cfg,resume_pipeline=True,stop_after='condition'))
                self.assertEqual(len(calls),4);self.assertEqual(result['stopped_after'],'condition')
                changed=copy.deepcopy(cfg);changed['phases']['condition']['max_steps']+=1
                with self.assertRaises(ValueError):run_pipeline(dict(changed,resume_pipeline=True))
                with self.assertRaises(ValueError):run_pipeline(dict(cfg,resume_pipeline=True,variant='C0'))

    def test_quality_failure_blocks_gaussian_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls=[];cfg=self.plan(Path(temporary),minimum_condition_dice=.8);execute=self.stage(calls)
            with patch('gta_avt.pipeline.pretrain',execute),patch('gta_avt.pipeline.train',execute),patch('gta_avt.pipeline.load_checkpoint',self.loaded):
                with self.assertRaisesRegex(RuntimeError,'condition Dice'):
                    run_pipeline(cfg)
            self.assertEqual([c['phase'] for c in calls],list(PHASE_ORDER[:4]))
            state=json.loads((Path(cfg['output_dir'])/'pipeline.json').read_text())
            self.assertEqual(state['status'],'failed');self.assertFalse(state['quality_checks']['condition']['passed'])

    def test_no_automatic_ablation_expansion_and_nongdf_budget_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls=[];cfg=self.plan(Path(temporary),variant='AR');execute=self.stage(calls)
            with patch('gta_avt.pipeline.pretrain',execute),patch('gta_avt.pipeline.train',execute),patch('gta_avt.pipeline.load_checkpoint',self.loaded):run_pipeline(cfg)
            self.assertEqual(len(calls),7)
            self.assertEqual(calls[4]['lambda_gdf'],1.)
            self.assertEqual([(c['gdf_weight_start'],c['gdf_weight_end']) for c in calls[5:]],[(0.,0.),(0.,0.)])

    def test_attribute_failure_never_launches_joint_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg=self.plan(Path(temporary))
            with patch('gta_avt.pipeline.pretrain',return_value={'successful_updates':1,'best_ready_checkpoint':None}),patch('gta_avt.pipeline.train') as train:
                with self.assertRaisesRegex(RuntimeError,'No attribute checkpoint'):run_pipeline(cfg)
                train.assert_not_called()

    def test_native_evaluation_uses_final_phase_and_ct_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'cases.json').write_text(json.dumps({'cases':[dict(id='x',image='ct.nrrd',label='gt.nrrd',split='VAL')]}))
            cfg=self.plan(root,evaluation_cases=str(root/'cases.json'),evaluation_endpoint='best');calls=[];execute=self.stage(calls)
            def infer(config):
                self.assertEqual(Path(config['checkpoint']).parts[-2:],('refine','best.pt'))
                self.assertNotIn('label',config);return {'prediction':'prediction.nii.gz'}
            def evaluate(config):
                self.assertEqual(config['cases'][0]['image'],'ct.nrrd');return {'per_case':[]}
            with patch('gta_avt.pipeline.pretrain',execute),patch('gta_avt.pipeline.train',execute),patch('gta_avt.pipeline.load_checkpoint',self.loaded),patch('gta_avt.pipeline.infer',infer),patch('gta_avt.pipeline.evaluate',evaluate):
                result=run_pipeline(cfg)
            self.assertEqual(len(result['completed']),8)


if __name__=='__main__':unittest.main()
