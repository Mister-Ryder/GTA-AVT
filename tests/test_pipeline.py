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
        cfg=dict(manifest='not-read',output_dir=str(root/'run'),seed=30,phases=phases,stop_after='condition')
        cfg.update(changes);return cfg

    def stage(self, calls):
        def execute(config):
            calls.append(copy.deepcopy(config));home=Path(config['output_dir']);home.mkdir()
            return dict(successful_updates=config['max_steps'],last_checkpoint=str(home/'last.pt'),
                best_checkpoint=str(home/'best.pt'),best_ready_checkpoint=str(home/'best_ready.pt'),
                validation={'summary':{'dice':{'mean':.75}}})
        return execute

    def loaded(self, *args):return {'validation':{'summary':{'dice':{'mean':.8}}}}

    def test_declared_prefix_keeps_indivisible_pair_handoffs(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls=[];cfg=self.plan(Path(temporary));execute=self.stage(calls)
            with patch('gta_avt.pipeline.pretrain',execute),patch('gta_avt.pipeline.train',execute),patch('gta_avt.pipeline.load_checkpoint',self.loaded):
                result=run_pipeline(cfg)
                self.assertEqual(result['status'],'staged')
                self.assertEqual([c['phase'] for c in calls],list(PHASE_ORDER[:4]))
                self.assertEqual(Path(calls[1]['predictor_checkpoint']).parts[-2:],('attribute','best_ready.pt'))
                self.assertEqual(Path(calls[2]['parent_checkpoint']).parts[-2:],('base_warm','last.pt'))
                self.assertEqual(Path(calls[3]['parent_checkpoint']).parts[-2:],('base','best_ready.pt'))
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

    def test_uncalibrated_generic_gaussian_pipeline_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg=self.plan(Path(temporary),stop_after='complete')
            with patch('gta_avt.pipeline.train') as train:
                with self.assertRaisesRegex(ValueError,'calibrated prepare/run'):
                    run_pipeline(cfg)
                train.assert_not_called()

    def test_attribute_failure_never_launches_joint_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg=self.plan(Path(temporary))
            with patch('gta_avt.pipeline.pretrain',return_value={'successful_updates':1,'best_ready_checkpoint':None}),patch('gta_avt.pipeline.train') as train:
                with self.assertRaisesRegex(RuntimeError,'No attribute checkpoint'):run_pipeline(cfg)
                train.assert_not_called()



if __name__=='__main__':unittest.main()
