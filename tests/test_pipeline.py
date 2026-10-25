import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gta_avt.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_fixed_four_arm_sequence_and_matching(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            def stage(config):
                calls.append(dict(config))
                home = Path(config['output_dir']); home.mkdir()
                return dict(successful_updates=config['max_steps'], last_checkpoint=str(home/'last.pt'),
                            best_checkpoint=str(home/'best.pt'), best_ready_checkpoint=str(home/'best_ready.pt'))
            config = dict(manifest='not-read.json', output_dir=str(Path(temporary)/'run'), seed=30,
                          pretrain_steps=7, base_steps=9, gdf_head_steps=3, local_steps=11, num_workers=4)
            with patch('gta_avt.pipeline.pretrain', stage), patch('gta_avt.pipeline.train', stage):
                result = run_pipeline(config)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(calls), 7)
            self.assertEqual([call['stage'] for call in calls[:3]], ['pretrain', 'base', 'gdf_head'])
            self.assertEqual(Path(calls[2]['base_checkpoint']).parts[-2:], ('base', 'best.pt'))
            branches = calls[3:]
            self.assertEqual([call['arm'] for call in branches], ['C0', 'AR', 'C0', 'AR'])
            self.assertEqual([call['lambda_gdf'] for call in branches], [0., 0., .1, .1])
            for key in ('gdf_head_checkpoint', 'predictor_checkpoint', 'seed', 'max_steps', 'use_gdf',
                        'local_shared_lr', 'local_gate_lr', 'local_gdf_lr', 'validation_seed'):
                self.assertEqual(len({call[key] for call in branches}), 1, key)
            self.assertEqual(branches[0]['validation_seed'], 0)
            with patch('gta_avt.pipeline.pretrain', side_effect=AssertionError('duplicate')):
                resumed = run_pipeline(dict(config, resume_pipeline=True))
            self.assertEqual(resumed['status'], 'complete')

    def test_prerequisite_failure_never_launches_segmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = dict(manifest='not-read.json', output_dir=str(Path(temporary)/'run'), seed=30)
            with patch('gta_avt.pipeline.pretrain', return_value=dict(successful_updates=20000,
                    best_ready_checkpoint=None)), patch('gta_avt.pipeline.train') as train:
                with self.assertRaisesRegex(RuntimeError, 'No attribute checkpoint'):
                    run_pipeline(config)
                train.assert_not_called()
            state = json.loads((Path(config['output_dir'])/'pipeline.json').read_text())
            self.assertEqual(state['status'], 'failed')

    def test_native_eval_includes_source_ct_and_declared_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'cases.json').write_text(json.dumps({'cases':[
                dict(id='x', image='ct.nrrd', label='gt.nrrd', split='VAL')]}))
            def stage(config):
                home=Path(config['output_dir']);home.mkdir()
                return dict(successful_updates=config['max_steps'],last_checkpoint=str(home/'last.pt'),
                    best_checkpoint=str(home/'best.pt'),best_ready_checkpoint=str(home/'best_ready.pt'))
            def infer(config):
                self.assertEqual(Path(config['checkpoint']).name, 'best.pt')
                self.assertNotIn('label',config)
                return dict(prediction='prediction.nii.gz')
            def evaluate(config):
                self.assertEqual(config['cases'][0]['image'],'ct.nrrd')
                self.assertEqual(config['cases'][0]['label'],'gt.nrrd')
                return {'per_case':[]}
            config=dict(manifest='not-read',output_dir=str(root/'run'),seed=30,
                evaluation_cases=str(root/'cases.json'),evaluation_endpoint='best')
            with patch('gta_avt.pipeline.pretrain',stage),patch('gta_avt.pipeline.train',stage),\
                 patch('gta_avt.pipeline.infer',infer),patch('gta_avt.pipeline.evaluate',evaluate):
                result=run_pipeline(config)
            self.assertEqual(len(result['completed']),11)


if __name__ == '__main__': unittest.main()
