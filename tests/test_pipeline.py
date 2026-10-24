import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gta_avt.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_fixed_four_arm_sequence_and_matching(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls=[]
            def stage(config):
                calls.append(dict(config))
                home=Path(config['output_dir']);home.mkdir()
                return dict(successful_updates=config['max_steps'],last_checkpoint=str(home/'last.pt'),
                            best_ready_checkpoint=str(home/'best_ready.pt'))
            config=dict(manifest='not-read.json',output_dir=str(Path(temporary)/'run'),seed=30,
                        pretrain_steps=7,warmup_steps=9,adapt_steps=11,num_workers=4)
            with patch('gta_avt.pipeline.pretrain',stage),patch('gta_avt.pipeline.train',stage):
                result=run_pipeline(config)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(calls),6)
            branches=calls[2:]
            self.assertEqual([c['arm'] for c in branches],['C0','A_only','AR','AR'])
            self.assertEqual([c['lambda_gdf'] for c in branches],[0.,0.,0.,.1])
            for key in ('warmup_checkpoint','predictor_checkpoint','seed','max_steps','use_gdf','lr'):
                self.assertEqual(len({c[key] for c in branches}),1,key)
            # Explicit resume skips genuine completed stages; no duplicate job.
            with patch('gta_avt.pipeline.pretrain',side_effect=AssertionError('duplicate')):
                resumed=run_pipeline(dict(config,resume_pipeline=True))
            self.assertEqual(resumed['status'],'complete')

    def test_prerequisite_failure_never_launches_segmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=dict(manifest='not-read.json',output_dir=str(Path(temporary)/'run'),seed=30,pretrain_steps=7)
            with patch('gta_avt.pipeline.pretrain',return_value=dict(successful_updates=7,best_ready_checkpoint=None)),patch('gta_avt.pipeline.train') as train:
                with self.assertRaisesRegex(RuntimeError,'No attribute checkpoint'):
                    run_pipeline(config)
                train.assert_not_called()
            state=json.loads((Path(config['output_dir'])/'pipeline.json').read_text())
            self.assertEqual(state['status'],'failed')


if __name__=='__main__':unittest.main()
