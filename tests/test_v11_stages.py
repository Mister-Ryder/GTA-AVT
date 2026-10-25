"""Focused v1.1 stage/schedule/selection checks; no data or GPU required."""
import tempfile
from pathlib import Path
import unittest

import torch
from gta_avt import engine
from test_engine import TinyPredictor, TinySegmentor


class StageContractTests(unittest.TestCase):
    def config(self, stage='local_adapt', **changes):
        value=engine._parameters(dict(manifest='unused',output_dir='unused',stage=stage))
        value.update(changes);return value

    def test_delayed_shape_schedule_keeps_early_conditional_learning_separate(self):
        c=self.config(lambda_gdf=.1,gdf_delay_steps=500,gdf_ramp_steps=500)
        self.assertEqual(engine._gdf_weight(c,500,'train'),0.)
        self.assertAlmostEqual(engine._gdf_weight(c,750,'train'),.05)
        self.assertAlmostEqual(engine._gdf_weight(c,1000,'train'),.1)
        self.assertAlmostEqual(engine._gdf_weight(c,2000,'train'),.1)
        self.assertEqual(engine._gdf_weight(self.config('base',lambda_gdf=9.),2000,'train'),0.)
        self.assertEqual(engine._gdf_weight(self.config('gdf_head'),1,'train'),1.)
        self.assertEqual(engine._alpha(c,250,'train'),1.)
        self.assertEqual(engine._alpha(self.config('gdf_head'),250,'train'),0.)

    def test_local_optimizer_groups_restore_distinct_constant_learning_rates(self):
        model=TinySegmentor(TinyPredictor({}));model.set_stage('local_adapt')
        optimizer=engine._optimizer(model,self.config())
        self.assertEqual({g['name']:g['lr'] for g in optimizer.param_groups},
                         dict(shared=.0001,gates=.001,gdf=.0001))
        expected={id(v) for v in model.parameters() if v.requires_grad}
        self.assertEqual({id(v) for g in optimizer.param_groups for v in g['params']},expected)
        model.train();self.assertFalse(model.backbone[1].training)
        self.assertFalse(model.predictor.training)

    def test_head_only_fit_preserves_entire_base_state(self):
        model=TinySegmentor(TinyPredictor({}));model.set_stage('gdf_head')
        frozen=engine._frozen_digest(model)
        optimizer=engine._optimizer(model,self.config('gdf_head'))
        self.assertEqual([g['name'] for g in optimizer.param_groups],['gdf'])
        x=torch.randn(2,1,8,8,8);output=model(x)
        output['gdf'].square().mean().backward();optimizer.step()
        self.assertEqual(engine._frozen_digest(model),frozen)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_checkpoint_schema_rejects_old_v1_even_if_weights_have_similar_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'old.pt';torch.save({'schema':'GTA-AVT-v1-checkpoint-1'},path)
            with self.assertRaisesRegex(ValueError,'v1.1 checkpoint'):engine.load_checkpoint(path)

    def test_optional_diagnostics_nulls_survive_loss_record(self):
        record=engine._loss_record({'loss':torch.tensor(1.),'gdf_fg_mse':None,'gdf_fg_voxels':0})
        values,counts=engine._mean_records([record])
        self.assertIsNone(values['gdf_fg_mse']);self.assertEqual(counts['gdf_fg_mse'],0)
        self.assertEqual(values['gdf_fg_voxels'],0)


if __name__=='__main__':unittest.main()
