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

    def test_phase_schedules_preserve_conditioning_then_strengthen_and_refine_geometry(self):
        condition=self.config(phase='condition')
        self.assertEqual(engine._alpha(condition,250,'train'),1.)
        self.assertEqual(engine._gdf_weight(condition,10000,'train'),0.)
        head=self.config('gdf_head',phase='gdf_head')
        self.assertEqual(engine._alpha(head,1,'train'),1.)
        self.assertEqual(engine._gdf_weight(head,1,'train'),1.)
        geometry=self.config(phase='geometry',gdf_weight_start=.1,gdf_weight_end=1.,gdf_ramp_steps=5000)
        self.assertEqual(engine._alpha(geometry,1,'train'),1.)
        self.assertAlmostEqual(engine._gdf_weight(geometry,1,'train'),.1)
        self.assertAlmostEqual(engine._gdf_weight(geometry,5000,'train'),1.)
        refine=self.config(phase='refine',gdf_weight_start=1.,gdf_weight_end=.2,gdf_ramp_steps=5000)
        self.assertEqual(engine._alpha(refine,1,'train'),1.)
        self.assertAlmostEqual(engine._gdf_weight(refine,1,'train'),1.)
        self.assertAlmostEqual(engine._gdf_weight(refine,5000,'train'),.2)
        self.assertEqual(engine._gdf_weight(self.config('base',phase='base'),99,'train'),0.)

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
            with self.assertRaisesRegex(ValueError,'GDF-joint checkpoint'):engine.load_checkpoint(path)

    def test_optional_diagnostics_nulls_survive_loss_record(self):
        record=engine._loss_record({'loss':torch.tensor(1.),'gdf_fg_mse':None,'gdf_fg_voxels':0})
        values,counts=engine._mean_records([record])
        self.assertIsNone(values['gdf_fg_mse']);self.assertEqual(counts['gdf_fg_mse'],0)
        self.assertEqual(values['gdf_fg_voxels'],0)


if __name__=='__main__':unittest.main()
