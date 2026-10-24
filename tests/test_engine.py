"""Fast CPU checks of scientific masks, checkpoint lineage and exact resume."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gta_avt import engine
from gta_avt.losses import attribute_loss, segmentation_loss
from gta_avt.metrics import AttributeMetrics, binary_metrics


class TinyPredictor(nn.Module):
    def __init__(self,statistics):
        super().__init__();self.conv=nn.Conv3d(1,2,1);self.register_buffer('constant',torch.tensor(1.))
    def forward(self,x):
        fields=self.conv(F.avg_pool3d(x,2))
        return {'a_logits':fields[:,:1],'z_radius':fields[:,1:]}


class TinySegmentor(nn.Module):
    def __init__(self,predictor,arm='AR',use_gdf=False):
        super().__init__();self.predictor=predictor;self.predictor.requires_grad_(False)
        self.predictor.eval();self.conv=nn.Conv3d(1,1,1);self.arm=arm
    def train(self,mode=True):
        super().train(mode);self.predictor.eval();return self
    def forward(self,x,alpha=1.,condition_override=None):
        with torch.no_grad():
            attr=self.predictor(x)
        return dict(attr,logits=self.conv(x))


class TinyDataset(torch.utils.data.Dataset):
    statistics={'mu':0.,'sigma':1.,'pi':.5,'gdf_scale':1.}
    cases=[{'id':'a'},{'id':'b'}]
    def __init__(self):
        self.epoch=0
    def set_epoch(self,epoch):
        self.epoch=epoch
    def __len__(self):
        return 4
    def __getitem__(self,index):
        generator=torch.Generator().manual_seed(100*self.epoch+index)
        image=torch.randn(1,8,8,8,generator=generator)
        a=(F.avg_pool3d(image,2)>0).float();r=F.avg_pool3d(image,2)
        return dict(image=image,label=(image>0).float(),valid=torch.ones_like(image),
            a=a,r=r,a_valid=torch.ones_like(a),r_valid=a,
            gdf=torch.zeros_like(image),gdf_valid=torch.ones_like(image),case_id='ab'[index%2])


class TestScientificLosses(unittest.TestCase):
    def test_fine_centerline_uses_native_spacing_and_empty_support_is_null(self):
        target=np.zeros((11,11,11),bool)
        target[2,2:7,2:7]=True;target[8,8,8]=True
        predicted=target.copy();predicted[8,8,8]=False
        nodes=np.zeros_like(target);nodes[2,4,4]=True;nodes[8,8,8]=True
        with patch('skimage.morphology.skeletonize',lambda x,method:nodes & x):
            anisotropic=binary_metrics(predicted,target,spacing_zyx=(3.,1.,1.))
            isotropic=binary_metrics(predicted,target,spacing_zyx=(1.,1.,1.))
            target[8,8,8]=False
            no_fine=binary_metrics(predicted,target,spacing_zyx=(3.,1.,1.))
        self.assertEqual(anisotropic['fine_centerline_supported_voxels'],1)
        self.assertEqual(anisotropic['fine_centerline_recall'],0.)
        self.assertEqual(isotropic['fine_centerline_supported_voxels'],2)
        self.assertEqual(isotropic['fine_centerline_recall'],.5)
        self.assertEqual(no_fine['fine_centerline_supported_voxels'],0)
        self.assertIsNone(no_fine['fine_centerline_recall'])

    def test_cldice_booleanizes_legacy_lee_output(self):
        vessel=np.zeros((5,5,5),bool);vessel[1:4,2,2]=True
        with patch('skimage.morphology.skeletonize',lambda x,method:x.astype(np.uint8)*255):
            measured=binary_metrics(vessel,vessel)
        self.assertEqual(measured['cldice'],1.)
        self.assertEqual(measured['centerline_recall'],1.)
        self.assertEqual(measured['target_skeleton_voxels'],3)
        empty=binary_metrics(np.zeros_like(vessel),np.zeros_like(vessel))
        self.assertEqual(empty['cldice'],1.)
        self.assertIsNone(empty['centerline_recall'])
        self.assertEqual(binary_metrics(vessel,np.zeros_like(vessel))['cldice'],0.)

    def test_gdf_does_not_change_checkpoint_selection_formula(self):
        config={'dice_weight':1.,'bce_weight':1.,'deep_supervision_weight':0.}
        a={'loss':2.3,'dice_loss':.3,'segmentation_bce':.5,'auxiliary_bce':0.,'gdf_mse':15.}
        b=dict(a,loss=100.,gdf_mse=999.)
        self.assertEqual(engine._selection_score(a,config,'train'),engine._selection_score(b,config,'train'))
        self.assertEqual(engine._selection_score(a,config,'pretrain'),2.3)

    def test_unknown_radius_is_not_zero_supervision(self):
        a=torch.tensor([[[[[1.,0.]]]]]);r=torch.tensor([[[[[2.,float('nan')]]]]])
        z=torch.tensor([[[[[1.,400.]]]]],requires_grad=True)
        output={'a_logits':torch.zeros_like(a,requires_grad=True),'z_radius':z}
        batch={'a':a,'r':r,'a_valid':torch.ones_like(a),'r_valid':a}
        loss=attribute_loss(output,batch);loss['loss'].backward()
        self.assertEqual(float(loss['radius_mse']),1.)
        self.assertEqual(float(z.grad.flatten()[1]),0.)

    def test_gdf_ignores_unknown_before_math(self):
        logits=torch.zeros(1,1,1,1,2,requires_grad=True)
        gdf=torch.tensor([[[[[1.,5.]]]]],requires_grad=True)
        batch={'label':torch.zeros_like(logits),'valid':torch.ones_like(logits),
            'gdf':torch.tensor([[[[[0.,float('nan')]]]]]),'gdf_valid':torch.tensor([[[[[1.,0.]]]]])}
        loss=segmentation_loss({'logits':logits,'gdf':gdf},batch,gdf_weight=1.)
        loss['loss'].backward();self.assertEqual(float(gdf.grad.flatten()[1]),0.)

    def test_metric_constant_condition_cannot_pass_spatial_test(self):
        metrics=AttributeMetrics(.5)
        a=torch.ones(1,1,2,2,2);radius=torch.arange(8.).reshape_as(a)-3.5
        metrics.update({'a_logits':a*20,'z_radius':torch.zeros_like(a)},
            {'a':a,'r':radius,'a_valid':a,'r_valid':a,'case_id':['x']})
        report=metrics.report()
        self.assertAlmostEqual(report['per_case'][0]['case_centered_radius_ratio'],1.)
        self.assertFalse(report['readiness']['passed'])


class TestEngineLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.patches=[patch.object(engine,'AttributePredictor',TinyPredictor),
                      patch.object(engine,'GTAAVT',TinySegmentor),
                      patch.object(engine,'_datasets',lambda c:(TinyDataset(),TinyDataset()))]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def config(self,name,**changes):
        return dict(manifest='unused.json',output_dir=str(self.root/name),seed=31,device='cpu',
            patch_size=32,batch_size=2,max_steps=2,checkpoint_every=1,validate_every=1,
            amp=False,**changes)
    def test_complete_stages_and_matched_clone(self):
        p=self.config('p');engine.pretrain(p)
        predictor=str(self.root/'p'/'best.pt')
        w=self.config('warm',stage='warmup',arm='C0',predictor_checkpoint=predictor,
                      require_attribute_readiness=False)
        engine.train(w)
        warmup=str(self.root/'warm'/'last.pt')
        digests=[]
        for arm in ('C0','A_only','AR'):
            c=self.config(arm,stage='adapt',arm=arm,predictor_checkpoint=predictor,
                warmup_checkpoint=warmup,require_attribute_readiness=False)
            engine.train(c)
            checkpoint=engine.load_checkpoint(self.root/arm/'last.pt')
            self.assertEqual(checkpoint['step'],2)
            digests.append(checkpoint['lineage']['segmentation_init']['warmup_state_sha256'])
        self.assertEqual(len(set(digests)),1)
        with self.assertRaises(FileExistsError):engine.pretrain(p)
    def test_exact_resume_matches_uninterrupted(self):
        baseline=self.config('baseline');engine.pretrain(baseline)
        interrupted=self.config('interrupted');save=engine._save
        def stop_after_first(path,state):
            save(path,state)
            if path.name=='last.pt' and state['step']==1:
                raise RuntimeError('intentional test interruption')
        with patch.object(engine,'_save',stop_after_first):
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                engine.pretrain(interrupted)
        resumed=dict(interrupted,resume=str(self.root/'interrupted'/'last.pt'))
        engine.pretrain(resumed)
        a=engine.load_checkpoint(self.root/'baseline'/'last.pt')
        b=engine.load_checkpoint(self.root/'interrupted'/'last.pt')
        self.assertEqual(a['step'],b['step'])
        for name,value in a['model'].items():
            self.assertTrue(torch.equal(value,b['model'][name]),name)

    def test_prefetched_workers_preserve_batches_and_resume(self):
        # Four updates cross an epoch boundary in this picklable stateless
        # fixture. Two workers may prepare unused future samples before stop.
        baseline=self.config('serial',num_workers=0);baseline['max_steps']=4
        parallel=self.config('parallel',num_workers=2);parallel['max_steps']=4
        engine.pretrain(baseline);engine.pretrain(parallel)
        interrupted=self.config('prefetch_resume',num_workers=2);interrupted['max_steps']=4
        save=engine._save
        def stop_after_third(path,state):
            save(path,state)
            if path.name=='last.pt' and state['step']==3:
                raise RuntimeError('intentional prefetched interruption')
        with patch.object(engine,'_save',stop_after_third):
            with self.assertRaisesRegex(RuntimeError,'prefetched'):
                engine.pretrain(interrupted)
        engine.pretrain(dict(interrupted,resume=str(self.root/'prefetch_resume'/'last.pt')))
        checkpoints=[engine.load_checkpoint(self.root/name/'last.pt')
                     for name in ('serial','parallel','prefetch_resume')]
        for other in checkpoints[1:]:
            self.assertEqual(other['step'],4)
            self.assertEqual(checkpoints[0]['scheduler'],other['scheduler'])
            self.assertTrue(torch.equal(checkpoints[0]['rng']['torch'],other['rng']['torch']))
            for name,value in checkpoints[0]['model'].items():
                self.assertTrue(torch.equal(value,other['model'][name]),name)


if __name__=='__main__':
    unittest.main()
