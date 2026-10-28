"""Analytic class-mass, edge-case and engine wiring checks for ratio revision."""
import math
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from gta_avt import engine
from gta_avt.losses import _masked_bce, attribute_loss, segmentation_loss


def tensor(values, **kwargs):
    return torch.tensor(values, dtype=torch.float32, **kwargs).reshape(1, 1, 1, 1, -1)


def test_foreground_mass_is_unchanged_by_duplicating_identical_background():
    foreground_gradients = []
    values = []
    ordinary_values = []
    for negatives in (1, 99):
        logits = tensor([-2.] + [1.] * negatives).requires_grad_()
        target = tensor([1.] + [0.] * negatives)
        balanced, diagnostics = _masked_bce(logits, target, torch.ones_like(target), 'region_equal')
        balanced.backward()
        foreground_gradients.append(float(logits.grad.flatten()[0]))
        values.append(float(balanced.detach()))
        ordinary_values.append(float(diagnostics['bce_unweighted'].detach()))
        assert float(diagnostics['bce_fg_to_bg_weight_ratio']) == pytest.approx(negatives)
    assert values[0] == pytest.approx(values[1])
    assert foreground_gradients[0] == pytest.approx(foreground_gradients[1])
    assert ordinary_values[0] != pytest.approx(ordinary_values[1])


def test_fractional_occupancy_uses_soft_class_mass_without_thresholding():
    logits = tensor([-1., 2., .5])
    target = tensor([.125, .5, 0.])
    loss, details = _masked_bce(logits, target, torch.ones_like(target), 'region_equal')
    fg = (target * F.softplus(-logits)).sum() / target.sum()
    bg = ((1-target) * F.softplus(logits)).sum() / (1-target).sum()
    assert float(loss) == pytest.approx(float((fg+bg)/2), rel=1e-6)
    assert float(details['foreground_fraction']) == pytest.approx(.625/3)


@pytest.mark.parametrize('value', [0., 1.])
def test_single_present_class_reduces_to_ordinary_bce(value):
    logits = tensor([-2., 0., 2.]).requires_grad_()
    target = torch.full_like(logits, value)
    loss, details = _masked_bce(logits, target, torch.ones_like(target), 'region_equal')
    assert float(loss.detach()) == pytest.approx(float(F.binary_cross_entropy_with_logits(logits, target)))
    assert details['bce_fg_to_bg_weight_ratio'] is None
    loss.backward()
    assert bool(torch.isfinite(logits.grad).all())


def test_invalid_nan_is_removed_before_arithmetic_and_has_zero_gradient():
    logits = tensor([1., float('nan')]).requires_grad_()
    target = tensor([.25, float('nan')])
    valid = tensor([1., 0.])
    loss, details = _masked_bce(logits, target, valid, 'region_equal')
    loss.backward()
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.flatten()[1]) == 0.
    assert float(details['foreground_fraction']) == .25
    empty = torch.zeros_like(valid)
    loss, details = _masked_bce(logits, target, empty, 'region_equal')
    assert float(loss.detach()) == 0.
    assert details['foreground_fraction'] is None


def test_sparse_foreground_gradient_is_amplified_and_legacy_default_unchanged():
    target = tensor([1.] + [0.]*99)
    logits = torch.zeros_like(target, requires_grad=True)
    legacy, details = _masked_bce(logits, target, torch.ones_like(target))
    assert torch.equal(legacy, F.binary_cross_entropy_with_logits(logits, target))
    legacy.backward();old_gradient = float(logits.grad.flatten()[0]);logits.grad.zero_()
    balanced, _ = _masked_bce(logits, target, torch.ones_like(target), 'region_equal')
    balanced.backward()
    assert float(logits.grad.flatten()[0]) / old_gradient == pytest.approx(50.)


def test_each_case_equal_even_when_valid_voxel_counts_differ():
    logits = torch.tensor([0., 0., 0., 0., -2., 0., 0., 0.]).reshape(2,1,1,1,4)
    target = torch.ones_like(logits)
    valid = torch.tensor([1.,1.,1.,1.,1.,0.,0.,0.]).reshape_as(logits)
    loss, _ = _masked_bce(logits,target,valid,'region_equal')
    assert float(loss) == pytest.approx((math.log(2.) + float(F.softplus(torch.tensor(2.))))/2)


def test_main_and_deep_supervision_receive_same_balancing():
    shape=(1,1,2,2,2)
    target=torch.zeros(shape);target.flatten()[0]=1.
    logits=torch.zeros(shape,requires_grad=True)
    head=torch.full(shape,1.,requires_grad=True)
    output=dict(logits=logits,aux_logits=[head,head])
    result=segmentation_loss(output,dict(label=target,valid=torch.ones(shape)),
        deep_supervision_weight=1.,bce_class_balance='region_equal')
    expected,_=_masked_bce(head,target,torch.ones(shape),'region_equal')
    assert float(result['auxiliary_bce'].detach()) == pytest.approx(.75*float(expected.detach()))
    assert float(result['auxiliary_bce_unweighted'].detach()) != pytest.approx(float(result['auxiliary_bce'].detach()))
    result['loss'].backward()
    assert bool(torch.isfinite(logits.grad).all()) and bool(torch.isfinite(head.grad).all())


def test_engine_routes_balance_to_pretrain_segmentation_and_joint_attributes():
    cfg=engine._parameters(dict(manifest='unused',output_dir='unused',stage='base',
        bce_class_balance='region_equal'))
    zeros=torch.zeros(1,1,1,1,1)
    fake_seg=dict(loss=zeros.sum(),dice_loss=zeros.sum(),segmentation_bce=zeros.sum(),gdf_mse=zeros.sum())
    fake_attr=dict(loss=zeros.sum())
    with patch('gta_avt.engine.attribute_loss',return_value=fake_attr) as attr, \
         patch('gta_avt.engine.segmentation_loss',return_value=fake_seg) as seg:
        engine._loss({}, {},cfg,'pretrain')
        assert attr.call_args.kwargs['bce_class_balance']=='region_equal'
        engine._loss({}, {},cfg,'train')
        assert seg.call_args.kwargs['bce_class_balance']=='region_equal'
        assert attr.call_args.kwargs['bce_class_balance']=='region_equal'
    assert engine._parameters(dict(manifest='x',output_dir='x'))['bce_class_balance']=='none'
    with pytest.raises(ValueError):engine._parameters(dict(manifest='x',output_dir='x',bce_class_balance='invalid'))


def test_attribute_keeps_radius_mse_while_balancing_support():
    shape=(1,1,1,1,4)
    output=dict(support_logits=tensor([-1.,2.,.5,0.]),radius_mean=torch.ones(shape)*2)
    batch=dict(support=tensor([.25,0.,0.,0.]),support_valid=torch.ones(shape),
               r=torch.ones(shape),r_valid=tensor([1.,0.,0.,0.]))
    result=attribute_loss(output,batch,bce_class_balance='region_equal')
    assert float(result['radius_mse'])==1.
    assert float(result['support_bce']) != pytest.approx(float(result['support_bce_unweighted']))


def test_airway_ratio_matches_historical_exponential_soft_mass_reference():
    logits=tensor([-2.,1.,.25,-.5])
    target=tensor([.125,.5,0.,1.])
    valid=torch.ones_like(target)
    loss,details=_masked_bce(logits,target,valid,'airway_ratio')
    fraction=target.mean()
    wf=(1-fraction).exp()/(fraction.exp()+(1-fraction).exp())
    wb=fraction.exp()/(fraction.exp()+(1-fraction).exp())
    expected=(wf*(target*F.softplus(-logits)).sum()+wb*((1-target)*F.softplus(logits)).sum())/(wf*target.sum()+wb*(1-target).sum())
    assert float(loss)==pytest.approx(float(expected),rel=1e-6)
    assert float(details['bce_fg_to_bg_weight_ratio'])==pytest.approx(math.exp(1-2*float(fraction)),rel=1e-6)


def test_airway_foreground_weight_ratio_is_bounded_by_e_at_sparse_support():
    target=tensor([1e-6,0.,0.,0.])
    loss,details=_masked_bce(torch.zeros_like(target),target,torch.ones_like(target),'airway_ratio')
    ratio=float(details['bce_fg_to_bg_weight_ratio'])
    assert 1.<ratio<=math.e+1e-6
    assert ratio==pytest.approx(math.exp(1-2*float(target.mean())),rel=1e-6)
    assert float(loss)==pytest.approx(math.log(2.),rel=1e-6)


def test_airway_mean_weight_normalization_preserves_zero_logit_bce_scale():
    for fraction in (0.,.001,.025,.5,.9,1.):
        target=torch.full((2,1,1,1,8),fraction)
        loss,_=_masked_bce(torch.zeros_like(target),target,torch.ones_like(target),'airway_ratio')
        assert float(loss)==pytest.approx(math.log(2.),rel=1e-6)


def test_airway_single_class_and_invalid_nan_are_finite_with_correct_gradient():
    for value in (0.,1.):
        logits=tensor([-2.,1.,float('nan')]).requires_grad_()
        target=tensor([value,value,float('nan')]);valid=tensor([1.,1.,0.])
        loss,details=_masked_bce(logits,target,valid,'airway_ratio')
        ordinary=F.binary_cross_entropy_with_logits(logits.flatten()[:2],target.flatten()[:2])
        assert float(loss.detach())==pytest.approx(float(ordinary.detach()),rel=1e-6)
        assert details['bce_fg_to_bg_weight_ratio'] is None
        loss.backward()
        assert bool(torch.isfinite(logits.grad).all()) and float(logits.grad.flatten()[-1])==0.
    logits=tensor([float('nan')]).requires_grad_()
    loss,_=_masked_bce(logits,tensor([float('nan')]),torch.zeros_like(logits),'airway_ratio')
    loss.backward()
    assert float(loss.detach())==0. and float(logits.grad)==0.


def test_airway_engine_wiring_and_shipped_phase_defaults():
    import json
    from pathlib import Path
    cfg=engine._parameters(dict(manifest='unused',output_dir='unused',stage='base',bce_class_balance='airway_ratio'))
    zero=torch.tensor(0.)
    fake_seg=dict(loss=zero,dice_loss=zero,segmentation_bce=zero,gdf_mse=zero)
    fake_attr=dict(loss=zero)
    with patch('gta_avt.engine.attribute_loss',return_value=fake_attr) as attr, \
         patch('gta_avt.engine.segmentation_loss',return_value=fake_seg) as seg:
        engine._loss({}, {},cfg,'pretrain')
        assert attr.call_args.kwargs['bce_class_balance']=='airway_ratio'
        engine._loss({}, {},cfg,'train')
        assert seg.call_args.kwargs['bce_class_balance']=='airway_ratio'
        assert attr.call_args.kwargs['bce_class_balance']=='airway_ratio'
    root=Path(__file__).resolve().parents[1]
    phases=json.loads((root/'configs/phases.json').read_text())
    assert len(phases)==7 and all(c['bce_class_balance']=='airway_ratio' for c in phases.values())
