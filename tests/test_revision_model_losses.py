"""Analytic checks of the inherited segmentation and deterministic-radius objectives."""
import math

import pytest
import torch

from gta_avt.losses import attribute_loss, segmentation_loss


def test_original_dice_bce_smoothing_and_two_deep_supervision_weights():
    shape = (1, 1, 2, 2, 2)
    logits = torch.zeros(shape, requires_grad=True)
    near = torch.full(shape, math.log(3.), requires_grad=True)
    far = torch.full(shape, -math.log(3.), requires_grad=True)
    batch = dict(label=torch.ones(shape), valid=torch.ones(shape, dtype=torch.bool))
    result = segmentation_loss(dict(logits=logits, aux_logits=[near, far]), batch,
                               deep_supervision_weight=1.)
    # Eight positive voxels. Old +1 smoothing gives Dice losses 4/13,
    # 2/15 and 6/11 at probabilities .5, .75 and .25 respectively.
    main = math.log(2.) + 4./13.
    auxiliary = .5 * (math.log(4./3.) + 2./15.) + .25 * (math.log(4.) + 6./11.)
    assert float(result['dice_loss'].detach()) == pytest.approx(4./13., rel=1e-6)
    assert float(result['auxiliary_loss'].detach()) == pytest.approx(auxiliary, rel=1e-6)
    assert float(result['loss'].detach()) == pytest.approx(main + auxiliary, rel=1e-6)
    result['loss'].backward()
    assert all(float(value.grad.abs().sum()) > 0 for value in (logits, near, far))


def test_radius_mse_is_case_equal_and_excludes_cases_without_measurements():
    shape = (2, 1, 1, 1, 4)
    prediction = torch.tensor([2., 2., 2., 2., 1., 1., 9., 9.]).reshape(shape).requires_grad_()
    measured = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool).reshape(shape)
    target = torch.zeros(shape); target[~measured] = float('nan')
    output = dict(support_logits=torch.zeros(shape), radius_mean=prediction)
    batch = dict(support=torch.ones(shape), support_valid=torch.ones(shape, dtype=torch.bool),
                 r=target, r_valid=measured)
    result = attribute_loss(output, batch, support_weight=0.)
    assert float(result['radius_mse'].detach()) == pytest.approx(2.5)
    result['loss'].backward()
    assert torch.equal(prediction.grad[~measured], torch.zeros_like(prediction.grad[~measured]))
    batch['r_valid'][1] = False
    result = attribute_loss(output, batch, support_weight=0.)
    assert float(result['radius_mse'].detach()) == pytest.approx(4.)


def test_invalid_dice_or_deep_supervision_weights_are_rejected():
    shape = (1, 1, 1, 1, 1)
    output = dict(logits=torch.zeros(shape), aux_logits=[torch.zeros(shape)])
    batch = dict(label=torch.ones(shape), valid=torch.ones(shape, dtype=torch.bool))
    with pytest.raises(ValueError):
        segmentation_loss(output, batch, dice_smooth=0.)
    with pytest.raises(ValueError):
        segmentation_loss(output, batch, deep_supervision_weight=1., deep_supervision_weights=(-.5,))
