import torch
from torch.nn import functional as F


def dice_bce(logits, target):
    # Half-precision sums overflow above65504, easily reached by a96^3 patch.
    probability = logits.float().sigmoid()
    target = target.float()
    axes = tuple(range(2, logits.ndim))
    dice = (2. * (probability * target).sum(axes) + 1.) / (probability.sum(axes) + target.sum(axes) + 1.)
    return F.binary_cross_entropy_with_logits(logits, target) + 1. - dice.mean()


def objective(output, batch, auxiliary_weight=.1):
    segmentation = dice_bce(output['logits'], batch['mask'])
    for weight, logits in zip((.5, .25), output['aux_logits']):
        full_logits = F.interpolate(logits, size=batch['mask'].shape[2:], mode='trilinear', align_corners=False)
        segmentation = segmentation + weight * dice_bce(full_logits, batch['mask'])
    attributes = output['logits'].new_zeros(())
    details = {'segmentation': float(segmentation.detach())}
    for name in ('scale', 'structure'):
        if name + '_logits' in output:
            values = -(batch[name] * F.log_softmax(output[name + '_logits'], 1)).sum(1)
            # Geometry may support radius while a rejected junction candidate
            # leaves structure unknown. Each head needs its own denominator;
            # an all-zero target is unknown, never an empty/background class.
            valid = batch.get('attribute_valid', torch.ones_like(values)) * (batch[name].sum(1) > 0).to(values.dtype)
            loss = (values * valid).sum() / valid.sum().clamp_min(1.)
            attributes = attributes + loss
            details[name] = float(loss.detach())
    total = segmentation + auxiliary_weight * attributes
    details['total'] = float(total.detach())
    return total, details


def analytic_attribute_validity_checks():
    """Check independent head validity and mixed-batch normalization."""
    torch.manual_seed(20261022)
    checks = {}
    for mixed in (False, True):
        logits = torch.zeros(2, 1, 8, 8, 8, requires_grad=True)
        scale_logits = torch.randn(2, 5, requires_grad=True)
        structure_logits = torch.randn(2, 3, requires_grad=True)
        mask = torch.zeros_like(logits)
        mask[:, :, 2:6, 2:6, 2:6] = 1.
        structure_target = torch.zeros(2, 3)
        if mixed:
            structure_target[1] = torch.tensor([0., .6, .4])
        batch = {'mask': mask, 'scale': torch.tensor([[0., .1, .2, .3, .4]] * 2),
                 'structure': structure_target, 'attribute_valid': torch.ones(2)}
        output = {'logits': logits, 'aux_logits': [], 'scale_logits': scale_logits,
                  'structure_logits': structure_logits}
        loss, details = objective(output, batch, auxiliary_weight=.1)
        assert torch.isfinite(loss)
        expected_structure = float(-(structure_target[1] * F.log_softmax(structure_logits[1], 0)).sum().detach()) if mixed else 0.
        assert abs(details['structure'] - expected_structure) < 1e-6
        loss.backward()
        assert logits.grad.abs().sum() > 0. and scale_logits.grad.abs().sum() > 0.
        assert torch.count_nonzero(structure_logits.grad[0]) == 0
        if mixed:
            assert structure_logits.grad[1].abs().sum() > 0.
        else:
            assert torch.count_nonzero(structure_logits.grad) == 0
        checks['mixed_one_valid_structure' if mixed else 'valid_scale_unknown_structure'] = 'passed'
    return checks


if __name__ == '__main__':
    import json
    print(json.dumps(analytic_attribute_validity_checks()), flush=True)
