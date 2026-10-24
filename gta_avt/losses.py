"""Explicit masked losses. Unknown geometry is removed before arithmetic."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_case_mean(value, valid):
    """Mean each supported case, then mean supported cases; empty => graph zero."""
    mask = valid.bool()
    safe = torch.where(mask, value, torch.zeros_like(value))
    count = mask.flatten(1).sum(1)
    means = safe.flatten(1).sum(1) / count.clamp_min(1)
    supported = count > 0
    return means[supported].mean() if supported.any() else safe.sum() * 0


def attribute_loss(output, batch, *, availability_weight=1.0, radius_weight=1.0):
    """BCE over known applicability cells plus standardized R MSE where emitted.

    A0 never supplies a radius-zero target. Each term gives supported cases equal
    weight. Applicability is operational geometry availability, not vessel class.
    """
    logits, predicted = output['a_logits'].float(), output['z_radius'].float()
    a = batch['a'].float()
    if logits.shape != a.shape or predicted.shape != batch['r'].shape:
        raise ValueError('Attribute output and half-resolution target shapes differ')
    a_valid, r_valid = batch['a_valid'].bool(), batch['r_valid'].bool()
    safe_a = torch.where(a_valid, a, torch.zeros_like(a))
    la = masked_case_mean(F.binary_cross_entropy_with_logits(logits, safe_a,
                          reduction='none'), a_valid)
    target = torch.where(r_valid, batch['r'].float(), torch.zeros_like(predicted))
    residual = torch.where(r_valid, predicted - target, torch.zeros_like(predicted))
    lr = masked_case_mean(residual.square(), r_valid)
    loss = availability_weight * la + radius_weight * lr
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite attribute loss')
    return {'loss': loss, 'availability_bce': la, 'radius_mse': lr}


def segmentation_loss(output, batch, *, dice_weight=1.0, bce_weight=1.0,
                      gdf_weight=0.0, deep_supervision_weight=0.0):
    logits = output['logits'].float()
    label, valid = batch['label'].float(), batch['valid'].bool()
    if logits.shape != label.shape:
        raise ValueError('Segmentation output and full-resolution label shapes differ')
    safe_label = torch.where(valid, label, torch.zeros_like(label))
    bce = masked_case_mean(F.binary_cross_entropy_with_logits(logits, safe_label,
                           reduction='none'), valid)
    prob = torch.where(valid, logits.sigmoid(), torch.zeros_like(logits))
    truth = safe_label
    numerator = 2 * (prob * truth).flatten(1).sum(1)
    denominator = prob.flatten(1).sum(1) + truth.flatten(1).sum(1)
    case_valid = valid.flatten(1).any(1)
    dice_per_case = 1 - (numerator + 1e-5) / (denominator + 1e-5)
    dice = dice_per_case[case_valid].mean() if case_valid.any() else logits.sum() * 0
    gdf = logits.sum() * 0
    if gdf_weight and output.get('gdf') is None:
        raise ValueError('Nonzero GDF loss requires use_gdf=True')
    if output.get('gdf') is not None:
        mask = batch['gdf_valid'].bool()
        target = torch.where(mask, batch['gdf'].float(), torch.zeros_like(logits))
        error = torch.where(mask, output['gdf'].float()-target, torch.zeros_like(logits))
        gdf = masked_case_mean(error.square(), mask)
        if not gdf_weight:
            gdf = gdf.detach()  # A lambda-zero diagnostic must not update the unused head.
    auxiliary = logits.sum() * 0
    if deep_supervision_weight:
        heads = output.get('aux_logits', [])
        if not heads:
            raise ValueError('Nonzero deep supervision needs auxiliary logits')
        for head in heads:
            head = F.interpolate(head.float(), size=label.shape[2:], mode='trilinear',
                                 align_corners=False)
            auxiliary = auxiliary + masked_case_mean(
                F.binary_cross_entropy_with_logits(head, safe_label, reduction='none'), valid)
        auxiliary = auxiliary / len(heads)
    total = dice_weight*dice + bce_weight*bce + gdf_weight*gdf + deep_supervision_weight*auxiliary
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite segmentation loss')
    return {'loss': total, 'dice_loss': dice, 'segmentation_bce': bce,
            'gdf_mse': gdf, 'auxiliary_bce': auxiliary}
