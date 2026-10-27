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


def _case_dice(probability, target, valid, smooth=1e-5):
    probability = torch.where(valid, probability, torch.zeros_like(probability))
    target = torch.where(valid, target, torch.zeros_like(target))
    numerator = 2 * (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    supported = valid.flatten(1).any(1)
    values = 1 - (numerator + smooth) / (denominator + smooth)
    return values[supported].mean() if supported.any() else probability.sum() * 0


def _region_case_mean(value, foreground, background):
    """Case equal, then equal existing foreground/background region means."""
    terms, present = [], []
    for mask in (foreground, background):
        count = mask.flatten(1).sum(1)
        safe = torch.where(mask, value, torch.zeros_like(value))
        terms.append(safe.flatten(1).sum(1) / count.clamp_min(1))
        present.append(count > 0)
    denominator = present[0].float() + present[1].float()
    per_case = (terms[0] * present[0] + terms[1] * present[1]) / denominator.clamp_min(1)
    good = denominator > 0
    return per_case[good].mean() if good.any() else torch.where(foreground | background, value, 0.).sum() * 0


def attribute_loss(output, batch, *, availability_weight=1.0, radius_weight=1.0,
                   support_weight=None):
    """Support Dice+BCE and measured-radius MSE, case equal.

    The deterministic revision changes only what is measured and consumed.
    Unknown radii are excluded before arithmetic, never labelled as zero.
    """
    logits = output['support_logits'].float()
    predicted = output['radius_mean'].float()
    support = batch['support'].float()
    valid, rv = batch['support_valid'].bool(), batch['r_valid'].bool()
    if any(value.shape != logits.shape for value in (predicted, support, valid, rv, batch['r'])):
        raise ValueError('Attribute outputs and half-resolution targets must match')
    if bool((rv & ~valid).any()):
        raise ValueError('Radius supervision cannot include unobserved cells')
    if bool(((support[valid] < 0) | (support[valid] > 1)).any()):
        raise ValueError('Support target must be a real-cell occupancy in [0,1]')
    safe_support = torch.where(valid, support, torch.zeros_like(support))
    safe_logits = torch.where(valid, logits, torch.zeros_like(logits))
    support_bce = masked_case_mean(F.binary_cross_entropy_with_logits(safe_logits, safe_support,
                                  reduction='none'), valid)
    support_dice = _case_dice(safe_logits.sigmoid(), safe_support, valid)
    mu = torch.where(valid, predicted, torch.zeros_like(predicted))
    # Eliminate unknown target values BEFORE subtracting, including NaNs.
    target = torch.where(rv, batch['r'].float(), torch.zeros_like(predicted))
    residual = torch.where(rv, mu - target, torch.zeros_like(mu))
    radius_mse = masked_case_mean(residual.square(), rv)
    sw = availability_weight if support_weight is None else support_weight
    if min(sw, radius_weight) < 0:
        raise ValueError('Attribute loss weights must be nonnegative')
    loss = sw * (support_bce + support_dice) + radius_weight * radius_mse
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite attribute loss')
    return dict(loss=loss, support_bce=support_bce, support_dice_loss=support_dice,
                radius_mse=radius_mse)


def segmentation_loss(output, batch, *, dice_weight=1.0, bce_weight=1.0,
                      gdf_weight=0.0, deep_supervision_weight=0.0,
                      deep_supervision_weights=(.5, .25), dice_smooth=1.0):
    if not isinstance(dice_smooth, (int, float)) or not 0 < dice_smooth < float('inf'):
        raise ValueError('Dice smoothing must be finite and positive')
    logits = output['logits'].float()
    label, valid = batch['label'].float(), batch['valid'].bool()
    if logits.shape != label.shape:
        raise ValueError('Segmentation output and full-resolution label shapes differ')
    safe_label = torch.where(valid, label, torch.zeros_like(label))
    safe_logits = torch.where(valid, logits, torch.zeros_like(logits))
    bce = masked_case_mean(F.binary_cross_entropy_with_logits(safe_logits, safe_label,
                           reduction='none'), valid)
    prob = torch.where(valid, safe_logits.sigmoid(), torch.zeros_like(logits))
    truth = safe_label
    numerator = 2 * (prob * truth).flatten(1).sum(1)
    denominator = prob.flatten(1).sum(1) + truth.flatten(1).sum(1)
    case_valid = valid.flatten(1).any(1)
    dice_per_case = 1 - (numerator + dice_smooth) / (denominator + dice_smooth)
    dice = dice_per_case[case_valid].mean() if case_valid.any() else safe_logits.sum() * 0
    gdf = safe_logits.sum() * 0
    gdf_details = dict(gdf_fg_mse=None, gdf_bg_mse=None, gdf_fg_voxels=0,
                       gdf_bg_voxels=0, gdf_valid_cases=0)
    if gdf_weight and output.get('gdf') is None:
        raise ValueError('Nonzero GDF loss requires use_gdf=True')
    if output.get('gdf') is not None:
        mask = batch['gdf_valid'].bool() & valid
        if output['gdf'].shape != label.shape or batch['gdf'].shape != label.shape:
            raise ValueError('GDF prediction and native training-grid targets must match')
        target = torch.where(mask, batch['gdf'].float(), torch.zeros_like(logits))
        prediction = torch.where(mask, output['gdf'].float(), torch.zeros_like(logits))
        error = prediction - target
        fg, bg = mask & (safe_label > 0), mask & (safe_label == 0)
        gdf = _region_case_mean(error.square(), fg, bg)
        gdf_details.update(gdf_fg_voxels=int(fg.sum().detach()), gdf_bg_voxels=int(bg.sum().detach()),
                           gdf_valid_cases=int(mask.flatten(1).any(1).sum().detach()))
        for name, region in (('gdf_fg_mse', fg), ('gdf_bg_mse', bg)):
            if bool(region.any()):
                gdf_details[name] = float(masked_case_mean(error.detach().square(), region))
        if not gdf_weight:
            gdf = gdf.detach()  # A lambda-zero diagnostic must not update the unused head.
    auxiliary_bce = safe_logits.sum() * 0
    auxiliary_dice = safe_logits.sum() * 0
    if deep_supervision_weight:
        heads = output.get('aux_logits', [])
        if not heads:
            raise ValueError('Nonzero deep supervision needs auxiliary logits')
        if len(deep_supervision_weights) < len(heads) or any(
                not isinstance(weight, (int, float)) or not 0 <= weight < float('inf')
                for weight in deep_supervision_weights):
            raise ValueError('Each deep-supervision head needs an explicit finite nonnegative weight')
        for weight, head in zip(deep_supervision_weights, heads):
            source_valid = F.adaptive_max_pool3d(valid.float(), head.shape[2:]) > 0
            head = torch.where(source_valid, head.float(), torch.zeros_like(head, dtype=torch.float32))
            if head.shape[2:] != label.shape[2:]:
                head = F.interpolate(head, size=label.shape[2:], mode='trilinear', align_corners=False)
            head = torch.where(valid, head, torch.zeros_like(head))
            auxiliary_bce = auxiliary_bce + weight * masked_case_mean(
                F.binary_cross_entropy_with_logits(head, safe_label, reduction='none'), valid)
            auxiliary_dice = auxiliary_dice + weight * _case_dice(head.sigmoid(), safe_label, valid,
                                                                smooth=dice_smooth)
    auxiliary = auxiliary_bce + auxiliary_dice
    total = dice_weight*dice + bce_weight*bce + gdf_weight*gdf + deep_supervision_weight*auxiliary
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite segmentation loss')
    return {'loss': total, 'dice_loss': dice, 'segmentation_bce': bce,
            'gdf_mse': gdf, 'auxiliary_bce': auxiliary_bce, 'auxiliary_dice': auxiliary_dice,
            'auxiliary_loss': auxiliary, **gdf_details}
