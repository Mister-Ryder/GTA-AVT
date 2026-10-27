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



def _masked_bce(logits, target, valid, class_balance='none'):
    """Valid-cell BCE with optional explicit class-mass weighting.

    Soft occupancy contributes y and (1-y) mass to the two classes. Every
    supported case has equal outer weight; a missing class is not fabricated.
    Target masses are label-only constants. Padding is removed before softplus.
    ``airway_ratio`` transfers the historical bounded exponential ratio,
    normalized by its actual weighted target mass so the BCE scale is retained.
    ``region_equal`` is a separate, stronger explicit research option. Diagnostic
    ordinary BCE remains available even when weighting is selected.
    """
    if class_balance not in ('none', 'airway_ratio', 'region_equal'):
        raise ValueError('bce_class_balance must be none, airway_ratio or region_equal')
    if logits.shape != target.shape or target.shape != valid.shape:
        raise ValueError('BCE logits, targets and validity must align')
    valid = valid.bool()
    z = torch.where(valid, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    y = torch.where(valid, target.float(), torch.zeros_like(target, dtype=torch.float32))
    if not bool(torch.isfinite(y[valid]).all()) or bool(((y[valid] < 0) | (y[valid] > 1)).any()):
        raise ValueError('Valid BCE targets must be finite in [0,1]')
    positive = y
    negative = torch.where(valid, 1. - y, torch.zeros_like(y))
    foreground_mass = positive.flatten(1).sum(1)
    background_mass = negative.flatten(1).sum(1)
    counts = valid.flatten(1).sum(1)
    supported = counts > 0
    fg_present, bg_present = foreground_mass > 0, background_mass > 0
    # clamp avoids zero division only; it does not replace a small real mass.
    tiny = torch.finfo(torch.float32).tiny
    foreground_sum = (positive * F.softplus(-z)).flatten(1).sum(1)
    background_sum = (negative * F.softplus(z)).flatten(1).sum(1)
    fg_mean = foreground_sum / foreground_mass.clamp_min(tiny)
    bg_mean = background_sum / background_mass.clamp_min(tiny)
    present = fg_present.float() + bg_present.float()
    per_case = (fg_mean * fg_present + bg_mean * bg_present) / present.clamp_min(1.)
    zero = z.sum() * 0.
    balanced = per_case[supported].mean() if bool(supported.any()) else zero
    ordinary = masked_case_mean(F.binary_cross_entropy_with_logits(z, y, reduction='none'), valid)
    fraction = foreground_mass / counts.clamp_min(1)
    exp_foreground, exp_background = fraction.exp(), (1. - fraction).exp()
    weight_foreground = exp_background / (exp_foreground + exp_background)
    weight_background = exp_foreground / (exp_foreground + exp_background)
    weighted_mass = weight_foreground * foreground_mass + weight_background * background_mass
    airway_cases = (weight_foreground * foreground_sum + weight_background * background_sum) / weighted_mass.clamp_min(tiny)
    airway = airway_cases[supported].mean() if bool(supported.any()) else zero
    both = fg_present & bg_present
    diagnostics = dict(
        bce_unweighted=ordinary,
        bce_foreground=fg_mean[fg_present].mean().detach() if bool(fg_present.any()) else None,
        bce_background=bg_mean[bg_present].mean().detach() if bool(bg_present.any()) else None,
        foreground_fraction=fraction[supported].mean().detach() if bool(supported.any()) else None,
        bce_fg_to_bg_weight_ratio=(background_mass[both] / foreground_mass[both]).mean().detach()
            if class_balance == 'region_equal' and bool(both.any()) else
            (weight_foreground[both] / weight_background[both]).mean().detach()
            if class_balance == 'airway_ratio' and bool(both.any()) else
            z.new_tensor(1.) if class_balance == 'none' and bool(both.any()) else None)
    selected = balanced if class_balance == 'region_equal' else airway if class_balance == 'airway_ratio' else ordinary
    return selected, diagnostics


def attribute_loss(output, batch, *, availability_weight=1.0, radius_weight=1.0,
                   support_weight=None, bce_class_balance='none'):
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
    support_bce, bce_details = _masked_bce(safe_logits, safe_support, valid, bce_class_balance)
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
                radius_mse=radius_mse,
                **{'support_' + key: value for key, value in bce_details.items()})


def segmentation_loss(output, batch, *, dice_weight=1.0, bce_weight=1.0,
                      gdf_weight=0.0, deep_supervision_weight=0.0,
                      deep_supervision_weights=(.5, .25), dice_smooth=1.0,
                      bce_class_balance='none'):
    if not isinstance(dice_smooth, (int, float)) or not 0 < dice_smooth < float('inf'):
        raise ValueError('Dice smoothing must be finite and positive')
    logits = output['logits'].float()
    label, valid = batch['label'].float(), batch['valid'].bool()
    if logits.shape != label.shape:
        raise ValueError('Segmentation output and full-resolution label shapes differ')
    safe_label = torch.where(valid, label, torch.zeros_like(label))
    safe_logits = torch.where(valid, logits, torch.zeros_like(logits))
    bce, bce_details = _masked_bce(safe_logits, safe_label, valid, bce_class_balance)
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
    auxiliary_bce_unweighted = safe_logits.sum() * 0
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
            head_bce, head_details = _masked_bce(head, safe_label, valid, bce_class_balance)
            auxiliary_bce = auxiliary_bce + weight * head_bce
            auxiliary_bce_unweighted = auxiliary_bce_unweighted + weight * head_details['bce_unweighted']
            auxiliary_dice = auxiliary_dice + weight * _case_dice(head.sigmoid(), safe_label, valid,
                                                                smooth=dice_smooth)
    auxiliary = auxiliary_bce + auxiliary_dice
    total = dice_weight*dice + bce_weight*bce + gdf_weight*gdf + deep_supervision_weight*auxiliary
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite segmentation loss')
    return {'loss': total, 'dice_loss': dice, 'segmentation_bce': bce,
            'gdf_mse': gdf, 'auxiliary_bce': auxiliary_bce, 'auxiliary_dice': auxiliary_dice,
            'auxiliary_loss': auxiliary, 'auxiliary_bce_unweighted': auxiliary_bce_unweighted,
            **{'segmentation_' + key: value for key, value in bce_details.items()}, **gdf_details}
