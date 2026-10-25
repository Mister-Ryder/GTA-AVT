# GTA-AVT v1.1 method

This is a new algorithm candidate with no v1.1 dataset results. Existing v0.1
and v1 measurements motivate the design and are never relabelled.

## Reliability-aware anatomical attributes

Keep the TRAIN-standardized physical log-radius target on geometrically measurable
sites. Old operational applicability becomes the loss-only `r_valid` mask. The CT
predictor outputs `support_logits`, `radius_mean`, `radius_log_variance`, and its
original multiscale features. Support is supervised by lumen occupancy in
half-resolution cells with case-equal Dice+BCE. It is a support score, not claimed
calibrated probability. Radius uses masked Gaussian heteroscedastic NLL. Unknown
real foreground/background receive separately normalized weak KL-to-standard-normal
prior regularization, rather than fabricated measured radius-zero observations.

For support score v, standardized mean m and log variance s:

    q = relu((1-exp(s))/(1+exp(s)))
    cV = v
    cR = v*q*clamp(m,-4,4).

q is relative informativeness against the TRAIN reference. Calibration needs
empirical evaluation; the clipping bound is an explicit numerical safeguard.

## Bounded local conditional fusion

Preserve four original branches (dilation 1/1/2/3), their three bounded contrasts
and 576 zero-initialized coefficients:

    gj = tanh(bj + uj*cV + wj*cR)
    F = F0 + alpha*g1*D1 + alpha*(1-|g1|)*(g2*D2+g3*D3).

Equivalent branch coefficients are in [0,2] and sum to four in real arithmetic.
A static control consumes zero coordinates while retaining the same attribute
features and global biases. No hard applicability threshold vetoes conditioning.
Normal training and inference consume CT predictions, never label-derived inputs.

## Gaussian geometry supervision and staged adaptation

Keep v1's full-case physical Gaussian target, but normalize its squared error
separately within valid foreground/background and equally across present regions
and cases. Prepare held-out foreground fields too; fit statistics only on TRAIN.

The segmentation and Softplus Gaussian heads share the final 32-channel feature.
After attribute pretraining, train a neutral-condition segmentation base. Freeze
it to prewarm only the Gaussian head. Then freeze the backbone, remaining layers
and BN/dropout while training only local branches, fusion projection, 576 route
coefficients, final segmentation projection and Gaussian head: 115491 parameters.
Frozen layers pass input Jacobians, allowing Gaussian supervision to affect
conditional routing. Use separate gate/shared learning rates and delay/ramp GDF
loss after the conditional warmup. Choose segmentation checkpoints primarily by
validation Dice; report GDF reconstruction separately.

C0/AR/C0_GDF/AR_GDF form the minimal matched 2x2 design, but do not by themselves
identify attribute pretraining's segmentation benefit against a plain backbone.
The common GTA principle is anatomical attributes -> conditional feature
increments -> geometric supervision. Original airway operators retain their own
implementation/evidence; no new-operator, Dice or topology guarantee is inferred.

## Related methodological foundations

Input-dependent regression uncertainty has established precedents, including
[Kendall and Gal (2017)](https://arxiv.org/abs/1703.04977). Conditional feature
modulation also predates this method, including
[FiLM](https://arxiv.org/abs/1709.07871). This design does not claim to invent
either general technique. Its intended contribution is task-grounded separation
of anatomical support, measurable scale and relative reliability, their bounded
local consumption, and staged coupling to Gaussian geometry supervision. The
new combination and its segmentation effect require evaluation.
