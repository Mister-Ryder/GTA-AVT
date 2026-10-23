# Completed attribute-only warm-refinement screen

This preparation route failed its prespecified entry screen. It stopped after
2,250 successful updates at the fixed validation-plateau rule and selected
step 750 by the raw minimum of monitor-A KL. The single refined guard-B
evaluation followed immutable checkpoint selection. No prepared-predictor
segmentation run was authorized or launched.

This was **attribute-only warm refinement of the existing joint classifier**,
which had already been trained within the 10,000-update joint model. It was not
independent classifier training from random initialization, a replication of
the airway training history, or proof that the original predictor had converged.

| Prespecified attribute screen | Observed result | Decision |
|---|---:|---|
| Pooled case-equal foreground ordinal MSE | 0.01645803 → 0.01566042; 4.8463% improvement | Fail: required absolute gain 0.00689766, equivalent to 41.9106% here |
| Cases with improved pooled foreground MSE | 5 / 12 | Fail: at least 8 required |
| Monitor-A KiTS foreground ordinal MSE | 0.01678432 → 0.02088805 | Fail: every source must be nonworse in each half |
| Guard-B all-strata structure KL | 0.20042984 → 0.20975022 | Fail: each head must be nonworse in each half |

The required gain was fixed as `max(10% of baseline MSE, 2 × eta_MSE)`.
`eta_MSE = 0.0034488303` is the sample SD of the twelve baseline case-wise
monitor-minus-guard differences divided by √12. It describes patch-sampling
resolution on the same cases; it is not a confidence interval. Its numerical
value was fixed before refinement. No threshold was relaxed after the result.

The two measurements serve different purposes. Checkpoint selection used the
case-equal sum of scale and structure KL over all 16 sampled slots per
case and half: eight foreground-centered patches and eight CT-uniform patches.
Each head excludes its own unknown targets. The ordinal MSE entry metric uses
the eight known-nonempty foreground patches, separately per case and half.
KL is soft-target cross-entropy minus normalized target entropy; training CE
alone does not measure attribute quality or convergence. The original geometry-derived
soft scale and structure targets, radius edges and ordinal consumer were retained;
they are operational pseudo-attributes rather than new clinical ground-truth labels.

At the selected checkpoint, monitor-A all-strata KL improved from 0.61425927
to 0.60102301. Its reduction came from empty CT-uniform blocks: the nonempty
foreground and CT-uniform contributions increased. Guard-B all-strata KL
increased. This explains the all-strata selection score; it does **not** explain
away the separately measured 4.8463% foreground ordinal-MSE improvement.
The foreground conditional-composition MSE improved modestly, while the
foreground presence-Brier error worsened. Neither result supplies a dynamic
segmentation intervention or a GTA efficacy claim.

Only the other 24 TRAIN cases supplied gradient-bearing samples. All twelve
monitor/guard cases had already been seen by the original joint classifier;
the new patch bank is development monitoring, not unseen-case validation.
The halves reuse the same twelve cases. Parameters and buffers of the location
and guidance heads stayed fixed in evaluation mode, while their input features
could change through the trained shared encoder.

The separate primary matched segmentation pair with the **old whole classifier
strictly frozen** is still running at this report's preparation time. This
attribute screen does not provide its result, establish dynamic efficacy, or
authorize additional seeds. The earlier negative ordinal-consumption round
remains in [its completed report](../ordinal_consumption_v1_VAL_20261023/README.md).

See [the Chinese interpretation](readout_zh.txt) and the [table definitions](tables/README.txt).
The seven small [CSV tables](tables/manifest.json) contain selected derived numerical measurements only. They omit CT,
labels, prediction arrays, weights, patch coordinates, feature signatures and
private execution infrastructure. They support inspection of the reported
arithmetic, not a complete model recomputation or independent proof of remote
execution. No new VAL8 or TEST evaluation was used to select this refinement.
