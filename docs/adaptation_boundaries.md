# What is being transferred from GTA?

The transferable hypothesis is that image-predicted anatomical attributes can
condition feature processing at appropriate levels. A change in organ does not
require retaining airway class names or enabling every historical component.
It does require a measurable attribute, an inference-available condition, a
specified feature operation, and a comparison that isolates the condition's
contribution.

The present implementation retains the independent attribute encoder, historical
segmentation backbone, GAAM/SAAM convolutional paths, decoder sites and final
guidance-map fusion. It changes attribute semantics and implements both hard
and soft condition arithmetic. The current reduced four-model scope does not
evaluate the hard anchor; implementation is not a completed hard-versus-soft
comparison. Random joint training is a common development
recipe, not a reproduction of the historical staged training procedure.

| Element | Permissible adaptation | Required evidence or disclosure |
|---|---|---|
| Attribute definition | Airway generation to physical vessel radius; structure classes to operational junction neighborhoods | Units, geometric construction, invalid/unknown policy and training-only reliability checks |
| Condition calculation | Hard category, composition probability or a continuous measurement | Exact formula, gradient path, and actual distribution of the consumed condition |
| Operation parameters | Kernel, dilation, channel count or modulation strength | Physical feature lattice and matched controls; a larger operation is not assumed better |
| Component set | Omit an unsupported spatial condition or independent output regularizer | Explicit model identity and limited claims about omitted components |
| Broader architecture | Local attribute fields, new encoder or different fusion sites | A separately named variant and corresponding capacity/auxiliary-supervision controls |

The framework relationship remains `CT -> predicted attributes/features ->
conditional feature operation -> segmentation`. An auxiliary classifier that
never changes segmentation features does not test this relationship. A model
requiring a reference centerline or reference tree during inference does not
provide an image-only transfer.

## Why these AVT attributes?

Radius is measured in physical space, rather than inferred from graph depth.
The training-only analysis found that changing the operational root could change
the direction of the depth-radius relationship. Truncated fields of view and
cycles in some annotation-derived graphs also limit an ideal rooted-tree model;
these graph cycles are not independently verified anatomical loops.

The scale head predicts the composition of voxel-skeleton radius samples in a patch,
with a separate empty class. The structure head predicts empty, nonjunction and
supported junction-neighborhood composition. Rejected junction candidates remain
unknown. These targets are operational geometric summaries, not clinical branch
names, true curvature or segmentation volume proportions.
Radius is the physical Euclidean distance to the mask boundary sampled at the
centerline, an inscribed-radius measure rather than a clinical cross-sectional
diameter. Composition is not weighted by physical branch arc length.

No spatial condition is active in this initial study. Image coordinates describe
the scan and can change under cropping or field-of-view changes. A tree-relative
condition would need an image-only estimated tree at inference and an explicit
training/inference consistency protocol. Neither spatial robustness nor
tree-relative localization is established here.

## What the current global condition cannot establish

Each patch shares one scale vector and one structure vector. Soft composition
preserves mixed classes, but does not locate a small branch within the patch.
The explicit additional amplitudes are `0.2*p_small_radius` and `0.1*p_junction`;
their actual values and variation must be measured from predictions. A small
ground-truth fraction alone does not establish a small predicted effect, because
sampling and segmentation gradients can change head outputs.

The `p_empty` availability term may have a larger effect than the nonempty
composition terms. The completed [same-weight availability-only
validation intervention](../reports/minimal4_VAL_diagnostics_20261023/composition_intervention/readout.txt) preserves each predicted `p_empty` and redistributes the
remaining mass uniformly. It is a sensitivity diagnostic, not another trained
model or proof of anatomical causation. Its normal-minus-intervention Dice
difference is only +0.024479 percentage points, with no positive geometry-group
mean recall change; this does not identify which component dominates.
[Measured head errors and actual consumed conditions](../reports/minimal4_VAL_diagnostics_20261023/attributes/readout.txt)
retain partial proxy associations, negative-correlation cases and unknown support.

Dense local attribute fields are a possible later GTA extension. They are not
implemented or evaluated by this global-composition study. In particular, wall
distance at every foreground voxel must not be silently renamed local vessel
radius; centerline radius projection and unknown geometry need explicit rules.

## Interpretable internal comparisons

The current primary contrasts are MAIN full soft A versus ordinary attribute
fusion C, and SECOND full-A bridge versus the same operators with constant
conditions. MAIN A-C changes specialized operations and predicted conditioning
jointly, with the same auxiliary supervision. SECOND A-constant measures overall
predicted-condition value, including empty-class mass and gradient-path changes.
Without MAIN constant, the SECOND contrast cannot decompose a MAIN gain into
condition value. Single-condition and hard-anchor jobs belonged to the original
seven-job plan and are now deferred, not completed. The two A models are platform
bridges with the same seed, not extra seeds or a platform correction. Registered
parameter equality does not imply equal active computation.

Useful evidence includes case-paired whole-case metrics, supported small-radius
and junction recall, false-positive cost, and attribute/condition diagnostics.
Low common segmentation quality must be investigated before interpreting a small
module difference. A null or adverse contrast remains a result. This protocol
does not guarantee a positive transfer, seed stability, cross-validation accuracy
or superiority to the different published AVT protocols.

The [case-independence limitation](avt_case_independence.md) applies to these
comparisons: released scan IDs are the observational units, while distinct
subjects have not been verified. Paired case summaries remain descriptive;
their bootstrap uncertainty does not establish subject-independent generalization.
