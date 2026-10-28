# GTA-AVT v1.2 ratio-r2


## Airway-derived foreground/background ratio revision

For each case patch, pF=sum(y)/Nreal and pB=1-pF. Follow the historic airway weighting law wF=exp(pB)/(exp(pF)+exp(pB)), wB=exp(pF)/(exp(pF)+exp(pB)). BCE is the weighted soft-target logistic sum divided by wF*sum(y)+wB*sum(1-y). Thus wF/wB=exp(1-2*pF), at most e for sparse foreground. Mean-weight normalization preserves the loss scale; total class contributions are not forced equal. Padding is excluded, soft occupancy retained, and single-class patches reduce to ordinary BCE. Historical entries compute ratios per batch; AVT retains per-case outer weighting.

All formal phase configs explicitly select airway_ratio for P support, main segmentation and both DS BCE terms. Ordinary Dice, measured-radius MSE, existing region-balanced Gaussian MSE, architecture, attributes, sampler, seeds and phase budgets are unchanged. region_equal remains an explicit unused research option, not the current recipe. No power-Dice or new sampling hierarchy was added.

Historical airway code applies ratio weights in weighted/power Dice. AVT keeps its Dice+BCE objective and transfers that ratio law to BCE, rather than copying the historical numerator-only weighted Dice asymmetry. The completed original P is kept separately; the revised P starts randomly, never initialized from that control. Common VAL metrics, rather than differently defined total losses, support comparison. No efficacy improvement is claimed before AVT results.

This implementation keeps image-predicted anatomical attributes, conditional feature fusion and Gaussian geometry supervision. Its primary revision is the training system, rather than additional network modules.

The predictor emits soft lumen occupancy support and deterministic standardized physical log radius. Case-equal support Dice+BCE and measured-radius MSE supervise these outputs. Radius validity is a loss mask, not a foreground label. Production training and inference always consume CT predictions.

Training proceeds through independent attribute pretraining (30k), joint neutral-base warmup (20k), joint-base maturation (30k), local conditional adaptation (10k), Gaussian-head prewarming (2k), geometry reinforcement (20k), and lower-rate balancing (10k). These 122k successful-update defaults are a configurable starting recipe, not an optimum or convergence claim.

The neutral base jointly learns P/S using main Dice+BCE, two Dice+BCE deep-supervision heads weighted .5/.25, and .1 times the attribute objective. The selected mature base is an indivisible P/S pair. Later stages must not reload the earlier pretrained predictor.

The four original convolution branches and 576 bounded routing coefficients remain. Local stages update 115491 parameters with the predictor, backbone and normalization state frozen. Gaussian-head prewarming occurs after conditional adaptation and uses the same active conditions. Geometry refinement therefore does not reset learned condition strength.

TRAIN sampling uses stage-specific mixtures of vessel, fine-radius, supported-junction, boundary, high-intensity background and uniform anchors. Missing candidates cause explicitly logged fallbacks. Anchor proportions are not foreground voxel proportions; both source counts and actual valid-voxel foreground fractions are recorded. Validation sampling stays fixed and production inference never uses labels.

The spatial Gaussian target remains a full-case physical centreline kernel field. Foreground/background region means and cases are balanced. Its weight rises from .1 to1 over5k steps during geometry reinforcement, then falls from1 to.2 over5k steps during final balancing. Main segmentation weights remain fixed. This is geometric supervision, not a topology guarantee.

By default the pipeline stops after the mature base. Explicit stage limits allow continued work without changing its registered recipe. Only one requested variant is run; factorial controls are available but are never dispatched automatically.

The common GTA formulation is anatomical state prediction, conditional feature adaptation and geometric supervision. Airway and AVT retain their task-specific attributes, operators and experimental identities. No v1.2 AVT efficacy results are included.
