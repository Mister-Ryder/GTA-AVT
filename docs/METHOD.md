# GTA-AVT v1.2

This implementation keeps image-predicted anatomical attributes, conditional feature fusion and Gaussian geometry supervision. Its primary revision is the training system, rather than additional network modules.

The predictor emits soft lumen occupancy support and deterministic standardized physical log radius. Case-equal support Dice+BCE and measured-radius MSE supervise these outputs. Radius validity is a loss mask, not a foreground label. Production training and inference always consume CT predictions.

Training proceeds through independent attribute pretraining (30k), joint neutral-base warmup (20k), joint-base maturation (30k), local conditional adaptation (10k), Gaussian-head prewarming (2k), geometry reinforcement (20k), and lower-rate balancing (10k). These 122k successful-update defaults are a configurable starting recipe, not an optimum or convergence claim.

The neutral base jointly learns P/S using main Dice+BCE, two Dice+BCE deep-supervision heads weighted .5/.25, and .1 times the attribute objective. The selected mature base is an indivisible P/S pair. Later stages must not reload the earlier pretrained predictor.

The four original convolution branches and 576 bounded routing coefficients remain. Local stages update 115491 parameters with the predictor, backbone and normalization state frozen. Gaussian-head prewarming occurs after conditional adaptation and uses the same active conditions. Geometry refinement therefore does not reset learned condition strength.

TRAIN sampling uses stage-specific mixtures of vessel, fine-radius, supported-junction, boundary, high-intensity background and uniform anchors. Missing candidates cause explicitly logged fallbacks. Anchor proportions are not foreground voxel proportions; both source counts and actual valid-voxel foreground fractions are recorded. Validation sampling stays fixed and production inference never uses labels.

The spatial Gaussian target remains a full-case physical centreline kernel field. Foreground/background region means and cases are balanced. Its weight rises from .1 to1 over5k steps during geometry reinforcement, then falls from1 to.2 over5k steps during final balancing. Main segmentation weights remain fixed. This is geometric supervision, not a topology guarantee.

By default the pipeline stops after the mature base. Explicit stage limits allow continued work without changing its registered recipe. Only one requested variant is run; factorial controls are available but are never dispatched automatically.

The common GTA formulation is anatomical state prediction, conditional feature adaptation and geometric supervision. Airway and AVT retain their task-specific attributes, operators and experimental identities. No v1.2 AVT efficacy results are included.
