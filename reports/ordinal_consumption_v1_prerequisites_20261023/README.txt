Ordinal condition: actual prerequisites, not segmentation efficacy
23 October 2026. Original protocol and fresh coordinates were frozen before new predictions.
The round changes only the consumer of existing scale probabilities: c=p1+(2*p2+p3)/3.
The constant uniform-five-class reference is .4; labels, head,256 zero parameters,
backbone, common7406 start, auxiliary losses and training schedule are unchanged.

All576 fresh TRAIN patch rows are retained (36 cases x8 foreground-jitter +8 CT-uniform).
Fit24 supplies only a case-equal GT-mean constant; probe12 is not unseen by the old head.
The raw ordinal score MSE on probe12 foreground patches is .016874 versus .046205
for its fit-mean baseline (63.48% reduction); within-case correlations are positive
in12/12, with case-centered pooled r=.824835. By comparison the old fine-only p1
MSE is .017353 versus its own fine-target baseline .017020 (1.95%worse), with7/12
positive and centered r=.279016. Raw MSE across different targets is not a fair
comparison; use each score's own baseline and within-case relations.

At exactly the original55,294 cached skeleton anchors, native-derived physical
radii and full-case1mm-mask EDT yield ordinal patch-target discrepancy MSE.0007561
versus within-case target variance.0274534, ratio.0275421 below the prespecified.25.
All anchors remain foreground. The across36 case-equal radius MAE is.25436mm;
8.38% of node bin assignments differ across the grids. This checks consistency of
an operational surrogate, not expert anatomical truth or clinical diameter.

Three actual128-cube/B2/CUDA AMP engineering checks and byte-only TRAIN/VAL input
audits passed. These are prerequisites, not dynamic benefit, stability or zero-shot
results. The final constant/dynamic pairs are running under the locked protocol;
no endpoint metrics are included here. Existing VAL8/TEST12 were used previously;
new seeds cannot restore a pristine patient holdout. No new TEST arrays are read.

Run python recompute.py to independently recompute the semantic and geometry gates
using stdlib only. This does not replay models, EDT computation or patient data.
JSON/CSV files contain derived diagnostic measurements, not raw images or weights.
Original private scientific identities remain separate from future public mappings.
