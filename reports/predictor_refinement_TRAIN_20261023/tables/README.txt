Selected derived tables only. All MSE, KL, entropy, CE and Brier values are dimensionless.
Before=original joint classifier; prepared=fixed selected750 attribute-only warm-refined classifier.
FG8 means eight known-nonempty foreground-centered slots per case/half.
All16 means all supported foreground+CT-uniform slots; each head excludes its own unknowns.
Source and pooled-half values remain descriptive; they reuse the same12 TRAIN cases.
KL_contributions uses 1/(12 * per-case all16 valid-head slots) for each row; group contributions add to all16 case-equal KL.
These contributions are not conditional group means; absent group support contributes zero to the additive decomposition.
Head_metrics instead gives supported case-equal head means for its stated scope.
Monitor_selection covers monitor-A only; no other prepared checkpoint has a published/new guard evaluation.
CSV arithmetic is inspectable; these tables do not reproduce model inference or prove remote execution.
No CT, label/prediction arrays, weights, coordinates, feature signatures or private machine fields are included.
