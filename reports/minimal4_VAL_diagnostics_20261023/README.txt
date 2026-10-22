Validation diagnostic reporting candidate (not yet a public release)

These measured diagnostics accompany the four normal validation models so that
small, unfavorable and mixed mechanism evidence remains visible. Validation was
already used for selection; these are development diagnostics, not independent
TEST confirmation. No TEST results are included or used in this candidate.

Composition intervention
  composition_intervention/readout.txt explains all 10 metrics, 6 GT-recall groups
  and all 8 cases. The original 7 CSV numerical tables preserve every result,
  support and null. numerical_summary.json copies the original estimates and
  exploratory case-bootstrap intervals without adding statistical calculations.
  evaluations/availability_only contains 8 original metrics plus original summary.
  The already published normal bridge 9 files are linked by relative paths and
  exact hashes in composition_intervention/declaration.json; they are not
  duplicated here. Once installed under reports/minimal4_VAL_diagnostics_20261023,
  their report directory is ../minimal4_reference_VAL_20261023/evaluations/
  ablations_A_soft_platform_bridge/.

  Main finding: normal-minus-availability-only Dice is only +0.024479 pp;
  all 8 cases reduce FP but increase FN, and no GT geometry-group recall mean
  improves. Availability-only retains each predicted p_empty and replaces only
  the remaining composition. It is not a newly trained constant model.

Classifier attributes
  attributes/readout.txt explains the 4 strata separately. case_scores/ contains
  the 4 byte-identical original score JSON files (512 case-score rows total).
  aggregates.json preserves 64 aggregate rows, including predicted and actual
  consumed conditions. support.json has 4 stratum summaries and 32 shared
  case-stratum records (8 cases × 4 strata), not 32 additional cases. ranges.json
  includes descriptive ranges and missing-correlation reasons.

  There are 160 shared patches TOTAL across 8 cases, not 160 per case;4 models produce
  640 classifier samples. CT_uniform has 64 patches, with 57 valid targets per head.
  Its 7 foreground patches without centerline support are unknown/ignored, not
  empty background. Each GT-enriched layer has 32 patches; enrichment is not
  natural prevalence/calibration. C consumes no composition probabilities;
  constant consumes uniform vectors despite having trained predictive heads.
  Both A models consume their predictions. Partial association and visible
  variation do not establish reliable anatomical meaning or segmentation gain.

  MAE/SSE are probability-vector errors, not CE/entropy or gradient measures.
  Spearman is within-case descriptive; undefined values and reasons are kept.
  Legacy patient_* fields and old aggregation wording denote case IDs/summaries;
  independent patients have not been verified. No correlation p-values or new
  intervals are added. Overlapping patches/support are not independent samples.

Evidence and privacy boundary
  declaration.json discloses the 4 declared model identities, selected steps,
  weights/source/data/split hashes and a safe software/device runtime projection.
  Composition declarations also retain original normal/availability protocol
  FILE hashes. Hashes bind disclosed files/projections to author artifacts;
  they do not independently prove physical servers, executed weights, data-byte
  identity, authorization or experimental chronology. Original full protocols,
  private index/after proofs, raw banks, per-patch inputs/predictions, CT input
  hashes, images, labels, masks, native paths and geometry arrays are omitted.

  Neither diagnostic is subtracted from trained-A versus trained-constant or
  pooled with the other training platform. Neither identifies that p_empty or
  extra operators dominate, nor that conditioning is useless. No change to
  weights, thresholds, checkpoint selection or the existing protocol follows
  from this package.

Inventory
  candidate_manifest.json lists every payload file hash and its source artifact
  hashes. It excludes its own hash to avoid recursion; the authors record that FILE
  hash separately. References to external normal metrics are resolved from
  composition_intervention/declaration.json. This candidate was prepared locally;
  it is not an evaluation controller, new helper, private freeze marker or proof
  of publication. The authors review it before any public-tree copy or push.
