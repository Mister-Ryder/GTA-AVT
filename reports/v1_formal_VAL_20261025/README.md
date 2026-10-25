# GTA-AVT v1 fixed formal VAL results

Training source commit: `e17e91465022fd38f8ca1fb82a73b3c504f050d0`.

Three independent initializations (seeds 20261030, 20261031, 20261032), four matched arms, eight native VAL cases per arm: 12 trained endpoints and 96 case measurements. No new TEST evaluation is included. Case identifiers do not establish patient independence.

Each seed used 20,000 successful attribute-predictor updates, a frozen predictor, 10,000 common segmentation warmup updates, and 5,000 updates per arm. Each arm independently loaded its own seed's final common warmup with an optimizer reset. All five fixed attribute prerequisites passed for each selected predictor. A finite update budget is not a convergence claim.

| Arm | Consumed conditions | GDF loss weight |
| --- | --- | --- |
| C0 | Zero condition inputs; learned static biases allowed | 0 |
| A_only | Predicted operational applicability A | 0 |
| AR | Predicted applicability A and local log-radius R | 0 |
| AR_GDF | Same predicted A and R | 0.1 |

All arms share pretrained predictor features, backbone, guidance path and adapter capacity. These contrasts measure condition consumption and the Gaussian auxiliary within that shared design; there is no pure backbone-only comparator in this release.

## Objective native VAL measurements

Values below are percentages: equal-weight mean of the three within-seed eight-case means ± population SD of those three means. Two runs used RTX 4080 and one used vGPU-32GB; the between-run SD is descriptive across these hardware blocks, not three homogeneous-device repeats.

| Arm | Dice (%) | clDice (%) | Fine centerline recall (%) |
| --- | ---: | ---: | ---: |
| C0 | 79.7667 ± 4.5740 | 57.3324 ± 9.2947 | 44.3368 ± 1.9731 |
| A_only | 79.4893 ± 4.8196 | 57.8267 ± 9.3786 | 44.1169 ± 1.5235 |
| AR | 79.5326 ± 4.7108 | 56.5800 ± 10.0361 | 44.4315 ± 2.4545 |
| AR_GDF | 79.8579 ± 4.5055 | 57.0938 ± 10.2102 | 44.6060 ± 1.1846 |

The CSV files retain every positive, zero and negative observed difference. `native_summary.csv` records each seed's case population SD, extrema and worst-20% mean; for eight defined cases, the latter averages two cases. `paired_deltas.csv` uses to minus from on matched seed/case pairs; multiply score deltas by 100 for percentage points. Undefined metrics remain blank/null and are not replaced by zero.

Centerline metrics use native-grid Lee skeleton voxel overlap, not physical branch length. Fine centerline recall uses reference skeleton sites with native physical EDT radius ≤2 mm. FP/FN volumes are in mL. `across_seed_summary.csv` reports the mean of available seed means, their population SD, the mean within-seed case SD and mean worst-20% value; defined seed and case counts accompany every metric.

## Evaluation engineering erratum

The original three pipelines completed all six training stages and then failed the native C0 prediction/label header check. The original failed statuses remain recorded in `manifest.json`. The check used an absolute header-component tolerance of 0.00001 (origin and spacing in mm); NIfTI float32 serialization of original CT geometry exceeded that tolerance for some fields. An external evaluator accepted only exact CT header values or their exact float32 conversions after verifying all eight original CT/label grids. It reused certified original C0 predictions and called the unchanged original inference function for the remaining arms. It applied the original metric functions to the unchanged prediction arrays and original labels with the original label spacing and 0.5 threshold. No resampling, prediction voxel editing, training/model changes or score-based setting selection was performed.

External evaluator SHA256: `93900d714dc3927aa0f416509ba7309d353cc0a86f2761e6b27e6e4d03cbf9ea`. Its exact code and focused tests are in `tools/`; the immutable training capsule is identified by the source commit above. Each run explicitly has `pipeline_status=failed`, `status=complete_with_evaluation_erratum` and `external_evaluation_status=complete`.

## Files

- `native_per_case.csv`: all 96 native case measurements.
- `native_summary.csv`: per-seed, per-arm case summaries for all exported metrics.
- `paired_deltas.csv`: case and seed-mean paired contrasts.
- `across_seed_summary.csv`: descriptive summaries across the three seed means.
- `training_history.csv`: numeric update history and explicitly named sampled-VAL losses; these are not native metrics.
- `manifest.json`: fixed recipes, selected attribute readiness, training lineage, original failures and external evaluation provenance.
- `package_manifest.json`: packaged file hashes and source history hashes.
- `tools/`: exact evaluation, export, packaging and validation code.
