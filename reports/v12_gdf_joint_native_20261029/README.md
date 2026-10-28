# GTA-AVT gdf-joint-r3: completed fixed-seed native VAL8 readout

This directory contains only the unchanged F base and the two revised full-S endpoints. Old local-adaptation endpoints are not included. All eight fixed VAL cases have equal weight; no TEST cases are used.

## Native endpoint scores and system-level net change

| Endpoint | Dice % | Case SD (pp) | Dice minus F (pp) | clDice % | Fine centerline recall % |
|---|---:|---:|---:|---:|---:|
| F | 93.055603 | 2.976870 | 0.000000 | 86.325340 | 69.141034 |
| AR | 93.298116 | 2.273389 | 0.242513 | 88.413948 | 59.483125 |
| AR_GDF | 93.051245 | 2.518763 | -0.004358 | 86.883716 | 59.510969 |

F is the unchanged selected base before condition adaptation, with 47,000 segmentation-weight lineage updates. The revised pair inherits the clean AR condition checkpoint (57,000 S updates), shares one 2,000-update head-only warmup (zero S updates), and each executes geometry 20,000 plus refine 10,000 full-S updates. Each final S lineage is 87,000; total new shared-head-plus-two-arms training compute is 62,000. Both final native evaluations use the prescribed LAST checkpoint.

F-to-final is the observed system-level net change after additional training, not a budget-matched isolated component ablation. If the GDF endpoint exceeds the matched control but remains below F, that comparison alone does not establish improvement beyond the base.

## Case-paired contrasts

System-level net changes are listed first, followed by the matched Gaussian-supervision contrast. Differences are calculated within each case before aggregation.

| Contrast | Mean Dice difference (pp) | Case SD (pp) | Positive cases | Negative cases | Equal cases |
|---|---:|---:|---:|---:|---:|
| AR-F | 0.242513 | 1.183247 | 4 | 4 | 0 |
| AR_GDF-F | -0.004358 | 1.093431 | 2 | 6 | 0 |
| AR_GDF-AR | -0.246871 | 0.571987 | 2 | 6 | 0 |

## Files and measurement scope

- endpoint_per_case.csv and endpoint_summary.csv contain all three endpoints, all five metrics, native support counts, equal-case means, population SDs, minima and worst-ceil(20%) case means.
- paired_per_case.csv and paired_summary.csv contain the three revised comparisons, including fine centerline recall and the Dice signs across cases.
- objective_readout.json records definitions and validation; provenance.json contains allowlisted source, plan, checkpoint and input identities. No machine paths, credentials, images, labels or weights are copied.

Scores are percentages; SDs and paired differences are percentage points. Population SD is across cases within one seed, not across seeds. Fine centerline recall uses native GT Lee-skeleton sites with physical EDT radius at most 2 mm; clDice measures native skeleton voxel overlap, not physical branch length. Undefined supported metrics remain null and are listed rather than filled with zero.

This local aggregation checks reported GT support and metric arithmetic; it does not independently compare voxelwise label-file hashes. Fixed VAL64 training blocks and auxiliary CPU VAL16 metrics are separate from native whole-case results. GDF error reduction is not itself evidence of connectivity improvement, and these final endpoints do not identify which intermediate stage caused any change.

The run-specific [receipt validator and aggregator](../../tools/v12_gdf_joint_objective_readout.py) uses [shared native-metric arithmetic](../../tools/v12_core_objective_readout.py). It accepts a locally collected completion receipt and F readout; full machine receipts remain private. The public per-case CSVs can be inspected and summarized without model inference.
