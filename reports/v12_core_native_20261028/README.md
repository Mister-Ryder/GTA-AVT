# GTA-AVT-v1.2 ratio-r2: completed fixed-seed native VAL readout

This directory contains the five planned evaluation endpoints for seed 20261032 on the same eight fixed VAL cases. No TEST cases are included.

The frozen source commit is `beab40c2b63345336c2eda8556d0caf99abad7f7`. F is the unchanged selected joint P/S base before condition and GDF stages. Its segmentation weights inherit 20,000 warm updates and the selected base update 27,000 (47,000 S updates). The base stage computed 30,000 updates; checkpoint selection and executed compute are separate quantities.

C0, AR, C0_GDF and AR_GDF use that same parent checkpoint and each execute the fixed 42,000-update tail: condition 10,000, GDF head 2,000, geometry 20,000, refine 10,000. All four final evaluations use LAST. C0 retains the predictor and attribute-feature fusion, with a learnable static gate. These are the original local-adaptation branches; the subsequently revised full-S Gaussian pair is excluded.

## Native VAL8 endpoint summaries

| Endpoint | Dice % | Across-case SD (pp) | Precision % | Recall % | clDice % | Fine centerline recall % |
|---|---:|---:|---:|---:|---:|---:|
| F | 93.055603 | 2.976870 | 92.563637 | 93.842593 | 86.325340 | 69.141034 |
| C0 | 92.941607 | 3.133809 | 92.435315 | 93.754484 | 85.607784 | 65.445656 |
| AR | 93.002647 | 3.053030 | 92.607354 | 93.686186 | 86.251096 | 64.806248 |
| C0_GDF | 92.969656 | 3.063573 | 92.443352 | 93.806186 | 86.004568 | 66.810578 |
| AR_GDF | 93.022498 | 3.058228 | 92.608566 | 93.729838 | 85.892956 | 66.347925 |

## Fixed-seed paired native contrasts

Each difference is computed within each case before aggregation. Units are percentage points. Positive/negative counts describe case-level Dice differences.

| Contrast | Mean Dice difference (pp) | Across-case SD (pp) | Positive cases | Negative cases | Equal cases |
|---|---:|---:|---:|---:|---:|
| AR-C0 | 0.061040 | 0.237373 | 4 | 4 | 0 |
| AR_GDF-C0_GDF | 0.052842 | 0.306897 | 4 | 4 | 0 |
| C0_GDF-C0 | 0.028050 | 0.099526 | 4 | 4 | 0 |
| AR_GDF-AR | 0.019851 | 0.034958 | 4 | 4 | 0 |
| interaction | -0.008198 | 0.121395 | 5 | 3 | 0 |

## Files and measurement scope

- `endpoint_per_case.csv` and `endpoint_summary.csv`: all five endpoints, per-case metrics, native support counts, means, population SDs, minima and the lowest ceil(20%) case means.
- `paired_per_case.csv` and `paired_summary.csv`: all within-case contrasts, including the interaction and final-minus-F pipeline differences.
- `objective_readout.json`: definitions, support validation and normalized source/seed/parent receipts.
- `provenance.json`: allowlisted identities and observation hashes; no machine paths or credentials.

Scores in the CSVs are percentages; differences and SDs are percentage points. SD is population SD across cases, not across seeds. This one-seed readout does not assess seed stability or supply confidence intervals. Cases receive equal weight; undefined metrics remain null and are listed. Fine centerline recall is measured at native GT Lee-skeleton sites with physical EDT radius at most 2 mm. clDice uses native Lee-skeleton voxel overlap rather than physical branch length.

These native whole-case metrics are distinct from the fixed VAL64 training-block metrics and auxiliary CPU VAL16 metrics. They must not be combined on one scale or subtracted as if the evaluation procedures were identical. F-to-final differences include the additional tail training and are not budget-matched isolated component ablations. The validation confirms reported native GT support and metric count arithmetic; it does not certify voxelwise label-file hashes.
