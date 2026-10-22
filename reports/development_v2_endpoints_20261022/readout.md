# Original-v2 first two endpoints: development validation only

Snapshot: **22 October 2026, 16:10 UTC**. These are the two genuine first jobs from the original v2 plan, each completing 10,000 successful updates and full VAL8 at updates 5,000 and 10,000. They retain their original version and have not been retrained or relabeled as new reduced-scope jobs. The original seven-job queues were administratively closed before their other five jobs started. MAIN A and SECOND constant-A have now started under the [unchanged-recipe four-model scope](../../docs/minimal4_scope_20261022.md); their results are not present here.

**No fair GTA conclusion or TEST result:** C and the A bridge trained on different roles/hardware. Their difference is not an algorithm effect. Both use auxiliary attribute supervision; C is not a single-task baseline. These are original training-time validation outputs, not the final common-reference evaluations. Original VAL backend flags were not independently recorded completely.

All **32 original case/timepoint metrics and four original summaries** are copied byte for byte under `main/C_concat/` and `ablations/A_soft_platform_bridge/`. [The source manifest](../stage_source_manifest_20261023.json) binds each file SHA. [casewise.csv](casewise.csv) gives the same 32 rows with raw probability fractions, ml volumes, mm distances and geometry support. [summary.json](summary.json) retains endpoint means, sample SDs, missing/support counts, all within-job pairs and same-job V1 associations. No CPU-receipt contents, private execution paths, predictions, images or weights are distributed.

## Complete means and case dispersion

Probabilities below are percent; FP/FN are ml and distances mm. Entries are mean ± sample SD across eight released case IDs. All ten metrics are defined for all eight cases at every timepoint. Subject independence is unverified; these SDs are case dispersion, not uncertainty across seeds. No CI or p-value is computed.

| Metric | MAIN C 5,000 | MAIN C 10,000 | SECOND A bridge 5,000 | SECOND A bridge 10,000 |
|---|---|---|---|---|
| dice | 65.693 ± 7.584 | 80.380 ± 8.980 | 66.987 ± 9.115 | 85.949 ± 5.307 |
| precision | 64.032 ± 14.626 | 75.292 ± 13.621 | 58.203 ± 13.669 | 85.596 ± 8.032 |
| recall | 73.256 ± 15.802 | 88.097 ± 7.232 | 82.716 ± 8.430 | 86.953 ± 6.953 |
| fp_ml | 172.110 ± 131.548 | 127.364 ± 116.236 | 224.924 ± 100.172 | 59.419 ± 40.440 |
| fn_ml | 115.117 ± 106.700 | 47.145 ± 35.879 | 69.399 ± 45.636 | 52.423 ± 36.421 |
| assd_mm | 29.226 ± 9.450 | 13.539 ± 7.181 | 28.745 ± 10.300 | 8.486 ± 4.790 |
| hd95_mm | 129.104 ± 18.678 | 92.999 ± 23.667 | 123.515 ± 21.102 | 75.654 ± 35.925 |
| centerline_recall | 63.238 ± 13.035 | 79.507 ± 12.396 | 72.676 ± 11.494 | 77.874 ± 10.964 |
| centerline_precision | 50.234 ± 30.390 | 54.108 ± 13.602 | 35.430 ± 17.937 | 65.222 ± 17.726 |
| cldice | 51.391 ± 20.828 | 63.023 ± 11.271 | 46.087 ± 15.623 | 70.360 ± 14.517 |

## Every endpoint case

Each row is the 10,000-update endpoint. Probability metrics are percent. All 5,000-update rows remain in the byte-identical source files and CSV.

| Role/job | Case | Dice | Precision | Recall | FP ml | FN ml | CL precision | CL recall | clDice |
|---|---|---|---|---|---|---|---|---|---|
| main/C_concat | Dongyang_D2 | 78.122 | 84.503 | 72.638 | 60.653 | 124.587 | 63.775 | 77.158 | 69.831 |
| main/C_concat | Dongyang_D5 | 85.116 | 78.336 | 93.180 | 41.722 | 11.042 | 53.064 | 78.939 | 63.465 |
| main/C_concat | KiTS_K13 | 66.125 | 52.078 | 90.547 | 316.307 | 35.885 | 28.278 | 91.094 | 43.159 |
| main/C_concat | KiTS_K15 | 84.087 | 83.809 | 84.367 | 42.043 | 40.323 | 57.881 | 68.524 | 62.755 |
| main/C_concat | KiTS_K18 | 85.999 | 78.720 | 94.760 | 98.106 | 20.070 | 49.974 | 87.921 | 63.726 |
| main/C_concat | KiTS_K8 | 67.726 | 56.120 | 85.385 | 310.029 | 67.867 | 47.621 | 56.150 | 51.535 |
| main/C_concat | Rider_R10 | 90.606 | 89.305 | 91.946 | 67.611 | 49.455 | 75.370 | 83.152 | 79.070 |
| main/C_concat | Rider_R9 | 85.255 | 79.468 | 91.951 | 82.442 | 27.931 | 56.903 | 93.118 | 70.639 |
| ablations/A_soft_platform_bridge | Dongyang_D2 | 82.245 | 95.940 | 71.972 | 13.867 | 127.618 | 68.298 | 70.610 | 69.435 |
| ablations/A_soft_platform_bridge | Dongyang_D5 | 92.248 | 91.401 | 93.112 | 14.183 | 11.153 | 75.923 | 76.783 | 76.350 |
| ablations/A_soft_platform_bridge | KiTS_K13 | 80.947 | 76.737 | 85.646 | 98.567 | 54.492 | 58.562 | 78.918 | 67.233 |
| ablations/A_soft_platform_bridge | KiTS_K15 | 79.427 | 74.771 | 84.700 | 73.718 | 39.465 | 43.581 | 76.741 | 55.592 |
| ablations/A_soft_platform_bridge | KiTS_K18 | 91.030 | 88.128 | 94.129 | 48.566 | 22.486 | 71.188 | 90.594 | 79.727 |
| ablations/A_soft_platform_bridge | KiTS_K8 | 83.201 | 79.942 | 86.737 | 101.062 | 61.592 | 36.463 | 56.606 | 44.355 |
| ablations/A_soft_platform_bridge | Rider_R10 | 86.005 | 83.695 | 88.446 | 105.799 | 70.945 | 82.474 | 82.722 | 82.598 |
| ablations/A_soft_platform_bridge | Rider_R9 | 92.490 | 94.152 | 90.885 | 19.588 | 31.629 | 85.284 | 90.018 | 87.587 |

## All within-job changes and V1 counterexamples

Positive probability differences are percentage points, volume differences ml. These are descriptive same-job comparisons. V1-to-v2 changed training patch 96→128, update budget/poly schedule 5,000→10,000, validation timing and SECOND software; they cannot isolate any one factor. Equal seeds do not replay cross-version patches or augmentations.

| Role | Case | V2 10k−5k Dice | V2 FP | V2 FN | V2 clDice | V2 10k−V1 5k Dice | vs V1 FP | vs V1 FN |
|---|---|---|---|---|---|---|---|---|
| main | Dongyang_D2 | 16.508 | -10.502 | -96.327 | 9.857 | 7.532 | 3.815 | -51.362 |
| main | Dongyang_D5 | 19.205 | -69.985 | -16.372 | -2.340 | 7.142 | -29.631 | -1.815 |
| main | KiTS_K13 | 11.789 | -126.399 | -36.996 | 9.097 | -13.606 | 216.624 | -25.992 |
| main | KiTS_K15 | 18.444 | -91.832 | -26.188 | 24.953 | 10.737 | -68.600 | -4.149 |
| main | KiTS_K18 | 14.426 | -107.873 | -34.689 | 26.904 | 10.266 | -102.271 | -7.402 |
| main | KiTS_K8 | 0.248 | 41.545 | -23.347 | 27.680 | 5.321 | -8.778 | -41.301 |
| main | Rider_R10 | 30.865 | 27.150 | -285.803 | 7.600 | 25.362 | -51.310 | -209.700 |
| main | Rider_R9 | 6.004 | -20.069 | -24.049 | -10.696 | -0.463 | 7.476 | -2.576 |
| ablations | Dongyang_D2 | 14.761 | -100.526 | -37.575 | 28.028 | 9.684 | -24.190 | -46.779 |
| ablations | Dongyang_D5 | 40.374 | -244.384 | -3.500 | 43.552 | 27.668 | -133.318 | -3.199 |
| ablations | KiTS_K13 | 24.293 | -317.828 | -10.531 | 30.839 | 9.082 | -99.380 | -1.200 |
| ablations | KiTS_K15 | 11.594 | -69.561 | -12.561 | 14.929 | 10.016 | -49.235 | -16.025 |
| ablations | KiTS_K18 | 20.824 | -182.571 | -28.325 | 30.335 | 22.616 | -231.055 | -16.006 |
| ablations | KiTS_K8 | 15.465 | -200.342 | -10.606 | 12.859 | 26.443 | -361.822 | -35.374 |
| ablations | Rider_R10 | 4.960 | -39.046 | -26.061 | 4.879 | 19.170 | -32.792 | -165.344 |
| ablations | Rider_R9 | 19.427 | -169.781 | -6.650 | 28.758 | 6.553 | -51.632 | -0.279 |

Both jobs improve all eight Dice values from their own 5,000-update point. MAIN C improves 14.686 pp, but K8 improves only 0.248 pp while FP increases 41.545 ml. Its same-job V1-to-v2 mean improves 6.536 pp (six cases improve, two decline), yet **K13 falls 13.606 pp and FP increases 216.624 ml**; R9 also falls 0.463 pp. R10 contributes a large FN reduction. Mean improvement does not remove these counterexamples.

The bridge improves 18.962 pp from its own v2 midpoint and 16.404 pp over its own V1 endpoint, with all eight Dice values increasing in both comparisons. Relative to V1, D5/K8/K18 supply 58.467% of the sum of case Dice increments and 73.843% of total FP-volume reduction. R10 supplies 58.177% of total FN-volume reduction. Absolute ml contributions naturally depend on case volume; these are descriptive decompositions, not causal effects or independent samples.

At the bridge endpoint, D2 still has recall 71.972% and FN 127.618 ml. K8 has Dice 83.201% but CL precision 36.463%, CL recall 56.606% and clDice 44.355%. K15 has the lowest Dice, 79.427%, and HD95 134.967 mm; its HD95 improves only 0.008 mm from V1. K13/K8/R10 retain FP volumes near or above 100 ml. Native-v2 endpoint predictions were not saved, so these metrics do not locate a named anatomical trunk or demonstrate visual completeness.

## Fine-radius regressions despite aggregate improvement

Group means are equal-case GT-centerline recall, not group precision. GT support for each case/group is unchanged across these timepoints and versions. No skeleton node or patch adds a case.

| Role | Group | Own V1 endpoint recall % | V2 10k recall % | Change pp |
|---|---|---|---|---|
| main | radius_le_2mm | 50.321 | 45.401 | -4.920 |
| main | radius_2_to_4mm | 72.511 | 67.991 | -4.520 |
| main | radius_4_to_8mm | 77.209 | 82.928 | 5.719 |
| main | radius_gt_8mm | 88.377 | 97.536 | 9.159 |
| main | nonjunction | 76.821 | 78.647 | 1.826 |
| main | junction_neighbourhood | 81.326 | 86.473 | 5.147 |
| ablations | radius_le_2mm | 53.983 | 42.897 | -11.086 |
| ablations | radius_2_to_4mm | 73.151 | 66.797 | -6.353 |
| ablations | radius_4_to_8mm | 76.794 | 83.697 | 6.902 |
| ablations | radius_gt_8mm | 87.450 | 94.682 | 7.232 |
| ablations | nonjunction | 77.779 | 77.198 | -0.582 |
| ablations | junction_neighbourhood | 78.002 | 83.457 | 5.455 |

For the bridge, <=2 mm recall falls **53.983→42.897% (−11.086 pp)** and 2–4 mm falls **73.151→66.797% (−6.353 pp)**; each declines in six of eight cases. D2 <=2 mm falls 41.475→18.894%; K8 ends at 22.581% for <=2 mm and 19.366% for 2–4 mm. Overall bridge CL recall increases only 1.165 pp over V1, whereas CL precision increases 22.072 pp and clDice 17.039 pp. The observed improvement is largely FP/centerline-precision cleanup plus middle/large-scale coverage, **not evidence that fine-vessel recall improved**. MAIN C also loses 4.920 pp in <=2 mm and 4.520 pp in 2–4 mm recall relative to its own V1 endpoint.

Radius is physical inscribed EDT radius at GT skeleton nodes, not a clinical diameter or named vessel class. The <=2 mm stratum has only 19–537 supported nodes per case. Lee coverage is a voxel approximation and may omit branches; neither it nor clDice establishes arc-length detection, complete topology, connectivity or all branches. Operational junction labels are not validated bifurcation annotations. Depth, curvature and tree-relative space are not all answered by this study.

## Optimization evidence and remaining limits

- main/C_concat: last 1,000 successful updates have mean segmentation loss 0.602, weighted auxiliary loss 0.169 (ratio 0.282), AMP retries 1 and preclip norm >12 on 31.300% of steps. These scalar losses do not establish convergence or gradient conflict; soft-target CE includes a target-entropy floor.
- ablations/A_soft_platform_bridge: last 1,000 successful updates have mean segmentation loss 0.590, weighted auxiliary loss 0.172 (ratio 0.291), AMP retries 0 and preclip norm >12 on 28.500% of steps. These scalar losses do not establish convergence or gradient conflict; soft-target CE includes a target-entropy floor.

The two planned comparisons remain MAIN A-C (operators and predicted conditioning jointly) and SECOND A-constant (overall predicted conditions including empty mass/gradient paths). Missing MAIN constant prevents decomposing a MAIN gain into condition value. A bridges are not additional seeds or a platform correction. Fixed-model validation diagnostics and a separately authorized all-four TEST freeze are still pending; no TEST or automatic larger matrix is inferred from these development results.

The [original V1 report and its terminology erratum](../development_v1_20261022/reporting_erratum_20261022.json) remain unchanged. The [same-weight extent diagnostic](../development_extent_v1_20261022/readout.md) is separate development evidence and does not test v2 training efficacy.
