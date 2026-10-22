# Fixed V1 same-weight deployment extent: all 12 evaluations

This completed development probe uses the genuine V1 5,000-update C and A-bridge weights, each held identical across 96-cube and 128-cube inference, on the three already declared cases `Dongyang_D2`, `KiTS_K13`, `Rider_R10`. Four conditions × three cases completed. It is not a new training experiment, a final performance estimate, another seed, or a C-versus-bridge causal comparison.

[summary.json](summary.json) binds source-record hashes, original weight identities, complete means, six same-weight pairs, geometry support and retest differences. [casewise.csv](casewise.csv) contains all 12 numerical rows with raw probability fractions. No native masks, images or weight files are distributed. All original 35 private artifacts were SHA/size verified; the completed frozen helper checked the entire original 270-tensor/BN-buffer inventory after every condition. This packaging does not independently reload weights or decode predictions.

All conditions use one MAIN Python 3.8/PyTorch 1.11 runtime/device, AMP, TF32 disabled, batch two, overlap 0.5, threshold 0.5, native-grid restoration, predicted conditions and no postprocessing. The original source/cache identities and all case/group GT supports are unchanged. All-56 file bytes were checked for identity, including TEST file bytes, but only these three VAL arrays were decoded and evaluated; no TEST inference or metrics occurred.

## Complete means

Probability means are percent; probability differences are pp; volume is ml and distance mm. Each mean gives three cases with equal weight. All reported metrics have three defined cases; no CI or p-value is computed.

### C

| Metric | Inference 96 | Inference 128 | 96−128 |
|---|---|---|---|
| dice | 70.931 | 71.855 | -0.924 |
| precision | 69.297 | 78.038 | -8.741 |
| recall | 76.809 | 67.617 | 9.192 |
| fp_ml | 186.742 | 91.814 | 94.928 |
| fn_ml | 114.127 | 165.660 | -51.533 |
| assd_mm | 18.215 | 11.683 | 6.532 |
| hd95_mm | 106.420 | 83.799 | 22.621 |
| centerline_recall | 77.099 | 74.310 | 2.790 |
| centerline_precision | 46.854 | 55.131 | -8.277 |
| cldice | 56.235 | 62.316 | -6.081 |
### A_bridge

| Metric | Inference 96 | Inference 128 | 96−128 |
|---|---|---|---|
| dice | 67.458 | 70.420 | -2.962 |
| precision | 63.560 | 74.432 | -10.872 |
| recall | 77.746 | 69.515 | 8.230 |
| fp_ml | 262.377 | 124.868 | 137.509 |
| fn_ml | 107.728 | 155.460 | -47.732 |
| assd_mm | 21.376 | 13.541 | 7.835 |
| hd95_mm | 103.545 | 85.722 | 17.823 |
| centerline_recall | 80.184 | 78.147 | 2.037 |
| centerline_precision | 28.849 | 53.784 | -24.934 |
| cldice | 40.427 | 62.166 | -21.739 |

## Every same-weight pair

| Model | Case | Dice pp | Precision pp | Recall pp | FP ml | FN ml | ASSD mm | HD95 mm | CL recall pp | CL precision pp | clDice pp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C | Dongyang_D2 | 2.991 | -2.864 | 6.593 | 19.400 | -30.021 | 2.570 | 7.606 | 0.744 | -8.350 | -6.016 |
| C | KiTS_K13 | -19.569 | -29.718 | 1.819 | 275.319 | -6.905 | 15.943 | 20.122 | 5.411 | -12.843 | -11.413 |
| C | Rider_R10 | 13.806 | 6.358 | 19.164 | -9.934 | -117.673 | 1.083 | 40.135 | 2.213 | -3.638 | -0.814 |
| A_bridge | Dongyang_D2 | 0.456 | -9.006 | 6.133 | 43.728 | -27.924 | 4.610 | 13.306 | 2.232 | -12.277 | -7.874 |
| A_bridge | KiTS_K13 | -19.593 | -24.284 | -0.562 | 331.893 | 2.134 | 17.012 | 3.919 | -1.804 | -19.326 | -23.567 |
| A_bridge | Rider_R10 | 10.250 | 0.673 | 19.121 | 36.905 | -117.406 | 1.883 | 36.242 | 5.682 | -43.200 | -33.777 |

Both models show mixed case directions, not a common advantage for 96. The three-case Dice means change by −0.924 pp for C and −2.962 pp for the bridge. Recall rises with 96, but FP rises by 94.928/137.509 ml, and clDice falls 6.081/21.739 pp. K13 loses about 19.6 Dice pp for both weights and adds 275.319/331.893 ml FP. R10 gains Dice but the bridge loses 43.200 pp in centerline precision and 33.777 pp in clDice. These costs prevent interpreting higher recall or a selected improved case as a common repair.

## Geometry support and recall

GT support for every case/group is identical between sizes and matches the original V1 evaluation; full support counts and per-case recalls remain in JSON/CSV. Groups provide recall only, without group precision.

| Model | Group | Three-case mean 96−128 recall pp |
|---|---|---|
| C | radius_le_2mm | -1.320 |
| C | radius_2_to_4mm | 3.948 |
| C | radius_4_to_8mm | 6.970 |
| C | radius_gt_8mm | 1.784 |
| C | nonjunction | 4.073 |
| C | junction_neighbourhood | -3.366 |
| A_bridge | radius_le_2mm | -0.012 |
| A_bridge | radius_2_to_4mm | -0.931 |
| A_bridge | radius_4_to_8mm | 4.316 |
| A_bridge | radius_gt_8mm | 4.588 |
| A_bridge | nonjunction | 1.668 |
| A_bridge | junction_neighbourhood | 0.113 |

For C/R10, the 13.806 pp Dice gain at 96 accompanies only 1.523 pp >8 mm centerline-recall improvement (65.412→66.935%); large-radius coverage remains low. For bridge/R10, >8 mm coverage improves 72.581→86.111%, but centerline precision/clDice deteriorate strongly. D2 >8 mm recall is almost unchanged at either size for both models. A larger global overlap value does not demonstrate restoration of a named main trunk.

## Inference-128 retest against the original same-weight V1 endpoint

C reproduces all ten metrics exactly on all three cases. The bridge differs only slightly. The table reports new128 minus original128, with probability differences in pp; full ten-metric values are retained in JSON. Original training-VAL backend flags were not independently fully recorded; the old bridge also used Python 3.11/PyTorch 2.7 and another GPU. Agreement on three cases does not prove runtime equivalence or reproducibility for every case.

| Model | Case | Dice pp | FP ml | FN ml | CL precision pp | CL recall pp | clDice pp |
|---|---|---|---|---|---|---|---|
| C | Dongyang_D2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C | KiTS_K13 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C | Rider_R10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| A_bridge | Dongyang_D2 | -0.000083 | -0.001166 | 0.001166 | 0.000000 | 0.000000 | 0.000000 |
| A_bridge | KiTS_K13 | -0.000358 | 0.011078 | -0.003693 | 0.025868 | 0.000000 | 0.026120 |
| A_bridge | Rider_R10 | -0.000382 | 0.000309 | 0.003090 | -0.064745 | 0.000000 | -0.035959 |

## Interpretation boundary

96→128 simultaneously changes context, stride 48→64, tile locations/coverage, Gaussian **tile-blending** sigma 12→16 mm, patch condition composition and normalization inputs. It is not a pure context or BatchNorm experiment; this Gaussian is distinct from the omitted GTA Gaussian output supervision. All12 completed evaluations are retained, including 96 failures. Their runtime records are costs under a fixed execution order/cache state, not a fair speed benchmark.

This does not compare 96-versus128 training, prove a universally preferable deployment size, justify tuning TEST, or establish GTA effectiveness. It supplied development evidence against a common rollback to96 before the [scope was reduced](../../docs/minimal4_scope_20261022.md). The original-v2 training recipe is unchanged; remaining four-model matched comparisons and final evaluation are separate.
