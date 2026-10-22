# Four-model common-reference TEST12 measurements

**Measurements completed and reviewed on 23 October 2026.** This packet retains all 48 original case-metric files and four original summaries, a safe evaluation declaration, the unchanged public selection declaration, numerical analysis and three PNG/PDF figures. It contains no CT images, labels, prediction masks or weights.

The four models each completed 10,000 successful updates with the common v2 recipe; all selected checkpoints were update 10,000. TEST uses 12 frozen released case IDs, separate from VAL8, with native-grid evaluation, patch 128, batch 2, overlap 0.5, threshold 0.5 and no postprocessing. All four measurements use one common reference runtime/device declaration. No TEST intervention or post-TEST model selection is included.

## Results and adverse evidence

All means are equal-case means. Percentage metrics are in percent; differences below are percentage points (pp).

| Training role and model | Dice | Precision | Recall | FP (ml) | FN (ml) | Centerline recall | Centerline precision | clDice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MAIN C | 76.7130 | 72.8774 | 83.4883 | 102.5451 | 55.7904 | 73.9366 | 59.0103 | 63.7187 |
| MAIN soft-A | 81.9991 | 82.8065 | 81.9625 | 49.3184 | 60.2610 | 74.5362 | 67.0360 | 69.9148 |
| SECOND soft-A bridge | 80.3267 | 79.7407 | 82.3999 | 62.0360 | 59.7257 | 75.5104 | 68.9048 | 70.9657 |
| SECOND constant-A | 83.9775 | 84.8275 | 83.5991 | 43.9691 | 55.9624 | 73.4607 | 70.5329 | 70.8500 |

The two within-role comparisons are separate estimands. C receives the same auxiliary supervision as A. MAIN A−C changes specialized operations and predicted conditioning jointly; absent MAIN constant prevents attributing its gain specifically to informative conditions. SECOND bridge−constant changes overall predicted conditions, including empty-class mass, nonempty composition and training gradient paths. The two A runs share a seed and are not independent seeds. Do not pool the comparisons, subtract a platform effect or treat their cross-role difference as an algorithm effect.

| Paired change, mean [exploratory 95% interval] | MAIN A−C | SECOND bridge−constant |
|---|---:|---:|
| Dice (pp) | +5.2861 [1.5484, 10.0400] | −3.6508 [−8.4292, 0.4112] |
| Precision (pp) | +9.9291 [4.5944, 16.7810] | −5.0868 [−12.4426, 1.4760] |
| Recall (pp) | −1.5258 [−4.2578, 0.6231] | −1.1992 [−3.4931, 0.5245] |
| FP (ml) | −53.2266 [−96.5686, −20.3129] | +18.0669 [−7.9621, 47.5346] |
| FN (ml) | +4.4706 [−2.3357, 14.0580] | +3.7633 [−2.2878, 12.2546] |
| ASSD (mm) | −4.1599 [−9.1671, 0.0882] | +3.4780 [−0.4875, 8.8243] |
| HD95 (mm) | −7.4884 [−29.3879, 14.6374] | +1.1561 [−14.5484, 17.2894] |
| Centerline recall (pp) | +0.5996 [−1.9065, 3.4887] | +2.0497 [0.0400, 4.2931] |
| Centerline precision (pp) | +8.0257 [1.2739, 15.6586] | −1.6281 [−11.7079, 8.4758] |
| clDice (pp) | +6.1960 [0.5039, 12.9093] | +0.1157 [−5.7492, 6.1721] |

MAIN Dice improves in 10 cases and declines in two. Its mean FP reduction accompanies greater mean FN, rather than uniform improvement: Rider_R14 loses 4.9628 pp Dice and adds 50.1760 ml FN; KiTS_K5 loses 1.0848 pp Dice and adds 4.8866 ml FP. Rider_R13 improves Dice by 2.8112 pp while clDice falls 5.1439 pp.

SECOND Dice improves in seven cases and declines in five, yet its equal-case mean is lower. Rider_R16 loses 21.7544 pp Dice, adds 134.3770 ml FP and loses 17.2584 pp clDice; Rider_R14 loses 16.0748 pp Dice and adds 76.8210 ml FP. Removing any single case leaves the SECOND mean Dice difference negative (range −4.3344 to −2.0050 pp); this sensitivity range is not a confidence interval. Mean FP+FN changes by −48.7560 ml for MAIN and +21.8302 ml for SECOND.

| GT-defined centerline-recall change (pp) | Supported case pairs | MAIN A−C | SECOND bridge−constant |
|---|---:|---:|---:|
| Radius ≤2 mm | 10 | +4.9091 | +3.4227 |
| Radius 2–4 mm | 12 | +5.5739 | +3.7050 |
| Radius 4–8 mm | 12 | +1.0678 | +4.2674 |
| Radius >8 mm | 12 | −2.5210 | −2.3946 |
| Nonjunction | 12 | +0.7142 | +1.6016 |
| Junction neighborhood | 12 | +1.2445 | +2.4291 |

Fine-group and centerline recall can improve while bulk recall, large-radius recall or global false positives deteriorate. These results do not establish that predicted conditions or their nonempty anatomical composition are superior to trained constant conditions. Geometry tables measure GT-group recall only, with global FP and centerline precision retained separately. Rider_R14 and Rider_R16 have zero ≤2 mm support and remain explicit nulls; Dongyang_D14 has only three such nodes. The two largest SECOND Dice-loss cases therefore do not contribute to the fine-group mean; that mean cannot offset their whole-volume failures. Nodes are not branches, physical arc length or additional independent cases. Full intervals, support, missingness, cohort summaries and every paired case are preserved in [analysis/paired.json](analysis/paired.json) and [casewise.csv](analysis/casewise.csv).

Intervals use the unchanged paired-case bootstrap (20,000 draws, seed 20261022). They are exploratory: independence of the released case IDs as patients is unverified, there is one seed, and many endpoints are reported. Neither positive intervals nor aggregate Dice prove an anatomical mechanism or clinical/topological correctness. The [VAL measurements](../minimal4_reference_VAL_20261023/README.md) and [mixed condition diagnostics](../minimal4_VAL_diagnostics_20261023/README.md) remain separate evidence; no TEST-guided redesign or extra ablation is supplied here.

## Files, source level and reproduction

- `evaluations/<slot>/summary.json` and `<case>/metrics.json`: 52 byte-identical original numerical files.
- `configs/evaluation_declaration.json`: safe metadata and original numerical-file hashes; no original private evaluation protocol.
- `configs/declared_selection_before_TEST.json`: unchanged public declaration, SHA256 `1bc254120496e2cdbb68890019caeb4f3df680bc22104f4764428c21cf855cf7`.
- `analysis/`: portable paired results, complete casewise CSV, fixed adverse-example selection and readout.
- `figures/`: all-case Dice, all ten metric differences and all six geometry-group differences, with source CSV/JSON and file hashes. The plotter uses stored intervals and does not rerun bootstrap; the three numerical PNGs have received author visual review; see the [release review](release_review.md).

This is the public **declaration/file-self-consistency** level, not independent attestation of author physical servers, unavailable weights/data bytes or selection chronology. `source_marker_FILE_sha256` is null. The public selection file is not an author-private freeze or TEST authorization. True numerical parity was checked against the author-source analysis for eight shared result fields, without equating their different provenance levels.

From a released repository checkout containing this packet at `reports/minimal4_reference_TEST_20261023`, use fresh output directories:

```bash
python analysis/analyze_minimal4_portable.py --execute-analysis --declaration reports/minimal4_reference_TEST_20261023/configs/evaluation_declaration.json --declaration-sha256 5c97413c15c5dbe2e12ac3ab64dff145430224d8a26e4a863908ca429e742018 --selection-record reports/minimal4_reference_TEST_20261023/configs/declared_selection_before_TEST.json --selection-sha256 1bc254120496e2cdbb68890019caeb4f3df680bc22104f4764428c21cf855cf7 --output results/recomputed_TEST
python analysis/plot_minimal4_portable.py --execute-plot --analysis reports/minimal4_reference_TEST_20261023/analysis/paired.json --analysis-sha256 66fa40f7379f1699e055fee51d4525b09e7dd4d97d833589d2f959cc86c54b77 --declaration reports/minimal4_reference_TEST_20261023/configs/evaluation_declaration.json --declaration-sha256 5c97413c15c5dbe2e12ac3ab64dff145430224d8a26e4a863908ca429e742018 --selection-record reports/minimal4_reference_TEST_20261023/configs/declared_selection_before_TEST.json --selection-sha256 1bc254120496e2cdbb68890019caeb4f3df680bc22104f4764428c21cf855cf7 --output results/recomputed_TEST_figures
```

These commands only reanalyze existing numbers or plot them; they do not train, load weights or run TEST. A newly recomputed paired file has its own actual file hash. See the repository's `analysis/MINIMAL4_PORTABLE_README.txt` and `analysis/MINIMAL4_PLOT_PORTABLE_README.txt` for the declaration contract.
