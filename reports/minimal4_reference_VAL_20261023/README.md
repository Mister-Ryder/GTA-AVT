# Four-model development VAL8 results

These are completed common-reference validation measurements, with two separate within-role comparisons. They are development evidence: the same eight released cases were used for checkpoint selection. The fixed four-model TEST12 evaluation has now started; no TEST result is included here.

Read the [full numerical interpretation](README.txt), [all paired results and exploratory intervals](analysis/paired.json), and [source declaration](configs/evaluation_declaration.json). The unchanged [pre-TEST declaration](configs/declared_selection_before_TEST.json) is distinct from the author-private formal selection marker.

The MAIN Dice gain accompanies less false-positive volume and more false-negative volume. The smaller SECOND condition effect is sensitive to individual cases. Neither contrast establishes stable fine-vessel or junction recovery, and they cannot be pooled or subtracted.

[Per-case Dice](figures/01_case_Dice_by_training_role.png), [all ten metrics](figures/02_all_metric_paired_changes.png), and [six geometry groups](figures/03_GT_geometry_recall_changes.png) have PNG/PDF exports with source CSV/JSON and a figure manifest. All three actual public figures were visually reviewed. Publication inputs establish declaration/file self-consistency; they do not independently attest author servers, tensor contents, data bytes or selection timing.

The README.txt statement about not viewing images refers to its numerical explanation stage. A separate private review subsequently checked all four cases selected by the frozen adverse-example rule, using three GT-volume-quartile slices per case on the native grid, full field of view, fixed display window and TP/FP/FN colors. Those images are not released. Three slices cannot locate every 3D error or establish topology or clinical cause; in particular, the numerical clDice declines are not explained away by these views.

The candidate manifest preserves the reviewed preparation snapshot. No CT, label, prediction mask, weight, private evaluation protocol or private execution receipt is included. The two A runs are not additional seeds, and released-case independence is unverified.
