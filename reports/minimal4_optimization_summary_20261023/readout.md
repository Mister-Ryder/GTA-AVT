# Four-model optimization evidence: completed training, descriptive only

This safe whitelist summarizes existing verified training records. It contains no TEST inputs or results, full logs, CT, predictions, weights, private runtime receipts or deployment paths. It does not independently attest a server or replace the private selection freeze. No new experiment, calculation framework, model selection or training change is introduced.

The four weight identities remain bound to the unchanged declared-selection record (FILE SHA256 `1bc254120496e2cdbb68890019caeb4f3df680bc22104f4764428c21cf855cf7`; canonical identity `73ffd3e0d9261049ba345dcd8d2c6ff96afa3636183b96a912d336e7965c7630`). All selected steps are10000, chosen from the scheduled original training-VAL means at5000/10000; no common-reference or TEST reselection occurs. Original C/bridge retain their original V2 origins; the two later jobs use the same source and complete training recipe, not a new algorithm version.

| Slot | VAL Dice5000% | VAL Dice10000% | Same-run gain pp | AMP attempts/retries/retry-bearing updates | Median step seconds, excluding first10 |
| --- | ---: | ---: | ---: | --- | ---: |
| main_C_concat | 65.693246 | 80.379503 | 14.686257 | 10010 / 10 / 9 | 0.968016 |
| main_A_soft | 71.647316 | 86.289518 | 14.642203 | 10010 / 10 / 9 | 1.058770 |
| ablations_A_soft_platform_bridge | 66.986704 | 85.949031 | 18.962327 | 10011 / 11 / 8 | 1.092890 |
| ablations_A_soft_constant | 66.362719 | 84.731827 | 18.369108 | 10010 / 10 / 7 | 1.080834 |

All four logs contain10000 consecutive successful updates with finite recorded values. Attempts include retries of the same batch after numerical overflow; they are not extra successful updates. The JSON preserves their original log FILE hashes and existing window retry/scaler summaries.

| Slot | Segmentation mean1-500 ->9001-10000 | Scale CE | Structure CE | Total mean |
| --- | --- | --- | --- | --- |
| main_C_concat | 1.784817 -> 0.601905 | 1.568450 -> 1.212892 | 0.726875 -> 0.481644 | 2.014349 -> 0.771359 |
| main_A_soft | 1.788735 -> 0.579850 | 1.561851 -> 1.156794 | 0.730266 -> 0.457388 | 2.017947 -> 0.741268 |
| ablations_A_soft_platform_bridge | 1.764997 -> 0.589656 | 1.563856 -> 1.226826 | 0.729577 -> 0.490244 | 1.994340 -> 0.761363 |
| ablations_A_soft_constant | 1.799586 -> 0.528658 | 1.556338 -> 1.199268 | 0.720401 -> 0.472615 | 2.027260 -> 0.695846 |

The JSON also retains the two last500-update windows separately, the original all-update/excluding-first10/last1000 time summaries, and all ten original scheduled VAL metric summaries. Overlapping loss windows are descriptive views of the same updates, not independent samples. Scalar losses depend on targets and minibatch composition.

C also receives scale/structure auxiliary supervision. MAIN C and A have very similar5000-to10000 Dice gains (14.686257pp and14.642203pp); these two points do not support calling multi-task convergence slower. Reaching10000 updates or observing falling loss does not establish convergence, a late-stage turning point, or that extending training would help. Both A runs use the same seed and are not additional independent seeds.

Update timing includes sampling/data construction and all bounded AMP attempts through CUDA synchronization, but excludes subsequent log writing, VAL and checkpoint saving. It is a description of these runs, not a controlled efficiency benchmark. More seconds per update is not the same quantity as optimization convergence speed; no cross-role efficiency effect is inferred.

Soft-target CE includes a target-entropy term that was not logged. Nonzero CE is not an error that should approach zero, and loss/auxiliary ratios or global norm cannot establish task-gradient conflict. Lower CE does not by itself demonstrate correct anatomical conditions. These records support neither completed convergence nor successful fine-vessel/junction semantics.

Original training-time VAL endpoints remain separate from common-reference VAL and TEST. The shared auxiliary heads make C an attribute-supervised comparator, not a single-task baseline. Full FP/FN, centreline and geometry tradeoffs are reported elsewhere; a better Dice or training loss must not erase those counterexamples. Released case IDs are not verified independent patients.

`optimization_summary.json` is a safe author-reported numerical projection with original log/analysis FILE hashes; the raw private logs and proofs are deliberately omitted. Hashes bind this account but do not let a reader independently recompute the omitted logs. No TEST metric or TEST-driven choice enters the projection.
