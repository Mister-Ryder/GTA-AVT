# Four selected GTA-AVT weight identities before TEST

This author-declared record identifies the four checkpoints selected by the original training-VAL rule. Each completed 10,000 successful updates and selected step 10,000: choose the larger mean Dice from the scheduled step-5,000/10,000 validation evaluations, with an exact tie selecting the earlier step. No checkpoint is reselected from common-reference VAL, attribute diagnostics or TEST.

The linked [declared_selection.json](declared_selection.json) uses the released portable reader's six-field declared-selection schema. It retains separate role/job/method, selected step, tensor-weight SHA256, provenance SHA256, original-checkpoint SHA256 and full training-recipe SHA256 for each slot; the fixed case split, 24-file source map, canonical data identity and safe common-reference declarations are also bound.

| Slot | Selected step | Tensor-weight SHA256 |
| --- | ---: | --- |
| main_C_concat | 10000 | `4f038b9a459ea8a53b779ee70c9297ef70787ead821b5f8cd14442af37e39b6c` |
| main_A_soft | 10000 | `e09609fff90b735b7cc268aee942143aa8b540a7a3ca019d9c63373028db81f1` |
| ablations_A_soft_platform_bridge | 10000 | `5d7ea03b9f0005d0d6eb4fe17f421d8983392f80300e3a874b262af1c8852eb5` |
| ablations_A_soft_constant | 10000 | `7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6` |

At preparation, the complete five-task common-reference VAL is still pending, as are classifier diagnostics, the separate formal private all-four selection freeze, and TEST. This record contains no TEST metrics and claims no completed four-model scientific comparison. It records the existing choice; it does not launch evaluation, freeze weights, grant TEST permission or change any selection rule.

The reference fields describe a declared common evaluation environment, not either physical training origin. Only whitelisted software, precision and selected-device properties are disclosed. The original full runtime is represented by its canonical SHA256; private hostnames, UUIDs, drivers, executable/deployment paths, relocation siblings and CPU-proof records are excluded.

This is a publication-level author declaration. Its timestamp and hashes alone do not independently prove physical execution, weight contents, source-data bytes or that selection preceded TEST. It is not the author-private all-four freeze marker and adds no source_marker field. A later publication of actual TEST evaluations should use this same unchanged safe record and the matching declared identities; the portable reader checks its FILE SHA and canonical selection identity separately.

The two comparisons remain separate: MAIN A minus C changes specialized operations and predicted conditioning jointly; SECOND bridge minus constant changes the overall condition path, including empty mass and gradient paths. Both A runs use the same training seed and are not extra independent seeds. This selection record establishes no performance advantage.

The frozen split contains 36 training, 8 validation and 12 held-out released case IDs. Subject independence is unverified; the record does not identify 56 independent patients.

- Declared creation UTC: `2026-10-22T20:00:25.186099+00:00`.
- Declaration FILE SHA256: `1bc254120496e2cdbb68890019caeb4f3df680bc22104f4764428c21cf855cf7`.
- Canonical selection identity SHA256: `73ffd3e0d9261049ba345dcd8d2c6ff96afa3636183b96a912d336e7965c7630`.
- Source-map canonical SHA256: `7c5b29ff069f5b33fe2212d721945c6b2e468bc5d15a9f87f6f12e1661f7fe96`.
- Canonical data SHA256: `7f091053f2633f18f1f6e63fd1b206a4c28ee052cef38ff2a63dd512fe14d71c`.
- Fixed split FILE SHA256: `df9647da8b331260fb9f66b294eced01180c1eaf812de27b5148b6800a344a3b`.
