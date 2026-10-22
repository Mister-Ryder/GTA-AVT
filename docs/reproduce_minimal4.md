# Reproducing the four-model AVT study

Run all commands from the repository root. All four 10,000-update models and their 4 × 12 TEST measurements have completed. There are two separate activities: **train your own models and evaluate them**, or **recompute the published VAL and TEST statistics without weights**. Author weights, CT images, labels and prediction masks are not distributed.

The current scope has two within-role comparisons: MAIN soft-A versus C, and SECOND soft-A bridge versus constant-A. Keep these comparisons separate. C has the same auxiliary supervision as A; A–C changes specialized operations and conditioning jointly. Missing MAIN constant prevents decomposing a MAIN gain into condition value. The two A runs share a seed and are not additional independent seeds. The historical seven-method matrix and three-MAIN-classifier protocol remain incomplete.

## Train and export your own models

First follow the repository README for official AVT acquisition, the frozen 36/8/12 case split, the Python 3.8/PyTorch 1.11 environment and `data/cache_1mm/manifest.json`. Released case IDs are not verified independent patients. These four configurations use the same v2 recipe: random initialization, seed 20261022, 128³ input, batch 2, 10,000 successful updates and complete VAL8 at updates 5,000 and 10,000.

```bash
python code/train.py --config configs/bounded_20261022_v2/formal_main_C_concat.json --manifest data/cache_1mm/manifest.json --output runs/repro_MAIN_C
python code/train.py --config configs/bounded_20261022_v2_minimal4/formal_main_A_soft.json --manifest data/cache_1mm/manifest.json --output runs/repro_MAIN_A
python code/train.py --config configs/bounded_20261022_v2/formal_ablations_A_soft_platform_bridge.json --manifest data/cache_1mm/manifest.json --output runs/repro_SECOND_bridge
python code/train.py --config configs/bounded_20261022_v2_minimal4/formal_ablations_A_soft_constant.json --manifest data/cache_1mm/manifest.json --output runs/repro_SECOND_constant
```

These commands create new reproduction runs. The authors' original-v2 C and bridge retain their original identities; this scope reduction is not algorithm v3. A reproduction selects its own genuine best checkpoint by scheduled VAL mean Dice, with the earlier checkpoint retained on an exact tie. Do not force update 10,000 or copy an author checkpoint identity. Equal seeds do not ensure bitwise training equivalence across hardware.

For each completed run, capture its actual training environment and export its genuine `best.pt`. Replace uppercase path placeholders with your own paths and use fresh output directories.

```bash
python tools/reference_weights_support.py capture-environment --output ENV.json
python tools/export_reference_weights.py --run RUN --manifest ORIGINAL_TRAIN_CACHE/manifest.json --training-environment-json ENV.json --output BUNDLE
```

The exporter verifies the original run, source and data identities and writes CPU tensor-only weights plus provenance and verification JSON files. It performs no model forward. Keep the original training manifest for export; a relocated reference manifest has a different file hash even when canonical data bytes are unchanged. See [reference-weight instructions](../tools/REFERENCE_WEIGHTS_README.txt).

## Common-reference VAL and condition diagnostics

Evaluate all four bundles in the same actual reference environment and device, using separate output directories:

```bash
python tools/evaluate_reference_weights.py --bundle BUNDLE --manifest REFERENCE_CACHE/manifest.json --split val --device cuda --output RESULTS
```

The evaluator fixes patch 128, batch 2, overlap 0.5, threshold 0.5, native-grid metrics, no postprocessing and TF32 off. It verifies source and canonical data identity. This command does not launch TEST.

For the fixed same-weight **SECOND bridge** availability-only diagnostic, use that bridge's normal-evaluation bundle again:

```bash
python tools/evaluate_reference_weights.py --bundle BRIDGE_BUNDLE --manifest REFERENCE_CACHE/manifest.json --split val --device cuda --intervention availability_only --output results/bridge_availability
```

This preserves each predicted empty-class probability and uniformly distributes its nonempty mass. It is a runtime intervention, not another trained model or a causal decomposition of A versus trained constant-A. The historical `analysis/analyze_condition_intervention.py` has MAIN-specific reporting and is not recommended for relabeling SECOND results. See the published [SECOND intervention and classifier diagnostics](../reports/minimal4_VAL_diagnostics_20261023/README.md).

For classifier-only diagnostics, fill `configs/minimal4_bundle_index.example.json` with four genuine generic-export bundle paths and their actual three-file hashes; retain all four distinct slots.

```bash
python tools/attribute_patch_diagnostics_minimal4.py --index configs/my_four_bundles.json --manifest REFERENCE_CACHE/manifest.json --output results/classifier4_val --device cuda --execute-frozen-validation
```

Without the explicit execution flag, this prints a plan without opening models or data. The measurement uses fixed VAL8, 128³ patches, batch at most 2 and at most 160 shared patches; it does not execute the segmentation decoder. CT-uniform and GT-enriched strata remain separate, with unknown targets excluded per head. Follow [the four-slot instructions](../tools/ATTRIBUTE_MINIMAL4_README.txt); do not substitute SECOND bundles into the preserved three-MAIN protocol.

## Recompute the published VAL numbers and plots

These commands need the published summary/metric files, NumPy and, for plotting, Matplotlib—not author weights. They retain both within-role comparisons, all cases, geometry support, missing values and adverse examples.

```bash
python analysis/analyze_minimal4_portable.py --execute-analysis --declaration reports/minimal4_reference_VAL_20261023/configs/evaluation_declaration.json --declaration-sha256 fe486e6092f8d9286f68193529ff6fad6a2c73455a2ae07d5e63dcb880c91bf8 --output results/recomputed_VAL
python analysis/plot_minimal4_portable.py --execute-plot --analysis reports/minimal4_reference_VAL_20261023/analysis/paired.json --analysis-sha256 b2468592f02afd94b7d1ac38d6bfd94bd5071966f6426ee7db068136392939a7 --declaration reports/minimal4_reference_VAL_20261023/configs/evaluation_declaration.json --declaration-sha256 fe486e6092f8d9286f68193529ff6fad6a2c73455a2ae07d5e63dcb880c91bf8 --output results/recomputed_VAL_figures
```

The plot command binds the released `paired.json`; to plot your newly generated file, supply its own actual SHA256. Both output directories must be new. See the [analysis](../analysis/MINIMAL4_PORTABLE_README.txt) and [plotting](../analysis/MINIMAL4_PLOT_PORTABLE_README.txt) instructions. These tools perform numerical analysis or plotting only, with no training, inference or TEST launch.

The public [pre-TEST selection declaration](../reports/minimal4_selection_before_TEST/declared_selection.json), SHA256 `1bc254120496e2cdbb68890019caeb4f3df680bc22104f4764428c21cf855cf7`, records the existing four weight identities. Public declarations and file hashes support consistency checks; they do not independently attest author hardware, unavailable tensor/data bytes or selection timing, and do not replace the separate author execution proof. The completed [four-model TEST12 report](../reports/minimal4_reference_TEST_20261023/README.md) retains mixed evidence: MAIN A−C mean Dice is +5.2861 pp, whereas SECOND bridge−trained-constant is −3.6508 pp; this does not establish superiority of predicted anatomical composition.

## Recompute the published TEST numbers and plots

With the complete TEST packet at `reports/minimal4_reference_TEST_20261023`, the commands below reanalyze numerical files or plot their saved intervals. They do not load author weights, run inference or launch TEST. The selection record is the unchanged public declaration, not an author-private freeze. Keep VAL and TEST analyses separate and use fresh output directories.

```bash
python analysis/analyze_minimal4_portable.py --execute-analysis --declaration reports/minimal4_reference_TEST_20261023/configs/evaluation_declaration.json --declaration-sha256 5c97413c15c5dbe2e12ac3ab64dff145430224d8a26e4a863908ca429e742018 --selection-record reports/minimal4_reference_TEST_20261023/configs/declared_selection_before_TEST.json --selection-sha256 1bc254120496e2cdbb68890019caeb4f3df680bc22104f4764428c21cf855cf7 --output results/recomputed_TEST
python analysis/plot_minimal4_portable.py --execute-plot --analysis reports/minimal4_reference_TEST_20261023/analysis/paired.json --analysis-sha256 66fa40f7379f1699e055fee51d4525b09e7dd4d97d833589d2f959cc86c54b77 --declaration reports/minimal4_reference_TEST_20261023/configs/evaluation_declaration.json --declaration-sha256 5c97413c15c5dbe2e12ac3ab64dff145430224d8a26e4a863908ca429e742018 --selection-record reports/minimal4_reference_TEST_20261023/configs/declared_selection_before_TEST.json --selection-sha256 1bc254120496e2cdbb68890019caeb4f3df680bc22104f4764428c21cf855cf7 --output results/recomputed_TEST_figures
```

The plot command binds the released TEST `paired.json`; a newly recomputed paired file has its own actual SHA256. These entry points verify public declaration/file consistency, not unavailable author weights or physical execution evidence.
