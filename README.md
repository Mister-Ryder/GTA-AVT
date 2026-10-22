# GTA-AVT

An original-operator migration of GTA to binary aortic vessel-tree segmentation on the public 56-case AVT dataset.

**Status:** research implementation and a bounded, one-seed internal ablation protocol. Segmentation results are pending. This repository does not establish a performance improvement, a completed cross-validation study, or a clinical application. It contains no dataset, patient images, trained weights, manuscript drafts, or private execution infrastructure.

## Scientific identity

The primary implementation retains the historical `SegAirwayModel` backbone, independent anatomical attribute encoder, generation-aware adaptive module (GAAM), structure-aware adaptive module (SAAM), decoder placement, and final anatomical guidance-map fusion. Four historical sources are copied byte for byte; their provenance and hashes are in [code/historical](code/historical/).

AVT attributes describe physical vessel scale and operational local junction structure. They are patch-level composition targets, not voxel-wise anatomical classes. Reference masks and centerlines define training targets; inference takes CT images only.

This is an **original-operator migration**, not a reproduction of the entire historical training procedure. The current recipe starts from random weights and trains jointly. It does not claim airway checkpoint transfer or the original separate classifier pretraining and staged freezing. Decoder deep supervision and a common Dice+BCE recipe are shared across the present comparisons.

The primary study is explicitly **no-space/no-Gaussian**: it omits the location-adaptive module and Gaussian output supervision. This does not imply that CT features contain no spatial context. The historical classifier still registers an unused location head; registered and active parameter counts must be distinguished.

### Compared methods

| CLI method | Active computation | Question |
|---|---|---|
| `faithful_baseline` | Original segmentation backbone without attribute injection | Common reconstruction reference |
| `faithful_concat` | Same attribute encoder and original base projection/concatenation/residual fusion at both decoder sites | Does ordinary feature injection explain the benefit? |
| `faithful_joint` | Original hard GAAM/SAAM decisions and confidence arithmetic | Traceable original-operator anchor |
| `faithful_constant` | Same hard operators with uniform condition vectors | Hard-condition control |
| `faithful_soft` | Same GAAM/SAAM convolutional operators with the composition arithmetic below | Targeted AVT adaptation |
| `faithful_soft_constant` | Exactly the same registered model as `faithful_soft`, with uniform condition vectors | Matched control for informative soft conditions |
| `faithful_soft_scale`, `faithful_soft_structure` | Only the named soft condition varies | Separate scale and structure effects |

`faithful_scale` and `faithful_structure` provide analogous hard-condition ablations. C registers the same specialized modules as A to make their entire initial states identical, but executes only base fusion; specialized kernels are inactive in C. Thus C/A have equal registered capacity but different active computation. A-versus-C alone does not isolate condition information; the matched constant-A control is necessary.

The alternative `baseline`/`gta_*` family in `model.py` uses a different soft-FiLM implementation and is retained for traceability. It is not the primary original-operator experiment and its results must remain separate.

### Hard and soft conditioning

Scale classes are empty, radius <=2 mm, 2-4 mm, 4-8 mm, and >8 mm. The hard adapter maps them to historical generation predicate values `[0, 5, 1, 2, 3]`, activating the original `>=5` fine-vessel rule only for the measured small-radius class. Graph depth is not treated as vessel caliber. Structure classes empty/nonjunction/junction-neighborhood map to `[0, 1, 3]`.

Hard argmax decisions and maximum mixture mass are detached, as in the historical operators. Segmentation gradients still reach classifier features. Maximum mixture mass is **not calibrated uncertainty** for these mixed-composition targets.

The soft candidate keeps the original base fusion, depthwise enhancement, learned gate, and four morphology paths. With fused feature `f`, original enhancement `M`, learned gate `u`, morphology residual `m`, and predicted composition probabilities:

```text
enhanced = f * M * (1 + 0.2 * p_small_radius)
GAAM     = enhanced * (1 - u * p_empty) + f * u * p_empty
SAAM     = f + (1 - p_empty) * m * (1 + 0.1 * p_junction)
```

This explicitly changes conditional arithmetic while retaining the original convolutional operators. It is not mathematically identical to the hard anchor. Probabilities provide composition/foreground-availability signals, and are computed in float32 for the soft path under mixed precision.

## Data and frozen split

Download the original [AVT dataset](https://doi.org/10.6084/m9.figshare.14806362) from Figshare. It contains 20 KiTS, 18 RIDER, and 18 Dongyang cases. The record lists **CC BY 4.0** and also asks users to respect the applicable licenses of the underlying collections. Read the [official dataset record](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362) and cite [Radl et al., Data in Brief, 2022](https://doi.org/10.1016/j.dib.2022.107801). No dataset license is granted by this source repository.

[configs/split_seed20261022.json](configs/split_seed20261022.json) fixes 36 training, eight validation, and 12 held-out patients. The development pilot uses three predeclared validation cases, one per source: `Dongyang_D2`, `KiTS_K13`, and `Rider_R10`. It never evaluates the held-out test set. A single seed and a small validation subset cannot establish seed stability or formal generalization.

The official NRRDs mix HU and stored HU+1024 conventions. Preprocessing uses CT-only air evidence and stops on ambiguous cases. Three CT-only reviewed exceptions in [configs/ct_only_reviewed_offset_overrides.json](configs/ct_only_reviewed_offset_overrides.json) are locked to the original image SHA256 and detector SHA256. Private review images are not distributed. These corrections are intensity-convention estimates, not verification of unavailable DICOM calibration metadata.

The image-defined crop uses the largest body component above -600 estimated HU, an x/y margin of 10 mm, and the full z extent. Held-out labels do not define inference crops. Images use linear and masks nearest-neighbor resampling. The examples below use a **1.0 mm isotropic** cache; inspect training-only native-grid round-trip coverage before selecting a final resolution.

## Environment

Use Python 3.11 for the current environment target and install a CUDA-compatible PyTorch 2.7.1 build before the remaining packages. [PyTorch's version-specific installation page](https://pytorch.org/get-started/previous-versions/) provides wheel choices for the installed driver. `requirements.txt` pins the tested non-PyTorch packages; it is not a complete transitive or hardware-specific lock file.

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The CUDA 12.8 command above requires a compatible NVIDIA driver. Both development environments passed actual GPU and auxiliary-loss fixtures:

| Environment | Python | PyTorch | NumPy | SciPy | scikit-image | SimpleITK |
|---|---|---|---|---|---|---|
| Current target | 3.11.17 | 2.7.1+cu128 | 1.26.4 | 1.15.3 | 0.25.2 | 2.5.2 |
| Legacy target | 3.8.10 | 1.11.0+cu113 | 1.22.4 | 1.10.1 | 0.21.0 | 2.3.1 |

These are separate environments: PyTorch 1.11 is not a Python 3.11 installation choice. The cache is generated once in the legacy environment and transferred byte-identically, so comparisons do not silently regenerate targets under different skeleton-library versions. Exact optimization equivalence across environments is not asserted. Core A/C/constant comparisons are grouped on one GPU; the other group includes a repeated full-A bridge alongside scale/structure/hard variants. Cross-group comparisons require that bridge and an explicit hardware/software limitation. Final evaluation is intended to use one reference environment, or to report evaluator-version sensitivity. These are protocol intentions, not completed segmentation results.

Checkpoint loading handles the different `weights_only` APIs and is intended only for trusted checkpoints produced by this code.

The legacy training and reference-evaluation environment can instead install `torch==1.11.0+cu113` in Python 3.8 and use [requirements-legacy.txt](requirements-legacy.txt). The two requirements files describe separate tested environments.

## Reproducible commands

Run commands from the repository root. Use a fresh cache after changing preprocessing or geometry code, and separate immutable run directories for every method/configuration.

### 1. Download and extract official archives

```bash
python tools/download_avt.py --destination data/downloads
python -m zipfile -e data/downloads/Dongyang.zip data/raw
python -m zipfile -e data/downloads/KiTS.zip data/raw
python -m zipfile -e data/downloads/Rider.zip data/raw
```

The public downloader reads the official Figshare API, uses its download URLs, supports interrupted downloads when the server honors byte ranges, and verifies each archive's recorded size and MD5. It has no credentials, alternate host, or private mirror. `--metadata-only` downloads only public metadata. A network or checksum failure is a failure, not a completed dataset.

### 2. Validate the implementation and prepare the cache

```bash
python code/faithful_smoke.py --device cuda --patch-size 96 --batch-size 2 --steps 2 --output results/faithful_smoke.json
python code/prepare.py --data-root data/raw --split configs/split_seed20261022.json --output data/cache_1mm --spacing 1.0 --offset-overrides configs/ct_only_reviewed_offset_overrides.json
```

The smoke checks historical source identity, common initialization, original operator responses, finite optimization, expected gradient paths, and native-grid metric fixtures. It uses synthetic data; passing it is not a segmentation result. Cache records retain original geometry, source hashes, preprocessing hashes, cached-file hashes, intensity decisions, and training-only geometry/round-trip diagnostics.

### 3. Run bounded internal ablations

```bash
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_B --method faithful_baseline
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_C --method faithful_concat
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_Ahard --method faithful_joint
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_Asoft --method faithful_soft
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_Asoft_constant --method faithful_soft_constant
```

The pilot is 200 updates with one fixed seed and three validation patients. It tests pipeline behavior and the need for longer, controlled experiments. It does not fill a final results table. The bounded initial delivery plan in [configs/bounded_20261022](configs/bounded_20261022/) fixes 5,000 updates per internal ablation, one seed, and all eight validation patients at 2,500 and 5,000 updates. Main comparisons are ordinary fusion, full soft GTA, and matched constant soft GTA; a second runtime runs a full-GTA bridge, single-condition ablations, and the original hard anchor. This phase requires a separate explicit invocation after pilot review. The 10,000-update template is retained as an unused template, not the initial delivery budget.

Resume an interrupted run with the same command and `--resume`. Configuration, source identity, manifest, and patient lists must match. The training snapshot recursively preserves the vendored sources too.

An AMP overflow retries the same batch with a smaller loss scale, with a maximum of eight attempts. Batch-normalization buffers and Torch CPU/CUDA random states are restored before a retry; only successful optimizer updates advance the declared budget. Invalid forward loss, invalid FP32 gradients, persistent overflow, and norm overflow fail explicitly. `amp_overflow.jsonl` preserves events. The first real-data pilot exposed the former unconditional-overflow failure; its partial run was archived and all replacement pilots started fresh. The actual GPU fixture below passed on both tested runtimes:

```bash
python tools/amp_retry_fixture.py --project-root . --output results/amp_retry_fixture.json
```

### 4. Inspect validation predictions

```bash
python code/evaluate.py --run runs/pilot_Asoft --split val --output results/pilot_Asoft_val --checkpoint best.pt
python runs/pilot_Asoft/source/evaluate.py --run runs/pilot_Asoft --split val --output results/pilot_Asoft_val --checkpoint best.pt --reuse
```

The first command evaluates all eight validation cases, beyond the three used during the pilot. These remain development results. The second resumes exactly the same evaluation with its frozen source; reuse refuses a different checkpoint, condition mode, source, or manifest. To intervene on scale/structure conditions, use a separate output directory and `--condition-mode constant` or `--condition-mode permuted`; the latter reverses class entries, not patient identities. Pilot checkpoints are ineligible for final-test reporting by default. Do not use held-out test performance to choose methods, thresholds, preprocessing, or geometry targets.

## Metrics and geometry limitations

- Predictions are restored to the original CT physical grid before evaluation. Full prediction foreground, including distant false positives, is retained. The threshold is 0.5 and no connected-component cleanup is applied to segmentation endpoints.
- Dice, precision, recall, FP/FN volumes, pooled bidirectional 6-neighbor boundary-voxel ASSD, and maximum-directional HD95 are explicit. ASSD is not surfel-area weighted or the unweighted mean of two directional means. Missing distance values remain missing, with counts reported.
- Lee skeleton coverage is a voxel-count approximation, not physical arc-length detection. Lee thinning can lose branches or produce empty skeletons for nonempty masks under grid phase/symmetry changes. Such missing skeleton support is not a zero model score.
- Native physical EDT radii are sampled at resampled skeleton nodes. This does not recover branches already lost in resampling. Positive patches without reliable centerline support are excluded from attribute supervision.
- Junction labels are operational local-graph proxies. A degree candidate requires three supported incident arms after contracting a connected 3 mm core, with arm support to 9 mm inside a 12 mm neighborhood. Rejected neighborhoods are unknown, and per-head loss denominators exclude unknown targets. These are not validated anatomical bifurcation annotations. Straight-versus-curved adaptation and tree-relative spatial conditions are not implemented.
- Geometry strata currently report GT-centerline recall; false-positive precision is global. Patient-level reporting is required. Skeleton nodes and overlapping patches are not independent patients.

## Published reference values

These are literature references, **not reproduced results or a shared-protocol ranking**. They concern AVT or its use as the public SEG.A. training data, and must not be confused with the separate AortaSeg24 dataset.

| Primary study | Evaluation | Reported result |
|---|---|---|
| [Yuan & Yang, AVTSeg, 2023](https://arxiv.org/pdf/2305.09833) | Public 56 cases, random five-fold evaluation | Dice 90.7% with original intensities; 92.0% after reported KiTS/RIDER intensity correction. Table 1 does not report HD. |
| [Myronenko et al., Auto3DSeg/SegResNet, 2023 preprint / 2024 chapter](https://arxiv.org/pdf/2310.04114) | Public five-fold single-model evaluation | Dice 94.312%. The separate fourth-center hidden-test submission used a 15-model ensemble and reported Dice 92.0%, HD95 6.013. |
| [Wodzinski & Müller, 3D ResUNet, 2023 preprint / 2024 chapter](https://arxiv.org/pdf/2310.15827) | Public 56 cases, five-fold evaluation, 400³ input | Dice 94.41%, HD95 1.89 mm. Smaller-volume and augmentation ablations use different recipes. |
| [Li et al., HybridNet, PLOS One, 2026](https://doi.org/10.1371/journal.pone.0345549) | 2D model; 38 training / 18 test patients | Dice 87.91%, HD95 7.06 mm; ASSD 1.35 mm uses an average of directional means. |

Patient splits, training populations, 2D/3D inputs, resolution, optimization budgets, initialization, ensemble cost, and distance implementations differ from this repository. Aggregate Dice does not demonstrate small-branch or junction-specific benefit. New GTA claims require matched comparisons here, with failures and uncertainty retained.

## Results and provenance

No AVT performance table is released in this scaffold. Future result records should identify the code hash, immutable configuration, split, checkpoint, evaluator, environment, and prediction provenance. A single-seed internal ablation must retain that identity and cannot be described as independent multi-seed or complete five-fold evidence.

[release_source_hashes.json](release_source_hashes.json) records the copied source bytes. [NOTICE](NOTICE) and the historical third-party notices explain source lineage and licensing boundaries. No blanket MIT license is selected for this repository or for later GTA additions.
