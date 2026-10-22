# GTA-AVT

An original-operator migration of GTA to binary aortic vessel-tree segmentation on the public 56-case AVT dataset.

**Current stage (23 October 2026, China time):** all four genuine 10,000-update runs, five common-reference VAL tasks, the 160-shared-patch classifier diagnostic and four fixed TEST12 evaluations have completed. The [complete TEST report](reports/minimal4_reference_TEST_20261023/README.md) includes all 48 case-metric files, four summaries, paired analysis, geometry support, adverse examples and numeric figures. No TEST-guided model, threshold or selection change followed. The original seven-method comparison remains incomplete; the [reduced scope](docs/minimal4_scope_20261022.md) retains the exact v2 recipe and all 24 source files, rather than introducing algorithm v3. This repository contains no dataset, CT images, trained weights, manuscript drafts or private execution infrastructure.

The held-out results are mixed. MAIN soft-A versus ordinary attribute fusion C improves mean Dice by **5.2861 percentage points** (81.9991% versus 76.7130%), with less FP but greater mean FN. SECOND soft-A bridge versus matched constant-A changes Dice by **−3.6508 points** (80.3267% versus 83.9775%): centerline and some small-radius coverage improve while global FP/FN cost increases. These two within-role contrasts cannot be pooled or subtracted. The MAIN contrast supports the complete adaptation path in this setting, while the two contrasts do not establish stable overall value of informative anatomical conditions or a local causal mechanism. See [reproduction instructions](docs/reproduce_minimal4.md) for training your own models or recomputing published numbers without weights.

Supporting [condition diagnostics](reports/minimal4_VAL_diagnostics_20261023/README.md) retain the tiny same-weight composition effect and imperfect attribute predictions, including negative evidence. The [training summary](reports/minimal4_optimization_summary_20261023/readout.md) records loss windows, AMP attempts and scheduled validation endpoints; it does not establish convergence speed or task-gradient conflict.

The [shared GTA principle](docs/gta_framework_principle_20261023.md) specifies the attribute-to-condition-to-feature relationship and the limits of its present AVT instance. The [pre-TEST weight-identity declaration](reports/minimal4_selection_before_TEST/readout.md) records the existing choices and is distinct from the private formal freeze marker.

## Scientific identity

The primary implementation retains the historical `SegAirwayModel` backbone, independent anatomical attribute encoder, generation-aware adaptive module (GAAM), structure-aware adaptive module (SAAM), decoder placement, and final anatomical guidance-map fusion. Four historical sources are copied byte for byte; their provenance and hashes are in [code/historical](code/historical/).

AVT attributes describe physical vessel scale and operational local junction structure. They are patch-level composition targets, not voxel-wise anatomical classes. Reference masks and centerlines define training targets; inference takes CT images only.

[Adaptation boundaries and interpretation](docs/adaptation_boundaries.md) explain which attribute, condition-calculation and component changes remain within the GTA hypothesis, and what the present global conditions cannot establish.

This is an **original-operator migration**, not a reproduction of the entire historical training procedure. The current recipe starts from random weights and trains jointly. It does not claim airway checkpoint transfer or the original separate classifier pretraining and staged freezing. Decoder deep supervision and a common Dice+BCE recipe are shared across the present comparisons.

The primary study is explicitly **no-space/no-Gaussian**: it omits the location-adaptive module and Gaussian output supervision. This does not imply that CT features contain no spatial context. The historical classifier still registers an unused location head; registered and active parameter counts must be distinguished.

### Implemented methods (not all trained)

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

The alternative `baseline`/`gta_*` family in `model.py` uses a different soft-FiLM implementation and is retained for traceability. It is not the primary original-operator experiment. Its initial mixed-precision smoke produced zero segmentation-to-attribute-head gradients; that candidate has not passed the full validation gate and supplies no primary result here. The tested original-operator family uses the `faithful_*` names above.

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

At the 1 mm input grid, GAAM acts on a 4 mm feature lattice: its three-tap depthwise kernel spans 8 mm between the outer kernel centers. SAAM acts on a 2 mm lattice; dilation rates 1/1/2/3 span 4/4/8/12 mm between outer centers. These are the added operators' sampling spans, not the full encoder receptive field or the radius-label thresholds. This initial study retains the historical kernels.

## Data and frozen split

Download the original [AVT dataset](https://doi.org/10.6084/m9.figshare.14806362) from Figshare. It contains 20 KiTS, 18 RIDER, and 18 Dongyang cases. The record lists **CC BY 4.0** and also asks users to respect the applicable licenses of the underlying collections. Read the [official dataset record](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362) and cite [Radl et al., Data in Brief, 2022](https://doi.org/10.1016/j.dib.2022.107801). No dataset license is granted by this source repository.

[configs/split_seed20261022.json](configs/split_seed20261022.json) fixes 36 training, eight validation, and 12 held-out cases with non-overlapping released case IDs. The 56 scans are not verified to represent 56 independent patients. The cited RIDER source has a test-retest design, but no particular AVT repeat pair or cross-split subject overlap has been demonstrated; see the [case-independence limitation and reporting erratum](docs/avt_case_independence.md). The development pilot uses three predeclared validation cases, one per source: `Dongyang_D2`, `KiTS_K13`, and `Rider_R10`. It never evaluates the held-out test set. A single seed and a small validation subset cannot establish seed stability or subject-independent generalization.

The [recorded official archive identities](references/AVT_archives_20261022.json) fix the three download URLs, sizes and MD5 hashes used here. Direct server access was blocked or very slow, so existing local official archives were transferred and verified against those identities and ZIP integrity checks.

The official NRRDs mix HU and stored HU+1024 conventions. Preprocessing uses CT-only air evidence and stops on ambiguous cases. Three CT-only reviewed exceptions in [configs/ct_only_reviewed_offset_overrides.json](configs/ct_only_reviewed_offset_overrides.json) are locked to the original image SHA256 and detector SHA256. Private review images are not distributed. These corrections are intensity-convention estimates, not verification of unavailable DICOM calibration metadata.

The image-defined crop uses the largest body component above -600 estimated HU, an x/y margin of 10 mm, and the full z extent. Held-out labels do not define inference crops. Images use linear and masks nearest-neighbor resampling. The examples below use a **1.0 mm isotropic** cache; inspect training-only native-grid round-trip coverage before selecting a final resolution.

## Environment

The v2 training and common reference-evaluation environment uses **Python 3.8 / PyTorch 1.11.0+cu113**. Both training roles passed actual 128-cube synthetic and real-TRAIN GPU checks with the versions below. [PyTorch's version-specific installation page](https://pytorch.org/get-started/previous-versions/) provides the official CUDA 11.3 wheel command. [requirements-legacy.txt](requirements-legacy.txt) pins the tested non-PyTorch packages; it is not a complete transitive or hardware-specific lock file.

```bash
python3.8 -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements-legacy.txt
```

The installed NVIDIA driver must support the selected CUDA runtime. The distinct v1 secondary environment remains documented for the original development results:

| Environment | Python | PyTorch | NumPy | SciPy | scikit-image | SimpleITK |
|---|---|---|---|---|---|---|
| V2 both training roles and common evaluator; v1 primary | 3.8.10 | 1.11.0+cu113 | 1.22.4 | 1.10.1 | 0.21.0 | 2.3.1 |
| V1 secondary only | 3.11.17 | 2.7.1+cu128 | 1.26.4 | 1.15.3 | 0.25.2 | 2.5.2 |

These are separate environments: PyTorch 1.11 is not a Python 3.11 installation choice. The cache is generated once in the legacy environment and transferred byte-identically. The current four-model scope has two distinct within-role comparisons: MAIN A versus C, and SECOND full-A bridge versus constant-A. MAIN constant, scale-only, structure-only and hard-anchor training are deferred. Matching core software does not eliminate hardware effects; the two A models are platform bridges with the same seed, not additional seeds. They are not pooled or used to subtract a platform effect. The four normal VAL and four normal TEST evaluations, fixed same-weight VAL bridge intervention and four-slot classifier diagnostic completed on one reference environment and device. Completed-stage results are indexed in [Results and provenance](#results-and-provenance). The legacy runtime warns that some backward operators lack deterministic implementations; identical seeds do not imply bitwise deterministic training.

Checkpoint loading handles the different `weights_only` APIs and is intended only for trusted checkpoints produced by this code.

`requirements.txt` retains the separate tested Python 3.11 / PyTorch 2.7.1 environment used by the original v1 secondary role. It is not the v2 study environment.

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
python code/faithful_smoke.py --device cuda --patch-size 128 --batch-size 2 --steps 2 --output results/faithful_smoke.json
python code/prepare.py --data-root data/raw --split configs/split_seed20261022.json --output data/cache_1mm --spacing 1.0 --offset-overrides configs/ct_only_reviewed_offset_overrides.json
```

The smoke checks historical source identity, common initialization, original operator responses, finite optimization, expected gradient paths, and native-grid metric fixtures. It uses synthetic data; passing it is not a segmentation result. Cache records retain original geometry, source hashes, preprocessing hashes, cached-file hashes, intensity decisions, and training-only geometry/round-trip diagnostics.

Actual [TRAIN36 preprocessing evidence](reports/preprocessing_20261022/readout.txt) is available with per-case records. Native-grid round-trip Dice averaged 97.69%, all-centerline retention 99.83%, and radius <=2 mm retention 98.78% across 35 supported cases. The latter is 96.08% when weighted by native skeleton-node support, with 91.07% and 93.14% in two RIDER cases. These are preprocessing-retention measurements, not model scores. Unknown structure accounts for 12.05% of resampled training skeleton nodes and is masked from structure supervision.

### 3. Run bounded internal ablations

```bash
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_B --method faithful_baseline
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_C --method faithful_concat
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_Ahard --method faithful_joint
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_Asoft --method faithful_soft
python code/train.py --config configs/faithful_pilot.json --manifest data/cache_1mm/manifest.json --output runs/pilot_Asoft_constant --method faithful_soft_constant
```

The pilot is 200 updates with one fixed seed and three validation cases. It checks pipeline behavior and does not fill a final results table. The original [v1 plan](configs/bounded_20261022/) used 5,000 updates; only its first two models completed, with [actual development results](reports/development_v1_20261022/readout.txt). Its remaining jobs were closed before starting.

The [common v2 engineering revision](docs/common_recipe_v2.md) originally declared seven internal jobs, retained all frozen source bytes, used fresh 128-cube training for 10,000 successful updates, and evaluated all VAL8 at updates 5,000 and 10,000. The [original exact v2 recipes and plan](configs/bounded_20261022_v2/) remain historical, incomplete records. After both first endpoints and a fixed same-weight deployment-size diagnostic completed, the scope was reduced transparently to [four model tuples](configs/bounded_20261022_v2_minimal4/scientific_plan.json): existing MAIN C and SECOND A bridge retain their original v2 identities; only MAIN A and SECOND constant-A start fresh. Their [two byte-identical recipe copies and portable commands](configs/bounded_20261022_v2_minimal4/) do not define a new training recipe. MAIN A-C measures specialized operations and predicted conditioning jointly, with the same auxiliary supervision. SECOND A-constant measures overall predicted conditions, including empty-class mass and gradient-path changes. Missing MAIN constant prevents decomposing a MAIN gain into condition value. The revision also aligns core software. Its nominal input-voxel budget is 4.74 times v1; it is a combined engineering change, not a single-factor context experiment. Sampling and augmentation algorithms are preserved, but the new patch size changes realized starts, jitter, inputs and target compositions; equal seeds do not imply identical v1/v2 samples. The old `faithful_formal_template.json` remains a template and does not identify the executed v2 configuration.

Resume an interrupted run with the same command and `--resume`. Configuration, source identity, manifest, and case-ID lists must match. The training snapshot recursively preserves the vendored sources too.

An AMP overflow retries the same batch with a smaller loss scale, with a maximum of eight attempts. Batch-normalization buffers and Torch CPU/CUDA random states are restored before a retry; only successful optimizer updates advance the declared budget. Invalid forward loss, invalid FP32 gradients, persistent overflow, and norm overflow fail explicitly. `amp_overflow.jsonl` preserves events. The first real-data pilot exposed the former unconditional-overflow failure; its partial run was archived and all replacement pilots started fresh. The actual GPU fixture below passed on both tested runtimes:

```bash
python tools/amp_retry_fixture.py --project-root . --output results/amp_retry_fixture.json
```

### 4. Inspect validation predictions

```bash
python code/evaluate.py --run runs/pilot_Asoft --split val --output results/pilot_Asoft_val --checkpoint best.pt
python runs/pilot_Asoft/source/evaluate.py --run runs/pilot_Asoft --split val --output results/pilot_Asoft_val --checkpoint best.pt --reuse
```

The first command evaluates all eight validation cases, beyond the three used during the pilot. These remain development results. The second resumes exactly the same evaluation with its frozen source; reuse refuses a different checkpoint, condition mode, source, or manifest. To intervene on scale/structure conditions, use a separate output directory and `--condition-mode constant` or `--condition-mode permuted`; the latter reverses class entries, not case identifiers. Pilot checkpoints are ineligible for final-test reporting by default. Do not use held-out test performance to choose methods, thresholds, preprocessing, or geometry targets.

## Metrics and geometry limitations

The [reference-evaluation tools](tools/REFERENCE_WEIGHTS_README.txt) export genuine completed checkpoints to tensor-only weights, verify exact source/cache identities, and evaluate all trained methods in the same Python 3.8/PyTorch 1.11 environment. Test evaluation requires an explicit command and an eligible formal model. A saved TRAIN-only FP32 input bank supports a numerical runtime bridge. These tools are separate from the full training-checkpoint evaluator above.

[Case-level analysis](analysis/analyze_key_ablations.py) and [figure exports](analysis/plot_key_ablations.py) implement the [frozen comparison protocol](docs/key_ablation_analysis_protocol_20261022.txt): paired case differences, 20,000 bootstrap draws, explicit missing/support counts, and separate training-runtime groups. The [case-unit erratum](docs/avt_case_independence.md) corrects the older protocol's patient terminology without changing its mathematics or compatibility keys. Bootstrap intervals are exploratory and conditional on unverified case independence; same-subject dependence could affect their uncertainty and held-out generalization. Plotting additionally requires Matplotlib. Synthetic helper tests are not experimental results.

The reduced four-model study has separate [portable paired-case analysis](analysis/MINIMAL4_PORTABLE_README.txt) and [figure tools](analysis/MINIMAL4_PLOT_PORTABLE_README.txt). They retain the two within-role contrasts, full VAL8 or TEST12 case sets, ten metrics, six geometry groups, missing support and fixed adverse-example rules. Their public input contract verifies safe declarations and original summary/metric file hashes; it does not independently attest author servers, weights, data bytes or selection timing. The [final readout rules](docs/minimal4_final_readout_rules_20261023.txt) are published before final TEST execution. Completed [VAL8 measurements](reports/minimal4_reference_VAL_20261023/README.md) remain development evidence; the [fixed TEST12 results](reports/minimal4_reference_TEST_20261023/README.md) are reported separately. The preserved historical seven-model helpers remain separate.

The preserved [attribute-faithfulness protocol](docs/attribute_faithfulness_protocol_20261022.txt) is designed to check three completed formal MAIN C/A/constant classifiers against fixed VAL8 geometric targets, distinguishing predicted heads from the conditions actually consumed. It has not been completed. Its [v2 input amendment](docs/attribute_faithfulness_v2_input_amendment_20261022.txt) requires genuine v2 primary-role exports and the complete 128-cube/10,000-update recipe; matching software alone cannot identify the primary A rather than its second-role bridge. The reduced four-model scope has no MAIN constant and therefore does not satisfy that three-MAIN-bundle protocol: do not substitute the SECOND bridge or SECOND constant into it. The separate [portable four-model classifier diagnostic](tools/ATTRIBUTE_MINIMAL4_README.txt) now declares this input contract using genuine generic exports and an explicit four-slot index; its role assignments are reproduction declarations, not author-server attestations. Measured four-slot classifier diagnostics on 160 shared VAL patches are available in the [diagnostic report](reports/minimal4_VAL_diagnostics_20261023/attributes/readout.txt). They do not satisfy or relax the preserved three-MAIN-bundle protocol. The preserved original text refers to v1's 96-cube training. Diagnostic inputs remain 128-cube. Public commands below document the preserved three-MAIN helper. Public commands use `tools/`; an `ops/` path in the preserved authoring protocol denotes its corresponding public tool here.

```bash
python tools/attribute_patch_diagnostics.py --bundle-c exports/C --bundle-a exports/A --bundle-constant exports/constant --manifest data/cache_1mm/manifest.json --output results/attribute_val --device cuda --execute-frozen-validation
python analysis/analyze_condition_intervention.py --normal results/A_val --availability-only results/A_val_availability --output results/A_condition_analysis
```

Omit `--execute-frozen-validation` from the first command to print its nonexecuting gate without opening models or data. The first helper measures classifier outputs and computed class-condition vectors; it does not execute the segmentation decoder or learned feature gate. Uniform constant vectors also alter empty-class mass, so this separately trained control is not availability-preserving. The second command reads completed same-weight normal and availability-only segmentation evaluations; it does not run inference. The same-weight SECOND A intervention and four-slot classifier diagnostics have completed, with small and mixed effects retained in the [diagnostic report](reports/minimal4_VAL_diagnostics_20261023/README.md). The intervention is not another trained model or an isolated causal test of anatomical composition. These diagnostics permit no post-diagnostic tuning or causal decomposition of trained-A versus constant-A.

For the four-model classifier entry point, fill the [bundle-index example](configs/minimal4_bundle_index.example.json) with actual export hashes and run:

```bash
python tools/attribute_patch_diagnostics_minimal4.py --index configs/my_four_bundles.json --manifest data/cache_1mm/manifest.json --output results/classifier4_val
```

This default prints a plan without opening models or data. Add `--device cuda --execute-frozen-validation` only for the explicit classifier-only VAL measurement after all four exports exist. See the dedicated instructions for its trust boundary and score interpretation. It neither evaluates TEST nor replaces the final two-comparison segmentation analysis.

- Predictions are restored to the original CT physical grid before evaluation. Full prediction foreground, including distant false positives, is retained. The threshold is 0.5 and no connected-component cleanup is applied to segmentation endpoints.
- Dice, precision, recall, FP/FN volumes, pooled bidirectional 6-neighbor boundary-voxel ASSD, and maximum-directional HD95 are explicit. ASSD is not surfel-area weighted or the unweighted mean of two directional means. Missing distance values remain missing, with counts reported.
- Lee skeleton coverage is a voxel-count approximation, not physical arc-length detection. Lee thinning can lose branches or produce empty skeletons for nonempty masks under grid phase/symmetry changes. Such missing skeleton support is not a zero model score.
- Native physical EDT radii are sampled at resampled skeleton nodes. This does not recover branches already lost in resampling. Positive patches without reliable centerline support are excluded from attribute supervision.
- Junction labels are operational local-graph proxies. A degree candidate requires three supported incident arms after contracting a connected 3 mm core, with arm support to 9 mm inside a 12 mm neighborhood. Rejected neighborhoods are unknown, and per-head loss denominators exclude unknown targets. These are not validated anatomical bifurcation annotations. Straight-versus-curved adaptation and tree-relative spatial conditions are not implemented.
- Geometry strata currently report GT-centerline recall; false-positive precision is global. Report equal-case summaries with explicit support. Skeleton nodes and overlapping patches do not add cases, and distinct released case IDs do not establish distinct patients.

## Published reference values

These are literature references, **not reproduced results or a shared-protocol ranking**. They concern AVT or its use as the public SEG.A. training data, and must not be confused with the separate AortaSeg24 dataset.

| Primary study | Evaluation | Reported result |
|---|---|---|
| [Yuan & Yang, AVTSeg, 2023](https://arxiv.org/pdf/2305.09833) | Public 56 cases, random five-fold evaluation | Dice 90.7% with original intensities; 92.0% after reported KiTS/RIDER intensity correction. Table 1 does not report HD. |
| [Myronenko et al., Auto3DSeg/SegResNet, 2023 preprint / 2024 chapter](https://arxiv.org/pdf/2310.04114) | Public five-fold single-model evaluation | Dice 94.312%. The separate fourth-center hidden-test submission used a 15-model ensemble and reported Dice 92.0%, HD95 6.013. |
| [Wodzinski & Müller, 3D ResUNet, 2023 preprint / 2024 chapter](https://arxiv.org/pdf/2310.15827) | Public 56 cases, five-fold evaluation, 400³ input | Dice 94.41%, HD95 1.89 mm. Smaller-volume and augmentation ablations use different recipes. |
| [Li et al., HybridNet, PLOS One, 2026](https://doi.org/10.1371/journal.pone.0345549) | 2D model; 38 training / 18 test patients | Dice 87.91%, HD95 7.06 mm; ASSD 1.35 mm uses an average of directional means. |

Study partitions, training populations, 2D/3D inputs, resolution, optimization budgets, initialization, ensemble cost, and distance implementations differ from this repository. Attributed patient wording in the literature table does not verify subject independence for this release. Aggregate Dice does not demonstrate small-branch or junction-specific benefit. New GTA claims require matched comparisons here, with failures and uncertainty retained.

## Results and provenance

The [complete four-model TEST12 report](reports/minimal4_reference_TEST_20261023/README.md) provides held-out measurements and both favorable and adverse evidence. The [VAL8 report](reports/minimal4_reference_VAL_20261023/README.md), [condition diagnostics](reports/minimal4_VAL_diagnostics_20261023/README.md), and [training summary](reports/minimal4_optimization_summary_20261023/readout.md) remain separate development evidence. The [reproduction guide](docs/reproduce_minimal4.md) gives the exact current four-model entry points. Historical records below preserve earlier versions and scope decisions.

[V1 development measurements](reports/development_v1_20261022/summary.json) include 32 case/timepoint/model rows in [casewise.csv](reports/development_v1_20261022/casewise.csv), with original run, source and checkpoint hashes. The original summary's `statistical_unit: "patient"` label is inaccurate: its observations are eight released case IDs with unverified subject independence; the [reporting erratum](reports/development_v1_20261022/reporting_erratum_20261022.json) preserves the original file identity and numerical values. At 5,000 updates, C achieved VAL8 Dice 73.8430% and the full-A bridge 69.5451%; both retained substantial errors. They trained under different hardware/software roles, so this difference does not estimate the GTA effect.

[Original-v2 first-endpoint records](reports/development_v2_endpoints_20261022/readout.md) release all 32 raw case/timepoint/model metrics and four original summaries, plus casewise tables. At 10,000 updates, MAIN C reached Dice 80.3795% and SECOND A bridge 85.9490%; this cross-role difference is not a GTA effect. Both improved all eight Dice values from their own 5,000-update point. Relative to its own V1 endpoint, C still regressed strongly on K13 (Dice -13.606 pp, FP +216.624 ml), and the bridge's <=2 mm centerline recall decreased 11.086 pp despite its better overall Dice. The [fixed V1 same-weight extent record](reports/development_extent_v1_20261022/readout.md) reports all 12 development evaluations: 96 versus 128 has mixed case effects and lowers both three-case mean Dice values, with larger FP/centerline-precision costs. No native images or predictions are released. These records informed the reduced scope; they are not final four-model or TEST results.

Result records identify the immutable configuration, full training recipe, code, split, checkpoint, evaluator, environment and prediction provenance. A single-seed internal ablation cannot be described as independent multi-seed or complete five-fold evidence.

[release_source_hashes.json](release_source_hashes.json) records the copied source bytes. [NOTICE](NOTICE) and the historical third-party notices explain source lineage and licensing boundaries. No blanket MIT license is selected for this repository or for later GTA additions.
