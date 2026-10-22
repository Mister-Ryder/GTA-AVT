Portable four-model classifier diagnostic: input contract and commands

This small overlay adds tools/attribute_patch_diagnostics_minimal4.py and a manual
configs/minimal4_bundle_index.example.json to the public GTA-AVT checkout. It does
not replace the preserved MAIN3 protocol or seven-model analysis. No real four-model VAL
classifier result is reported here. No training or TEST is run.

Inputs and trust boundary

Create four genuine completed formal models with the released v2 recipes:
  main_C_concat: configs/bounded_20261022_v2/formal_main_C_concat.json
  main_A_soft: configs/bounded_20261022_v2_minimal4/formal_main_A_soft.json
  ablations_A_soft_platform_bridge:
    configs/bounded_20261022_v2/formal_ablations_A_soft_platform_bridge.json
  ablations_A_soft_constant:
    configs/bounded_20261022_v2_minimal4/formal_ablations_A_soft_constant.json
Each is fresh random seed20261022,128-cube,batch2,10000 successful updates, with
the frozen36/8/12 case split and complete VAL8 selection at5000 and10000. An
external four-run reproduction must not relabel its new C/bridge as the authors'
imported original-v2 runs. Keep the intended within-role comparisons on their
declared training hardware. The bridge is not an additional seed.

In each model's actual training interpreter, use the existing public commands:
  python tools/reference_weights_support.py capture-environment --output ENV.json
  python tools/export_reference_weights.py --run RUN --manifest CACHE/manifest.json --training-environment-json ENV.json --output BUNDLE

The generic exporter produces weights_tensor_only.pt,provenance.json and
export_verified.json. Use those actual three files unchanged. No private transfer
receipt, queue proof, server root or deployment helper is required or fabricated.
The portable gate verifies the generic export's genuine best/signature/selected
summary/completed-best evidence; it does not independently re-open training
checkpoints or all historical validation summaries. Actual tensor loading also
checks the full inventory and strict frozen model state before classifier use.

Copy the index example, retain its four exact slot/role/job/method assignments,
set bundle paths relative to the index file, and replace all12 placeholders with
actual SHA256 values. Relative parent components are allowed for normal layouts
such as configs/ plus exports/. Absolute paths and drive/URI references are not.
There is deliberately no automatic builder or method/recipe override.

Roles are explicit user declarations. Generic provenance verifies method,
recipe, checkpoint, source, cache and recorded runtime; it cannot magically prove
which physical training server supplied an identically named faithful_soft model.
This index is not an author-server attestation. Author provenance audits and
third-party declared reproduction roles must not be described as interchangeable.

Execution from this public checkout

Default plan (stdlib only; no index/bundle/manifest/array is opened):
  python tools/attribute_patch_diagnostics_minimal4.py --index configs/my_four_bundles.json --manifest data/cache_1mm/manifest.json --output results/classifier4_val

Explicit validation-only classifier measurement:
  python tools/attribute_patch_diagnostics_minimal4.py --index configs/my_four_bundles.json --manifest data/cache_1mm/manifest.json --output results/classifier4_val --device cuda --execute-frozen-validation

Use the original common Python3.8/PyTorch1.11 software with the five released
package versions. The original training/export software receipt must match it.
CPU is available only by explicit --device cpu; CUDA failure never falls back.
Actual execution records selected device/autocast/runtime after setting reference
precision. Use a nonexistent output directory; outputs cannot be resumed or
overwritten. When running the tool outside a checkout, --repository may
point to the unmodified public checkout that supplies the existing tools/code.

Unchanged measurement

The eight released stable_rng/build_geometry/unique_starts/image_patch/make_bank/
predict_bank/spearman/score_bank functions are imported unchanged from the pinned
public tools/attribute_patch_diagnostics.py. The old MAIN3 admission/main routine
is never invoked. No copies of model or geometry mathematics are introduced.

One shared128-cube VAL8 bank has at most160 unique patches: per case8 CT-uniform,
4 fine-centred,4 junction-centred,4 large-nonjunction-centred. Duplicate starts
are merged with all their stratum labels. Geometry is computed separately in the
diagnostic output with frozen native EDT/physical resampling/unknown rules. It
is only target/sampling evidence; the classifier receives normalized CT only.
TRAIN/TEST arrays are not opened. Current canonical cache checking is metadata
only here; each generic export previously verified all56 actual raw/cache bytes,
and the bank code verifies the VAL files it actually opens. No cache is modified.

Per-head validity is unchanged. Outputs distinguish classifier probabilities from
actually computed conditions: C has no consumed composition; both soft-A slots
use their own predictions; constant-A uses uniform5/3 conditions while its heads
still have learned predictions. Four separate slot keys/files prevent the two
faithful_soft instances from colliding. This is classifier-only inference: no
segmentation decoder or learned GAAM/SAAM feature gate is executed.

Report equal-case MAE/SSE, fine/junction component MAE and defined support;
within-case Spearman is descriptive only. GT-centred sampling is not population
prevalence. Operational skeleton geometry is not validated anatomical truth.
Legacy patient-named score fields mean released scan/case IDs; patient
independence remains unverified, and overlapping patches/nodes are not new cases.
Uniform constants also change empty mass, so this is not an availability-only
intervention. Do not infer segmentation benefit, feature gain or calibrated
uncertainty from these outputs, or retune after viewing them.

This helper performs no checkpoint selection, threshold search, training, TEST,
automatic follow-up, private queue validation or final two-pair analysis. The
portable analysis adapter is a separate input contract.

CPU-only synthetic review (no real arrays/tensors/forward):
  python tools/verify_attribute_patch_diagnostics_minimal4.py --repository PUBLIC_CHECKOUT --output QA.json
The verifier checks admission failures, exact public split conversion, default
no-heavy-import gating, unchanged function bytes and small synthetic score
vectors. A passing verifier is not an actual four-model diagnostic result.
