GTA-AVT: common-reference evaluation and cross-runtime numerical bridge

Purpose

The second GPU trains in Python3.11/torch2.7. Final metrics are computed in the
main Python3.8/torch1.11 environment using the unchanged frozen model,
inference.py/native_images, metrics.py and geometry.py. No external network,
optimizer/RNG fabrication, training-checkpoint shim, automatic test evaluation,
hyperparameter search, or frozen-code edit is performed by these helpers.

1. Export a genuine selected model in its original training runtime

Required owned-run artifacts: config.json, complete.json, genuine best.pt and
validation/step_SELECTED/summary.json. The run must have completed its declared
budget. best.pt must match the run signature, selected validation summary and
completed best validation Dice. No latest.pt or arbitrary checkpoint selector is
offered. An overflow/partial run cannot be promoted into a completed model.

Use the actual Python executable used for that run. Retain its real software
receipt (the existing results/environment_versions.json is accepted when its
Python and all five package versions match). To capture a receipt in that same
runtime, without claiming retrospective training backend settings:

  python tools/reference_weights_support.py capture-environment --output results/TRAIN_ENV.json

Then export, replacing RUN/CACHE/BUNDLE with actual paths:

  python tools/export_reference_weights.py --run RUN --manifest CACHE/manifest.json --training-environment-json results/TRAIN_ENV.json --output BUNDLE

Export verifies the exact original training manifest SHA256, current frozen
code signature, all56 raw CT/mask files, all cache arrays, exact36/8/12 patient
IDs/order, and preprocessing source. Every state_dict entry becomes a detached,
contiguous CPU tensor. weights_tensor_only.pt uses pickle_protocol=2 and legacy
Torch serialization for torch1.11. No optimizer/RNG/signature object is placed
inside that tensor-only file. provenance.json carries the original checkpoint
SHA256, method/stage/selectedstep/recipe, source hashes, genuine run records,
software receipt and per-tensor dtype/shape/raw-byte SHA256.

The exported tensor-only artifact is NOT passed to code/evaluate.py, which
expects a complete training checkpoint. Use the separate reference wrapper.

2. Common cache/source identity across relocated paths

Each verification reads every actual source/cache file and checks its saved
SHA256. The canonical manifest sorts cases by ID and excludes only cache_dir,
image/label path strings and preprocessing_seconds from case identity. Source
image/label hashes remain included. All other case/manifest metadata, including
crop/recovery geometry, intensity correction and geometry-target bytes, remain
identity-bearing. The reference cache can be relocated but cannot be rebuilt
with merely similar arrays. Cache_bridge's byte-identical handoff is suitable.

Optional inspectable full receipt:

  python tools/reference_weights_support.py cache-identity --manifest CACHE/manifest.json --output results/CACHE_IDENTITY.json

Full-file hashing is intentional and can take time on shared storage. A path
change alone is accepted; any raw/cache byte or semantic metadata change fails.

3. Explicit reference evaluation

Copy the entire BUNDLE directory, unchanged frozen code, and the byte-identical
cache to the main reference environment. Use its Python3.8/torch1.11 executable;
the evaluator refuses other runtime versions and silent CPU fallback.

  python tools/evaluate_reference_weights.py --bundle BUNDLE --manifest REFERENCE_CACHE/manifest.json --split val --output RESULTS/VAL

Model loading is strict. The weight-file SHA256 and every tensor's dtype,
shape/raw-byte hash are checked before use and again after model loading.
Frozen patient/source/data identity must match before inference. A pilot uses
only its original predeclared validation subset, never all8 by accident. A
formal model uses its original complete8-case validation selection.

Fixed inference: patch128, overlap0.5, batch2, predicted conditions, native-grid
threshold0.5, no postprocessing. The wrapper imports frozen evaluate_cases;
therefore CUDA inference retains that function's AMP autocast, and CPU does not
use autocast. Both TF32 backends are disabled, deterministic algorithms enabled,
and selected execution device/autocast/runtime/hardware are recorded. This
precision policy is the same across all reference-evaluated methods.

Test is a separate manual command, available only for a completed, genuinely
eligible formal model selected using all8 frozen validation cases:

  python tools/evaluate_reference_weights.py --bundle FORMAL_BUNDLE --manifest REFERENCE_CACHE/manifest.json --split test --allow-formal-test --output RESULTS/TEST

No helper runs this command automatically. A pilot is rejected even when the
explicit flag is supplied. Reuse an interrupted evaluation only with --reuse
and an exactly matching immutable evaluation_protocol.json. Never mix methods,
devices, source versions or interventions in one output directory.

4. Saved TRAIN-only FP32 numerical bridge

Build once from the common cache. Six input patches are fixed in advance: D1,
K2,R3 (all training patients), each with one centre and one seeded-uniform start.
Only CT arrays/metadata are used for these starts; no GT geometry, masks,
validation patients or test patients are used. Bank patch size96 (the training
patch), seed20261022; final sliding-window evaluation remains patch128.

  python tools/reference_input_bank.py build --manifest CACHE/manifest.json --output BANK

BANK contains actual normalized FP32 input/coordinate arrays in inputs.npz and
bank.json with source case IDs, starts, common-data identity and whole-array /
per-sample byte hashes. Copy that exact BANK; do not regenerate it independently.

In the ORIGINAL TRAINING runtime, using the exported genuine model:

  python tools/reference_input_bank.py logits --bundle BUNDLE --bank BANK --role training --output ORIGINAL_LOGITS

In the MAIN REFERENCE runtime:

  python tools/reference_input_bank.py logits --bundle BUNDLE --bank BANK --role reference --output REFERENCE_LOGITS

These diagnostics explicitly disable autocast and both TF32 backends, use FP32
model/input/logits and batch2, and save actual logits.npz plus environment,
selecteddevice, weights and input identity receipts. Reference role enforces
Python3.8/torch1.11. Training role verifies the recorded training software.

Finally, compare the same saved inputs and weights:

  python tools/reference_input_bank.py compare --bank BANK --training-logits ORIGINAL_LOGITS --reference-logits REFERENCE_LOGITS --output results/FP32_BRIDGE.json

The output reports overall and per-patch maximum/mean absolute logit difference.
It does not tune a tolerance, alter weights/inputs/thresholds, choose a method,
or claim segmentation improvement. It is a numerical bridge, not an empirical
cross-domain generalization experiment. Hardware and runtime differences are
both retained in the receipt.

5. Validation-only foreground-availability intervention

Optional mechanism diagnostic for a trained faithful_soft model only:

  python tools/evaluate_reference_weights.py --bundle SOFT_BUNDLE --manifest REFERENCE_CACHE/manifest.json --split val --intervention availability_only --output RESULTS/VAL_AVAILABILITY_ONLY

The runtime conditions hook keeps each head's original predicted p_empty=q and
replaces only positive-class composition by uniform. Scale becomes
[q,(1-q)/4,(1-q)/4,(1-q)/4,(1-q)/4], structure becomes
[q,(1-q)/2,(1-q)/2]. It uses no GT and changes no weights. The trained model name
remains faithful_soft; it is explicitly NOT a separately trained method.
evaluation_protocol.json records formula and hook file SHA256. Any attempt to
use this intervention on test, or on another method, is rejected. It should
inform interpretation on validation, never select the final method using test.

6. Local verification and actual-runtime boundary

  python tools/verify_reference_schema.py

This NumPy-only synthetic fixture checks Python3.8 syntax, location-invariant
canonical identity,260 raw/cache fixture hashes, corruption/spatial/split
rejection, pilot/formal/test/intervention gates and the probability formula.
results/reference_schema_QA.json explicitly states that actual Torch loading,
protocol2 compatibility, GPU inference and cross-runtime numerical agreement
have NOT been tested by that local fixture. Execute the completed pilot export,
strict reference load and FP32 bridge on the two real runtimes before using
formal metrics. No optimizer/RNG state is fabricated even for that test.
