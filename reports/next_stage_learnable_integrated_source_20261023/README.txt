Learnable interaction plus physical Gaussian field: completed engineering check
23 October 2026. Formal candidate status: UNTRAINED, STOPPED, NOT LAUNCH AUTHORIZED.

Actual outcome and scope

The new combined helper completed on the ORIGINAL private37 scientific snapshot:
CUDA128-cube/batch2 AMP, normal detached image-predicted conditions, one successful
original AdamW update, natural exit0, elapsed11.458112478256226s. It needed2 AMP
attempts: the first finite-forward/nonfinite-gradient attempt skipped the optimizer,
reduced scale65536 to32768 and restored BN/RNG before retry. No AVT arrays,
VAL/TEST, checkpoint, formal training or segmentation efficacy result was produced.
The normal prediction path used a synthetic image; no GT target was a model input.
This is evidence that the named new joint computation can update, not that GDF or
dynamic conditions improve AVT segmentation. It does not reopen failed gates.

The model, attribute definitions, data pipeline and GDF implementation remain
unchanged from frozen learnable36. The isolated integrated trainer adds the fixed
GDF auxiliary weight.1 with voxel normalization. Global scale/structure CE weights
are.1/.1, the structure condition stays uniform, and normal raw-fine consumption
is detached. The recipe retains128/batch2,2,000 successful updates, poly decay,
original7406 constant warmstart and alpha_start0/ramp500. Alpha1 in this QA is
explicit synthetic branch coverage, not a change to that untrained recipe.

The actual joint components before the successful update were:
  segmentation1.8454604148864746
  scale CE2.9491565227508545; structure CE2.001149892807007
  voxel GDF MSE.7145082950592041
  total2.4119420051574707 = segmentation + .1*(scale+structure+GDF),
  subject to FP32 accumulation rounding (stdlib check tolerance5e-7).
The preclip gradient norm was8.918920516967773 with clip threshold12.
Peak allocated/reserved CUDA memory was13747663872/19732103168 bytes.

All eight representative parameter groups had finite nonzero joint gradients and
actual parameter changes: segmentation head, scale head, structure head, GDF head,
shared segmentation feature, shared attribute encoder, interaction weight and bias.
Both128-element interaction tensors changed. Tables record the inspected parameter
and its exact counts/magnitudes; they do not assert every parameter in each module
changed or independently attribute a shared gradient to every task.

A separate GDF-only AMP VJP, diagnostic scale1024, made NO optimizer update and
released its graph before the joint update. Five representative routes were
nonzero: GDF head, shared segmentation feature, shared attribute encoder and both
interaction tensors. Three terminal routes were zero as expected: segmentation,
scale and structure heads. BN/RNG were restored. This tests a physical synthetic
fixture's gradient wiring, not gradient dominance, conflict, convergence or benefit.

Physical fixtures and units

The synthetic128mm cubes contain cylinders of mask radius3 and6mm at1mm spacing.
Their foreground counts were2755/10735 and Lee skeleton supports93/89 nodes.
Physical EDT proxy ranges were2–3.1622776985168457mm and4–6.082762718200684mm;
end-cap/grid effects mean these are not uniformly the nominal cylinder radii.
The unchanged complete-seed cumulative Gaussian was foreground-masked, log1p
transformed and divided by fixed TRAIN scale2.8707158419291448, with no per-case
normalization or truncation. Kernel evaluations were256215/955415 and target maxima
.7625421285629272/.9711592197418213. Fields and MSE are dimensionless; radii and
spacing are mm, memory is bytes, time is seconds. These synthetic geometries are
not anonymized patient samples. No target field was supplied as a model input.

Scientific status and boundaries

The earlier constant-condition Kp/Gp GDF pair failed its continuation gate; see
../next_stage_GDF_VAL_20261023/readout.txt. The older global joint QA did NOT
cover this newer interaction; this actual new check now supplies its own evidence.
Current Clearn/Dlearn have GDF0, and no final paired/correspondence results are
included here. This report establishes no AVT gain, stability, zero-shot behavior,
local+GDF result or successful full scientific transfer. A one-step synthetic pass
and interface compatibility do not authorize a combined training experiment.
The named integrated scientific candidate remains STOPPED and UNTRAINED.

Source identity

learnable_integrated_train.py, learnable_integrated_joint_qa.py and the recipe
retain their original frozen bytes. The original private candidate manifest SHA
is b08643e41e8e405127d280d30ebd23f34cb81774dd2438eb85e41abc521aed0c.
The separately named PUBLIC manifest changes only one existing metadata-source
SHA in each37/36 source map: mechanism_probe.py's previously published output-
origin label adaptation. Model/loss/trainer bytes are unchanged. Original/public
hash maps and manifest hashes are separate in source_identity_mapping.json.
The original private manifest is not copied or overwritten. Neither this release
identity nor a public binding pass replaces immutable private experiment identity.
The actual CUDA check used the original private37/b086 manifest. The mapped public
manifest73a393f8278db2c3a81a66c67225ddab3674db7a147ba6d2af94e6b6412bba3c
and the expanded public repository were NOT CUDA-executed. The full repository
is not the isolated37-file scientific tree.

Independent arithmetic and inert plans

QA_summary.json and the three CSV tables preserve the actual components, eight
parameter groups and two physical fixtures. Run recompute.py --check with Python
stdlib to verify weighted-loss arithmetic, table bindings and physical log1p
checks. Tolerances are5e-7 for FP32 total-loss rounding and1e-7 for FP32 target
values; this is not a bit-exact Python replay of CUDA. Optional --project SNAPSHOT
--qa-helper QA_FILE also invokes the public-source binding and two inert plans.

After applying the source overlay, run these metadata-only commands from the
public repository root (no --execute). They do not validate the expanded tree:

python next_stage/learnable_integrated_joint_qa.py --manifest next_stage/learnable_integrated_candidate_public_manifest_20261023.json --recipe next_stage/learnable_integrated_candidate_UNTRAINED_20261023.json
python next_stage/learnable_integrated_train.py --candidate-manifest next_stage/learnable_integrated_candidate_public_manifest_20261023.json --config next_stage/learnable_integrated_candidate_UNTRAINED_20261023.json --manifest NOT_OPENED_BY_PLAN.json --output NOT_CREATED_BY_PLAN

The separately named manifest must be passed explicitly. The unchanged default
filename identifies the private manifest, which is absent from this source overlay.
For the binding check, prepare an isolated snapshot containing ONLY the37 mapped
scientific files, the recipe and the public manifest; keep the QA helper external.
Run check_public_binding.py --project SNAPSHOT --qa-helper QA_FILE. It checks
all37 corresponding public source hashes and the external helper, reconstructs
the original manifest byte hash, checks its expected rejection of the adapted
public tree, and runs both inert plans. It uses only Python stdlib;
it never imports Torch, executes CUDA, creates a preflight or opens study arrays.

No weights, patient arrays, field arrays, original target metadata or private
FULL preflight are distributed. The pinned7406 constant state and exact GDF target
manifest18820f0b0cfdc2316aa2563bdad44a6209fb8cd35d1d9c4d04a4c3943e1fb1e9
remain controlled dependencies. Sources, definitions and hashes do not supply
exact private replay of the actual original37 CUDA result. The target-generation code is available, but regenerating
metadata at a different location does not imply the original manifest file SHA.
A hypothetical new execution would need its own reviewed source/input/runtime
binding, actual new joint QA, input and GDF audits, and explicit authorization.
The frozen formal guard rejects this STOPPED candidate before model execution;
never fabricate a passed FULL preflight or reuse an old GDF0/old-operator receipt.
