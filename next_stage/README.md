# Next-stage GTA-AVT source status

This source overlay extends the completed historical minimal4 study with bounded
attribute-supervision, Gaussian-field and local-condition investigations. The
implementation is preserved separately from the evidence that a mechanism works.
This source release does not redefine any frozen experiment identity.

At this documentation cutoff (23 October 2026):

- Frozen local readout and fixed TRAIN-only local oracle gates failed. Raw local A
  is **stopped**; no formal L0/Lp segmentation training was launched. Implemented
  local files and configurations are traceability records, not positive results.
- The 2,000-update Kp/Gp GDF pair completed and failed its screening gate;
  independent arithmetic review agrees. See the [GDF VAL report](../reports/next_stage_GDF_VAL_20261023/readout.txt).
  The attribute-supervision U arm completed; its matched K arm remains running,
  with no final paired result here.
- The new global preflight passed actual CUDA128, batch2, FP32+AMP QA on
  an isolated source snapshot. See [engineering scope](../reports/next_stage_engineering_20261023/README.txt).
  No local preflight, exact checkpoint replay or efficacy is certified by that pass.
- A separately frozen [learnable-interaction study](../reports/next_stage_learnable_preflight_20261023/README.txt)
  is training constant/predicted-condition pairs on two platforms, with the same
  seed and a fixed 2,000-update budget per arm. Its support probe and engineering
  checks passed; adverse attribute groups remain included. Clearn is a new
  matched control, not historical ordinary fusion C. No fixed-endpoint result,
  independent-seed variance estimate or full-GDF validation is supplied yet.
- The separate [global attribute-plus-GDF engineering check](../reports/next_stage_global_joint_engineering_20261023/README.txt)
  completed one synthetic joint update. Its named scientific candidate remains
  untrained and stopped; this check does not reverse the failed GDF gate.

The [completed diagnostic report](../reports/next_stage_diagnostics_20261023/readout.txt)
retains the negative readout/oracle results and the limited GDF field audit.
The cold oracle does not test a jointly adapted local segmenter and does not
refute all GTA implementations. It also supplies no authorization to expand the
stopped local experiment. These TRAIN diagnostics do not establish zero-shot,
independent-patient gains, repeated-seed variance or whole-stage completion.

Read [reproduction notes](../docs/reproduce_next_stage.txt) for prerequisites,
reference versions, inventory scope and remaining exact-replay limitations.
Of the original 22 next-stage artifacts (21 Python and one gate JSON), 21
retain exact bytes; mechanism_probe.py changes only its reported GPU-origin
label to reference. All 24 core artifacts and all model/trainer bytes are unchanged. The 46 deployed artifacts include the gate JSON;
the trainer's 45-file source inventory does not. New documentation hashes must
not replace immutable source/data/weight identities from original experiments.

The [historical minimal4 TEST report](../reports/minimal4_reference_TEST_20261023/README.md)
remains a distinct study with its original within-role contrasts and limitations.

Four later diagnostic artifacts are added byte for byte: local_oracle_probe.py,
local_oracle_probe_checks.py, local_oracle_gate_20261023.json and gdf_scale_audit.py.
They were not part of the historical training inventory. The pinned original
oracle plan and trained checkpoints are absent, so these sources alone do not
provide an exact replay. Default metadata plans require NumPy and genuine input
manifests; see the reproduction notes and the weight-free numerical report.
