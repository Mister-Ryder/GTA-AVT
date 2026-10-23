# Reference global engineering QA

The revised global preflight completed in one observed reference execution on
23 October 2026. Both actual QA subprocesses exited zero. An independent local
consistency review checked the seven collected original JSON files, their hashes,
source maps, command scopes, output bindings and phase records. This verifies the
received evidence; it is **not independent remote-execution attestation**.

[The public summary](engineering_summary.json) contains only whitelisted numerical
QA metadata, code hashes, backend settings and limitations. Original execution
receipts, data manifests, case identifiers, machine details and file locations are
not distributed here.

- TRAIN audit: 36 cases; 144 image/mask/geometry/GDF files fully rehashed,
  totaling 13,977,659,043 bytes. Held-out case arrays were not opened. Selected
  GDF values agreed with the same-source CPU64 reference at 288 centerline seed
  positions, and 288 selected background values were zero.
- Model QA: actual 128-cubed input, batch two, FP32 and AMP; all 16 phases passed,
  including exact constant/alpha-zero logits and two deep-supervision outputs,
  attribute-routing gradients and Gaussian-head/shared-feature gradients. No
  optimizer update or study-image input was used by the model QA.
- Sampler QA: 24 synthetic samples, including 16 padded and eight with unknown
  foreground geometry; common inputs/targets matched the frozen reference.
- Whole preflight elapsed time: 52.522 seconds; model QA: 25.158 seconds. Maximum
  reported CUDA allocated/reserved peaks were 21,837,980,672 / 28,735,176,704 bytes.
  These are observations, not runtime, memory-capacity or training guarantees.

The actual QA snapshot contained 47 trainer-inventory source files: the frozen
overlay's original 45 remained byte-identical, and two memory-safe QA files were
added. Its deployment also retained one non-executable gate JSON, making 48
artifacts. The portable tools were outside that snapshot. The assembled public
superset contains 52 executable/historical source files, including later
diagnostics and the portable tools; that entire superset was **not** the source
tree executed in this reference run. Matching entrypoint/QA bytes do not certify
a different inventory. Each reader must run a new source-bound preflight locally.

The old graph-retaining QA failed with an OOM and remains in the audit history.
The separately identified replacement releases unused graphs between checks;
scientific model/trainer bytes and assertions were retained. Its deterministic
forward profile uses TF32 off and CUBLAS `:4096:8`; backward uses deterministic
algorithms with `warn_only=True`. This does not promise bitwise training repeats.

This engineering pass establishes no segmentation improvement, Gaussian-field
benefit, topology guarantee, stability or zero-shot result. It does not reproduce
training endpoints, certify local-learning preflight or reverse the failed local
oracle gate. Raw local A and its predicted-normalization successor remain stopped.
