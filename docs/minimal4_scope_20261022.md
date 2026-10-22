# Reduced four-model scope with the unchanged v2 recipe

Snapshot: **22 October 2026, 16:10 UTC**. This scope was adopted after development evidence, not declared before that evidence was observed. The original seven-job v2 queues were administratively closed after their first model on each training role completed. Their remaining five jobs did not start. The original seven-model plan is preserved and remains incomplete.

The [portable scientific plan](../configs/bounded_20261022_v2_minimal4/scientific_plan.json) fixes exactly these model tuples:

| Training role | Job | Method | Origin and status at snapshot |
|---|---|---|---|
| MAIN | `C_concat` | `faithful_concat` | Genuine original-v2 10,000-update endpoint; retained without retraining |
| MAIN | `A_soft` | `faithful_soft` | Fresh training started; no completed result yet |
| SECOND | `A_soft_platform_bridge` | `faithful_soft` | Genuine original-v2 10,000-update endpoint; retained without retraining |
| SECOND | `A_soft_constant` | `faithful_soft_constant` | Fresh training started; no completed result yet |

All four use the exact v2 recipe after removing only the method name: random initialization, seed 20261022, 1 mm cache, 128-cube training, batch two, 10,000 successful updates, the same auxiliary scale/structure supervision, AdamW and polynomial schedule, and VAL8 at updates 5,000 and 10,000. All 24 frozen source files retain their bytes. The [two new recipe files](../configs/bounded_20261022_v2_minimal4/) are exact byte copies of the existing v2 A/constant recipes. This does not define algorithm v3. Equal registered parameter inventories do not imply equal active computation.

The two within-role comparisons have separate estimands:

- **MAIN A minus C:** specialized operators and predicted conditioning jointly versus ordinary attribute-feature fusion, with the same auxiliary supervision. C is not a single-task baseline. Missing MAIN constant prevents decomposing a MAIN improvement into condition value.
- **SECOND A bridge minus constant-A:** overall predicted conditions versus uniform vectors within the same registered soft-GTA model. Uniform vectors also change empty-class mass and gradient paths. This does not isolate radius/junction semantics or local anatomical targeting.

Both A models use the same seed and serve as platform bridges; they are not extra seeds. Do not pool them, subtract their difference as a platform correction, or compare MAIN C directly with SECOND A as a GTA effect. Core software matches Python 3.8/PyTorch 1.11; training hardware still differs.

MAIN constant, scale-only, structure-only and the hard anchor are deferred. No external architecture, additional seed or larger matrix is introduced. The original seven-job protocol and its strict all-seven tools must not be relabeled as completed by four models. The historical attribute-faithfulness helper requiring three MAIN C/A/constant bundles is not satisfied; do not substitute a SECOND bundle into it.

The [first two v2 endpoint records](../reports/development_v2_endpoints_20261022/readout.md) and [fixed V1 extent diagnostic](../reports/development_extent_v1_20261022/readout.md) motivated this narrower scope. Their complete per-case errors and fine-group regressions are retained. The old C and bridge keep their original version, selection and source identities; no new training is claimed for them. Neither development record establishes fair GTA effectiveness, successful small-vessel adaptation, complete anatomical branches, or a final four-model result.

The intended final assessment evaluates four normal models in one reference environment/device, followed by a validation-only availability-preserving intervention on the same SECOND A weights. It preserves each predicted empty mass and uniformly redistributes the remaining composition; it is not a fifth trained model or an isolated causal proof. Only a separately invoked all-four selection freeze may precede four-model TEST12 evaluation. No TEST has informed this scope choice, and no TEST or next job starts automatically.

This remains one split/seed and released case IDs with unverified subject independence. Geometry groups give GT-centerline recall, not group precision or anatomical ground truth. Global patch conditions cannot establish localized adaptation, and the initial depth/scale, curvature/junction and spatial questions have not all been completed.
