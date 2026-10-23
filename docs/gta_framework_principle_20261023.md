# A shared GTA principle and its AVT instance

GTA can be framed as a testable relationship between anatomical attributes and
feature processing. The airway implementation, including its generation-,
structure- and location-related components, is one instance of that relationship.
Those three choices need not become mandatory components of every instance.
This is a working research definition, not a theorem or a claim that the original
airway experiments established a general framework. Further work in the present
scope concerns AVT; no additional airway experiment is implied.

Three commitments make this definition substantive:

1. **A measurable attribute with a processing rationale.** Define its units,
   reference construction and reliability, then explain why a particular feature
   operation could address the associated segmentation demand. Neither branch
   depth nor a junction label automatically establishes small calibre or difficulty.
2. **Conditions available from CT at inference.** Reference annotations may
   supervise attribute learning, but the deployed condition must come from the
   image. A reference tree supplied at inference would test a different claim.
3. **An explicit effect on features.** Specify the path
   `CT -> predicted attributes/features -> conditional feature operation -> segmentation`,
   its layer, formula and gradient paths. Permit evidence that the condition is
   inaccurate, uninformative or harmful.

An auxiliary attribute loss can improve features without useful conditioning.
Likewise, calling attention weights “anatomical” does not establish an
attribute-to-operation relationship. The computation and appropriate controls
must distinguish the claim from extra capacity, auxiliary supervision or generic
attention. This relationship alone also establishes no novelty or causal proof.

When the deployed condition is a deterministic estimate from the same full CT
input, it adds no independent observation beyond that input. Its proposed value
is an inductive bias, access to context or a useful processing choice in a model
with finite capacity. Reliable geometric meaning, image predictability and an
appropriate feature operation are separate requirements; this framing does not
guarantee improvement or constitute a new mathematical theorem.

| Adaptation axis | What must be disclosed |
| --- | --- |
| Attributes | Geometric meaning, units, classes or continuous targets, empty/unknown policy and reliability |
| Conditions | Hard or soft representation, spatial granularity, exact arithmetic, availability terms and gradient paths |
| Operations and components | Feature sites, operators and their parameters, fusion paths, active computation, omitted components and regularizers |
| Training | Initialization, staged or joint learning, auxiliary losses, optimization and selection protocol |

These axes may change within a named, traceable GTA instance. Changing several
at once does not isolate any one contribution. A substantially new operator or
architecture requires its own algorithm description and validation; retaining
only an anatomical motivation would justify “GTA-inspired,” rather than the
identity or evidence of the original implementation.

The historical minimal4 AVT instance retains the historical backbone, independent attribute
encoder, GAAM/SAAM convolutional paths and guidance fusion. It uses physical
centreline-radius composition and operational junction-neighbourhood composition
with soft condition arithmetic. Each patch shares its condition vectors: this
does not locate individual small branches. The geometric proxies are not
clinical diameters or verified anatomical branch classes, and the empty-class
term is not calibrated uncertainty. Location conditioning and Gaussian output
supervision are omitted; spatial feature maps and sliding-window blending still
exist. Random joint training and differentiable soft conditions differ from the
historical training procedure and hard decisions. The airway results therefore
do not validate the AVT soft formulas or their individual mechanisms.

That four-model study answers two limited questions separately. MAIN
A-minus-C tests specialized operations and predicted conditioning jointly; C
also receives the same auxiliary supervision. SECOND bridge-minus-constant
tests the overall predicted-condition path, including empty mass, nonempty
composition and changed gradient paths. Without MAIN constant, the latter
cannot decompose a MAIN gain. The two A runs are not extra seeds, and the two
contrasts cannot be pooled or subtracted. Measured validation differences include
trade-offs; Dice alone cannot establish fine-vessel or junction recovery.
Interpretation requires case-level errors, clDice, FP/FN cost, supported geometry
groups and condition diagnostics. The [completed TEST12 report](../reports/minimal4_reference_TEST_20261023/README.md)
shows a favorable MAIN joint-path contrast and an unfavorable mean SECOND
predicted-versus-constant Dice contrast, with coverage/error trade-offs retained.
It does not establish stable overall value of informative conditions or a
single-component causal effect. One split/seed and unverified subject independence
remain limitations; mixed or negative evidence must narrow the claim.

The next-stage [learnable interaction](../reports/next_stage_learnable_preflight_20261023/README.txt)
is a separately named realization of the same attribute-to-feature hypothesis.
It retains the original constant GAAM output `G0` and learns
`Y = G0 + alpha * G0 * tanh(b + w*c)` at the same feature site. The predicted
raw fine-class probability `c=(1-q)*r_f`, including predicted nonempty mass,
can select signed channel gains; zero-initialized
`b,w` preserve the starting output. This is an explicit change to the conditional
operator and needs its own evidence. The condition comes from the same CT as the
segmentation features, so the experiment tests how the model processes that
input, rather than an additional source of information at inference.

The new Clearn control learns the same parameters with constant `c=.2`.
It is distinct from historical ordinary fusion C. Both new arms inherit the same
warmstart and receive the same attribute losses and gradual introduction of the
interaction. Their difference tests predicted versus constant conditioning in
this realization; it does not isolate attribute supervision or the value of the
training schedule. The independent U/K contrast addresses attribute supervision.
That [completed amended pilot](../reports/next_stage_attribute_VAL_20261023/readout.txt)
shows improved mean overlap, case tail and false-positive control with joint
auxiliary supervision under constant conditions, while fine-branch recall and
four case Dice values decline. It uses different matched-specification physical
GPUs and one seed. No shuffled-label control separates label-specific semantics
from generic auxiliary regularization; no separate head contribution is identified.
Case dispersion and two same-seed platform blocks do not measure training-seed
variance or establish zero-shot behavior. The [completed fixed-endpoint report](../reports/next_stage_learnable_VAL_20261023/README.txt)
shows lower mean Dice and clDice with predicted conditions in both platform
blocks; both continuation gates failed. Small case-SD reductions accompanying
mean and adverse-case losses do not establish a stability benefit.

A useful predicted-condition contrast would still need evidence that the
predicted composition has the stated meaning and that its correspondence to the
input matters. The fixed TRAIN diagnostic preserves each patch's predicted
empty mass while re-pairing its conditional composition. It retains all original
cases and distinguishes unsupported interventions from measured zero effects.
It does not preserve the raw-condition marginal distribution, and its seen-TRAIN
results cannot establish independent generalization or a complete causal
decomposition. Auxiliary supervision, static modulation, variable modulation and
anatomical correspondence therefore remain distinct claims.

At the two completed endpoints, every original TRAIN case supports an effective
intervention and all 48 primary conditions change. Correct-minus-re-paired Dice
effects are approximately -0.000371 and -0.000036 percentage points. Both
interaction tensors learned nonzero parameters, but useful anatomical
correspondence was not demonstrated. These fixed-bank, seen-TRAIN diagnostics
do not refute all possible GTA instances or provide a reason to extend this one.

Gaussian-field supervision is a further geometric training constraint. The
[completed constant-condition GDF pair](../reports/next_stage_GDF_VAL_20261023/readout.txt)
failed its continuation gate, with coverage and tail tradeoffs retained. The
[older global joint engineering check](../reports/next_stage_global_joint_engineering_20261023/README.txt)
demonstrates one combined update for that named operator only. The [newer interaction-plus-GDF implementation](../reports/next_stage_learnable_integrated_source_20261023/README.txt)
now has its own separately identified synthetic joint-update check. Its complete
scientific candidate remains stopped and untrained. Neither interface
compatibility nor that successful engineering step establishes its scientific
benefit or overrides the negative conditional/GDF continuation evidence.

See the [adaptation boundaries](adaptation_boundaries.md),
[v2 recipe](common_recipe_v2.md), [four-model scope](minimal4_scope_20261022.md)
and [case-independence limitation](avt_case_independence.md).
