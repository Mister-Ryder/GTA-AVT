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

The present AVT instance retains the historical backbone, independent attribute
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

The current four-model study answers two limited questions separately. MAIN
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

See the [adaptation boundaries](adaptation_boundaries.md),
[v2 recipe](common_recipe_v2.md), [four-model scope](minimal4_scope_20261022.md)
and [case-independence limitation](avt_case_independence.md).
