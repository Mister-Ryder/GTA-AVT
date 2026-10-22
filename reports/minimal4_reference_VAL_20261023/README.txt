Four-model common-reference VAL8: actual numerical results, development evidence

These are the four normal model evaluations on the same reference software and
selected device. Each model completed the original10000-update recipe and was
selected at step10000 by its scheduled training-VAL rule. The same VAL8 cases
were used for selection; this is development evidence, not final TEST evidence.
The bank/classifier diagnostics and private all-four freeze are separate stages.
TEST has not been run at preparation of this candidate. Nothing here launches it.

Comparison                   Before Dice%  After Dice%  Delta pp
MAIN A minus C                 80.379503    86.289518     5.910016
SECOND bridge minus constant   84.731827    85.949031     1.217204

MAIN: the Dice gain accompanies precision+12.040832pp and FP-75.695406ml, with
recall-2.235693pp, FN+9.409222ml and centreline recall-1.848708pp. Dice improves
in6/8 cases; FN increases in7/8. Fine<=2mm macro recall is-0.511385pp and
junction recall-1.245079pp. Six cases improve fine recall, one ties and one
falls; the negative macro mean does not mean every fine-vessel case worsens.
K13 improves Dice18.961861pp but loses17.249154pp centreline recall; its29
fine GT nodes have recall72.413793%->41.379310%. K8 improves Dice13.932374pp
while clDice falls14.422429pp. R10 loses4.020528pp Dice with FP/FN both higher.

SECOND: mean Dice+1.217204pp accompanies FP-6.442915ml/FN-2.437617ml, but
ASSD+0.569664mm and HD95+3.242548mm. Global clDice changes only+0.029505pp;
fine macro recall-0.328284pp, junction+1.958288pp. Dice improves in5/8 cases.
Leaving out K8 changes the mean Dice difference to-0.133432pp; all8 remains the
primary result and leave-one-out is sensitivity, not a confidence interval.
K15 loses7.176588pp Dice, gains48.007763ml FP and41.903270mm HD95 despite
higher junction recall. D2 improves Dice1.910366pp but loses13.127424pp clDice;
its fine recall34.562212%->18.894009% over217 GT nodes. R10 loses3.762917pp Dice.

All ten metrics, six groups, case values, support and exploratory paired-case
intervals are in analysis/paired.json and analysis/readout.txt. The Dice95%
percentile case-bootstrap intervals are[1.2025,10.9437]pp and[-2.4118,5.0257]pp.
They use20000 paired resamples/seed20261022, assume case independence and do not
include seed/selection effects or multiplicity adjustment. Released case IDs
are not verified independent patients; no confirmatory significance is claimed.
Group recall is GT Lee centreline voxel coverage, not group precision, physical
length, branch count or anatomical causation. Interpret it with global FP and
centreline precision. No CT/prediction image was viewed for these explanations.

MAIN changes specialized operators and predicted conditioning jointly; C also
has the same auxiliary supervision. SECOND changes overall predicted versus
uniform conditioning, including empty mass/composition and gradient paths.
Missing MAIN constant prevents decomposition of MAIN gains. The two contrasts
are separate and never pooled/subtracted; the two soft models are not extra
independent seeds. A common reference runtime does not erase training histories.
These mixed VAL results do not establish fine-vessel/junction adaptation success.

Publication scope and reproducibility

The files under evaluations/ are byte-identical original summary.json and
case metrics.json for the four normal models. No masks, CT, labels, checkpoints,
weights, raw private evaluation_protocol.json or private deployment receipts are
included. Original protocol/runtime SHA values in the declaration bind safe
metadata without exposing their private plaintext. Timing fields are retained
from original metrics, not a controlled performance benchmark.

configs/evaluation_declaration.json is the released portable safe declaration.
It supports declaration-and-file self-consistency only. The author's internal
source/transfer/execution proofs were checked separately; this public artifact
does not independently attest their physical server, tensor contents, data bytes
or chronology. analysis/paired.json therefore explicitly uses declared source
level, not author-private verified attestation. Its source_marker_FILE_sha256 is
null. The public declared selection record is a separate pre-TEST declaration,
not a private all-four freeze marker, TEST authorization or proof of timing.

configs/declared_selection_before_TEST.json preserves the previously declared
record byte-for-byte (FILE SHA2561bc254120496e2cdbb68890019caeb4f3df680bc22104f4764428c21cf855cf7).
It is not consumed by VAL analysis and records no TEST result.

The released b936 portable reader was actually run on these genuine files.
All private427 numerical/case/support/figure-selection/statistical-method
fields exactly match the portable output, including the stored exploratory
intervals. Their provenance levels and runtime disclosure deliberately differ.
Use the published tools and manifest FILE hashes to rerun this VAL analysis.
Figures/native review and later TEST require their separately authorized stages.
