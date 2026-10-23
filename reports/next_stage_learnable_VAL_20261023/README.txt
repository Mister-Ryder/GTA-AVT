Learnable signed conditional interaction: two platform blocks, fixed VAL endpoints
23 October 2026. Scientific continuation decision: STOP; both pilot gates FAILED.

What was tested

Each platform ran the prospectively fixed Clearn then Dlearn pair from the same
original constant warmstart, with fresh AdamW state and the same seed20261023.
Both use the original constant operator G0(q0=.2,s0=.2) and the same256 added
parameters:128-channel w and b initialized exactly zero. The output is
  Y = G0 + alpha * G0 * tanh(b + w*c).
Clearn consumes c=.2; Dlearn consumes the detached global raw fine probability
from the image classifier. Scale and structure auxiliary CE weights are.1/.1;
the structure condition remains uniform. GDF weight is ZERO in these four runs.
The only requested recipe differences within each pair are conditioning and role.
Both arms retain128-cube/batch2, foreground sampling probability.5, fixed2000
successful updates, learning rate.0001 with the original poly schedule, weight
decay.0001, clip12 and alpha_start0/ramp500. No best-step or best-platform selection.
Clearn is a learned constant-interaction control, not ordinary feature fusion.

PRIMARY is the original primary hardware/software block. SECONDARY is the other
prospectively declared platform block. Each C/D comparison used the same physical
GPU and runtime within its own block. The blocks differ in device/runtime; they
are reported separately, never pooled or selected by performance. The same seed
across platforms is NOT independent training-seed replication. Public runtime
profiles retain software/device model/backend, excluding machine identifiers.

Native VAL8 results: Dlearn minus Clearn

PRIMARY mean Dice86.60512409 ->86.22885612%, delta-.37626797 percentage points;
clDice73.78446368 ->73.32603559%, delta-.45842809pp. Case Dice SD decreases
.18978563pp and own-worst-two Dice increases.04290077pp, but neither establishes
variance reduction in training or robustness. Dongyang_D5 Dice falls3.77511779pp:
precision falls8.00321378pp, FP volume increases17.46028431mL, ASSD rises4.70221930mm
and HD95 rises31.87613362mm. Its recall and clDice improve, illustrating why a
single overlap or centerline score is insufficient. Five cases improve Dice,
three decline; omitting D5 changes the seven-case mean to+.10928201pp. The full
eight-case result and all leave-one-out values remain primary/descriptive.

SECONDARY mean Dice86.56130248 ->85.46796019%, delta-1.09334229pp;
clDice73.56265533 ->72.06450469%, delta-1.49815063pp. Own-worst-two Dice declines
.62877604pp and D5 Dice falls5.66784325pp. Mean centerline recall rises.21184850pp
while centerline precision falls3.19948419pp. This is a descriptive precision/
recall tradeoff, not a causal decomposition of the nonlinear mean clDice.
Every seven-case leave-one-out mean Dice difference remains negative.

The unchanged prospective pilot gate requires: mean Dice and defined clDice
differences nonnegative; own-worst-two Dice difference nonnegative; no case Dice
decline beyond3pp; and at least one substantive signal (Dice+.3pp, clDice+.5pp,
worst-two+.5pp, or case Dice SD reduction1pp). Both gates fail. These thresholds
are exploratory continuation rules, not statistical significance or clinical
margins. No experiment is automatically resumed by this report.

TRAIN condition correspondence diagnostic

The same final Dlearn weights were evaluated twice, with no updates and with
model parameters, BN and RNG unchanged. The fixed bank comprises12 seen TRAIN
cases and96 patches, including four GT-centerline-plus-jitter primary patches
per case and four CT-uniform patches. There are74 known-nonempty,17 valid-empty
and5 unknown patches. Unknown is not empty. GT geometry defines diagnostics and
scoring only; normal inference conditions use the image, never GT.

The intervention keeps each patch's own predicted presence a=1-q, and half-ring
re-pairs the full predicted four-component conditional radius composition r among
same-case known-nonempty slots. It then consumes c'=a*r_fine(donor), holding the
weights and G0's q0=.2 fixed. All12 cases meet intervention-support criteria:
at least two known-nonempty patches, nonzero exact FP32 normal-c variance, and
at least one primary patch with changed consumed FP32 c. All case/support/null
records are retained; unsupported cases would have null mechanism effects, not0.

Correct-minus-repaired primary Dice is-.000371389635pp on PRIMARY and
-.000036116027pp on SECONDARY, averaged over the original four primary patches
per case then equally over12 cases. These tiny nonpositive effects supply no
positive evidence of useful correspondence at this endpoint. They are weak
negative diagnostics, not equivalence tests or proof that attributes can never
help. Learned w and b are nonzero in both runs; parameter motion does not imply
useful conditional information. Mean absolute logit responses are.000604317 and
.000475182, respectively, so the condition path is numerically active but weak.
The diagnostic mixes GT-jitter and nonempty CT-uniform donor strata within case.
It preserves conditional-r composition multisets, NOT the raw consumed-c multiset;
presence/composition reassociation and gain intensity cannot be fully separated.

Attribute semantics and pseudo-target limits

The radius attributes are geometric pseudo-targets from the original centerline
and physical EDT proxy, not clinical annotation classes or topology guarantees.
Bins are <=2,(2,4],(4,8],>8mm. A global fine probability combines presence and
composition. The tables retain both raw fine and conditional fine, all five
probabilities, GT availability and source. Known-empty has q1 and undefined r;
unknown q/r are null. Initial other predicted-r components were not recorded and
are null; only initial recorded conditional fine is reported. Final D r4 is
recorded. Final C bank predictions were not measured and are not fabricated.

On74 known-nonempty patches, INITIAL_SHARED raw/conditional fine MSE is
.03077681637/.02984386449. Final PRIMARY is.03181479703/.02973538933 and
SECONDARY.03145174249/.02944777498. Raw MSE slightly worsens; conditional MSE
improves only modestly. Final raw/conditional fine correlations are.17619/.29540
and.19925/.30527. Source-wise weaknesses remain: Rider conditional correlation
is negative initially and at both endpoints; final conditional MSE is worse than
the optimistic within-bank Rider GT-mean constant. Those in-bank constants are
descriptive lower-variance comparators, not TRAIN-fitted deployable baselines.
The support/nondegeneracy preflight did not require semantic accuracy and its
pass must not be relabeled as attribute validity or segmentation efficacy.
Prior cold-readout/oracle negative evidence and the failed constant-condition
Kp/Gp Gaussian-field continuation gate remain unchanged. The newer combined
learnable+GDF implementation has a separate one-step synthetic engineering pass;
it is still formally STOPPED/UNTRAINED and has no AVT efficacy endpoint here.
Idle hardware is not evidence for a combined rescue experiment.

Training windows and cost

training_windows.csv contains the fixed first100,last100 and four500-step
quarters, including population SD, median, range, losses, LR, alpha, pre-update
gain/condition diagnostics and post-update parameter norms. No favorable window
was selected. Rows are derived moments, not raw training logs, and cannot replay
the training trajectory or establish convergence/gradient dominance. Different
sampled patches and a changing LR prevent interpreting endpoint loss slopes as
controlled learning curves. GDF0 placeholder losses are zero because disabled,
not because a Gaussian target was learned. Gain diagnostics are relative feature
multipliers, not logit or causal task effects. CUDA nondeterministic backward
operations remain; no bitwise training claim is made.

Logged successful-update durations total9138.643489s (2.538512GPU-hours):
PRIMARY Clearn2220.303837s,Dlearn2216.407157s; SECONDARY Clearn2339.747168s,
Dlearn2362.185327s. AMP retries are10/8/8/7, respectively, with skipped optimizer
steps restored before retry. These durations include update-time retries but
exclude native VAL evaluation, dataset preparation, transfer, other engineering
checks and server rental overhead. Mechanism diagnostic times are separately
reported. Four runs do not constitute four independently sampled training seeds.

Files and independent recomputation

Run: python -B -I -S recompute.py --check
Only Python stdlib is required. The script verifies the exact file whitelist and
hashes, reconstructs macro means/sample SD, paired deltas, own-worst-two, LOO,
all gate parts, native FP/FN counts, all six geometric recall groups and null
support; recomputes all TRAIN case/intervention summaries and source-wise
attribute MSE/correlation from the released rows. arithmetic.py contains only
copied pure functions from the frozen numerical readers, with one function rename.
This is derived-statistic reproducibility, not execution attestation or model
replay. Training-window moments are bound/checked, not reconstructed from logs.

VAL proportions are percentages and their deltas are percentage points. Surface
distances are mm; physical FP/FN volumes mL; native voxel counts are counts.
Native Dice=2TP/max(P+T,1), while TRAIN patch Dice uses the historical1e-6 numerator/
denominator epsilon. Group metrics are GT-centerline recall only, not group Dice
or precision. Zero support remains null. Macro means weight cases equally; case
SD is sample SD (n-1). Native count totals mix voxel volumes and are not pooled
physical volumes. The mean clDice is the mean of casewise harmonic scores, not
the harmonic mean of macro precision/recall. Empty CSV fields mean null, not0.

provenance.json retains checkpoint identities, metric/source/report hashes and
the original private36 scientific inventory separately from the corresponding
public36 inventory (35 exact, one existing mechanism output-origin metadata
adaptation). It does not claim the expanded public tree was CUDA-executed. Recipes
are exact safe copies. The bank's earlier34-file support snapshot is separately
identified; it is not mislabeled as the later36-file training snapshot. No model
weights, patient CT/mask/field arrays, coordinates, features, private machine
identifiers, remote access details, raw receipts or TEST metrics are distributed.
Missing original weights and controlled data prevent exact end-to-end replay.

These are reused development VAL8 cases, not a new blind TEST. Patient-level
independence has not been established. All source cohorts occur in TRAIN; source
groups and case dispersion do not measure OOD or zero-shot generalization.
Single warmstart/seed, ordered C-then-D execution, small supports and cross-device
runtime differences limit causal and generalization claims. The adverse endpoints
are retained without post-hoc threshold, target, step, seed or platform selection.
