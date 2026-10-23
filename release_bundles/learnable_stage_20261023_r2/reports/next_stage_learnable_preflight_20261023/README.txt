Learnable global condition interaction: prospective source and completed preflight
23 October 2026. This package contains no fixed-endpoint segmentation results.

One new candidate uses the original constant GAAM output G0:
  Y = G0 + alpha * G0 * tanh(b + w*c)
There are128 channel weights and128 biases, all initialized at zero. The module
stays at the original128-channel GAAM site. It can learn signed channel gains;
it adds no local head, new geometry target, backbone or GDF supervision.

Contrast scope

Clearn is the NEW learnable-interaction constant-condition control, c=.2.
It is not the historical ordinary-feature-fusion method C. Dlearn consumes the
detached raw fine component of the global five-class CT-predicted scale vector.
The two recipes differ only in conditioning and role. Compare their own matched
endpoints; do not mix new Clearn numbers with historical C or subtract contrasts.
Equal parameter counts do not make the two effective function families identical.

Both inherit the original constant checkpoint's attribute-supervision history,
then use the same global auxiliary CE weights.1/.1. Clearn/Dlearn cannot test
whether attribute supervision itself helps; the independent random-init U/K
comparison addresses that question. Both use the same alpha0/500 introduction;
this pair does not separately prove that staged training is necessary. GDF weight
is0 in both, with SAAM uniform. This is not full-GDF scientific validation.

The fixed recipe is2,000 successful updates, original constant warmstart with
fresh optimizer/scaler, seed20261023, input/inference128-cubed, batch2, lr1e-4,
poly lr=lr0*(1-step/2000)^.9, clip12 and the original sampler/augmentation/AMP
retry rules. The same-card ordered pair is the primary platform block; another
ordered pair uses the same seed on a second GPU/driver platform. These are two
platform blocks of one candidate, not independent seeds or a variance study.
Mean Dice/clDice, case SD, worst-two, large regressions, FP/FN and supported
geometry groups retain the existing prospective decision rules. Final same-weight
TRAIN diagnostics re-pair conditional composition while preserving each patch's
predicted empty mass; an overall Dlearn gain alone will not prove fine-scale
anatomical correspondence. See the separately identified public protocol copy.

Completed support probe: a gate about support, not information or efficacy

The already fixed12 TRAIN cases and8patches/case completed once, without fitting
or resampling. Global128-patch attributes were used; the old local16mm readout
was not consumed. There were74 known-nonempty supported patches and11/12
informative cases under the prespecified variation thresholds. The support gate
passed, with overall raw-condition population SD.04851149289594932.
All original cases/strata are retained. TRAIN cases were already seen by the
historical backbone; these are not unseen-patient or zero-shot measurements.

Across74 known-nonempty patches (patch-weighted, fractions, MSE in fraction²):
  raw fine MSE.030776816367673453; Pearson.23613913042020676
  conditional fine MSE.02984386448734992; Pearson.2816214449877522
  fixed.2 MSE.036466270429562124
  in-bank GT-mean constant MSE.0323432826739335
The in-bank GT mean is descriptive and optimistic, not a fitted deployable
baseline. Small aggregate MSE differences do not establish information or useful
conditioning. In the48 primary GT-jitter patches, raw MSE.022029597326047038
is worse than their own in-bank mean baseline.02110731620997894; Pearson is
only.09730101834590223. Rider correlations are adverse: raw−.017502605995926134
and conditional−.21296604004100447. Cohort and per-case records remain included.

Earlier evidence remains: the separate global TRAIN16 diagnostic had8 nonempty
patches, raw-fine/GT correlation approximately−.073312, raw MSE.0137307 versus
fixed.2 MSE.0157496 and in-bank mean MSE.0097202. Different banks and supports
cannot be pooled or called a measured improvement. Earlier frozen-readout,
local-oracle and GDF continuation gates remain failed; this new support gate
does not reverse them. See ../next_stage_diagnostics_20261023/readout.txt and
../next_stage_GDF_VAL_20261023/readout.txt.

Completed engineering QA

An earlier CPU FP32, patch64/batch2 check observed exact zero-init outputs and
common preclip gradients in eval mode. It used the same model bytes and an
earlier QA-helper revision; it did not certify warmstart replay or exercise the
final CUDA/train-mode helper.
The two final CUDA128/batch2 AMP checks both passed using the same36-file
scientific snapshot. Zero-init outputs, train-mode outputs/BN/RNG and alpha0
fallback were exact. Common CUDA preclip gradients were NOT bitwise identical:
maximum differences.00048828125 were assessed against each platform's measured
baseline-repeat background, under the declared rule max(1e-6,2*repeat maximum).

Each platform independently restored the same starting model/BN/RNG and used
fresh optimizer/scaler for one constant-arm and one predicted-arm AMP update.
Both w and b had128 finite nonzero gradient entries and changed in both arms.
Each update needed3 attempts (2 AMP retries), with scale65536->16384. Elapsed
times were20.697596549987793s and22.027610540390015s. Deterministic algorithms
used warn_only; this is not a promise of bitwise training or repeat-seed stability.

The second platform's first attempt exited1 when a SciPy extension could not map
from a non-executable memory-backed environment. It did not produce a passed QA
result; some preceding model checks had run. Moving the environment to executable
storage, with the same scientific36 files and package versions, then passed.
The failed attempt is retained in the QA summary, rather than relabelled a pass.

Source identity and reproduction limits

The three new model/trainer/checks sources and the two requested recipe files are
byte-identical to the actual frozen artifacts. The public protocol is a metadata-
sanitized copy of the prospectively frozen protocol; original and public hashes
are separate. The actual GPU QA snapshot had36 files. In the corresponding public
map,35 match exactly and mechanism_probe.py retains the existing public-only
output-label adaptation. The new identity map does not replace the previously
published52-file identity, and the whole expanded public tree was not GPU-tested.

Original weights, patient arrays, private execution configurations and original
fixed-bank coordinates/readout plan are absent. Exact endpoint/bank replay is
therefore not supplied by this package. The trainer is locked to the named study
origin, not a general arbitrary-checkpoint training service. Run fresh source-bound
checks before a separately authorized execution; never fabricate a preflight.
The existing joint-GDF engineering bundle is included byte for byte as a separate
prior bundle. Its untrained/stopped candidate is distinct from this new study.

attribute_basis.txt and attribute_casewise.csv document the already completed
proxy audit. bank_groupwise.csv contains derived case/stratum moments, not image
data or patch coordinates. recompute.py --check uses stdlib arithmetic to verify
these numerical summaries and file hashes without private receipts. Optional
--project checks the corresponding public36 bytes only; it is not a GPU pass.
