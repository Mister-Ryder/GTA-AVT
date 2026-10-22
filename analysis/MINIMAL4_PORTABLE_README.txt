Portable paired-case analysis for four GTA-AVT evaluations

This candidate analyzes existing published summary/metrics files. It does not
train, load weights, run inference, launch TEST, choose a checkpoint or alter an
evaluation threshold. No four-model results are supplied in this candidate.

Install this directory's analysis/, configs/ and docs/ files into a checkout of
the released GTA-AVT repository, preserving relative paths. Alternatively use
--repository /path/to/released-checkout while leaving this directory intact.
Python 3.8 or later and NumPy are needed only for explicit numerical analysis.
The default plan uses stdlib, does not read inputs and does not import NumPy.

The released analysis/analyze_key_ablations.py dependency is pinned to SHA256
31f355f735f6744589e464bbfb50cef276bc4db650f5edc70c5f6938d834f3c4.
The released 24 code files and fixed36/8/12 case split are checked locally.
Checking their declarations does not verify source-data bytes or a model run.

Minimal input and source level

Copy configs/minimal4_evaluation_declaration.example.json to a new file and
replace every placeholder with actual metadata. The template is deliberately
invalid until completed. Keep each original summary.json and each original
<case_id>/metrics.json byte-for-byte in its declared relative directory.
Supply all four fixed slots and all VAL8 or all TEST12 cases in the released
order. Interventions and partial stages are not accepted. The two faithful_soft
models remain separate slots; their shared method name is not an identity.

The declaration is an independent safe schema, not the original generic
evaluation_protocol.json. Whitelist only its exact permitted fields when
preparing it from genuine outputs. Do not copy private paths, hostnames,
environment plaintext, credentials, images, predictions or weights. For the
original runtime and protocol, provide only the specified SHA256 values. Safe
reference fields retain software, precision and selected-device declarations.
The original runtime hash must be common to all four evaluations. Full typed
training recipes, declared selected steps and selected weight/provenance/
checkpoint hashes remain separate for each slot. The reader verifies the
declaration, actual original metric/summary FILE hashes, complete case sets,
fixed native evaluation fields, summary reconciliation and metric validity.

This source level establishes declaration/file self-consistency. It cannot
independently prove a generic export, physical training/reference host, original
data bytes, weight contents, checkpoint selection timing or an author-private
TEST freeze. Roles are declared for the reproduction, not server attestations.
Even a declaration generated from author outputs retains this narrower level.
Original raw files must be separately inspected for privacy before publication.

Execution

  python analysis/analyze_minimal4_portable.py

The command above prints only the plan. To analyze completed existing VAL files:

  python analysis/analyze_minimal4_portable.py --execute-analysis \
    --declaration configs/evaluation_declaration.json \
    --declaration-sha256 ACTUAL_DECLARATION_FILE_SHA256 \
    --output results/portable_val_analysis

The output directory must be new. Paths in the declaration are POSIX paths
relative to the declaration file; --repository can identify another released
checkout. Outputs are paired.json, casewise.csv, figure_selection.json and a
Chinese readout. The readout explicitly labels its declared source level.

For existing TEST files, first create a separate declared selection record from
the second template. Its selection must equal selection_identity(declaration),
and selection_identity_sha256 must equal canonical(selection). These pure
helpers are exposed by analyze_minimal4_portable.py. Record all four selected
identities and the fixed data/source/split/reference declarations. Then invoke
the analysis in a separate explicit command, additionally passing:

    --selection-record configs/declared_selection.json \
    --selection-sha256 ACTUAL_SELECTION_FILE_SHA256

The supplied record asserts pre-TEST selection; its timestamp does not prove
chronology. The reader does not run TEST or claim an author-private freeze.
source_marker_FILE_sha256 is always null. declared_selection_file_sha256 and
declared_selection_identity_sha256 are distinct fields for this local record.
VAL neither requires nor consumes a future TEST selection record.

Statistics and interpretation

minimal4_statistics.py contains five byte-identical functions extracted from
the fixed four-model analysis: distribution, scaled, figure_cases,
analyze_case_rows and text. Its source-function hashes are published in the
module. The fixed source SHA is provenance of the extraction, not a runtime
dependency on private files. Bootstrap and metric helpers come from the pinned
released31f dependency; old METHODS is never changed and no allow-partial path
is used. The candidate verifier checks the function-byte hashes.

For separate plotting code, load_verified_inputs(declaration_path, approved_sha,
repository, selection_path=None, selection_sha=None) performs the same admission
without paired estimates or bootstrap. It returns numeric_index, raw_rows,
declaration, functions, result_provenance and bound_file_sha256. The last field
contains internal resolved filenames solely for recheck_files(bound) after use;
never publish it. result_provenance is the safe output evidence/declared-level
metadata, not numerical results. analyze_files calls this loader, invokes the
unchanged pure analysis and then rechecks input bytes. Plotters can reconcile
previously saved results and their explicitly supplied FILE SHA without running
bootstrap again; this API does not independently authenticate a saved result.

The two comparisons are MAIN A minus C and SECOND bridge minus constant. They
are analyzed separately, never pooled or subtracted from each other. MAIN
changes specialized operations and predicted conditioning jointly; C retains
the same auxiliary supervision. SECOND tests the overall predicted-condition
path against uniform conditioning. It does not isolate radius/junction
semantics, and missing MAIN constant prevents decomposition of MAIN gains.
The two A models are not independent seeds. Equal-case paired bootstrap uses
seed20261022, 20000 resamples and 95% percentile intervals, under an unverified
case-independence assumption. These are exploratory intervals, not confirmatory
significance or subject-level evidence. VAL8 and TEST12 remain separate.

Reports preserve the ten metrics, case differences, FP+FN, win/tie/loss,
leave-one-case-out range and six GT geometry groups. Missing/unsupported
geometry and undefined distances remain explicit nulls; support is checked
across all four models when recorded. Geometry nodes are neither independent
samples, anatomical branches nor physical arc length. Fixed figure selection
identifies adverse examples; this program never opens images or renders them.

docs/minimal4_final_readout_rules_20261023.txt is an exact historical rule copy,
SHA256347d0850f9e54150f63c3cdee7a69216c22344eadc2b023a25e8fb92287ab565.
Its recorded experiment state describes the time it was written, not a claim
of present completion. The original bytes are retained for auditability.

Local verification

  python -B analysis/verify_minimal4_portable.py --repository RELEASED_CHECKOUT

The verifier uses temporary synthetic JSON files and released code/split
metadata. It performs no model, image, GPU or server operations and generates
no genuine result. A new --output path can preserve its QA receipt. Final
numerical parity against genuine four-model outputs remains a separate check
once all genuine evaluations are complete.
