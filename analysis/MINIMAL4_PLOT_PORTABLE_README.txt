Portable four-model figures from existing published evaluations

This candidate adds three static figures to the portable four-model analysis:
all-case Dice, ten paired metric changes, and six GT-group recall changes. Each
figure is exported as PNG and PDF with source CSV/JSON and a file-hash manifest.
No real four-model result is supplied. The program never opens images, arrays,
weights, servers or a GPU, performs inference, launches TEST or reselects a model.

Install analysis/plot_minimal4_portable.py and its verifier beside the portable
analyze_minimal4_portable.py and minimal4_statistics.py in a released repository.
The released analyze_key_ablations.py (31f355...) and plot_key_ablations.py
(9ce120...) files must remain byte-identical. NumPy and Matplotlib are needed
only for explicit admission/rendering. The default plan is stdlib-only and opens
no input files:

  python analysis/plot_minimal4_portable.py

Analyze original completed metric/summary files first with the separate portable
analysis entry point, retaining its paired.json. Then supply the actual FILE
SHA256 of that paired artifact and of the same evaluation declaration:

  python analysis/plot_minimal4_portable.py --execute-plot \
    --analysis results/portable_val_analysis/paired.json \
    --analysis-sha256 ACTUAL_PAIRED_FILE_SHA256 \
    --declaration configs/evaluation_declaration.json \
    --declaration-sha256 ACTUAL_DECLARATION_FILE_SHA256 \
    --output results/portable_val_figures

Output must be a new ordinary directory. --repository can point at a separate
released checkout; --reader-directory can identify the directory containing the
pinned portable reader and its statistics sibling when testing uninstalled
candidates. These paths are explicit local inputs, never server attestations.

For an existing TEST artifact additionally pass the same independently declared
selection record and its actual FILE hash used during its portable analysis:

    --selection-record configs/declared_selection.json \
    --selection-sha256 ACTUAL_DECLARED_SELECTION_FILE_SHA256

The reader rechecks every original summary/metrics FILE hash and all fixed
four-slot/full VAL8 or TEST12 metadata. TEST requires its separate declaration;
VAL rejects a future TEST selection record. Source admission remains at the
published-declaration/file-self-consistency level. It does not prove an author's
server/export, weight contents, source-data bytes or selection chronology.
source_marker_FILE_sha256 is always null, and declared-selection FILE and
canonical identity hashes remain distinct. Every figure and its manifest label
this narrower evidence level; nothing is upgraded to private author attestation.

The plot reconciles all case values, missing reasons, pairwise changes, slot
summaries, metric/group supports and descriptive means with the admitted raw
files. All ten metrics and six geometry groups remain present. Null/zero support
is not imputed, and explicit GT support is checked across all four slots. The
two platform-internal comparisons are never pooled or subtracted. Fraction
changes use percentage points, error volumes ml, and surface distances mm, on
separate axes. Source CSV has empty null fields; JSON preserves actual nulls.

Intervals are read from the explicit hash-locked paired artifact. Plotting checks
their support/finite bounds but never reruns bootstrap or independently attests
their calculation. Fixed case-selection records are also reconciled; this tool
does not render native prediction examples or establish local anatomical cause.
One seed and unverified case independence remain limitations.

The six plotting functions and four display constants are byte-identical copies
of the reviewed frozen four-model plotting implementation, identified by the
published extraction hashes. The private original is provenance of extraction,
not a runtime dependency. No private source-authentication gates are bypassed,
and no synthetic=True path is used to impersonate author-collected results.

Local verification uses only temporary, explicitly synthetic JSON and one
temporary watermarked VAL figure set. It makes no genuine results. For candidate
testing use the matching portable reader directory:

  python -B analysis/verify_minimal4_portable_plot.py \
    --repository RELEASED_CHECKOUT --reader-directory PORTABLE_READER_DIRECTORY

Temporary figures are deleted unless the verifier is explicitly given a new
--preview-directory for visual inspection. Such previews are watermarked
SYNTHETIC SELF-TEST - NOT RESULTS and must be removed after inspection. Actual
final figures require their own visual inspection once genuine evaluations
exist. Historical tool QA and synthetic previews are not experimental evidence.
