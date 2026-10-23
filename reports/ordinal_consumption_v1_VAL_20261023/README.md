# Completed ordinal-consumption development round

This candidate did **not** establish useful dynamic conditioning. All three
matched constant/dynamic pairs completed 2,000 successful updates per arm.
The primary and replication blocks failed utility gates; the locked third
block passed utility gates but failed both correspondence gates. It cannot
rescue the candidate. These blocks reuse the same VAL8 and common pretrained
state, and are not 24 independent cases or three independent pretraining runs.

| Block | Constant Dice | Dynamic Dice | Dice difference | clDice difference | Utility gate |
|---|---:|---:|---:|---:|---|
| A, primary | 86.4573% | 85.9448% | -0.5125 pp | -1.2892 pp | Fail |
| B, replication | 86.4453% | 86.6209% | +0.1755 pp | +0.4935 pp | Fail |
| C, locked confirmation | 85.6918% | 85.7844% | +0.0925 pp | +1.6580 pp | Pass |

The case-equal Dice effects of correct versus within-case permuted conditions
were +0.000349/+0.001283/+0.000474 pp for A/B/C. Preserving each patch's
predicted presence mass while permuting its nonempty composition gave
+0.000780/+0.001304/+0.001113 pp. Each intervention had support in all 12
prespecified TRAIN probe cases; none reached the frozen +0.1 pp criterion.
These are TRAIN patch diagnostics, not native VAL intervention effects.

See the [detailed Chinese interpretation](readout_zh.txt) and
[pretraining provenance and bounded next question](predictor_pretraining_zh.txt).
The latter records an uncovered training distinction, not results from a new
pretraining experiment.

The [numerical package](numbers/README.txt) contains all 48 native prediction
metric records (two arms times the same eight cases times three blocks), all
three fixed 576-row derived mechanism records, the original protocol, arithmetic
identities and a standard-library recomputation tool. No CT, label arrays,
prediction arrays, weights or patch coordinates are included. From the repository:

```bash
python reports/ordinal_consumption_v1_VAL_20261023/numbers/recompute.py --package reports/ordinal_consumption_v1_VAL_20261023/numbers --check
```

The default is inert. The package verifies membership and hashes before
recomputation. Two explicitly named descriptive variance fields alone permit
up to two binary64 ULPs of cross-platform comparison rounding; all effects,
counts, consumed conditions, support and scientific decisions remain exact.
This administrative compatibility repair did not change the frozen experiment,
recorded outcomes or scientific gates. Numerical recomputation is not an
independent observation of server execution.

The prospective source implementation remains separately documented in
[the reproduction guide](../../docs/reproduce_ordinal_source.txt). No larger
backbone, external baseline, new TEST evaluation or formal combined GDF training
was added in this round. Dynamic efficacy remains unverified.
