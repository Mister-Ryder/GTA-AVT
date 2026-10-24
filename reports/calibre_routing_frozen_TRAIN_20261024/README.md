# Frozen routing: attribute granularity and block matching

Three completed AR checkpoints were evaluated without optimization or state updates. The original pointwise candidate remains **STOP**. A block mean gives a more stable Dice/tail tradeoff and has a measurable block-matching effect, but its vascular losses prevent treating this frozen substitution as a validated replacement.

All original 192 patches per seed retain four fields: correct pointwise radius, its within-patch half-ring, own-patch mean log-radius, and the fixed FIT mean. A fifth field assigns a different patch mean from the same case, with donor `(slot+4)%8`, only on the original primary slots0–7. Recipient A/valid and images stay fixed. All eight primary blocks must have support. Uniform slots explicitly remain unmeasured for the fifth intervention.

The fifth intervention preserves eight equally weighted block scalars, **not** the site-weighted radius multiset. It estimates block-to-summary matching, rather than the previous point-to-radius spatial correspondence. Its effect size cannot be interpreted as an increase in the same mechanism estimate.

Actual128/B2 QA passed. All three formal jobs exited normally on 24 October 2026 at08:41:11/08:41:13/08:42:21UTC. Every original correct/half-ring count-derived metric reproduced its completed routing reference; BCE stayed within the fixed two-binary64-ULP tolerance. Full model state remained unchanged. This does not attest voxelwise parity of old masks, which were not stored.

## Measured tradeoff

All means are equal-case primary means. Differences are percentage points. Fine recall pools supported nodes within each case before averaging cases. SD is between-case Dice SD, not repeated-inference variance.

| Seed | Own mean Dice | Own mean SD | Own mean worst2 | Mean−point Dice | Mean−point fine | Mean−point worst2 |
|---|---:|---:|---:|---:|---:|---:|
|20261030|88.578490|6.695397|76.869057|+0.053568|−6.175416|+0.694840|
|20261031|88.042195|7.010659|75.748999|+0.009474|−7.371608|+0.742755|
|20261032|88.354888|7.105504|75.791149|+0.052468|−5.915736|+0.736828|

Relative to the matched trained A_only control, own-mean Dice improves by+0.291751/+0.950327/+0.135090pp. However, seed20261032 loses0.072784pp clDice,0.655838pp centerline recall and1.684209pp fine recall. Its uniform-known-nonempty centerline and fine recall lose0.399205/3.624084pp. Therefore the frozen mean replacement does not satisfy vascular protection. Uniform fine support is10 cases; KiTS_K9 and Rider_R15 remain explicit nulls.

## Correct block mean versus mismatched block mean

| Seed | Dice effect | Positive cases | Dongyang | KiTS | Rider | R15 | Worst2 effect |
|---|---:|---:|---:|---:|---:|---:|---:|
|20261030|+0.337841|9/12|+0.261693|+0.513659|+0.238170|+0.962414|+0.806363|
|20261031|+0.737379|9/12|+0.294997|+1.422723|+0.494416|+2.047288|+1.753572|
|20261032|+0.371638|9/12|+0.324042|+0.532583|+0.258288|+1.090419|+0.897871|

Every one of96 primary patch scalars and its actual FP32-consumed zR changes in every seed. All12 cases have all eight supported blocks. The comparison is not an empty intervention. Both mean variants have the same worst2 cases, R15/K12. Mean−cross SD changes are−0.223392/−0.612406/−0.257194pp.

Positive source Dice means do not imply uniform source benefit. Rider has only2/4,1/4,2/4 positive cases; its other three cases average−0.003245/−0.023209/−0.019089pp after excluding R15. The other11 cases across all sources still average+0.281061/+0.618296/+0.306294pp. KiTS fine recall is lower under the correct block mean by2.662164/6.318795/2.179428pp. These adverse observations remain part of the interpretation.

## What the R15 repair means

At slot5, replacing pointwise R with its own mean improves Dice by12.270958/13.657988/12.822313pp and reduces FN by9302/10060/9178 voxels. Yet own-mean minus FIT-mean Dice is only+0.090263/−0.210504/+0.093748pp. The own log-radius mean2.122885 is close to the fixed FIT mean2.068769. Most of that large repair supports removing the damaging within-patch variation and moving consumption nearer its neutral reference; it cannot all be credited to the correct block scalar.

The separate block-matching effect occurs elsewhere too. R15 slot7 contributes+4.179246/+12.523498/+4.714742pp when its own lower mean is used instead of slot3's higher mean. Slot1 contributes+1.567104/+3.420565/+2.122838pp. Slot5's own-versus-cross effect is only+1.543069/+0.256921/+1.841819pp. Within-patch smoothing and between-patch matching are different contrasts and do not form an additive causal decomposition.

## Consequence and limits

The next bounded hypothesis is to remove pointwise R variation while preserving meaningful differences between blocks. It requires matched training from the common initialization with the unchanged model, loss, normalization and budget. Frozen weights were optimized for pointwise fields, so their blood-vessel losses neither prove retraining will fail nor guarantee retraining will repair them. No CT predictor or successor training was authorized by this readout itself.

The mean is an average of a sparse, pipeline-defined native-EDT proxy over own A1, affected by support, sampling, vessel mixture and cropping. It is not a clinical anatomical class. Filling it only on own A1 is still a block summary with spatial support gating, not whole-patch broadcast. A lower-dimensional summary is not automatically CT-predictable. All data remain reused TRAIN/oracle data, with no independent native or zero-shot confirmation.

The [fixed diagnostic protocol](../../docs/calibre_routing_frozen_attribute_granularity_20261024.txt), [full analysis](analysis.json), [all900 case/stratum rows](casewise.csv), and [complete readout](readout.txt) retain all cases, source groups, regional confusion counts, supports and R15 slots. The inert [reader](../../next_stage/calibre_routing_frozen_readout.py) reproduces the aggregation without a GPU.
