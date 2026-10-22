# Portable reduced-scope v2 recipes

The [scientific plan](scientific_plan.json) describes four model tuples and two within-role contrasts. It contains no execution-host configuration. This is a transparent reduction after development evidence, with the unchanged v2 training recipe and frozen 24 source files, not algorithm v3.

The existing completed C and A bridge retain their original `bounded_20261022_v2` identity; this folder does not retrain or relabel them. The two new recipe files are byte-identical copies:

| New file | Original v2 recipe | SHA256 |
|---|---|---|
| `formal_main_A_soft.json` | `../bounded_20261022_v2/formal_main_A_soft.json` | `219611ef07097f068810c0bb0d1cd095f9b9c28f77df28c2c9a9187b9ef2b2f6` |
| `formal_ablations_A_soft_constant.json` | `../bounded_20261022_v2/formal_main_A_soft_constant.json` | `c317dbf25ef405e70f304bca2a98fe74c2deeaea8d17e78a0e2edc38e84d178d` |

Portable fresh-run examples, using a prepared common cache and new directories:

```bash
python code/train.py --config configs/bounded_20261022_v2_minimal4/formal_main_A_soft.json --manifest data/cache_1mm/manifest.json --output runs/minimal4_MAIN_A --method faithful_soft
python code/train.py --config configs/bounded_20261022_v2_minimal4/formal_ablations_A_soft_constant.json --manifest data/cache_1mm/manifest.json --output runs/minimal4_SECOND_constant --method faithful_soft_constant
```

These commands train only the requested model; they do not recreate private orchestration or automatically evaluate TEST. Keep each A/C contrast on its designated training role, preserve the exact recipe/source/data identities, and use the final common reference evaluator. Do not claim seven completed jobs, another seed, or a MAIN condition-value effect without MAIN constant. See [scope and interpretation](../../docs/minimal4_scope_20261022.md).
