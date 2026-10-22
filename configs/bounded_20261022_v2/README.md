# Bounded v2 internal comparison

These seven recipes and the split are byte-identical to the frozen experiment files. `plan.json` is a portable scientific description, with the private executed-plan hash and exact recipe hashes; it omits server paths and deployment infrastructure. It is not an executable private queue plan.

Read [the common revision decision](../../docs/common_recipe_v2.md) before interpreting results. All runs start fresh from seed 20261022, use 128-cube training at 1 mm, batch two and 10,000 successful updates, with all VAL8 at 5,000 and 10,000. The selected checkpoint maximizes mean validation Dice. The two role groups use matching Python 3.8 / PyTorch 1.11 core software. Use the prepared, verified common cache and run from the repository root.

Primary role, sequential jobs on one GPU:

```bash
for job in C_concat A_soft A_soft_constant; do
  python code/train.py --config configs/bounded_20261022_v2/formal_main_${job}.json --manifest data/cache_1mm/manifest.json --output runs/bounded_v2/main_${job} || break
done
```

Second role, sequential jobs on the other GPU:

```bash
for job in A_soft_platform_bridge A_soft_scale A_soft_structure A_hard_anchor; do
  python code/train.py --config configs/bounded_20261022_v2/formal_ablations_${job}.json --manifest data/cache_1mm/manifest.json --output runs/bounded_v2/ablations_${job} || break
done
```

Check the process exit status and genuine `complete.json` before advancing or exporting. The commands above stop their loop on failure; they do not implement the authors' private queue attestation or storage checks. Keep every run's source snapshot, original configuration and environment record. Use `--resume` only for an interrupted run with exactly the same configuration and identity, never to resume v1 weights into v2.

The primary effects are within-role A-minus-C and A-minus-constant-A. The second full-A is a hardware-role bridge, not a different seed. Scale, structure and hard-anchor comparisons use that second full-A. Final tensor-only reference evaluation uses one device/environment. Freeze all seven selections before an explicit TEST12 evaluation; these commands never run TEST12 or start another phase.
