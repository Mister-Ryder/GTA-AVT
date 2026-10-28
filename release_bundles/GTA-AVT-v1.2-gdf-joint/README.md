# GTA-AVT v1.2 — Gaussian joint-supervision revision

Implementation revision: `20261028-gdf-joint-r3`. This is the isolated corrected
Gaussian suffix. Its AVT efficacy has not yet been measured.

The attribute predictor and conditional fusion are inherited intact. The Gaussian
head reads the final two segmentation logits through Conv3d(2,1,3)+Sigmoid.
Targets use whole-case raw-Gaussian minmax followed by `0.5+0.5*u`. Geometry and
refinement jointly optimize the connected S network; P and BN running statistics
remain fixed. The original segmentation losses and sampling are retained.

- [完整训练阶段说明（中文）](docs/v12_gdf_joint_20261028/TRAINING_STAGES_ZH.md)
- [版本与实验对照](docs/v12_gdf_joint_20261028/EXPERIMENT_MATRIX.md)
- [变更记录](docs/v12_gdf_joint_20261028/CHANGELOG_20261028.md)
- [Frozen pre-GDF implementation](https://github.com/Mister-Ryder/GTA-AVT/tree/method/gta-avt-v1.2-ratio-r2-20261027)

From random initialization, run the frozen ratio-r2 prefix through `condition`,
then prepare a separate bounded target overlay and use this revision's explicit
runner. Do not import post-GDF control weights or incompatible Gaussian heads.

```sh
python scripts/prepare_gdf_overlay.py --source-manifest cache/manifest.json --output bounded_cache
python -m gta_avt.gdf_joint prepare --config configs/gdf_joint_plan.json
python -m gta_avt.gdf_joint run --config configs/gdf_joint_plan.json --arm AR
python -m gta_avt.gdf_joint run --config configs/gdf_joint_plan.json --arm AR_GDF
```

Fill the plan with your actual trusted condition checkpoint and its SHA256,
selected-base SHA256, data/evaluation paths and source identity. The common 2k
head warmup and TRAIN-only calibration execute once. Both arms then inherit the
same prepared checkpoint and execute 20k+10k successful S updates, followed by
native VAL8 evaluation. AR disables the GDF gradient; it retains P and all fusion.
Total new optimization is 62k updates, including the shared 2k head fit.

The four existing local-scope controls remain a separate frozen study. Comparing
their endpoints with this bundled correction cannot isolate individual changes.
No TEST tuning, new seeds, backbone comparisons or additional scope grid is part
of this revision. Synthetic QA is not efficacy evidence.
