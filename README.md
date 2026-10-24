# GTA-AVT-v1

从原始 CTA、独立属性预训练、完整分割训练到仅 CT 整例推理的独立项目。
这是用户认可的完整 v1 实现；早期 v0.1 的局部 oracle 适配结果不作为本版本的训练结果。
正式实验结果将单独发布，源代码包不捆绑或虚构训练权重。

## 方法

GTA 的共同原则是：**具有任务依据的几何属性 → 由图像预测属性 → 条件化特征适配**。
AVT 实例采用局部物理 log-radius 与“固定几何流程是否能够提供该半径”的适用性。
适用性不是血管标签或置信度；没有半径测量不等于半径为零。

一个独立预训练的属性网络同时提供多尺度特征和稠密条件，分割阶段冻结其全部参数和 BN。
完整分割网络及其引导适配器从随机初始化训练；中层使用有界、零初始化的三轴条件路由。
累计高斯场是独立开关的训练辅助，推理时不需要标注或高斯目标。

- [英文方法与公式](docs/METHOD.md)
- [与原始气道 GTA 合写一篇论文的说明](docs/UNIFIED_METHOD_ZH.md)
- [本轮固定正式实验协议](docs/FORMAL_V1_PROTOCOL.md)
- 模型：`gta_avt/models.py`；几何与数据：`geometry.py`、`data.py`
- 损失、训练、推理及评估：`losses.py`、`engine.py`、`inference.py`、`metrics.py`

## 安装与数据

推荐独立 Python 环境，按显卡安装匹配的 PyTorch，再在项目根目录运行：

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

工程验证环境为 Python 3.8、PyTorch 1.11/CUDA 11.3、NumPy 1.22、SciPy 1.10、scikit-image 0.21、SimpleITK 2.3。
依赖声明允许较新版本；换用新环境仍应执行检查。历史算子许可及来源见 `gta_avt/vendor/`。
项目新增研究代码未另外指定开源许可证，不将全部代码笼统声明为上游 MIT。

下载 [AVT 官方数据](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362)，解压后保留图像 `.nrrd` 和对应 `.seg.nrrd`。
以下命令均在项目根目录执行，配置路径相对于当前目录：

```bash
python -m gta_avt index --root /path/to/AVT/extracted --split-file configs/split_avt_20261022.json --output cases.json
python -m gta_avt prepare --cases cases.json --output cache
```

提供的历史划分为 TRAIN 36、VAL 8、TEST 12 个病例 ID，并非官方独立患者划分。
这些病例在前期开发中已被接触，不应称为新的盲测。任何均值、标准差和高斯归一化均只拟合 TRAIN。
默认全中心高斯准备可在 CPU 执行；大型病例可用 `--geometry-options` 指定 `{"gdf_backend":"cuda"}`，公式不变。
混合 HU/+1024 处理遇到歧义会停止，需在 `cases.json` 对该图像提供与图像 SHA256 绑定的 `intensity_override`。
这属于 CT 强度判定，不读取分割来决定裁剪或强度。

已经具有本项目历史 1mm 缓存时，可使用 `import-cache --help` 验证并重用。
该入口不需要旧模型、私人固定 patch bank 或 v0.1 公共权重；原始数据入口完整保留。

## 训练

一条命令执行预先声明的完整流程：

```bash
python -m gta_avt pipeline --config configs/pipeline.json
```

它先独立预训练属性预测器，再训练共同分割热身模型，然后分别从同一热身权重训练：

| 输出目录 | 条件 | 高斯损失 |
|---|---|---|
| C0 | 常量路由偏置，同一属性特征 | 0 |
| A_only | 预测适用性 | 0 |
| AR | 预测适用性和连续尺度 | 0 |
| AR_GDF | 与 AR 相同 | 0.1 |

所有组均保留同样的高斯预测头、特征、分割结构和训练预算。改变种子应重新训练 P 和 S；同一种子内共享 P 和热身检查点。
`pipeline.json` 的正式预算为属性 20,000、共同热身 10,000、每组后续 5,000 次成功更新。
独立阶段示例配置另见 `pretrain.json`、`warmup.json`、`adapt.json`；这些配置可单独运行，并非已完成的结果。

属性阶段输出 `best.pt`、`last.pt` 和满足固定前提时的 `best_ready.pt`。
若全部检查点均未通过非塌缩、受支持变化及超越常量尺度的前提，流水线停止并记录失败，不自动调整方法。
通过前提不能证明动态条件改善分割。预训练步数也不自动等同于已收敛；完整验证曲线会保留。

分割网络各组完整训练，而非只更新路由参数。路由强度前 1,000 次更新从零渐增。
各组用同一个不含高斯项的 VAL 分割标准保存 `best.pt`；共同固定终点保存 `last.pt`。
流水线主报告使用各组固定终点的原生空间 VAL，避免组间终点混杂。

支持失败后的显式续训：配置中增加 `resume` 检查点；流水线用 `resume_pipeline=true`，其余计划保持一致。
数据按种子、轮次、索引确定；多进程预取保留顺序。固定种子不等于 CUDA 的逐位确定性。

## 推理与结果

```bash
python -m gta_avt infer --config configs/infer.json
python -m gta_avt evaluate --config configs/evaluate.json
```

修改示例输入路径和检查点后执行。推理仅接受 CT，输出预测至 `output_dir/case_id/`，可选保存概率。
采用 1mm 网格、128³ 滑窗、0.5 重叠、概率融合，再还原原始物理坐标，固定阈值 0.5。
滑窗的 Gaussian 融合权重与 GDF 训练辅助是两件事。

评估单独读取标注，输出逐例 Dice、精确率、召回率、clDice、中心线指标、FP/FN 及病例分散度。
分数单位为 0–1；差值乘 100 后才是百分点。空目标或空预测造成未定义指标时以 null 保留。
`condition_mode` 支持显式常量半径或重配诊断；正常训练和推理均使用 CT 预测条件。

`scripts/smoke_cuda.py` 是不读取 AVT 的工程检查；它通过不代表科学有效性。
论文应分别报告属性特征、动态条件和高斯监督的证据，保留所有阴性结果。
