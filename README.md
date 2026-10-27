# GTA-AVT v1.2

**属性预训练 → 联合底座长训练 → 局部条件适配 → 高斯几何强化 → 平衡微调。**

v1.2把重点放在训练系统：阶段安排、稀疏前景采样、长训练及主辅目标的衔接。
保留属性、条件计算与融合、高斯场三个核心，不增加新的网络模块。本次交付代码与配方，未开展AVT训练。

## 七阶段配方

| 阶段 | 默认成功更新数 | 训练内容 | 高斯权重 |
| --- | ---: | --- | --- |
| attribute | 30000 | 独立预训练CT属性网络P | 0 |
| base_warm | 20000 | P与S共同建立中性条件分割底座 | 0 |
| base | 30000 | P与S继续长训练，深监督及属性辅助 | 0 |
| condition | 10000 | 冻结成熟P/主干/BN，训练局部条件融合 | 0 |
| gdf_head | 2000 | 在当前条件特征上仅预热高斯头 | 1，仅更新该头 |
| geometry | 20000 | 局部条件融合与高斯几何联合强化 | 0.1→1.0，5000步爬坡 |
| refine | 10000 | 较低学习率平衡分割与几何 | 1.0→0.2，5000步过渡 |

合计122000次成功更新。数值是可修改的首轮训练配方，不代表已验证最优或必然收敛。
共同底座采用主Dice+BCE、两级深监督Dice+BCE（0.5/0.25）和0.1倍属性目标。
成熟底座检查点按固定VAL区块的病例等权Dice优先选择，并要求当前联合P通过属性消费前提。
后续阶段继承同一个检查点内的完整P/S状态，不再加载最初预训练P。

## 类不平衡输入策略

TRAIN按病例等权，再按阶段选择血管、细血管、分叉邻域、边缘、高强度非血管背景、均匀视野区块。
候选类别不足时显式回退并记录；采样比例是区块来源比例，不是前景体素占比。
日志另外记录真实有效体素中的前景占比，避免误将“正样本区块比例”当成已解决体素不平衡。
VAL区块固定，原生全体积评价不使用标签指导裁剪或采样。

完整原理、阶段比例和继承关系见 [中文方法](docs/METHOD_ZH.md)。

## 运行

```bash
python -m pip install -e .
python -m gta_avt index --root /path/to/AVT --split-file configs/split_avt_20261022.json --output cases.json
python -m gta_avt prepare --cases cases.json --output cache
```

以下命令会训练，仅在明确决定开始实验后执行：

```bash
# 默认先到成熟底座，便于分阶段检查
python -m gta_avt pipeline --config configs/pipeline.json
# 配方不变，继续到指定阶段；已有阶段不会重复训练
python -m gta_avt pipeline --config configs/pipeline.json --set resume_pipeline=true --set stop_after='"condition"'
# 明确执行剩余完整流程
python -m gta_avt pipeline --config configs/pipeline.json --set resume_pipeline=true --set stop_after='"complete"'
```

仅执行上限stop_after可在恢复时改变；预算、采样、损失与学习率等科学配方必须保持一致。
默认只训练一个完整AR_GDF模型，不自动展开多组实验。C0/AR/C0_GDF/AR_GDF可显式选用，保持相同预算与采样。

```bash
python -m gta_avt infer --config configs/infer.json
python -m gta_avt evaluate --config configs/evaluate.json
python -m pytest -q
```

普通训练及推理始终以CT预测属性进入条件融合；标注几何只构造训练监督和评价目标。
高斯场用于训练期几何监督，不是推理时真值输入，也不同于滑窗高斯加权融合。
原生评价以Dice为主，同时报告clDice、中心线和细血管召回、误检漏检及病例分散度。
GDF是连续几何辅助，连通性改善需要实际指标支持。

工程核验范围与实际结果见ENGINEERING_QA.json；本项目不包含训练权重、原始影像或标注。

本次实际本地CPU核验：64项通过、1项因Windows符号链接权限跳过。包含合成阶段链、恢复一致性、采样回退、完整模型32立方前后向及真实模型检查点推理往返；未运行AVT或CUDA实验。
