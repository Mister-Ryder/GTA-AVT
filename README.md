# GTA-AVT v1.2 ratio-r2


## 前背景体素比例修订（ratio-r2）

本修订以原始GTA-airway实际训练入口的前背景占比加权为依据，不增加采样层级或网络模块。旧配方P完整训练后独立保留作对照；本修订重新随机初始化P，再执行完整七阶段，不导入旧P或分割权重。

对每个病例区块，在真实有效单元上计算pF=sum(y)/N，pB=1-pF，沿用历史指数比例关系：

    wF = exp(pB) / (exp(pF) + exp(pB))
    wB = exp(pF) / (exp(pF) + exp(pB))
    BCE_ratio = [wF*sum(y*softplus(-z)) + wB*sum((1-y)*softplus(z))]
                / [wF*sum(y) + wB*sum(1-y)]

相对前景权重wF/wB=exp(1-2*pF)，前景极少时最大约e倍。分母归一化保留损失量级，不把两类总贡献强制等权。单类区块退化为该类普通BCE，padding在算术前排除。V以连续占有率统计，不阈值化；病例区块保持等权。该实现按病例区块统计，历史入口按batch统计，避免把两者声称为逐项相同。

所有正式阶段显式启用airway_ratio，作用于P支撑、主分割和两级深监督BCE。普通Dice、半径掩码MSE、高斯区域等权MSE、网络、属性、六来源取块、种子和更新预算均不变。region_equal是代码中的显式研究选项，本轮正式配置全部不启用；未同时增加power-Dice或比例分层取块。

历史部分airway入口把权重用于weighted/power Dice；AVT保留现有Dice+BCE框架，将同一比例重权法接入BCE。历史dice_loss_weights仅对交集分子加权的不对称写法未复制。温和放大是否足以改善AVT需要实际结果，不能将更强补偿直接等同更好Dice。

普通未加权BCE、前背景误差、实际前景占比和相对权重另行记录。比较版本时优先采用相同固定VAL上的Dice、精确率、召回率和半径误差，不直接比较不同定义的总loss。两套源码、结果和TensorBoard分离，不在运行中途换目标。

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
共同底座采用主Dice＋前背景比例加权BCE、两级深监督Dice+BCE（0.5/0.25）和0.1倍属性目标。
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

本修订的实际工程核验数与范围见ENGINEERING_QA.json。
