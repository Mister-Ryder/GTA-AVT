# GTA-AVT v0.1 · 几何条件消费阶段版本

本阶段固定 `CalibreRoutingStageModel` 的三轴条件分支融合实现，以及已完成的 3 个种子 × 3 个条件组实验。模型与训练程序保留实际运行时的原始字节。另收录同一组 AR 终点权重上的五字段冻结测量。后续训练方案不属于本版本。

## 方法与接口

- 保留 GTA 的属性条件计算与特征融合：在已有四个 SAAM 形态分支及共享投影上，计算三个有界条件系数 `g_j=tanh(b_j+v_j*A+w_j*zR)`，输出为 `F0+alpha*g1*D1+alpha*(1-|g1|)*(g2*D2+g3*D3)`。三项方向分别对应 `[-,-,+,+]`、`[+,-,0,0]`、`[0,0,+,-]`。
- `A` 是几何属性可用性；`R=log(radius_mm/1mm)` 来自原生物理距离变换，在重采样 Lee 骨架节点取值并按可靠半径邻域关联。它是由标注构造的代理属性。无有效关联的位置保持未知并中性化，不能视作血管阴性或半径零。
- `C0` 使用常量条件；`A_only` 使用同一可用性；`AR` 使用可用性与标准化半径。三组使用相同的参数集合、共同初始权重、种子、训练采样计划和训练预算。这里没有属性辅助损失；GDF 头虽由基础类注册，本轮 GDF 损失权重为零。
- 学习范围为原有四个形态卷积、共享融合投影、末端两通道投影及 576 个条件系数，共 115458 个参数。其余参数及所有 BatchNorm/Dropout 状态固定在 evaluation 模式。输入为 1 mm、128³，batch size 2；每组 2000 次成功更新，原主输出 Dice+BCE，AdamW，已有卷积/投影学习率 1e-4、条件系数 1e-3、weight decay 1e-5，`alpha=min(1,step/250)`。
- `next_stage/continuous_ct_input.py` 提供不读取标注的 CT 预处理、铺块和原生空间输出接口。本阶段测得的 AR 消费由 GT 几何属性驱动；本阶段不含已完成对应实验的 CT 属性预测器，也不把该输入接口称为完整 CT 动态推理部署。

## 已测范围与客观汇总

本轮在既有 TRAIN36 缓存和固定 576 区块上执行：FIT24 用于更新，PROBE12 用于以下汇总，三种来源各 4 个 PROBE 病例。共同骨干此前接触过这些 TRAIN 病例，因此此处是 TRAIN 属性消费测量；表中数值不是独立 VAL、TEST、原生全体积或 zero-shot 结果。

主汇总为已知非空的 GT-jitter slots 0–7：先在病例内平均区块，再等权平均 12 病例。细节点召回先在病例内累加 `fine_covered/fine_nodes`，再等权平均有支持的病例。细节点来自现有 1 mm 缓存中半径不超过 2 mm 的节点；它不等同于独立临床小分支数量。SD 是病例间总体标准差；最低两病例由各组自身 Dice 排序产生。未定义项保持空值。

| 种子 | 条件 | Dice (%) | clDice (%) | 中心线召回 (%) | 细节点召回 (%) | 病例间 Dice SD (pp) | 最低两病例 Dice (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261030 | C0 | 85.699765 | 84.761984 | 82.516547 | 43.847861 | 9.494182 | 68.984383 |
| 20261030 | A_only | 88.286739 | 90.553193 | 90.418740 | 56.964286 | 6.925611 | 76.022849 |
| 20261030 | AR | 88.524923 | 92.078679 | 92.469270 | 67.214016 | 7.035477 | 76.174218 |
| 20261031 | C0 | 85.238267 | 84.025253 | 82.127589 | 44.485412 | 10.304800 | 67.372420 |
| 20261031 | A_only | 87.091869 | 89.753002 | 86.685709 | 42.626939 | 7.856121 | 73.734103 |
| 20261031 | AR | 88.032721 | 91.906471 | 90.192796 | 56.959984 | 7.403353 | 75.006244 |
| 20261032 | C0 | 85.157071 | 84.290663 | 82.012696 | 43.648354 | 10.640615 | 66.658974 |
| 20261032 | A_only | 88.219799 | 91.280201 | 91.322995 | 62.396676 | 7.358137 | 75.175005 |
| 20261032 | AR | 88.302421 | 91.973696 | 92.033572 | 66.628203 | 7.483109 | 75.054321 |

全部精度见 `objective_results/summary.csv`。训练和评估的所有 3 个种子、3 个条件组、病例、区块及数值方向均保留在该目录。

冻结字段包括：原逐点半径、同区块 A 支持内半圈重排、区块均值、FIT 均值，以及同病例固定 `donor=(slot+4)%8` 的跨区块均值。前四项各 192 个区块；第五项只测主区块的 96 个，其 uniform 项为空。冻结测量没有更新权重。区块均值比较保持每病例八个标量的集合，不保持按体素计权的半径集合，不能与逐点重排视作同一干预。

## 文件

- `code/`：公开 AVT 代码与历史 GTA 算子，保留原来源说明。
- `next_stage/`：本版本模型、实际训练程序、直接计算依赖、冻结字段程序、读出程序和 CT-only 输入接口。旧 `boundary_*` 与 `calibre_basis_*` 文件作为固定数学或状态加载依赖保留。
- `runtime_source42/`：实际程序严格要求的 42 个历史源文件及 `source_manifest.json`；此快照仅为原程序依赖，不增加本版本实验结果。
- `objective_results/pointwise_seed30|31|32/`：各组 2000 行 `training.jsonl`、192 行 `patches.jsonl`，AR 另有 192 行 `mechanism.jsonl`；每个种子包含固定归一化统计。
- `objective_results/frozen_seed30|31|32/diagnostics.jsonl`：192 行原始五字段测量，含区域计数、支持和未测标记。
- `summary.csv`、`frozen_summary.csv`：9 行训练组汇总、15 行冻结字段主汇总。
- `pointwise_casewise.csv`、`frozen_casewise.csv`：完整分层病例表。
- `pointwise_correspondence.csv`、`block_correspondence.csv`：所有病例、来源和指标的差值及支持数量；正负数值均保留。

实验数据文件只包含测量、训练记录和数值汇总；不收录分析报告、继续/停止决策或推断性结论。

## 运行前置

本仓库没有影像、标注、几何数组、训练权重或共同初始权重。数据从 [AVT 官方 Figshare](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362) 获得；`tools/download_avt.py` 是数据下载工具。运行本阶段原程序还须另行提供：

1. 1 mm 影像/标注/几何缓存和对应 `manifest.json`，缓存字节符合原固定区块表。
2. 原 TRAIN36/576 区块表，SHA256 `9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6`。
3. 原连续半径属性 bank，SHA256 `98c639be40fea8df7d72e9fc97835e8147c87c47fe1ce57e317f232794c2cde5`，及其引用的 A/R/valid 等属性数组。bank 中引用的文件须能从本机路径访问。
4. 共同初始 tensor-only 权重，SHA256 `7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6`。
5. 兼容 CUDA/PyTorch 环境与足够显存。原配方 128³/B2 已在约 16 GB 与 32 GB 显存服务器运行。`requirements_next_stage_reference.txt` 给出参考主包版本；PyTorch 须单独安装，它不是完整传递依赖锁文件。

`runtime_source42/source_manifest.json` 的 SHA256 为 `f311c9fbe6ea55be70e72ed220ab0562da08b3ee267863fc0d78509a30c3654e`。入口会使用此快照创建本地 request，包含当前机器的输入路径；不随本分支分发历史服务器 request。

```bash
python -m pip install -r requirements_next_stage_reference.txt
python -B scripts/run_stage.py --manifest /path/cache_1mm/manifest.json \
  --bank /path/fixed_train36_bank.json --oracle-bank /path/radius_oracle/run_bank.json \
  --weights /path/common7406.pt --seed 20261030 --output /path/fresh_run_seed30 --execute
```

分别用 `20261031`、`20261032` 和不同的新输出目录运行另两个种子。该命令会调用未修改的 `calibre_routing_pilot.py`，依次训练 C0、A_only、AR。默认不传 `--execute` 时不读取实验输入、不训练。此包装入口尚未在另一个全新环境重跑整个配方；它不能补足外部数据和权重。

完整读出程序 `calibre_routing_readout.py` 与 `calibre_routing_frozen_readout.py` 仍接受原始完整 collected/collection 结构和原 bank；本分支公开的数据目录是去除路径清单、权重与解释字段的测量子集，不能直接替代那些完整运行目录。冻结程序重跑还需对应 AR 终点权重；本分支不分发该权重。

项目及历史算子的授权说明见 `NOTICE` 与 `code/historical/`；数据许可单独处理。
