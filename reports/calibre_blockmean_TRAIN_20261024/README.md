# GTA-AVT 块均值条件：TRAIN 客观测量

新训练的 AR 使用区块内已支持 logR 的 FP64 均值，经 FP32 舍入后在该区块自身 A 支持内提供给原三轴条件融合器。模型、FIT 归一化、种子、训练计划和 2000 次成功更新预算保持原配方。C0、A_only 为原同种子的各 2000 次更新对照，本目录原样收录其训练和评估记录；它们没有在这一轮重新训练，不计作新的独立重复。

数据范围为既有 TRAIN36：FIT24 更新，PROBE12 测量，Dongyang、KiTS、Rider 各 4 个 PROBE 病例。每病例 16 个区块（GT-jitter 0–7，CT-uniform 8–15），常规分割每组各 192 个测量。主汇总使用已知非空 GT-jitter 区块。CSV 同时保留 all16、GT_jitter_all8、CT_uniform_all8、uniform_known_nonempty 分层。

AR 对应测量保留主区块 96 条记录，固定同病例 donor=(slot+4)%8，保留区块自身 A 与 valid，仅替换所提供的区块半径标量。uniform 对应测量没有执行。该比较保留八个区块标量集合，不保留按站点计权的半径集合。

半径为标注构造的原生物理距离变换代理值 log(radius_mm/1mm)；A 为属性可用性。未知位置中性化，不作为血管阴性或半径零。这里记录 GT-oracle 条件消费，未包含 CT 属性预测器、独立 VAL/TEST、原生全体积分割或 zero-shot 测量。

指标先在病例内平均区块，再等权平均有支持病例。fine_recall 在病例内先合并 fine_covered/fine_nodes；细节点是现有 1 mm 缓存中的半径不超过 2 mm 节点。Dice_population_SD 为病例间总体标准差，own_worst2 是各组自身最低两病例，matched_worst2 是同一病例集合上的配对值。

原始 JSONL 与 CSV 比例指标单位为 0–1，名称带 pp 的差值单位为百分点；FP/FN/TP 为区块体素计数或其病例内/病例间均值，BCE 为原实现均值。区块可能重叠。空值表示未定义或未测；保留原始数值精度、全部种子、病例、来源和正负方向。训练记录包括 loss、梯度、条件场和系数范数观测。

objective_*.csv 删除了判定列；嵌套 JSON 对象仅删除判定键，保留支持、条件实际改变和全部数值。目录不含解释报告、继续/停止结论、私有 request、权重、影像或数组。
