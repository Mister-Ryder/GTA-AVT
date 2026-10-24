# CLI：局部管径 oracle 与三臂共适应

以下是实际 worker 的可执行接口，使用仓库相对代码路径及用户自备资产路径；没有部署脚本、服务器标识或凭据。命令按 Linux shell 展示。Python 需安装 PyTorch/CUDA、NumPy、SciPy、scikit-image、SimpleITK；现有 `requirements.txt` 不是完整的传递环境锁。

所有入口默认惰性。先使用 `--help` 或不带 `--execute` 检查接口；CPU准备与 CUDA QA 通过后才运行正式模式。每个输出目录必须全新。

## 输入前提

- `assets/fixed_train_bank.json`：原固定 TRAIN36/576 patch bank，SHA256 为 `9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6`。
- `assets/request_part_0.json` 等三个互斥 TRAIN12 分片请求：沿用公开 `boundary_attribute_probe.py` 的请求 schema、settings、source/data/checkpoint 约束，各自填写本机资产位置。请求须引用正确缓存、来源清单和共同 tensor-only 权重；不得移除校验绕过身份限制。
- 请求中的原42文件来源身份仍是执行来源。公开 mapped42 具有单独身份；本次真实运行 QA 不认证任意重新映射的来源包。精确数据/权重资产不由这些新源码提供。

```bash
hash_file() { python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }
REQUEST=assets/request_part_0.json
BANK=assets/fixed_train_bank.json
MODEL=next_stage/calibre_coadapt_stage_model.py
ORACLE=next_stage/local_radius_oracle.py
```

## 1. 独立 CPU oracle 准备

三个请求各执行一次；`--shared-output` 是所有训练进程可读取的用户自定位置。

```bash
python next_stage/calibre_oracle_bank_prepare.py \
  --request "$REQUEST" \
  --helper next_stage/boundary_attribute_probe.py \
  --oracle-helper "$ORACLE" \
  --mode qa --output outputs/oracle_qa_part_0 \
  --shared-output assets/oracle_qa_part_0 --execute
```

QA模式只准备该分片第一个病例的原0、1槽。它不是正式样本选择。为工程 QA 创建明确的局部 bank，固定均值0、标准差1，仅用于一次性更新检查：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('assets/oracle_qa_part_0/manifest.json')
d=json.loads(p.read_bytes())
assert d['mode']=='qa' and d['disposable_engineering_only'] is True
d['normalization']={'mean':0.,'population_std':1.,'case_equal':False,'fit_cases':[]}
out=Path('assets/oracle_qa_bank.json')
with out.open('xb') as f: f.write((json.dumps(d,indent=2,allow_nan=False)+'\n').encode())
PY
```

正式 CPU 准备把 `--mode qa` 改为 `--mode run`，并使用独立的新目录；每个分片保留原12例×16槽。合并三份 `manifest.json` 的规则如下，可直接生成正式 global bank：

```bash
python - <<'PY'
import json,math
from pathlib import Path
bank=json.loads(Path('assets/fixed_train_bank.json').read_bytes())
parts=[json.loads(Path('assets/oracle_run_part_%d/manifest.json'%i).read_bytes()) for i in range(3)]
cases={}
for p in parts:
    assert p['mode']=='run' and p['status']=='verified_complete'
    assert p['bank_sha256']=='9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
    assert p['fit_cases']==bank['fit_cases'] and p['probe_cases']==bank['probe_cases']
    assert p['oracle_helper_sha256']==parts[0]['oracle_helper_sha256']
    assert p['producer_sha256']==parts[0]['producer_sha256']
    assert not set(cases).intersection(p['cases'])
    cases.update(p['cases'])
assert set(cases)==set(bank['case_ids']) and len(cases)==36
first=[]; second=[]
for cid in bank['fit_cases']:
    x=cases[cid]; n=x['case_supported_count']; assert n>0
    first.append(x['case_sum_R']/n); second.append(x['case_sum_R2']/n)
mu=math.fsum(first)/24.; var=math.fsum(second)/24.-mu*mu
assert math.isfinite(mu) and math.isfinite(var) and var>0.
d=dict(schema='GTA-AVT-calibre-oracle-bank-v1',bank_sha256=parts[0]['bank_sha256'],
       oracle_helper_sha256=parts[0]['oracle_helper_sha256'],producer_sha256=parts[0]['producer_sha256'],
       fit_cases=bank['fit_cases'],probe_cases=bank['probe_cases'],case_ids=bank['case_ids'],cases=cases,
       normalization=dict(mean=mu,population_std=math.sqrt(var),case_equal=True,fit_cases=bank['fit_cases']),
       disposable_engineering_only=False)
with Path('assets/oracle_run_bank.json').open('xb') as f:
    f.write((json.dumps(d,indent=2,allow_nan=False)+'\n').encode())
PY
```

病例统计来自存盘 FP32 可测 logR 的 FP64 累积。unknown保留在 state/reason 中，不参与半径矩；A=0处消费中性0不表示半径0。不得用 probe 或新阈值修正矩，也不得删除无支持 FIT 病例；无支持或非正方差应停止。

## 2. 实际 CUDA 工程 QA

```bash
OBANK=assets/oracle_qa_bank.json
python next_stage/calibre_coadapt_pilot.py \
  --mode qa --request "$REQUEST" --request-sha256 "$(hash_file "$REQUEST")" \
  --bank "$BANK" --bank-sha256 "$(hash_file "$BANK")" \
  --oracle-bank "$OBANK" --oracle-bank-sha256 "$(hash_file "$OBANK")" \
  --oracle-helper "$ORACLE" --oracle-helper-sha256 "$(hash_file "$ORACLE")" \
  --model "$MODEL" --model-sha256 "$(hash_file "$MODEL")" \
  --helper next_stage/boundary_attribute_probe.py \
  --old-pilot next_stage/boundary_field_pilot.py \
  --seed 20261030 --output outputs/calibre_qa_seed30 --execute
```

QA对每臂执行两次128³/B2/AMP一次性更新，alpha=1；核验原 StageModel 基点逐位一致、实际梯度/更新、禁用条件斜率零、其余参数及缓冲冻结。QA不是正式训练，也不建立科学效力。

## 3. 正式固定预算

上面命令改为 `--mode run`、`OBANK=assets/oracle_run_bank.json` 和全新输出目录；其余资产保持。一个进程依次完成同 seed 的 C0、A_only、AR，各2,000次成功更新。另两组只使用预锁 seed20261031、20261032与各自独立输出，不能根据首组数值改预算或挑最好组。

最终产物包括共同 `training_plan.json`、各臂 `training.jsonl`、小型 `trainable_final.pt`、`patches.jsonl`、AR的 `mechanism.jsonl` 以及汇总 `analysis.json`。正式 worker 只在端点评估 PROBE12；不会读 VAL/TEST、选 checkpoint、启动属性预测训练或更新目标状态。`trainable_final.pt` 不是完整部署模型，恢复须配合同一共同权重、源码、oracle与矩。

原可用性/血管指标/尾部保护及半环对应性门槛见方法说明。Oracle通过后仍需独立 CT-only 属性预训练和迁移验证；当前命令不提供 CT-only 推理接口。

## 独立的代理测量标定

第五个源码是固定54种合成直管配置的半径代理标定，不读取真实病例、不训练模型，也不拟合半径修正。它不代替条件效用试验：

```bash
python next_stage/analytic_radius_calibration.py \
  --output outputs/analytic_radius_calibration --execute
```
