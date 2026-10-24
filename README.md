# VLA 轻量微调：Octo-Small-1.5 × BridgeData V2（B 类 · 微调部分）

> 本仓库为机器学习课程大作业 **B 类（工程实践型）** 的**数据处理与轻量微调**部分。
> 评测（SimplerEnv 闭环、60 episodes）由同组组员在原生 Linux 上完成，相关代码、配置与协议一并在本仓库提供。

## 项目概述

| 项目 | 内容 |
|---|---|
| 项目类别 | B 类（工程实践型）|
| 基础模型 | Octo-Small **1.5**（27M 参数）|
| 基础 checkpoint | step **300000** |
| 训练数据 | BridgeData V2（`bridge_dataset`，WidowX 采集）|
| 微调方式 | **动作头微调（head_only）** 与 **完整微调（full）** 两种，各训练 1,000 步 |
| 训练硬件 | RTX 5070 Laptop（8 GB 显存）+ WSL2（12 GB 内存）|
| 评测任务 | T1 `widowx_spoon_on_towel`、T2 `widowx_put_eggplant_in_basket` |

本仓库覆盖作业第 9.3 节建议结构的以下内容：

```
vla_finetune_repo/
├── README.md                     ← 本文件（复现总说明）
├── requirements.txt              ← 依赖清单（含关键钉版本与安装顺序）
├── train.sh                      ← 训练脚本
├── train_config.py               ← 训练配置（数据/筛选/超参）
├── evaluate.py                   ← 评测脚本（组员使用）
├── configs/
│   ├── eval_config.json          ← 正式评测初始状态清单（已冻结）
│   └── checkpoint_metadata_*/    ← 两个 Final checkpoint 的配置与归一化统计量
├── scripts/
│   ├── octo_bridge_policy.py     ← **自包含策略适配层**（evaluate.py 使用）
│   ├── patch_simplerenv_octo.py  ← 备选：给 SimplerEnv 原生 OctoInference 打补丁
│   ├── summarize_results.py      ← 汇总 episodes.csv → results.csv
│   ├── task_config.py            ← T1/T2 任务参数
│   ├── generate_eval_config.py   ← 初始状态清单生成器
│   ├── placement_filter.py       ← 数据筛选函数
│   ├── package_checkpoint.sh     ← checkpoint 打包脚本（供上传）
│   ├── setup_native_linux.sh     ← 原生 Linux 一键安装
│   ├── setup_simplerenv.sh       ← SimplerEnv 环境安装
│   ├── patch_octo_jax06.sh       ← Octo 适配新版 JAX 的补丁
│   └── verify_render.py          ← 渲染自检
├── train_log_full.txt            ← 完整微调训练日志（含 20 个 loss 记录点）
├── train_log_head_only.txt       ← 动作头微调训练日志（含 20 个 loss 记录点）
├── checkpoint_link.txt           ← Final checkpoint 下载与加载说明
├── results.csv                   ← 结果汇总表（待评测组员填写）
├── episodes.csv                  ← 逐 episode 记录（待评测组员填写）
├── video_links.md                ← 代表性视频链接（待评测组员填写）
├── videos/                       ← 视频存放目录（当前为空）
└── docs/
    ├── EVAL_FIX.md               ← **评测 0 成功的根因与修复**（评测前必读）
    ├── REQUIREMENTS_CHECKLIST.md ← 作业要求逐条对照表（自查用）
    ├── MODEL_SELECTION.md        ← 模型选择与 WidowX 训练依据
    ├── DATA_PREPARATION.md       ← 数据处理细节
    ├── TRAINING_RECORD.md        ← 训练记录、可训练参数、loss 曲线
    └── REPRODUCTION.md           ← 完整复现步骤（含所有补丁说明）
```

---

## 1. 模型选择

| 项目 | 值 |
|---|---|
| 模型名称 | Octo-Small **version 1.5** |
| HuggingFace 仓库 | `rail-berkeley/octo-small-1.5` |
| checkpoint step | **300000** |
| 参数量 | 27M（ViT-S 等价）|
| 权重文件 | `300000/default/checkpoint`（约 546 MB）|

### 使用 WidowX 数据训练的依据（三条）

**依据 1：官方模型卡（随 checkpoint 发布）**

`README.md` 第 39–43 行明确列出训练数据构成：

```
| Dataset                          | Proportion of batch |
| Fractal (Brohan et al, 2022)     | 17.0%               |
| Kuka (Kalashnikov et al, 2018)   | 17.0%               |
| Bridge (Walke et al, 2023)       | 17.0%               |
```

其中 **Bridge (Walke et al., 2023) = BridgeData V2**，即使用 **WidowX 250s** 机械臂采集的机器人数据集。

来源：<https://huggingface.co/rail-berkeley/octo-small-1.5>

**依据 2：checkpoint 内的归一化统计量**

checkpoint 附带的 `dataset_statistics.json` 包含 `bridge_dataset` 条目：

| 字段 | 值 |
|---|---|
| `num_trajectories` | 60,064 |
| `num_transitions` | 2,195,527 |
| 动作 mask | `[True,True,True,True,True,True,False]`（第 7 维夹爪不归一化）|

说明该权重的训练管线确实包含 WidowX 数据。

**依据 3：SimplerEnv 官方仓库**

`scripts/octo_bridge.sh` 中：

```bash
policy_models=("octo-small" "octo-base")
robot=widowx
policy_setup=widowx_bridge
```

即 SimplerEnv 官方将其作为 **WidowX 评测的基准 checkpoint**。

来源：<https://github.com/simpler-env/SimplerEnv/blob/main/scripts/octo_bridge.sh>

> 更详细说明见 [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md)。

---

## 2. 数据与处理

| 项目 | 内容 |
|---|---|
| 数据名称 | BridgeData V2（TFDS 版本 `bridge_dataset`）|
| 来源 | <https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/> |
| 许可 | 遵循 BridgeData V2 原始许可（见官方页面）|
| 使用分片 | **30 个 TFRecord 分片**（官方共 1,024 train / 128 val）|
| 原始轨迹数 | **1,404 条**（train）/ **156 条**（val）|
| **筛选后轨迹数** | **717 条**（放置类任务）|
| **训练样本数** | **23,634 个**（717 条轨迹的总帧数；window=1 时每帧一个样本）|
| 每条轨迹帧数 | 均值 33.0，中位数 32，最小 11，最大 77 |
| 每 epoch 步数 | 23,634 / batch 8 = **2,954 步** |
| 训练/验证划分 | 按**完整轨迹**划分：27 个分片训练 + 3 个分片验证（无泄漏）|

### 字段映射

| 用途 | Bridge 原始字段 | 进入模型后 |
|---|---|---|
| 第三人称图像 | `observation.image_0` | `observation.image_primary`（256×256）|
| 腕部图像 | `observation.image_1` | `observation.image_wrist`（128×128）|
| 语言指令 | `language_instruction` | T5-Base 编码（max_length=16）|
| 动作 | `action`（7 维）| 位置增量 3 + 旋转增量 3 + 夹爪 1 |
| 本体状态 | `observation.state` | `proprio` |

### 动作约定

- 7 维：`[dx, dy, dz, droll, dpitch, dyaw, gripper]`
- 归一化：`(x - mean) / std`，**夹爪维度不归一化**（`action_normalization_mask = [T,T,T,T,T,T,F]`）
- 夹爪：0 = 闭合，1 = 张开（Bridge 约定）
- 单位：位置增量以米为单位；旋转增量以弧度为单位

### 数据筛选规则

使用 `octo.data.oxe.placement_filter:is_placement_task`，按语言指令的**首个动词**筛选：

```
put, place, move, moved, pick, take, picking, pickup, slide, remove, insert
```

- 筛选前：1,404 条
- 筛选后：**717 条**（保留 51.1%）
- 被排除：空指令、折叠、开/关抽屉、扫、转 等与"放置"无关的任务

> 详细统计与说明见 [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md)。

---

## 3. 训练方法

### 3.1 两种微调方式

| 方式 | 冻结 | 可训练参数 | 说明 |
|---|---|---|---|
| **head_only**（主方案）| 整个 Transformer（339 个参数叶子）| **27 个参数叶子**（动作头）| 保预训练视觉能力，避免小数据灾难性遗忘 |
| **full**（对照）| 无 | 全部 366 个参数叶子 | Octo 默认完整微调 |

选择 `head_only` 作为主方案的理由：训练数据仅 717 条，且与目标任务存在 domain gap（厨房场景 → 桌面放置），完整微调容易破坏预训练特征。

### 3.2 主要超参数

| 参数 | 值 |
|---|---|
| batch size | 8 |
| 训练步数 | 1,000 |
| 学习率调度 | cosine，`peak_value=3e-4`，`warmup_steps=100`，`decay_steps=1000` |
| 优化器 | AdamW，`weight_decay=0.01` |
| 保存间隔 | 每 250 步 |
| 随机种子 | 0 |
| 图像分辨率 | primary 256×256，wrist 128×128 |
| 动作预测长度 | 4 步 |

### 3.3 训练命令

```bash
# 主方案：动作头微调
MODE=head_only NUM_STEPS=1000 BATCH_SIZE=8 bash train.sh

# 对照：完整微调
MODE=full NUM_STEPS=1000 BATCH_SIZE=8 bash train.sh
```

### 3.4 训练日志

| 文件 | 内容 |
|---|---|
| [`train_log_head_only.txt`](train_log_head_only.txt) | 动作头微调（含 20 个 `[TRAIN] step=... loss=...` 记录点）|
| [`train_log_full.txt`](train_log_full.txt) | 完整微调（含 20 个记录点）|

#### ⚠️ 如何正确读这个 loss（避免误判）

Octo-Small 的动作头是**扩散策略**，`loss` 是**去噪误差**，且**已乘 action_dim=7**：

```
日志 loss = 7 × MSE(pred_eps, noise)     noise ~ N(0,1)
```

| 情形 | 日志 loss |
|---|---:|
| 完全不学（恒预测 0）| **7.0** |
| **本项目实测** | **0.92 – 2.38** |
| 完美预测 | 0 |

**所以 loss ≈ 1.6 不是"很差"，而是比 baseline 好约 77%**（实际 MSE = 1.6/7 ≈ 0.23）。

波动范围大（0.92–4.2）是扩散训练的常态：batch 仅 8、每步随机采样扩散时间步、任务种类多。

完整解读见 [`docs/TRAINING_RECORD.md`](docs/TRAINING_RECORD.md) 第 4.0 节。

---

## 4. Final checkpoint

### 4.1 选择规则

**采用"预先确定训练步数"规则**：两个模型都取 **step 750** 的权重。

依据（作业第 3.4 节）：
> "例如采用预先确定训练步数的权重"

**为什么是 750 步而不是 1000 步**：

训练确实跑满了 1,000 步（loss 记录到 step 1000），但：

| 模型 | step 750 | step 1000 |
|---|:---:|:---:|
| head_only | ✅ 298 MB 完整 | ✅ 298 MB 完整 |
| full | ✅ 297 MB 完整 | ❌ 132 MB（保存时被中断，权重未写完）|

为保证**两个模型步数完全一致**（公平对比只应体现"微调方式"的差异），统一采用 **step 750**。

选择 750 步的理由：
1. 717 条轨迹共 **23,634 个训练样本**（总帧数）；batch 8 → **2,954 步/epoch**，750 步 ≈ 0.25 epoch
2. 两个模型在该步数都有完整、可加载的权重
3. 训练环境（WSL2，12 GB 内存）在接近 1,000 步时出现内存压力，full 模式的最终保存未能完成

### 4.2 两个 Final checkpoint

| 名称 | 路径（训练机）| 可训练参数 | 大小 |
|---|---|---|---|
| **Final-A（主方案）** | `checkpoints/octo_finetune/octo_head_only_losslog_20261019_112942/750/` | 27 / 366 叶子 | 298 MB |
| **Final-B（对照）** | `checkpoints/octo_finetune/octo_full_losslog_20261019_113719/750/` | 366 / 366 叶子 | 297 MB |

> 两个 checkpoint 训练时的 `config.json`、`dataset_statistics.json`、`example_batch.msgpack` 位于各自目录的**根层级**（不在 `750/` 里面），打包分发时需一并包含。

下载链接与加载方式见 [`checkpoint_link.txt`](checkpoint_link.txt)。

### 4.3 加载验证（已完成）

三个模型均已验证可加载并正常推理：

| 模型 | 加载 | 推理输出 | 动作幅度（归一化）| 反归一化后 |
|---|:---:|:---:|---:|---|
| Base（step 300k）| ✅ | (1, 4, 7) | 0.287 | 约 ±3 mm |
| Final-A head_only（750）| ✅ | (1, 4, 7) | 3.717 | 约 ±5 cm |
| Final-B full（750）| ✅ | (1, 4, 7) | 3.884 | 约 ±5 cm |

```python
from octo.model.octo_model import OctoModel

path = "<checkpoint 目录>"
model = OctoModel.load_pretrained(path, step=750)   # ✅ 加载成功

import jax, jax.numpy as jnp
obs = {
    "image_primary": jnp.zeros((1, 2, 256, 256, 3), dtype=jnp.uint8),
    "image_wrist":   jnp.zeros((1, 2, 128, 128, 3), dtype=jnp.uint8),
    "timestep_pad_mask": jnp.ones((1, 2), dtype=jnp.bool_),
}
task = model.create_tasks(texts=["put the spoon on the towel"])
actions = model.sample_actions(obs, task, rng=jax.random.PRNGKey(0))
# ✅ 输出 shape (1, 4, 7)
```

> 加载时需要可访问 HuggingFace（或设置 `HF_ENDPOINT=https://hf-mirror.com`），因为会下载 T5-Base tokenizer。

---

## 5. 评测设置（由组员执行，配置已冻结）

| 项目 | 内容 |
|---|---|
| SimplerEnv 版本 | 官方仓库 commit **`c701ea271b16b80616d6fed9713c4a50a0adea3b`**（见 `scripts/setup_native_linux.sh` 的 `SIMPLERENV_COMMIT`）|
| 任务 | T1 `widowx_spoon_on_towel`、T2 `widowx_put_eggplant_in_basket` |
| **评测模型** | **3 个**：Base、Final-A（head_only）、Final-B（full）|
| episode 数 | 每模型每任务 **10** 次（共 **3 模型 × 2 任务 × 10 = 60**）|
| 初始状态清单 | [`configs/eval_config.json`](configs/eval_config.json)（**已冻结，未用于训练或调参**）|
| 控制频率 | 5 Hz（SimplerEnv WidowX 官方配置）|
| 最大步数 | T1 = **60 步**；T2 = **120 步** |
| 成功判定 | 环境原生 `done`（`env.step()` 返回的 `done=True` 才算成功）|
| 成功记录时机 | 每个 episode 结束时记录一次，取最终 `done` 值 |
| **终止逻辑** | ① 环境原生 `done=True` → 记成功；② 达到最大步数 `truncated=True` → 记失败；③ 策略输出 `terminate_episode>0` 仅记为 `terminated_by_policy`，**不当作成功**；④ 异常捕获 → 记失败并写入异常信息 |
| 模型 / 环境种子 | `model_seed=0`，`env_seed=0`（两者分别记录，见 `eval_config.json`）|
| 评分 | 成功率 = 成功数 / 评测数 × 100%；B-SimplerScore = (T1 + T2) / 2 |

> ✅ **本仓库已内置正确的策略适配层**（`scripts/octo_bridge_policy.py`），
> `evaluate.py` 直接使用它，**不需要修改 SimplerEnv 的任何文件**。
>
> SimplerEnv 原生 `OctoInference` 与 Octo 0.6.x 有两处不兼容，会导致
> **夹爪永远无法闭合、成功率恒为 0**（详见 [`docs/EVAL_FIX.md`](docs/EVAL_FIX.md)）。
> 若你因故仍需使用原生 `OctoInference`，可运行
> `python scripts/patch_simplerenv_octo.py` 打补丁（备选方案）。

### 运行命令

```bash
# Base（预训练，未经本次微调）
python evaluate.py --model-type octo-small \
  --checkpoint "$HOME/vla_workspace/models/octo-small-1.5" \
  --task widowx_spoon_on_towel --episodes 10 \
  --eval-config configs/eval_config.json --output-dir results/base_t1

# Final-A（动作头微调，主方案）
python evaluate.py --model-type octo-small \
  --checkpoint "<octo_head_only_step750 解压路径>" \
  --task widowx_spoon_on_towel --episodes 10 \
  --eval-config configs/eval_config.json --output-dir results/final_a_t1

# Final-B（全量微调，对照）
python evaluate.py --model-type octo-small \
  --checkpoint "<octo_full_step750 解压路径>" \
  --task widowx_spoon_on_towel --episodes 10 \
  --eval-config configs/eval_config.json --output-dir results/final_b_t1
```

对 `widowx_put_eggplant_in_basket` 重复同样命令（共 6 次调用）。

### 评测隔离声明

- `configs/eval_config.json` 在训练开始**之前**生成并冻结
- 训练、调参、checkpoint 选择**均未使用**该清单的任何状态
- 评测初始状态与训练数据（BridgeData V2）完全独立
- 不筛选失败 episode，不人工干预
- **3 个模型使用完全相同的初始状态清单与种子**

---

## 6. 结果（待评测组员填写）

### 6.1 汇总表

| 模型 | 训练模块 | 可训练参数 | T1 成功数 | T2 成功数 | B-SimplerScore |
|---|---|---|---:|---:|---:|
| **Base** | 未进行本次微调 | 0 | /10 | /10 | |
| **Final-A** | 动作头（head_only）| 27 / 366 叶子 | /10 | /10 | |
| **Final-B** | 全量（full）| 366 / 366 叶子 | /10 | /10 | |

**变化（单位：百分点）**：

| 对比 | T1 | T2 | B-SimplerScore |
|---|---:|---:|---:|
| Final-A − Base | | | |
| Final-B − Base | | | |
| Final-A − Final-B | | | |

> 第三行「Final-A − Final-B」是本项目额外对照的核心指标：**在 717 条小数据 + domain gap 条件下，冻结 Transformer 是否比全量微调更稳？**

### 6.2 逐 episode 记录

见 [`episodes.csv`](episodes.csv)（字段：模型、任务、episode 编号、环境 seed、模型 seed、成功标记、控制步数、异常信息）。

**预期行数**：3 模型 × 2 任务 × 10 episodes = **60 行**。

### 6.3 代表性视频

见 [`video_links.md`](video_links.md)（要求 2–4 个，覆盖 T1/T2，含失败样例）。

---

## 7. 复现步骤

完整步骤见 [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)，摘要：

```bash
# 1. 安装 Octo 训练环境（含 JAX 0.6.2 兼容补丁）
bash scripts/setup_native_linux.sh

# 2. 准备 BridgeData V2 数据（30 个分片，27 train + 3 val）
#    → 详见 docs/DATA_PREPARATION.md

# 3. 训练
MODE=head_only NUM_STEPS=1000 BATCH_SIZE=8 bash train.sh

# 4. 评测（需要 Vulkan GPU，WSL 不支持）
#    → 详见第 5 节
```

---

## 8. 分工

| 组员 | 负责内容 |
|---|---|
| （待填）| 数据处理、Octo 环境搭建、两种微调训练、checkpoint 验证、仓库整理 |
| （待填）| SimplerEnv 环境搭建、60 episodes 评测（3 模型）、视频录制、结果分析 |

---

## 9. 参考文献

1. [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213) — Octo 论文
2. [Octo-Small 1.5 模型卡](https://huggingface.co/rail-berkeley/octo-small-1.5)
3. [BridgeData V2](https://rail-berkeley.github.io/bridgedata/)
4. [SimplerEnv 官方仓库](https://github.com/simpler-env/SimplerEnv)
5. [SimplerEnv 论文](https://arxiv.org/abs/2405.05941)
6. [Open X-Embodiment](https://robotics-transformer-x.github.io/)
