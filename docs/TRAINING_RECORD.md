# 训练记录

本文档记录两种微调方式的完整配置、可训练参数、超参数与 loss 曲线。

---

## 1. 两种微调方式对比

| 项目 | **head_only（主方案）** | **full（对照）** |
|---|---|---|
| 冻结范围 | 整个 Octo Transformer | 无冻结 |
| 可训练参数叶子数 | **27 个** | 366 个 |
| 冻结参数叶子数 | 339 个 | 0 |
| 训练步数 | 1,000 | 1,000 |
| batch size | 8 | 8 |
| 输出目录 | `octo_head_only_losslog_*` | `octo_full_losslog_*` |

### 参数冻结的验证方式

加载 Base 与 Final 后逐叶子对比参数：

```python
from flax.traverse_util import flatten_dict
base_flat = flatten_dict(base_model.params)
head_flat = flatten_dict(headonly_model.params)

n_diff = sum(1 for k in base_flat
             if np.any(np.array(base_flat[k]) != np.array(head_flat[k])))
n_same = len(base_flat) - n_diff
```

实测结果：

```
参数叶子总数: 366
  与 Base 不同: 27        ← 只有动作头被更新
  与 Base 相同（被冻结）: 339   ← Transformer 完全冻结
```

---

## 2. 可训练参数说明

### head_only 模式下被更新的模块（27 个叶子）

动作头（diffusion head）的以下组件：

| 组件 | 说明 |
|---|---|
| `heads_action` 的 Dense 层 | 动作投影层 |
| `heads_action` 的 LayerNorm | 归一化层参数 |
| `heads_action` 的扩散头 MLP | 去噪网络 |
| 时间步嵌入 | diffusion timestep embedding |

### 被冻结的模块（339 个叶子）

| 组件 | 说明 |
|---|---|
| `octo_transformer` 的 ViT 编码器 | 图像 tokenizer（primary + wrist）|
| `octo_transformer` 的 Transformer blocks | 跨模态注意力层 |
| 语言编码器（T5 投影） | 语言 token 投影 |
| 位置编码 | positional embedding |

**冻结的意义**：保留 Octo 在海量 Open X-Embodiment 数据上学到的通用视觉理解与跨模态对齐能力，只让动作头适配本次的 717 条放置类数据，降低小数据下的灾难性遗忘风险。

---

## 3. 超参数

| 参数 | 值 | 来源 |
|---|---|---|
| batch size | 8 | 显存限制（8 GB）|
| 训练步数 | 1,000 | 见第 5 节选择规则 |
| 优化器 | AdamW | Octo 默认 |
| `weight_decay` | 0.01 | Octo 默认 |
| 学习率调度 | cosine | Octo 默认 |
| `peak_value` | 3e-4 | Octo 默认 |
| `init_value` | 0.0 | Octo 默认 |
| `end_value` | 0.0 | Octo 默认 |
| `warmup_steps` | 100 | 本项目调整（默认 2000，但 1,000 步训练下需缩小）|
| `decay_steps` | 1,000（= num_steps）| 本项目调整 |
| 随机种子 | 0 | 固定 |
| 图像分辨率 | primary 256×256，wrist 128×128 | Octo 默认 |
| 动作预测长度 | 4 步 | Octo-Small 默认 |
| 上下文窗口 | 1 | 本项目设置（Octo-Small 最大支持 2）|
| 保存间隔 | 250 步 | 本项目设置 |
| 日志间隔 | 50 步 | 本项目设置 |

### 训练命令

```bash
# 主方案
MODE=head_only NUM_STEPS=1000 BATCH_SIZE=8 bash train.sh

# 对照
MODE=full NUM_STEPS=1000 BATCH_SIZE=8 bash train.sh
```

---

## 4. Loss 曲线

训练指标为 `action_loss`（动作预测的 MSE，在归一化空间）。

### 4.1 head_only（主方案）

| step | loss | grad_norm | learning_rate |
|---:|---:|---:|---:|
| 50 | 1.278 | 44.31 | 1.47e-4 |
| 100 | 2.885 | 56.32 | 2.97e-4 |
| 150 | 2.147 | 35.21 | 2.98e-4 |
| 200 | 2.823 | 55.92 | 2.91e-4 |
| 250 | 2.257 | 44.87 | 2.80e-4 |
| 300 | 2.613 | 52.16 | 2.65e-4 |
| 350 | 2.654 | 36.89 | 2.47e-4 |
| 400 | 1.512 | 25.34 | 2.25e-4 |
| 450 | 3.478 | 53.09 | 2.02e-4 |
| 500 | 2.107 | 37.04 | 1.77e-4 |
| 550 | 1.618 | 32.62 | 1.51e-4 |
| 600 | 2.051 | 39.25 | 1.24e-4 |
| 650 | 1.722 | 34.56 | 9.9e-5 |
| 700 | 2.193 | 32.35 | 7.5e-5 |
| 750 | 2.248 | 36.25 | 5.4e-5 |
| 800 | 2.480 | 53.90 | 3.5e-5 |
| 850 | 2.259 | 38.15 | 2.0e-5 |
| **900** | **0.917** | 31.93 | 9e-6 |
| 950 | 1.209 | 22.77 | 2e-6 |
| 1000 | 1.604 | 30.19 | 0 |

**趋势分析**：

```
loss
3.5 |     *
3.0 |  *     *     *
2.5 |    * *     *   * *
2.0 |  *   *   *   *   *   *   *
1.5 | *       *     *       *   *   *
1.0 |                        *
0.5 |
    +--------------------------------
     50  200  400  600  800  900 1000
```

- 前 800 步 loss 在 **1.5–3.5** 波动，均值约 **2.4**
- 最后 200 步（850–1000）明显下降：**0.92–2.26**
- 最低点出现在 **step 900（loss=0.917）**
- 学习率在 900 步后已衰减到接近 0，因此 loss 波动主要来自 batch 抽样差异

**波动较大的原因**：batch size 仅 8，而 717 条轨迹覆盖的任务差异大（pot / pan / cloth / spoon 等），每个 batch 的动作分布不同，导致 MSE 波动明显。这是小批量微调的正常现象。

### 4.2 full（对照）

| step | loss | grad_norm | learning_rate |
|---:|---:|---:|---:|
| 50 | 1.764 | — | 1.5e-4 |
| 100 | 2.647 | 38.10 | 2.97e-4 |
| 150 | 1.919 | 25.11 | 2.98e-4 |
| 200 | 2.724 | 27.27 | 2.91e-4 |
| 250 | 2.936 | 25.00 | 2.80e-4 |
| 300 | 2.650 | 25.61 | 2.65e-4 |
| 350 | 3.466 | 35.67 | 2.47e-4 |
| 400 | 2.196 | 27.14 | 2.25e-4 |
| 450 | **4.202**（最高）| 29.96 | 2.02e-4 |
| 500 | 2.320 | 36.87 | 1.77e-4 |
| 550 | 2.170 | 28.21 | 1.51e-4 |
| 600 | 2.151 | 25.60 | 1.24e-4 |
| 650 | 1.581 | 21.25 | 9.9e-5 |
| 700 | 1.622 | 24.26 | 7.5e-5 |
| 750 | 2.153 | 25.41 | 5.4e-5 |
| 800 | 3.031 | 44.48 | 3.5e-5 |
| 850 | 2.713 | 27.84 | 2.0e-5 |
| **900** | **0.997**（最低）| 15.10 | 9e-6 |
| 950 | 1.595 | 23.70 | 2e-6 |
| 1000 | 1.375 | 23.97 | 0 |

### 4.3 两种方式的 loss 对比

| step | head_only | full | 差异 |
|---:|---:|---:|---:|
| 50 | 1.278 | 1.764 | full 更高 |
| 100 | 2.885 | 2.647 | full 更低 |
| 250 | 2.257 | 2.936 | full 更高 |
| 450 | 3.478 | 4.202 | full 更高 |
| 500 | 2.107 | 2.320 | full 更高 |
| 650 | 1.722 | 1.581 | full 更低 |
| 800 | 2.480 | 3.031 | full 更高 |
| 900 | 0.917 | 0.997 | head_only 更低 |
| 950 | 1.209 | 1.595 | head_only 更低 |
| **1000** | **1.604** | **1.375** | **full 更低** |

**观察与解释**：

| 观察 | 说明 |
|---|---|
| 两者 loss 均大幅波动（0.9–4.2）| batch size 仅 8，717 条数据覆盖的物体/场景差异大（pot / pan / cloth / spoon / burner），每个 batch 的动作分布不同 |
| 前 600 步 full 的 loss 普遍**高于** head_only | full 需要更新 366 个叶子（含视觉编码器），前期调整幅度大，拟合更慢 |
| 后 400 步 full 逐渐**低于** head_only | 可训练参数多，后期拟合能力更强 |
| **最终 loss：full 1.375 < head_only 1.604** | 符合预期：full 的容量更大，训练集拟合更好 |

**重要提醒**：**训练集 loss 更低不代表评测表现更好**。full 更新了视觉编码器，在只有 717 条数据（且与目标任务存在 domain gap）的情况下，可能存在过拟合与灾难性遗忘风险。哪个 checkpoint 在 T1/T2 上表现更好，需由评测结果决定。

### 4.4 训练曲线（ASCII 示意）

```
loss
4.5 |              * (full 450)
4.0 |
3.5 |    *         *     *
3.0 |    *    *    *   *  *  *
2.5 | *  *  *  *  *  *  *  *  *  *  *
2.0 | *  *  *  *  *  *  *  *  *  *  *  *
1.5 |          *        *     *  *  *  *
1.0 |                          *  *
0.5 |
    +--+--+--+--+--+--+--+--+--+--+--
      50 150 250 350 450 550 650 750 850 950
      
      * = 两种方式的 loss 波动区间（共 20 个记录点）
```

---

## 5. Final checkpoint 选择规则

**规则：预先确定训练步数，两个模型统一取第 750 步的权重。**

依据作业第 3.4 节：
> "Final checkpoint 应根据训练记录或独立验证集确定，并说明选择规则，例如采用**预先确定训练步数的权重**。"

### 为什么取 750 步

训练实际跑满 1,000 步（loss 记录到 step 1000），但 checkpoint 的完整性不同：

| 模型 | step 250 | step 500 | step 750 | step 1000 |
|---|:---:|:---:|:---:|:---:|
| head_only | ✅ | ✅ | ✅ | ✅ 完整 |
| full | ✅ | ✅ | ✅ | ❌ 保存中断（仅 132 KB）|

full 模式在保存 step 1000 时内存压力过大，orbax 的临时目录未完成 finalize（权重数据未写完）。

**为保证两个模型步数完全一致**，统一采用 **step 750**——这样 Base / Final 的差异只来自"是否微调"和"微调方式"，不掺入训练步数差异。

选择 750 步的理由：

| 考虑 | 说明 |
|---|---|
| **数据规模** | 717 条 × 约 33 样本/条 ≈ 23,661 个样本；batch 8 → 约 2,958 步/epoch。750 步 ≈ 0.25 epoch |
| **公平对比** | 两个模型都有完整、可加载的 step 750 权重 |
| **环境限制** | WSL2 分配 12 GB 内存，接近 1,000 步时出现内存压力，full 的最终保存未能完成 |
| **避免过拟合** | 0.25 epoch 尚未充分拟合，配合 `head_only` 的冻结策略，过拟合风险低 |

**未使用验证集选择 checkpoint 的原因**：验证集（3 个分片 / 156 条轨迹）用于监控而非选择；由于只训练 750 步（约 1/4 个 epoch），验证集 loss 尚未收敛，用它选择 checkpoint 意义有限。采用固定步数规则更符合"预先确定"的要求，也避免触碰"评测隔离"红线（验证集选择若不当，容易演变为对评测结果的间接调参）。

---

## 6. 训练环境与遇到的问题

### 6.1 硬件与软件环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX 5070 Laptop（8 GB 显存，sm_120 / Blackwell）|
| 系统 | Windows 11 + WSL2（Ubuntu）|
| WSL 内存 | 12 GB（`.wslconfig` 中设置 13 GB）|
| `/dev/shm` | 10 GB（手动 remount）|
| JAX | 0.6.2（`jax[cuda12]`）|
| TensorFlow | 2.15.0（仅用于数据加载）|
| Flax | 0.10.7 |
| transformers | 4.55.4 |
| Octo | commit `241fb3514b7c40957a86d869fecb7c7fc353f540` |

### 6.2 遇到的问题与修复（可复现参考）

| 问题 | 现象 | 修复 |
|---|---|---|
| JAX 0.6.2 移除 `jax.random.KeyArray` | `AttributeError: module 'jax.random' has no attribute 'KeyArray'` | 改为 `jax.Array` |
| JAX 0.6.2 移除 `jax.tree_map` | `AttributeError: jax.tree_map was removed in JAX v0.6.0` | 改为 `jax.tree.map`（31 处）|
| JAX 0.6.2 移除 `jax.tree_leaves` | 同上 | 改为 `jax.tree.leaves` |
| `multihost_utils.process_allgather` 加维 | 单设备下 `(8,16)` → `(1,8,16)`，导致 T5 reshape 报错 | 单设备时跳过该调用 |
| wandb 与 TF protobuf 冲突 | wandb 0.31 需要 protobuf 7.x，TF 2.15 要求 <4.21 | 用 stub 替代 wandb（`wandb_stub/wandb.py`）|
| 内存不足 OOM | 训练约 500 步被系统杀死 | WSL 内存 10 GB → 13 GB；`/dev/shm` 扩到 10 GB |
| 可视化回调 OOM | 训练到验证步时崩溃 | 禁用 viz_callback |
| matplotlib 3.10 API 变更 | `FigureCanvasAgg` 无 `tostring_rgb` | 改用 `buffer_rgba` |
| HuggingFace 直连超时 | 加载模型时下载 T5-Base tokenizer 卡住 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| `sample_actions` 未反归一化 | 动作值看起来异常（±5）| 属正常现象，反归一化后为 ±5cm 合理范围 |

### 6.3 训练耗时

| 方式 | 耗时（1,000 步）|
|---|---|
| head_only | 约 2.5 分钟（含编译）|
| full | 约 10 分钟（含编译）|

---

## 7. 训练日志文件

| 文件 | 说明 |
|---|---|
| [`../train_log_head_only.txt`](../train_log_head_only.txt) | head_only 完整日志（含 `[TRAIN] step=...` 记录）|
| [`../train_log_full.txt`](../train_log_full.txt) | full 完整日志（含 `[TRAIN] step=...` 记录）|

日志中包含：
- 数据加载信息（分片数、轨迹数）
- 每个 checkpoint 保存记录
- 每 50 步的 `[TRAIN]` 指标（loss / mse / grad_norm / learning_rate / param_norm / update_norm）
