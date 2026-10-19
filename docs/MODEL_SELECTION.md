# 模型选择与 WidowX 训练依据

## 1. 选定的模型

| 项目 | 值 |
|---|---|
| 模型名称 | Octo-Small **version 1.5** |
| HuggingFace 仓库 | [`rail-berkeley/octo-small-1.5`](https://huggingface.co/rail-berkeley/octo-small-1.5) |
| checkpoint step | **300000** |
| 参数量 | 27M（约等于 ViT-S）|
| 权重文件 | `300000/default/checkpoint`（约 546 MB）|
| 许可证 | MIT |
| 官方代码 | <https://github.com/octo-models/octo> |
| 论文 | [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213) |

### 模型结构要点（来自官方模型卡）

> Octo Small is trained with a window size of 2, predicting 7-dimensional actions 4 steps into the future using a diffusion policy. The model is a Transformer with 27M parameters (equivalent to a ViT-S). Images are tokenized by preprocessing with a lightweight convolutional encoder, then grouped into 16x16 patches. Language is tokenized by applying the T5 tokenizer, and then applying the T5-Base language encoder.

---

## 2. 使用 WidowX 数据训练的依据

作业第 3.1 节要求：**所选 checkpoint 必须已经使用 WidowX 机器人数据进行过预训练或微调**，并提供依据（论文、模型卡或官方仓库）。

以下提供三条相互独立的证据。

### 依据 1：官方模型卡的数据构成表

随 checkpoint 一起发布的 `README.md`（即 HuggingFace 模型卡）第 37–43 行：

```
This model was trained on a mix of datasets from the Open X-Embodiment dataset.

| Dataset                                                    | Proportion of batch |
|------------------------------------------------------------|---------------------|
| Fractal (Brohan et al, 2022)                               | 17.0%               |
| Kuka (Kalashnikov et al, 2018)                             | 17.0%               |
| Bridge (Walke et al, 2023)                                 | 17.0%               |
```

**关键点**：`Bridge (Walke et al., 2023)` 即 **BridgeData V2**。该数据集的采集平台就是 **WidowX 250s** 机械臂（6-DoF + 平行夹爪），因此 Octo-Small-1.5 在预训练阶段**直接见过 WidowX 的观测与动作数据**，且占比达 **17.0%**。

来源：
- 模型卡：<https://huggingface.co/rail-berkeley/octo-small-1.5>
- BridgeData V2 官方：<https://rail-berkeley.github.io/bridgedata/>

### 依据 2：checkpoint 内的归一化统计量

checkpoint 附带的 `dataset_statistics.json` 中，明确存在 `bridge_dataset` 条目：

| 字段 | 值 |
|---|---|
| `num_trajectories` | 60,064 |
| `num_transitions` | 2,195,527 |
| `action.mean` | `[0.00022, 0.00013, -0.00017, -0.00016, -0.00025, 0.00025, 0.58795]` |
| `action.std` | `[0.00963, 0.0135, 0.01251, 0.02815, 0.03028, 0.07586, 0.48772]` |
| `action.mask` | `[True, True, True, True, True, True, False]` |

这组统计量是该权重**训练时实际计算并使用**的归一化参数。其中：
- 轨迹数 60,064 与 BridgeData V2 官方公布的规模一致
- 动作维度为 7（位置增量 3 + 旋转增量 3 + 夹爪 1）
- 第 7 维（夹爪）`mask=False`，与 WidowX 的夹爪约定一致

**这说明该 checkpoints 的训练管线确实注入了 WidowX（Bridge）数据。**

### 依据 3：SimplerEnv 官方将其作为 WidowX 基准

SimplerEnv 仓库的 `scripts/octo_bridge.sh`：

```bash
policy_models=("octo-small" "octo-base")
...
ckpt_path=None            # 为 None 时默认从 HuggingFace 加载
robot=widowx
policy_setup=widowx_bridge
```

即 SimplerEnv 作者把 `rail-berkeley/octo-small` 作为 **WidowX 评测的默认基准 checkpoint**。

来源：<https://github.com/simpler-env/SimplerEnv/blob/main/scripts/octo_bridge.sh>

---

## 3. 为什么选 Octo-Small-1.5

| 考虑 | 说明 |
|---|---|
| **有 WidowX 数据依据** | 见上文三条证据，满足作业第 3.1 节硬性要求 |
| **有 SimplerEnv 现成评测** | SimplerEnv 官方 `octo_bridge.sh` 直接支持，动作接口已对齐 |
| **模型规模适合本地微调** | 27M 参数，8 GB 显存即可完成微调（本项目实测显存占用约 6 GB）|
| **提供语言条件接口** | T5-Base 编码语言指令，与两个目标任务的指令式输入匹配 |
| **动作空间兼容** | 7 维 `[dx, dy, dz, droll, dpitch, dyaw, gripper]`，与 WidowX 控制器一致 |
| **开源可复现** | MIT 许可，代码与权重全部公开 |

### 版本说明（重要）

| | SimplerEnv 默认 | 本项目使用 |
|---|---|---|
| 模型 | `octo-small`（1.0）| `octo-small-1.5`（1.5）|

**本项目使用 1.5 版**，理由：1.5 版在 1.0 基础上做了以下修复（见模型卡 "Updates for Version 1.5"）：

1. Language task tokens 现在在每个时间步重复
2. 用 GPT-3.5 改写了语言指令（增强语言多样性）
3. 三个 bug 修复：
   - 关闭扩散头的 dropout（与 layer norm 不兼容）
   - 修正注意力掩码的 off-by-one 错误
   - 修正不同图像增强未使用新的随机种子的问题

1.5 版的训练数据构成与 1.0 相同（Bridge 仍占 17%），因此 WidowX 依据依然成立。

---

## 4. 动作接口对齐（WidowX）

| 项目 | 值 |
|---|---|
| 动作维度 | 7 |
| 位置增量 | 3 维（米）|
| 旋转增量 | 3 维（弧度，axis-angle）|
| 夹爪 | 1 维（0 = 闭合，1 = 张开）|
| 控制频率 | 5 Hz（SimplerEnv WidowX 官方配置）|
| 控制模式 | `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` |

> 动作的旋转表示、尺度、夹爪符号以 SimplerEnv 的控制器与 Octo 的适配代码为准，详见 `evaluate.py`。

---

## 5. 参考链接

1. Octo 论文：<https://arxiv.org/abs/2405.12213>
2. Octo 官方代码：<https://github.com/octo-models/octo>
3. Octo-Small 1.5 模型卡：<https://huggingface.co/rail-berkeley/octo-small-1.5>
4. BridgeData V2：<https://rail-berkeley.github.io/bridgedata/>
5. SimplerEnv：<https://github.com/simpler-env/SimplerEnv>
6. SimplerEnv 论文：<https://arxiv.org/abs/2405.05941>
