# 评测结果为 0 的原因与修复方案

> ## ✅ 最新方案（推荐，无需打补丁）
>
> 仓库已内置自包含适配层 `scripts/octo_bridge_policy.py`，`evaluate.py` 已改为直接使用它，
> **不再依赖 SimplerEnv 的 `OctoInference`**。因此：
>
> - 从仓库重新部署后**自动生效**，不需要修改任何第三方代码
> - 下面第三节的 `patch_simplerenv_octo.py` 仅作为**备选**保留（适用于想继续用
>   SimplerEnv 原生 `OctoInference` 的情况）
>
> 自包含适配层修正了同样两处问题：
> 1. 用 `timestep_pad_mask`（而非 `pad_mask`）
> 2. **只对 mask=True 的维度反归一化**（夹爪保持原始值 0/1）
>
> 已用真实 Bridge 轨迹验证：模型在需要抓取的帧能正确输出 close
> （帧 15–17 判定 close，与真实标注一致），而修复前永远是 open。


## 一、现象

3 个模型（Base / Final-A / Final-B）在 T1、T2 上全部 **0/10**，所有 episode 都是 `truncated=True`（跑满 60/120 步），无异常。

## 二、根因：SimplerEnv 的 `OctoInference` 与 Octo 0.6.x 有两处不兼容

文件：`SimplerEnv/simpler_env/policies/octo/octo_model.py`

### Bug 1：`pad_mask` 键名错误（会直接 KeyError）

```python
# 现有代码（第 158 行）
input_observation = {"image_primary": images, "pad_mask": pad_mask}
```

但 Octo 0.6.x 的 `sample_actions` 读的是：

```python
# octo/model/octo_model.py 第 202 行
timestep_pad_mask = observations["timestep_pad_mask"]
```

→ `KeyError: 'timestep_pad_mask'`

> 若评测能跑起来而不报错，说明这处已被手工改过。但仍需检查下面的 Bug 2。

### Bug 2：夹爪被错误地反归一化（**这是 0 成功的真正原因**）

```python
# 现有代码（第 164 行）
raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]
```

**问题**：Octo 的 `action_normalization_mask = [T,T,T,T,T,T,False]`，第 7 维（夹爪）**训练时不归一化**，模型输出的就是原始值 0/1。

再对它做反归一化（Bridge 统计量 `mean=0.58795, std=0.48772`）：

| 模型输出（夹爪）| 反归一化后 | `> 0.5` 判定 | 正确吗 |
|---|---:|---|---|
| 1.0（open）| 1.076 | open | ✅ |
| **0.0（close）** | **0.588** | **open** | ❌ **错** |
| 0.2 | 0.685 | open | ❌ |
| -0.02 | 0.578 | open | ❌ |

**结论：模型输出任何 ≤ 0.8 的值，都被判成 open。夹爪物理上永远无法闭合。**

而任务成功判定的前提是 `is_src_obj_grasped`（抓住物体）—— 夹爪不闭合 → 抓不住 → `src_on_target` 永远 False → **成功率恒为 0**。

## 三、修复

### 一键修复（推荐）

```bash
# 1. 检查当前状态
python scripts/patch_simplerenv_octo.py --check

# 2. 应用修复（自动备份为 .bak）
python scripts/patch_simplerenv_octo.py

# 3. 验证
python scripts/patch_simplerenv_octo.py --verify
```

若 SimplerEnv 不在默认路径，用 `--path` 指定：

```bash
python scripts/patch_simplerenv_octo.py \
  --path /path/to/SimplerEnv/simpler_env/policies/octo/octo_model.py
```

### 手工修改（两处）

**修改 1**：键名 + dtype

```python
# 原
input_observation = {"image_primary": images, "pad_mask": pad_mask}

# 改为
input_observation = {
    "image_primary": images,
    "timestep_pad_mask": pad_mask.astype(bool),
}
```

**修改 2**：夹爪不反归一化

```python
# 原
raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]

# 改为
_n = norm_raw_actions.shape[-1]
_w = 6 if _n == 7 else _n            # 前 6 维归一化，第 7 维（夹爪）保持原值
raw_actions = norm_raw_actions.copy()
raw_actions[..., :_w] = (
    norm_raw_actions[..., :_w] * self.action_std[None, :_w]
    + self.action_mean[None, :_w]
)
```

## 四、修复后预期效果

用数值模拟验证（模型每步预测 `[O, O, O, C]` = `[1,1,1,0]`）：

| 情况 | 执行序列 | 夹爪判定 |
|---|---|---|
| **修复前** | [1.076, 1.076, 1.076, 0.954, ...] | O, O, O, O, ... **永远 open** |
| **修复后** | [1.0, 1.0, 1.0, 0.75, ...] | O, O, O, O（此预测仍 open，见下）|

若模型预测 `[O, C, C, C]` = `[1,0,0,0]`：

| 情况 | 执行序列 | 夹爪判定 |
|---|---|---|
| **修复前** | [1.076, 0.832, 0.751, 0.71, ...] | O, O, O, O **永远 open** ❌ |
| **修复后** | [1.0, 0.5, 0.333, 0.25, ...] | O, **C, C, C** ✅ **能闭合** |

**关键**：修复后只要模型在预测序列的前几步给出 close，ActionEnsembler 就能让夹爪真正闭合。

## 五、验证修复是否生效（建议先跑 1 个 episode）

```bash
python evaluate.py \
  --model-name VerifyFix \
  --model-type octo-small \
  --task widowx_spoon_on_towel \
  --episodes 1 \
  --eval-config configs/eval_config.json \
  --output-dir /tmp/verify_fix
```

**检查点**：
1. 不应出现 `KeyError: 'timestep_pad_mask'`
2. `episodes.csv` 里不应全是 `truncated=True`
3. 若仍全失败，请按第六节做诊断

## 六、若修复后仍为 0，请提供以下诊断信息

在 `run_episode` 的循环里（第一次 `model.step` 前后）加几行打印：

```python
# 1) 图像检查（确认不是全黑/全白）
print("[diag] image", image.shape, image.dtype, image.min(), image.max(), image.mean())

# 2) 指令检查
print("[diag] instruction:", repr(instruction))

# 3) 模型原始输出
print("[diag] raw world_vector:", raw_action["world_vector"])
print("[diag] raw rotation:", raw_action.get("rotation_delta"))
print("[diag] raw gripper:", raw_action["open_gripper"])

# 4) 实际下发给环境的动作
print("[diag] env action:", env_action)
```

把前 3 个 episode 的这段输出发回来即可定位。

**特别关注**：
- `image.mean()` 是否在 60–180 之间（若接近 0 说明图像全黑）
- `raw gripper` 是否在 0/1 附近（若在 −3~+3 说明模型输出是归一化值，则反归一化逻辑要反过来改）
- `instruction` 是否为空或与非任务描述不符

## 七、其他已排除的问题

| 检查项 | 结论 |
|---|---|
| `rgb_overlay_path` 绝对路径 | ✅ 可用（SimplerEnv 直接用 `os.path.exists` 检查，不拼接）|
| 视觉匹配 `rgb_overlay_mode` | ✅ 默认 `'background'`，合法 |
| 图像 dtype | ✅ ManiSkill2 返回 `uint8 (0–255)`，`_resize_image` 假设正确 |
| 图像尺寸 | ✅ `_resize_image` 统一 resize 到 256×256 |
| 语言指令来源 | ✅ `env.get_language_instruction()` 正常实现 |
| 成功判定 | ✅ 环境原生 `done`，与官方 `maniskill2_evaluator.py` 一致 |
| 初始状态 | ✅ 与 `eval_config.json` 一致（robot_init、obj_episode_id）|
| 控制频率 / 最大步数 | ✅ 5 Hz / 60 / 120，与官方 `octo_bridge.sh` 一致 |
