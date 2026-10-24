#!/usr/bin/env python
"""自包含的 Octo → WidowX Bridge 策略适配层。

**为什么需要这个文件**

SimplerEnv 自带的 `simpler_env/policies/octo/octo_model.py::OctoInference` 在
Octo 0.6.x 下有两处致命问题：

1. **键名错误**：传给 `sample_actions` 的是 `pad_mask`，但 Octo 0.6.x 读
   `observations["timestep_pad_mask"]` → `KeyError`。

2. **夹爪被错误反归一化**：Octo 的 `action_normalization_mask` 第 7 维为 False，
   夹爪在训练时**不归一化**，模型输出的就是原始值 0/1。再对它做
   `* std + mean` 会把 0（close）变成 0.588，判定为 open
   → **夹爪永远无法闭合 → 抓不住物体 → 成功率恒为 0**。

本模块用一个只依赖 `numpy + tensorflow + scipy + octo` 的实现替代它，
不修改 SimplerEnv 的任何文件，因此从仓库重新部署后**自动生效**，无需打补丁。

接口与 `OctoInference` 保持兼容（`reset` / `step`），便于替换。

用法：
    from octo.model.octo_model import OctoModel
    from octo_bridge_policy import OctoBridgePolicy

    model = OctoModel.load_pretrained("<ckpt>", step=750)
    policy = OctoBridgePolicy(model=model, policy_setup="widowx_bridge")
    policy.reset("put the spoon on the towel")
    raw_action, action = policy.step(image_uint8_hwc)
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# 动作集成（与 SimplerEnv 的 ActionEnsembler 行为一致）
# ---------------------------------------------------------------------------
class ActionEnsembler:
    """时序集成：把历史预测中"对应同一时刻"的动作做加权平均。

    例：每次预测 4 步，第 t 步执行时，收集
        pred_{t-3}[3], pred_{t-2}[2], pred_{t-1}[1], pred_t[0]
    取平均。这样能平滑扩散模型输出的随机性。
    """

    def __init__(self, pred_action_horizon: int, temp: float = 0.0):
        self.pred_action_horizon = pred_action_horizon
        self.temp = temp
        self.history = deque(maxlen=pred_action_horizon)

    def reset(self) -> None:
        self.history.clear()

    def ensemble(self, cur_action: np.ndarray) -> np.ndarray:
        """cur_action: (pred_action_horizon, action_dim)"""
        self.history.append(np.asarray(cur_action))
        n = len(self.history)
        # 取每条历史预测中"对应当前时刻"的那一步
        stacked = np.stack([self.history[i][n - 1 - i] for i in range(n)])
        weights = np.exp(-self.temp * np.arange(n))
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * stacked, axis=0)


# ---------------------------------------------------------------------------
# 主策略类
# ---------------------------------------------------------------------------
class OctoBridgePolicy:
    """Octo 模型在 WidowX Bridge 场景下的策略封装。

    Args:
        model: 已加载的 `OctoModel`
        policy_setup: "widowx_bridge" 或 "google_robot"
        dataset_id: 用于取归一化统计量的数据集键，Bridge 为 "bridge_dataset"
        horizon: 图像历史长度（Octo-Small 最大 2）
        pred_action_horizon: 每次预测的动作步数（Octo 默认 4）
        action_scale: 动作缩放系数（1.0 = 不缩放）
        ensemble_temp: 集成温度，0 表示等权平均
        init_rng: 初始随机种子
        use_action_ensemble: 是否启用时序集成
    """

    def __init__(
        self,
        model,
        policy_setup: str = "widowx_bridge",
        dataset_id: Optional[str] = None,
        horizon: int = 2,
        pred_action_horizon: int = 4,
        action_scale: float = 1.0,
        ensemble_temp: float = 0.0,
        init_rng: int = 0,
        use_action_ensemble: bool = True,
        image_size: int = 256,
    ) -> None:
        if policy_setup == "widowx_bridge":
            dataset_id = dataset_id or "bridge_dataset"
        elif policy_setup == "google_robot":
            dataset_id = dataset_id or "fractal20220817_data"
        else:
            raise NotImplementedError(f"不支持的 policy_setup: {policy_setup}")

        self.model = model
        self.policy_setup = policy_setup
        self.dataset_id = dataset_id
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.action_scale = action_scale
        self.image_size = image_size
        self.use_action_ensemble = use_action_ensemble
        self.ensembler = (
            ActionEnsembler(pred_action_horizon, ensemble_temp)
            if use_action_ensemble
            else None
        )

        # ---- 归一化统计量 ----
        stats = self.model.dataset_statistics[dataset_id]["action"]
        self.action_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["std"], dtype=np.float32)
        # 只对 mask=True 的维度反归一化（Bridge 是前 6 维，第 7 维夹爪保持原值）
        mask = stats.get("mask", None)
        if mask is None:
            self.norm_mask = np.ones_like(self.action_mean, dtype=bool)
        else:
            self.norm_mask = np.asarray(mask, dtype=bool)

        # ---- RNG（对齐 Octo server 的推理种子）----
        import jax

        self._rng = jax.random.PRNGKey(int(init_rng))
        for _ in range(5):
            self._rng, _ = jax.random.split(self._rng)

        self.task = None
        self.task_description = None
        self.image_history: deque = deque(maxlen=horizon)
        self.num_image_history = 0
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.sticky_gripper_num_repeat = 1 if policy_setup == "widowx_bridge" else 15

    # ------------------------------------------------------------------
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 → (image_size, image_size, 3) uint8"""
        if image.dtype != np.uint8:
            # 若是 0-1 浮点，先转回 0-255
            if image.size and image.max() <= 1.0:
                image = (image * 255.0).round()
            image = np.clip(image, 0, 255).astype(np.uint8)
        out = tf.image.resize(
            image, size=(self.image_size, self.image_size),
            method="lanczos3", antialias=True,
        )
        return tf.cast(tf.clip_by_value(tf.round(out), 0, 255), tf.uint8).numpy()

    def reset(self, task_description: str) -> None:
        """每个 episode 开始时调用。"""
        self.task = self.model.create_tasks(texts=[task_description])
        self.task_description = task_description
        self.image_history.clear()
        self.num_image_history = 0
        if self.ensembler is not None:
            self.ensembler.reset()
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0

    def _denormalize(self, norm_actions: np.ndarray) -> np.ndarray:
        """只对 norm_mask=True 的维度反归一化。

        这是整个模块最关键的修正：夹爪（mask=False）的模型输出本身就是原始
        值 0/1，若再乘 std 加 mean 会被推向 0.588（> 0.5）而永久判定为 open。
        """
        out = np.array(norm_actions, dtype=np.float32, copy=True)
        out[..., self.norm_mask] = (
            norm_actions[..., self.norm_mask]
            * self.action_std[self.norm_mask]
            + self.action_mean[self.norm_mask]
        )
        return out

    # ------------------------------------------------------------------
    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs):
        """执行一次策略推理。

        Args:
            image: (H, W, 3) uint8，第三人称视角图像
            task_description: 若与上次不同则自动 reset

        Returns:
            raw_action: dict，模型直接输出的动作（已反归一化）
            action: dict，可直接送入 ManiSkill2 环境的动作
        """
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        img = self._resize_image(np.asarray(image))
        self.image_history.append(img)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)
        images = np.stack(self.image_history, axis=0)[None]        # (1, h, H, W, 3)
        pad_mask = np.ones((1, images.shape[1]), dtype=bool)        # (1, h)

        # 更新 RNG（每次前向用不同的 key，对性能影响很大）
        import jax

        self._rng, key = jax.random.split(self._rng)

        # ---- 关键修正 1：用 timestep_pad_mask（不是 pad_mask）----
        observations = {
            "image_primary": images,
            "timestep_pad_mask": pad_mask,
        }
        norm_actions = self.model.sample_actions(observations, self.task, rng=key)
        norm_actions = np.asarray(norm_actions[0])                 # (pred_horizon, 7)

        # ---- 关键修正 2：夹爪不反归一化 ----
        raw_actions = self._denormalize(norm_actions)

        if self.ensembler is not None:
            raw_actions = self.ensembler.ensemble(raw_actions)[None]   # (1, 7)

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),          # 0/1
        }

        # ---- 转成环境动作 ----
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale

        # Bridge 的旋转是 euler(roll, pitch, yaw) → 环境需要 axis-angle * angle
        roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        action["rot_axangle"] = self._euler_to_rotvec(roll, pitch, yaw) * self.action_scale

        if self.policy_setup == "widowx_bridge":
            # 二值化：open(>=0.5) → +1，close → -1
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            ).astype(np.float32)
        else:
            # google_robot 用相对夹爪变化 + sticky 逻辑
            cur = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                rel = np.array([0.0])
            else:
                rel = self.previous_gripper_action - cur
            self.previous_gripper_action = cur
            if np.abs(rel) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = rel
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                rel = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action["gripper"] = np.array(rel, dtype=np.float32)

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    # ------------------------------------------------------------------
    @staticmethod
    def _euler_to_rotvec(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """欧拉角 → 旋转向量 (axis * angle)。

        等价于 transforms3d.euler.euler2axangle 的返回值
        （`ax * angle`），但只依赖 scipy，避免额外安装 transforms3d。

        注意：Bridge 的旋转约定为 static xyz（extrinsic x-y-z），
        对应 scipy 的 'xyz' 小写序列。
        """
        from scipy.spatial.transform import Rotation

        r = Rotation.from_euler("xyz", [roll, pitch, yaw])
        return r.as_rotvec().astype(np.float64)
