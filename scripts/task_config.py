"""
SimplerEnv WidowX 任务配置 —— 与 SimplerEnv 官方 `scripts/octo_bridge.sh` 保持一致。

本模块是纯 Python（不导入 sapien / torch / jax），因此可以在任何机器上导入校验，
也用于生成"评测初始状态清单"。

作业对应关系（B 类必做）：
    T1  widowx_spoon_on_towel            -> PutSpoonOnTableClothInScene-v0
    T2  widowx_put_eggplant_in_basket    -> PutEggplantInBasketScene-v0
    T3  widowx_carrot_on_plate           （B 类不要求）
    T4  widowx_stack_cube                （B 类不要求）

参数来源：https://github.com/simpler-env/SimplerEnv/blob/main/scripts/octo_bridge.sh
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TaskConfig:
    """单个 WidowX 任务的完整评测配置。"""

    key: str                        # 任务标识（SimplerEnv 注册名）
    env_name: str                   # ManiSkill2 环境类名
    scene_name: str
    robot: str
    rgb_overlay: str                # 视觉匹配用的真实照片贴图（仓库内相对路径）
    robot_init_x: float
    robot_init_y: float
    robot_init_quat: Tuple[float, float, float, float]  # w, x, y, z
    max_episode_steps: int
    control_freq: int = 5
    sim_freq: int = 500
    obj_episode_range: Tuple[int, int] = (0, 24)        # 官方脚本使用的对象变化范围
    instruction: str = ""                               # 留空则由环境返回原生指令

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["robot_init_quat"] = list(self.robot_init_quat)
        d["obj_episode_range"] = list(self.obj_episode_range)
        return d


# 控制模式：WidowX 固定使用该 control_mode（见 simpler_env/utils/env/env_builder.py）
WIDOWX_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"

# 用于 obs 取图的相机（WidowX 的第三人称视角）
OBS_CAMERA_NAME = "3rd_view_camera"

TASKS: Dict[str, TaskConfig] = {
    # ---------------- T1：把勺子放到毛巾上 ----------------
    "widowx_spoon_on_towel": TaskConfig(
        key="widowx_spoon_on_towel",
        env_name="PutSpoonOnTableClothInScene-v0",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        rgb_overlay="data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x=0.147,
        robot_init_y=0.028,
        robot_init_quat=(0.0, 0.0, 0.0, 1.0),
        max_episode_steps=60,
    ),
    # ---------------- T2：把茄子放进篮子 ----------------
    "widowx_put_eggplant_in_basket": TaskConfig(
        key="widowx_put_eggplant_in_basket",
        env_name="PutEggplantInBasketScene-v0",
        scene_name="bridge_table_1_v2",
        robot="widowx_sink_camera_setup",
        rgb_overlay="data/real_inpainting/bridge_sink.png",
        robot_init_x=0.127,
        robot_init_y=0.06,
        robot_init_quat=(0.0, 0.0, 0.0, 1.0),
        max_episode_steps=120,   # 官方：eggplant 用 120 步
    ),
    # ---------------- T3 / T4：B 类不要求，保留以备 A 类或扩展 ----------------
    "widowx_carrot_on_plate": TaskConfig(
        key="widowx_carrot_on_plate",
        env_name="PutCarrotOnPlateInScene-v0",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        rgb_overlay="data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x=0.147,
        robot_init_y=0.028,
        robot_init_quat=(0.0, 0.0, 0.0, 1.0),
        max_episode_steps=60,
    ),
    "widowx_stack_cube": TaskConfig(
        key="widowx_stack_cube",
        env_name="StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        rgb_overlay="data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x=0.147,
        robot_init_y=0.028,
        robot_init_quat=(0.0, 0.0, 0.0, 1.0),
        max_episode_steps=60,
    ),
}

# B 类必做任务
B_CLASS_TASKS: Tuple[str, ...] = ("widowx_spoon_on_towel", "widowx_put_eggplant_in_basket")
# A 类四任务
A_CLASS_TASKS: Tuple[str, ...] = (
    "widowx_spoon_on_towel",
    "widowx_put_eggplant_in_basket",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
)


def get_task(key: str) -> TaskConfig:
    if key not in TASKS:
        raise KeyError(f"未知任务 {key!r}；可选：{sorted(TASKS)}")
    return TASKS[key]


def iter_tasks(keys: Tuple[str, ...]) -> Tuple[TaskConfig, ...]:
    return tuple(get_task(k) for k in keys)
