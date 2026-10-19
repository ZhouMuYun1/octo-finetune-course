#!/usr/bin/env python
"""
生成"正式评测初始状态清单"。

作业第 6.2 节要求：
  · 在正式评测前确定并保存初始状态清单，记录选取或采样方法；
  · 不得根据评测结果只保留容易成功的状态；
  · 只要信息配合记录的环境版本能够重建初始条件，即无需保存完整仿真状态文件。

本脚本的做法：
  对每个任务，**确定性**地从官方对象变化范围 [0, 24) 中取前 N 个 episode_id，
  机器人初始位姿固定为官方值。因此初始状态完全由参数决定、可被精确重建，
  不需要保存仿真状态文件，也不涉及任何"按结果筛选"。

重要（评测隔离）：
  生成后请勿再修改本清单。该清单只用于正式评测，不得用于训练数据采集、
  调参或选择 checkpoint。若某组初始状态已用于开发/调参，必须另取一组。

用法：
    python generate_eval_config.py --episodes 10 --out-dir configs
    python generate_eval_config.py --episodes 10 --out-dir configs --tasks widowx_spoon_on_towel
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_config import B_CLASS_TASKS, get_task  # noqa: E402

# 环境版本信息：作业要求记录 SimplerEnv 版本/提交号
SIMPLER_ENV_VERSION = "github.com/simpler-env/SimplerEnv (main) + ManiSkill2_real2sim submodule"
ENV_SEED_SPEC = (
    "初始状态由 (robot_init_x, robot_init_y, robot_init_quat, obj_episode_id) 完全确定；"
    "环境随机种子用于对象/纹理采样的复现"
)


def build_config(tasks, episodes: int, env_seed: int, model_seed: int) -> dict:
    manifest = {
        "meta": {
            "purpose": "正式评测初始状态清单（B 类：T1/T2）",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selection_rule": (
                f"确定性选取：对每个任务取官方对象变化范围 obj_episode_range 的前 {episodes} 个 "
                "episode_id（0..N-1）；机器人初始位姿采用 SimplerEnv 官方 octo_bridge.sh 中的固定值。"
                "不进行任何基于评测结果的筛选。"
            ),
            "eval_isolation_note": (
                "本清单仅供正式评测使用。禁止用于选择 checkpoint、调参或筛选研究设置。"
                "开发与训练数据采集必须使用独立的初始状态。"
            ),
            "control_freq_hz": None,   # 由各任务填充（均为 5）
            "simpler_env_version": SIMPLER_ENV_VERSION,
            "env_seed_spec": ENV_SEED_SPEC,
            "env_seed": env_seed,
            "model_seed": model_seed,
            "episodes_per_task": episodes,
        },
        "tasks": {},
    }

    for key in tasks:
        cfg = get_task(key)
        lo, hi = cfg.obj_episode_range
        if episodes > (hi - lo):
            raise ValueError(
                f"{key}: 请求 {episodes} 个 episode，但官方对象变化范围只有 {hi - lo} 个"
            )
        spec = cfg.as_dict()
        spec["episodes"] = [
            {
                "episode_index": i,
                "episode_id": i,
                "env_seed": env_seed,
                "model_seed": model_seed,
                "robot_init_x": cfg.robot_init_x,
                "robot_init_y": cfg.robot_init_y,
                "robot_init_quat": list(cfg.robot_init_quat),
                "obj_episode_id": lo + i,
            }
            for i in range(episodes)
        ]
        manifest["tasks"][key] = spec

    manifest["meta"]["control_freq_hz"] = 5
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10, help="每个任务的 episode 数（B 类为 10）")
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--tasks", nargs="*", default=list(B_CLASS_TASKS))
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument(
        "--name",
        default="eval_config.json",
        help="输出文件名（正式评测请保持 eval_config.json 不变）",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = build_config(args.tasks, args.episodes, args.env_seed, args.model_seed)

    path = os.path.join(args.out_dir, args.name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"written: {path}")
    for key, spec in manifest["tasks"].items():
        print(
            f"  {key:34s} episodes={len(spec['episodes'])} "
            f"max_steps={spec['max_episode_steps']} freq={spec['control_freq']}Hz "
            f"overlay={spec['rgb_overlay']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
