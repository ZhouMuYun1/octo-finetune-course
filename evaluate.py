#!/usr/bin/env python
"""
SimplerEnv WidowX 评测脚本（B 类作业核心交付物）。

设计目标 —— 严格满足作业第 6.2 / 3.4 节的评测协议：
  1. 使用**预先固定并保存**的初始状态清单（--eval-config），不按结果筛选；
  2. Base 与 Final 必须在**独立进程**中分别加载运行（本脚本一次只跑一个模型）；
  3. 使用**环境原生成功判定**（env.step 的 done -> info['episode_stats']），
     不用策略输出的 terminate_episode 当作成功；
  4. **不筛除失败 episode**，失败/异常/超时一律记入结果；
  5. 模型 seed 与环境 seed 分别记录；
  6. 逐 episode 明细写入 episodes.csv，汇总写入 results.csv。

用法（Base 与 Final 各跑一次，两次调用是两个独立进程）：

    # Base（加载 HuggingFace 上的原始 octo-base，即"未经本次微调"）
    python evaluate.py \
        --model-name Base \
        --model-type octo-base \
        --task widowx_spoon_on_towel \
        --episodes 10 \
        --eval-config configs/eval_config.json \
        --output-dir artifacts/base

    # Final（加载本次微调产出的 checkpoint）
    python evaluate.py \
        --model-name Final \
        --model-type octo-base \
        --ckpt-path /root/vla_project/checkpoints/final \
        --task widowx_put_eggplant_in_basket \
        --episodes 10 \
        --eval-config configs/eval_config.json \
        --output-dir artifacts/final

注意：本脚本需要可用的 GPU Vulkan 渲染（SAPIEN）。
      在 WSL2 上会失败，请先运行 scripts/verify_render.py 确认环境。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# 仓库根目录（task_config.py 在这里）与 scripts/ 目录（octo_bridge_policy.py 在这里）
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scripts"))

from task_config import (  # noqa: E402
    OBS_CAMERA_NAME,
    WIDOWX_CONTROL_MODE,
    get_task,
)

EPISODES_CSV_FIELDS = [
    "model",
    "task",
    "episode_index",
    "episode_id",
    "env_seed",
    "model_seed",
    "robot_init_x",
    "robot_init_y",
    "robot_init_quat",
    "obj_episode_id",
    "success",
    "num_steps",
    "max_episode_steps",
    "terminated_by_policy",
    "truncated",
    "exception",
    "wall_time_s",
]


def _unwrap_info(info):
    """ManiSkill2 的 gym 包装可能把 info 变成 list，这里统一成 dict。"""
    if isinstance(info, (list, tuple)):
        return info[0] if len(info) else {}
    return info if isinstance(info, dict) else {}


def build_env(task, sim_freq, control_freq, repo_root):
    """构建 SimplerEnv 环境（视觉匹配的 WidowX 设置）。"""
    import simpler_env
    from simpler_env.utils.env.env_builder import build_maniskill2_env

    overlay_path = os.path.join(repo_root, "ManiSkill2_real2sim", task.rgb_overlay)
    if not os.path.exists(overlay_path):
        raise FileNotFoundError(
            f"缺少视觉匹配贴图: {overlay_path}\n"
            "请确认克隆 SimplerEnv 时带了 --recurse-submodules"
        )

    env = build_maniskill2_env(
        task.env_name,
        obs_mode="rgbd",
        robot=task.robot,
        sim_freq=sim_freq,
        control_mode=WIDOWX_CONTROL_MODE,
        control_freq=control_freq,
        max_episode_steps=task.max_episode_steps,
        scene_name=task.scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=overlay_path,
    )
    return env


def run_episode(model, env, task, ep, record_video, video_dir, model_name):
    """跑一个 episode，返回明细记录 dict。绝不因失败而丢弃该 episode。"""
    from simpler_env.utils.env.observation_utils import (
        get_image_from_maniskill2_obs_dict,
    )

    record = {
        "model": model_name,
        "task": task.key,
        "episode_index": ep["episode_index"],
        "episode_id": ep["episode_id"],
        "env_seed": ep["env_seed"],
        "model_seed": ep["model_seed"],
        "robot_init_x": ep["robot_init_x"],
        "robot_init_y": ep["robot_init_y"],
        "robot_init_quat": json.dumps(ep["robot_init_quat"]),
        "obj_episode_id": ep["obj_episode_id"],
        "success": False,
        "num_steps": 0,
        "max_episode_steps": task.max_episode_steps,
        "terminated_by_policy": False,
        "truncated": False,
        "exception": "",
        "wall_time_s": 0.0,
    }

    t0 = time.time()
    frames = []
    try:
        # 初始状态完全由这些参数决定，可在别处精确重建
        reset_options = {
            "robot_init_options": {
                "init_xy": np.array([ep["robot_init_x"], ep["robot_init_y"]]),
                "init_rot_quat": np.array(ep["robot_init_quat"]),
            },
            "obj_init_options": {"episode_id": ep["obj_episode_id"]},
        }
        obs, _ = env.reset(options=reset_options)

        instruction = task.instruction or env.get_language_instruction()
        image = get_image_from_maniskill2_obs_dict(
            env, obs, camera_name=OBS_CAMERA_NAME
        )
        if record_video:
            frames.append(image)

        model.reset(instruction)

        done, truncated = False, False
        step = 0
        info = {}
        while not (done or truncated):
            raw_action, action = model.step(image, instruction)
            # 策略自身的终止标记（仅用于诊断，不作为成功判据）
            if bool(action.get("terminate_episode", np.zeros(1))[0] > 0):
                record["terminated_by_policy"] = True

            env_action = np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            )
            obs, reward, done, truncated, info = env.step(env_action)

            image = get_image_from_maniskill2_obs_dict(
                env, obs, camera_name=OBS_CAMERA_NAME
            )
            if record_video:
                frames.append(image)

            step += 1
            if step >= task.max_episode_steps:
                truncated = True
                break

        info_dict = _unwrap_info(info)
        episode_stats = info_dict.get("episode_stats", {}) or {}

        # 环境原生成功判定：done=True 表示任务成功
        record["success"] = bool(done)
        record["num_steps"] = step
        record["truncated"] = bool(truncated)
        record["episode_stats"] = episode_stats

    except Exception as exc:  # 策略引发的异常按失败记录（作业第 6.2 节）
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["success"] = False
        traceback.print_exc()
    finally:
        record["wall_time_s"] = round(time.time() - t0, 2)
        if record_video and frames:
            try:
                import imageio

                os.makedirs(video_dir, exist_ok=True)
                tag = "success" if record["success"] else "failure"
                vp = os.path.join(
                    video_dir,
                    f"{model_name}_{task.key}_ep{ep['episode_index']:02d}_{tag}.mp4",
                )
                imageio.mimsave(vp, frames, fps=5)
                record["video"] = vp
            except Exception as exc:
                print(f"[warn] video write failed: {exc}")
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True, help="结果标签，如 Base / Final")
    ap.add_argument("--model-type", default="octo-base", choices=["octo-base", "octo-small"])
    ap.add_argument(
        "--ckpt-path",
        default="",
        help="微调 checkpoint 路径；留空 = 使用 HuggingFace 原始权重（Base）",
    )
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--eval-config", required=True, help="预先固定的初始状态清单 JSON")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--repo-root", default="/root/vla_project/SimplerEnv")
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("MS2_SIM_LOG_LEVEL", "error")
    os.environ.setdefault("MS2_RENDERER_LOG_LEVEL", "error")

    task = get_task(args.task)

    # ---- 读取冻结的初始状态清单 ----
    with open(args.eval_config, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if args.task not in manifest["tasks"]:
        print(f"[error] {args.task} 不在清单 {args.eval_config} 中")
        return 2
    spec = manifest["tasks"][args.task]
    episodes = spec["episodes"][: args.episodes]
    if len(episodes) < args.episodes:
        print(f"[warn] 清单中只有 {len(episodes)} 个 episode（请求 {args.episodes}）")

    print("=" * 70)
    print(f"model_name   : {args.model_name}")
    print(f"model_type   : {args.model_type}")
    print(f"ckpt_path    : {args.ckpt_path or '(pretrained, no finetune)'}")
    print(f"task         : {args.task}  ({task.env_name})")
    print(f"episodes     : {len(episodes)}")
    print(f"eval_config  : {args.eval_config}")
    print(f"output_dir   : {args.output_dir}")
    print(f"max_steps    : {task.max_episode_steps}   control_freq={task.control_freq}Hz")
    print("=" * 70)

    # ---- 加载策略 ----
    # 使用本仓库自带的适配层（scripts/octo_bridge_policy.py），而不是
    # SimplerEnv 的 OctoInference。原因见 docs/EVAL_FIX.md：
    #   · OctoInference 传 pad_mask（Octo 0.6.x 需要 timestep_pad_mask）→ KeyError
    #   · OctoInference 对夹爪也做反归一化 → 0(close) 变 0.588 → 永远判 open
    #   · 夹爪无法闭合 → 抓不住物体 → 成功率恒为 0
    # 自包含适配层不修改 SimplerEnv，从仓库重新部署即自动生效。
    from octo_bridge_policy import OctoBridgePolicy

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from octo.model.octo_model import OctoModel

    if args.ckpt_path:
        print(f"loading finetuned checkpoint: {args.ckpt_path}")
        loaded = OctoModel.load_pretrained(args.ckpt_path)
    else:
        hf_repo = f"hf://rail-berkeley/{args.model_type}"
        print(f"loading pretrained weights from HuggingFace: {hf_repo}")
        loaded = OctoModel.load_pretrained(hf_repo)

    model = OctoBridgePolicy(
        model=loaded,
        policy_setup="widowx_bridge",
        init_rng=int(manifest["meta"].get("model_seed", 0)),
    )

    # ---- 逐 episode 评测 ----
    video_dir = os.path.join(args.output_dir, "videos")
    records = []
    for ep in episodes:
        env = build_env(task, task.sim_freq, task.control_freq, args.repo_root)
        try:
            rec = run_episode(
                model, env, task, ep, args.record_video, video_dir, args.model_name
            )
        finally:
            try:
                env.close()
            except Exception:
                pass
        records.append(rec)
        status = "SUCCESS" if rec["success"] else "failure"
        extra = f"  EXC={rec['exception'][:60]}" if rec["exception"] else ""
        print(
            f"[{args.model_name}] {args.task} ep{rec['episode_index']:02d} "
            f"({rec['num_steps']} steps) -> {status}{extra}"
        )

    # ---- 写出 episodes.csv（追加，便于多次运行累积） ----
    ep_csv = os.path.join(args.output_dir, "episodes.csv")
    write_header = not os.path.exists(ep_csv)
    with open(ep_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EPISODES_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    # ---- 汇总 ----
    n = len(records)
    n_success = sum(1 for r in records if r["success"])
    n_exc = sum(1 for r in records if r["exception"])
    rate = 100.0 * n_success / n if n else 0.0
    summary = {
        "model": args.model_name,
        "task": args.task,
        "episodes": n,
        "success": n_success,
        "success_rate_pct": round(rate, 2),
        "exceptions": n_exc,
        "max_episode_steps": task.max_episode_steps,
        "control_freq_hz": task.control_freq,
        "eval_config": os.path.abspath(args.eval_config),
        "ckpt_path": args.ckpt_path or "(pretrained)",
        "model_seed": manifest["meta"].get("model_seed"),
        "env_seed": manifest["meta"].get("env_seed"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, f"summary_{args.task}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("")
    print("=" * 70)
    print(f"{args.model_name} / {args.task}: {n_success}/{n} = {rate:.1f}%  (exceptions={n_exc})")
    print(f"episodes.csv -> {ep_csv}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
