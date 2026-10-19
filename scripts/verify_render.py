#!/usr/bin/env python
"""
渲染验证脚本 —— 本项目的"生死关卡"检查。

作用：确认 SAPIEN 能在当前机器上真正完成 GPU 渲染，并能创建 SimplerEnv 的
      WidowX 环境。这一步如果失败，后续采集数据 / 训练 / 评测都无法进行。

背景（实测结论）：
  · SAPIEN 2.2.2 只支持 Vulkan GPU 渲染，没有 CPU 渲染后端。
  · WSL2 的 Vulkan 通常只暴露 llvmpipe（CPU 软件渲染），且缺少 dzn 驱动，
    因此 SAPIEN 会报 `vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent`。
  · 所以本脚本在 WSL2 上预期会失败，在真实 Linux GPU 服务器上预期会通过。

用法：
    python verify_render.py
    python verify_render.py --task widowx_put_eggplant_in_basket
退出码：0 = 渲染可用；1 = 渲染不可用（含原因）
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")


def section(title: str) -> None:
    print("")
    print("=" * 62)
    print(title)
    print("=" * 62)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="widowx_spoon_on_towel")
    ap.add_argument("--save-image", default="")
    args = ap.parse_args()

    section("1. environment")
    print("DISPLAY =", repr(os.environ.get("DISPLAY")))
    print("python  =", sys.version.split()[0])

    section("2. import sapien and report available render devices")
    try:
        import sapien  # noqa: F401
        import sapien.core as sapien_core

        print("sapien version:", sapien.__version__)
    except Exception as exc:  # pragma: no cover
        print("FAILED to import sapien:", repr(exc))
        traceback.print_exc()
        return 1

    # 探测：默认参数（走 Vulkan 自动选设备）
    try:
        sapien_core.SapienRenderer()
        print("SapienRenderer (default)      : OK")
    except Exception as exc:
        print("SapienRenderer (default)      : FAILED ->", exc)

    # 探测：显式指定 CUDA 设备
    try:
        sapien_core.SapienRenderer(device="cuda:0")
        print("SapienRenderer (device=cuda:0): OK")
    except Exception as exc:
        print("SapienRenderer (device=cuda:0): FAILED ->", exc)

    section("3. Vulkan physical devices (informational)")
    os.system(
        "vulkaninfo 2>/dev/null | grep -E 'deviceName|deviceType' | head -8 || "
        "echo 'vulkaninfo not available'"
    )

    section(f"4. create SimplerEnv environment: {args.task}")
    try:
        import simpler_env
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        print("registered environments:", len(simpler_env.ENVIRONMENTS))
        env = simpler_env.make(args.task)
        print("env created OK")
    except Exception as exc:
        print("FAILED:", type(exc).__name__, exc)
        traceback.print_exc()
        print("")
        print(">>> RENDER VERIFICATION FAILED — this machine cannot run SimplerEnv.")
        return 1

    section("5. reset + render an observation")
    try:
        obs, reset_info = env.reset()
        print("instruction:", env.get_language_instruction())
        image = get_image_from_maniskill2_obs_dict(env, obs)
        print("rendered image shape:", image.shape, "dtype:", image.dtype)
        print("pixel mean:", float(image.mean()))
        if args.save_image:
            import imageio

            imageio.imwrite(args.save_image, image)
            print("saved ->", args.save_image)
    except Exception as exc:
        print("FAILED:", type(exc).__name__, exc)
        traceback.print_exc()
        return 1

    section("6. step once with a random action (closed-loop smoke test)")
    try:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print("action shape:", action.shape)
        print("reward:", reward, "done:", done, "truncated:", truncated)
        print("episode_stats:", info.get("episode_stats"))
    except Exception as exc:
        print("FAILED:", type(exc).__name__, exc)
        traceback.print_exc()
        return 1
    finally:
        try:
            env.close()
        except Exception:
            pass

    section("RESULT: render verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
