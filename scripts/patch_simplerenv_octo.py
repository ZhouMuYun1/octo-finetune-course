#!/usr/bin/env python3
"""修复 SimplerEnv 的 OctoInference 与 Octo 0.6.x 的两处不兼容。

Bug 1：`pad_mask` 键名错误
    OctoInference 传给 model.sample_actions 的是 {"image_primary": ..., "pad_mask": ...}
    但 Octo 0.6.x 的 sample_actions 读的是 observations["timestep_pad_mask"]
    → 运行时报 KeyError: 'timestep_pad_mask'

Bug 2：夹爪被错误地反归一化（这是"所有 episode 都失败"的根因）
    SimplerEnv 对所有 7 维动作统一做反归一化：
        raw_actions = norm_actions * action_std + action_mean
    但 Octo 的 action_normalization_mask = [T,T,T,T,T,T,False]
    第 7 维（夹爪）在训练时**不做归一化**，模型输出的是原始值 0/1。
    再对它做反归一化：
        模型输出 0（close）→ 0*0.48772 + 0.58795 = 0.588 → >0.5 → 判为 open ❌
    结果：夹爪永远无法闭合，任何任务都无法抓取物体，成功率恒为 0。

修复方式：
    - Bug 1：键名改为 timestep_pad_mask，dtype 改为 bool（与 example_batch 一致）
    - Bug 2：只对前 6 维反归一化，第 7 维保持原值

用法：
    # 1. 先备份（脚本会自动备份为 .bak）
    python scripts/patch_simplerenv_octo.py --check     # 只检查，不修改
    python scripts/patch_simplerenv_octo.py             # 应用修复

    # 2. 回滚
    python scripts/patch_simplerenv_octo.py --revert

    # 3. 验证
    python scripts/patch_simplerenv_octo.py --verify
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

DEFAULT_PATH = os.environ.get(
    "SIMPLERENV_OCTO_POLICY",
    os.path.expanduser("~/vla_project/SimplerEnv/simpler_env/policies/octo/octo_model.py"),
)

# ---- Bug 1：pad_mask -> timestep_pad_mask ----
OLD_PAD = '''        input_observation = {"image_primary": images, "pad_mask": pad_mask}'''
NEW_PAD = '''        # [PATCH] Octo 0.6.x 读的是 timestep_pad_mask，且要求 bool 类型
        input_observation = {
            "image_primary": images,
            "timestep_pad_mask": pad_mask.astype(bool),
        }'''

# ---- Bug 2：夹爪不反归一化 ----
OLD_UNNORM = '''        raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]'''
NEW_UNNORM = '''        # [PATCH] 只反归一化前 6 维；第 7 维（夹爪）训练时 mask=False，模型输出已是原始值 0/1
        _n = norm_raw_actions.shape[-1]
        _w = 6 if _n == 7 else _n
        raw_actions = norm_raw_actions.copy()
        raw_actions[..., :_w] = (
            norm_raw_actions[..., :_w] * self.action_std[None, :_w]
            + self.action_mean[None, :_w]
        )'''


def check(path: str) -> int:
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        print("   请用 --path 指定，或设置环境变量 SIMPLERENV_OCTO_POLICY")
        return 1

    src = open(path, encoding="utf-8").read()
    has_pad_bug = OLD_PAD in src
    has_unnorm_bug = OLD_UNNORM in src
    has_pad_fix = "timestep_pad_mask" in src
    has_unnorm_fix = "[PATCH] 只反归一化前 6 维" in src

    print(f"文件: {path}")
    print()
    print("Bug 1 (pad_mask 键名):")
    print(f"  原始 bug 存在: {'是' if has_pad_bug else '否'}")
    print(f"  已修复:       {'是' if has_pad_fix else '否'}")
    print()
    print("Bug 2 (夹爪反归一化):")
    print(f"  原始 bug 存在: {'是' if has_unnorm_bug else '否'}")
    print(f"  已修复:       {'是' if has_unnorm_fix else '否'}")
    print()

    if has_pad_bug or has_unnorm_bug:
        print("⚠️  仍有未修复的 bug")
        return 2
    if has_pad_fix and has_unnorm_fix:
        print("✅ 两处均已修复")
        return 0
    print("⚠️  状态不明确（可能被手工改过），请人工检查")
    return 3


def apply(path: str) -> int:
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return 1

    src = open(path, encoding="utf-8").read()
    orig = src

    n_fix = 0
    if OLD_PAD in src:
        src = src.replace(OLD_PAD, NEW_PAD)
        print("✅ 已修复 Bug 1（pad_mask → timestep_pad_mask）")
        n_fix += 1
    elif "timestep_pad_mask" in src:
        print("ℹ️  Bug 1 已是修复状态，跳过")
    else:
        print("⚠️  未找到 Bug 1 的目标代码，请人工检查")

    if OLD_UNNORM in src:
        src = src.replace(OLD_UNNORM, NEW_UNNORM)
        print("✅ 已修复 Bug 2（夹爪不再反归一化）")
        n_fix += 1
    elif "[PATCH] 只反归一化前 6 维" in src:
        print("ℹ️  Bug 2 已是修复状态，跳过")
    else:
        print("⚠️  未找到 Bug 2 的目标代码，请人工检查")

    if src != orig:
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy(path, bak)
            print(f"已备份原文件到 {bak}")
        open(path, "w", encoding="utf-8").write(src)
        print(f"\n✅ 已写入修复（共 {n_fix} 处）")

    return 0


def revert(path: str) -> int:
    bak = path + ".bak"
    if not os.path.exists(bak):
        print(f"❌ 找不到备份: {bak}")
        return 1
    shutil.copy(bak, path)
    print(f"✅ 已从 {bak} 恢复")
    return 0


def verify(path: str) -> int:
    """静态验证：检查修复后代码的关键点。"""
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return 1
    src = open(path, encoding="utf-8").read()

    ok = True
    if "timestep_pad_mask" not in src:
        print("❌ 未找到 timestep_pad_mask")
        ok = False
    else:
        print("✅ 使用 timestep_pad_mask")

    if "_w = 6 if _n == 7 else _n" in src:
        print("✅ 夹爪维度不反归一化")
    else:
        print("❌ 未找到夹爪修复代码")
        ok = False

    # 语法检查
    try:
        compile(src, path, "exec")
        print("✅ 语法检查通过")
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        ok = False

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="修复 SimplerEnv OctoInference 的两处 bug")
    ap.add_argument("--path", default=DEFAULT_PATH, help="octo_model.py 路径")
    ap.add_argument("--check", action="store_true", help="只检查，不修改")
    ap.add_argument("--revert", action="store_true", help="从 .bak 回滚")
    ap.add_argument("--verify", action="store_true", help="验证修复结果")
    args = ap.parse_args()

    if args.check:
        return check(args.path)
    if args.revert:
        return revert(args.path)
    if args.verify:
        return verify(args.path)
    return apply(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
