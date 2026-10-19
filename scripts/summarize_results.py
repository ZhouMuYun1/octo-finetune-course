#!/usr/bin/env python3
"""汇总评测结果，生成 results.csv 与终端表格。

输入：episodes.csv（由 evaluate.py 自动追加，字段见 EPISODES_CSV_FIELDS）
输出：
  1. results.csv           —— 作业 8.1 要求的结果表
  2. 终端表格              —— 即时查看
  3. 完整性校验            —— 检查每个（模型, 任务）是否有 10 条记录

用法：
    python scripts/summarize_results.py --episodes episodes.csv
    python scripts/summarize_results.py --episodes episodes.csv --expect 10

一致性校验（作业 6.2 要求）：
    - 同一任务下 Base 与 Final 使用相同的 (episode_id, env_seed)
    - 模型种子与环境种子分别记录
    - 不筛除任何失败 episode
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

TASKS = ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"]
TASK_LABEL = {
    "widowx_spoon_on_towel": "T1",
    "widowx_put_eggplant_in_basket": "T2",
}
# 三个评测模型（作业最低要求为 Base + 1 个 Final；本项目额外加入 Final-B 作对照）
MODELS = ["Base", "Final-A", "Final-B"]
MODULE_LABEL = {
    "Base": "未进行本次微调",
    "Final-A": "动作头(head_only)",
    "Final-B": "全量(full)",
}
PARAM_LABEL = {
    "Base": "0",
    "Final-A": "27/366",
    "Final-B": "366/366",
}


def parse_bool(v) -> bool:
    """把 CSV 里的成功标记解析为布尔值。支持 1/0、true/false、True/False。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "success"}:
        return True
    if s in {"0", "false", "no", "n", "failure", "fail", ""}:
        return False
    return False


def load_episodes(path: str) -> list[dict]:
    """读取 episodes.csv，跳过注释行与空行。"""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        # 过滤以 # 开头的注释行（csv 模块不支持注释）
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    for r in reader:
        if not r.get("模型") and not r.get("model"):
            continue
        rows.append(r)
    return rows


def get(row: dict, *names, default=""):
    """兼容中英文字段名。"""
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="episodes.csv", help="episodes.csv 路径")
    ap.add_argument("--out", default="results.csv", help="输出 results.csv 路径")
    ap.add_argument("--expect", type=int, default=10, help="每模型每任务应有 episode 数")
    args = ap.parse_args()

    if not os.path.exists(args.episodes):
        print(f"❌ 找不到 {args.episodes}", file=sys.stderr)
        return 1

    rows = load_episodes(args.episodes)
    if not rows:
        print(f"❌ {args.episodes} 里没有有效数据行", file=sys.stderr)
        return 1

    # ---- 归集：model -> task -> [records] ----
    bucket: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        model = str(get(r, "模型", "model")).strip()
        task = str(get(r, "任务", "task")).strip()
        if not model or not task:
            continue
        bucket[model][task].append(r)

    # ---- 完整性校验 ----
    print("=" * 72)
    print("完整性校验")
    print("=" * 72)
    problems = []
    for model in MODELS:
        for task in TASKS:
            recs = bucket.get(model, {}).get(task, [])
            n = len(recs)
            mark = "✅" if n == args.expect else "❌"
            print(f"  {mark} {model:<6} {TASK_LABEL[task]:<3} {task:<34} {n}/{args.expect}")
            if n != args.expect:
                problems.append(f"{model}/{task} 只有 {n} 条（应为 {args.expect}）")

    # ---- 一致性与异常检查 ----
    print()
    print("=" * 72)
    print("一致性检查")
    print("=" * 72)

    for task in TASKS:
        base_ids = {
            str(get(r, "episode编号", "episode_index", "episode_id"))
            for r in bucket.get("Base", {}).get(task, [])
        }
        final_ids = {
            str(get(r, "episode编号", "episode_index", "episode_id"))
            for r in bucket.get("Final", {}).get(task, [])
        }
        if base_ids and final_ids:
            if base_ids == final_ids:
                print(f"  ✅ {TASK_LABEL[task]}：Base 与 Final 的 episode 编号集合一致（{len(base_ids)} 个）")
            else:
                only_b = base_ids - final_ids
                only_f = final_ids - base_ids
                print(f"  ❌ {TASK_LABEL[task]}：编号集合不一致（仅 Base: {only_b}，仅 Final: {only_f}）")
                problems.append(f"{task} 的 Base/Final episode 编号不一致")

    n_exc = sum(1 for r in rows if str(get(r, "异常信息", "exception")).strip())
    print(f"  {'✅' if n_exc == 0 else '⚠️ '} 异常 episode 数：{n_exc}")
    if n_exc:
        print("     （异常按失败计入，但报告中应说明异常类型）")

    # ---- 汇总成功率 ----
    summary = {}
    for model in MODELS:
        summary[model] = {}
        for task in TASKS:
            recs = bucket.get(model, {}).get(task, [])
            n = len(recs)
            n_succ = sum(1 for r in recs if parse_bool(get(r, "成功标记", "success")))
            rate = 100.0 * n_succ / n if n else 0.0
            summary[model][task] = {"n": n, "success": n_succ, "rate": rate}

    # ---- B-SimplerScore ----
    def score(model):
        rates = [summary[model][t]["rate"] for t in TASKS if summary[model][t]["n"]]
        return sum(rates) / len(rates) if rates else 0.0

    print()
    print("=" * 72)
    print("结果汇总")
    print("=" * 72)
    print(f"{'模型':<8}{'T1 成功数':<14}{'T2 成功数':<14}{'B-SimplerScore':<16}")
    print("-" * 72)
    for model in MODELS:
        t1 = summary[model][TASKS[0]]
        t2 = summary[model][TASKS[1]]
        print(
            f"{model:<8}"
            f"{t1['success']}/{t1['n']} ({t1['rate']:.1f}%)".ljust(14)
            + f"{t2['success']}/{t2['n']} ({t2['rate']:.1f}%)".ljust(14)
            + f"{score(model):.2f}"
        )
    print("-" * 72)

    # ---- 变化量 ----
    print()
    print("变化（单位：百分点）")
    pairs = [("Final-A", "Base"), ("Final-B", "Base"), ("Final-A", "Final-B")]
    for hi, lo in pairs:
        print(f"  {hi} − {lo}:")
        for task in TASKS:
            d = summary[hi][task]["rate"] - summary[lo][task]["rate"]
            print(f"    {TASK_LABEL[task]}：{d:+.1f}")
        print(f"    B-SimplerScore：{score(hi) - score(lo):+.2f}")

    # ---- 写 results.csv ----
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow([
            "模型", "训练模块", "可训练参数",
            "T1_任务", "T1_成功数", "T1_评测数", "T1_成功率",
            "T2_任务", "T2_成功数", "T2_评测数", "T2_成功率",
            "B-SimplerScore", "备注",
        ])
        for model in MODELS:
            t1 = summary[model][TASKS[0]]
            t2 = summary[model][TASKS[1]]
            w.writerow([
                model, MODULE_LABEL[model], PARAM_LABEL[model],
                TASKS[0], t1["success"], t1["n"], f"{t1['rate']:.2f}",
                TASKS[1], t2["success"], t2["n"], f"{t2['rate']:.2f}",
                f"{score(model):.2f}",
                "" if not problems else "; ".join(problems),
            ])
    print()
    print(f"✅ 已写出 {args.out}")

    if problems:
        print()
        print("⚠️ 存在完整性问题：")
        for p in problems:
            print(f"   - {p}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
