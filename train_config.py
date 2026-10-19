"""自定义 finetune 配置：BridgeData V2 + 放置类轨迹筛选。

用法（由 train.sh 调用）：
    cfg = get_config("head_only,language_conditioned")

路径可通过环境变量覆盖：
    VLA_WORKSPACE  工程根目录（默认 ~/vla_workspace）
    OCTO_DIR       Octo 代码目录（默认 $VLA_WORKSPACE/octo）
    BRIDGE_DATA    BridgeData V2 数据目录（默认 $VLA_WORKSPACE/bridge_data）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- 路径解析（优先环境变量，其次默认值）----
WORKSPACE = Path(os.environ.get("VLA_WORKSPACE", Path.home() / "vla_workspace"))
OCTO_DIR = Path(os.environ.get("OCTO_DIR", WORKSPACE / "octo"))
DATA_DIR = Path(os.environ.get("BRIDGE_DATA", WORKSPACE / "bridge_data"))

# Octo 的 finetune_config 位于 octo/scripts/configs/
sys.path.insert(0, str(OCTO_DIR / "scripts"))

from configs.finetune_config import get_config as _base_get_config  # noqa: E402
from octo.utils.spec import ModuleSpec  # noqa: E402


def get_config(config_string="full,language_conditioned"):
    """在官方 finetune 配置基础上，替换数据目录并加入放置类筛选。"""
    cfg = _base_get_config(config_string)

    # 1) 数据目录
    cfg.dataset_kwargs.data_dir = str(DATA_DIR)

    # 2) 只保留"放置类"轨迹（put / place / move / pick / take / slide / ...）
    cfg.dataset_kwargs.filter_functions = (
        ModuleSpec.create(
            "octo.data.oxe.placement_filter:is_placement_task",
        ),
    )
    return cfg


if __name__ == "__main__":
    # 自检：打印关键配置
    for mode in ("head_only", "full", "head_mlp_only"):
        cfg = get_config(f"{mode},language_conditioned")
        print(f"[{mode}]")
        print(f"  finetuning_mode : {cfg.finetuning_mode}")
        print(f"  data_dir        : {cfg.dataset_kwargs.data_dir}")
        print(f"  filter          : {cfg.dataset_kwargs.filter_functions}")
