#!/usr/bin/env bash
# =============================================================================
# 打包 Final checkpoint（用于上传到网盘 / 对象存储）
#
# 只打包 step 750 的权重 + 4 个配套文件，压缩后约 300 MB。
# dataset_statistics.json 必须包含（评测反归一化要用）。
#
# 用法：
#   bash scripts/package_checkpoint.sh              # 打包两个模型
#   bash scripts/package_checkpoint.sh head_only    # 只打包 head_only
#   bash scripts/package_checkpoint.sh full         # 只打包 full
# =============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/vla_workspace}"
CKPT_ROOT="${CKPT_ROOT:-$WORKSPACE/checkpoints/octo_finetune}"
OUT_DIR="${OUT_DIR:-$WORKSPACE/checkpoint_release}"
STEP="${STEP:-750}"

# 训练产出目录名（按需修改）
HEAD_DIR="${HEAD_DIR:-octo_head_only_losslog_20261019_112942}"
FULL_DIR="${FULL_DIR:-octo_full_losslog_20261019_113719}"

WHICH="${1:-all}"
mkdir -p "$OUT_DIR"

package_one() {
  local src_name="$1"     # 训练目录名
  local out_name="$2"     # 输出包名

  local src="$CKPT_ROOT/$src_name"
  if [ ! -d "$src" ]; then
    echo "❌ 找不到训练目录: $src"
    return 1
  fi
  if [ ! -d "$src/$STEP/default" ]; then
    echo "❌ 找不到 step $STEP 的权重: $src/$STEP/default"
    return 1
  fi

  local tmp="$OUT_DIR/$out_name"
  rm -rf "$tmp"; mkdir -p "$tmp"

  cp -r "$src/$STEP" "$tmp/"
  for f in config.json dataset_statistics.json example_batch.msgpack finetune_config.json; do
    if [ -f "$src/$f" ]; then
      cp "$src/$f" "$tmp/"
    else
      echo "⚠️  缺少 $f"
    fi
  done

  tar czf "$OUT_DIR/$out_name.tar.gz" -C "$OUT_DIR" "$out_name"
  echo "✅ $OUT_DIR/$out_name.tar.gz"
  du -h "$OUT_DIR/$out_name.tar.gz"
}

if [ "$WHICH" = "all" ] || [ "$WHICH" = "head_only" ]; then
  package_one "$HEAD_DIR" "octo_head_only_step${STEP}"
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "full" ]; then
  package_one "$FULL_DIR" "octo_full_step${STEP}"
fi

echo ""
echo "=== 产出 ==="
ls -lh "$OUT_DIR"/*.tar.gz 2>/dev/null
echo ""
echo "上传后请把链接填入 vla_finetune_repo/checkpoint_link.txt 第 2 节。"
