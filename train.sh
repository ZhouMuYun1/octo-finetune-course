#!/usr/bin/env bash
# =============================================================================
# Octo-Small-1.5 在 BridgeData V2 上的轻量微调
#
# 本项目实际使用的命令：
#   MODE=head_only NUM_STEPS=1000 BATCH_SIZE=8 bash train.sh   # 主方案
#   MODE=full      NUM_STEPS=1000 BATCH_SIZE=8 bash train.sh   # 对照
#   （Final checkpoint 取 step 750，原因见 docs/TRAINING_RECORD.md 第 5 节）
#
# 可用环境变量覆盖：
#   MODE             full / head_only / head_mlp_only   （默认 full）
#   NUM_STEPS        训练步数                            （默认 1000）
#   BATCH_SIZE       batch size                          （默认 8）
#   WARMUP_STEPS     学习率 warmup 步数                  （默认 NUM_STEPS/10，下限 10）
#   WORKSPACE        工程根目录                          （默认 ~/vla_workspace）
#   PRETRAINED       基础模型目录                        （默认 $WORKSPACE/models/octo-small-1.5）
#   PRETRAINED_STEP  基础模型 step                       （默认 300000）
#   SAVE_DIR         checkpoint 输出目录                 （默认 $WORKSPACE/checkpoints）
#   EXP_NAME         实验名（影响日志与输出子目录）      （默认 octo_finetune）
# =============================================================================
set -euo pipefail

# ---------- 可调参数 ----------
NUM_STEPS="${NUM_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MODE="${MODE:-full}"                       # full / head_only / head_mlp_only
WARMUP_STEPS="${WARMUP_STEPS:-$(( NUM_STEPS / 10 > 10 ? NUM_STEPS / 10 : 10 ))}"
WORKSPACE="${WORKSPACE:-$HOME/vla_workspace}"
PRETRAINED="${PRETRAINED:-$WORKSPACE/models/octo-small-1.5}"
PRETRAINED_STEP="${PRETRAINED_STEP:-300000}"
SAVE_DIR="${SAVE_DIR:-$WORKSPACE/checkpoints}"
EXP_NAME="${EXP_NAME:-octo_finetune}"
CONFIG="${CONFIG:-$WORKSPACE/train_config.py}"
# --------------------------------

# wandb_stub 必须放在 PYTHONPATH 最前面（避免 wandb 与 TF 的 protobuf 冲突）
export PYTHONPATH="$WORKSPACE/wandb_stub:$WORKSPACE/dlimp:$WORKSPACE/octo:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2

cd "$WORKSPACE"
mkdir -p "$SAVE_DIR"

# Octo 代码目录（可通过 OCTO_DIR 覆盖）
OCTO_DIR="${OCTO_DIR:-$WORKSPACE/octo}"
PYTHON="${PYTHON:-python}"

echo "======================================================="
echo " Octo-Small-1.5 微调"
echo "   mode        : $MODE"
echo "   num_steps   : $NUM_STEPS"
echo "   batch_size  : $BATCH_SIZE"
echo "   warmup      : $WARMUP_STEPS"
echo "   pretrained  : $PRETRAINED (step=$PRETRAINED_STEP)"
echo "   config      : $CONFIG"
echo "   save_dir    : $SAVE_DIR"
echo "======================================================="

"$PYTHON" "$OCTO_DIR/scripts/finetune.py" \
  --name="$EXP_NAME" \
  --config="${CONFIG}:${MODE},language_conditioned" \
  --config.pretrained_path="$PRETRAINED" \
  --config.pretrained_step="$PRETRAINED_STEP" \
  --config.batch_size="$BATCH_SIZE" \
  --config.num_steps="$NUM_STEPS" \
  --config.save_interval=250 \
  --config.eval_interval=5000 \
  --config.log_interval=50 \
  --config.save_dir="$SAVE_DIR" \
  --config.optimizer.learning_rate.warmup_steps="$WARMUP_STEPS" \
  --debug \
  2>&1 | tee "$WORKSPACE/train_log_${EXP_NAME}.txt"

# =============================================================================
# 参数说明
# =============================================================================
# --config.eval_interval=5000
#     大于训练步数，从而跳过验证回调。原因：验证回调在 12 GB 内存的 WSL2 上
#     会触发 OOM；且本项目用"固定训练步数"规则选 checkpoint，不依赖验证集。
#
# --debug
#     关闭 wandb 联网。本项目用 wandb stub 替代（见 docs/REPRODUCTION.md 1.3）。
#
# --config.optimizer.learning_rate.warmup_steps
#     Octo 默认 2000，但本项目只训练 1000 步，必须缩小，否则
#     decay_steps = num_steps - warmup_steps 会变成负数而报错。
#
# --config.log_interval=50
#     每 50 步在 stdout 打印一行 [TRAIN] step=... loss=... （需配合 finetune.py 的
#     补丁，见 docs/REPRODUCTION.md 1.3 的"修改 3"）。
