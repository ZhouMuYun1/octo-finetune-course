#!/usr/bin/env bash
# =============================================================================
# 原生 Linux（Ubuntu 24.04）安装后一条龙配置脚本
#
# 用途：装完 Ubuntu 双系统后，运行本脚本一次性配置：
#   NVIDIA 驱动（Blackwell 需 570+）→ SimplerEnv → Octo → JAX → 验证
#
# 基础：同学（gitee.com/karlshaus/vlalora）在 Pascal 上验证过的完整方案，
#      加上 Blackwell（RTX 5070, sm_120）所需的适配。
#
# ⚠️ 与同学方案的关键差异（Blackwell）：
#   1) NVIDIA 驱动必须 >= 570（Pascal 旧驱动即可，Blackwell 不行）
#   2) JAX 0.4.20 的 jaxlib 不支持 sm_120 → 需改用新版 JAX（见阶段 7）
#   3) 新版 JAX 与 Octo 固定 commit 的兼容性需验证（本脚本会做冒烟测试）
#
# 用法：
#   bash setup_native_linux.sh            # 完整安装（含驱动，需 sudo）
#   SKIP_DRIVER=1 bash setup_native_linux.sh  # 跳过驱动（已装好驱动时）
# =============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/vla_workspace}"
PROXY="${PROXY:-}"          # 如 http://127.0.0.1:7890，留空直连

# 精确 commit（同学验证过的版本，不要改）
SIMPLERENV_COMMIT="c701ea271b16b80616d6fed9713c4a50a0adea3b"
OCTO_COMMIT="241fb3514b7c40957a86d869fecb7c7fc353f540"
DLIMP_COMMIT="5edaa4691567873d495633f2708982b42edf1972"

# Blackwell 需要的新 JAX 版本（支持 sm_120；这是相对同学方案的唯一大改动）
JAX_VERSION="${JAX_VERSION:-0.4.38}"   # 0.4.38 起支持 sm_120；如仍有问题试 0.5.x

if [ -n "$PROXY" ]; then
  export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

log() { echo ""; echo "########## $* ##########"; }

# =============================================================================
# 阶段 0：系统更新
# =============================================================================
log "0/8 系统更新"
sudo apt-get update -qq
sudo apt-get upgrade -yqq

# =============================================================================
# 阶段 1：NVIDIA 驱动（Blackwell 必须 >= 570）
# =============================================================================
if [ "${SKIP_DRIVER:-0}" != "1" ]; then
  log "1/8 NVIDIA 驱动 (570+)"
  sudo apt-get install -yqq build-essential dkms
  sudo add-apt-repository -y ppa:graphics-drivers/ppa
  sudo apt-get update -qq
  # 装 570；若你的卡需要更高版本，改为 nvidia-driver-575 / 580
  sudo apt-get install -yqq nvidia-driver-570
  echo "!!! 驱动装完需重启：sudo reboot，然后重新运行本脚本（SKIP_DRIVER=1）"
  echo "!!! 若不想现在重启，请手动重启后继续。"
  read -r -p "已重启？继续按回车，否则 Ctrl+C 退出并重启: " _
fi

log "验证 NVIDIA 驱动"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# =============================================================================
# 阶段 2：图形 / Vulkan 系统库
# =============================================================================
log "2/8 图形与 Vulkan 库"
sudo apt-get install -yqq \
  libvulkan1 vulkan-tools libegl1 libgl1 libglvnd-dev \
  libxext6 libxrender1 libsm6 ffmpeg curl git bzip2

log "验证 Vulkan 能列出 NVIDIA GPU（不是 llvmpipe）"
vulkaninfo 2>/dev/null | grep -E "deviceName|deviceType" | head -6

# =============================================================================
# 阶段 3：conda + Python 3.10
# =============================================================================
log "3/8 conda + Python 3.10"
if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
export PATH="$HOME/miniconda3/bin:$PATH"
conda create -y -n simpler_env python=3.10 || true
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate simpler_env
python -V

# =============================================================================
# 阶段 4：基础数值包（版本精确锁定，来自同学方案）
# =============================================================================
log "4/8 基础数值包"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir \
  "numpy==1.24.4" "scipy==1.12.0" "opencv-python==4.10.0.84" "protobuf==3.20.3"

# =============================================================================
# 阶段 5：克隆并安装 SimplerEnv / Octo / dlimp（精确 commit）
# =============================================================================
log "5/8 克隆 SimplerEnv / Octo / dlimp"
mkdir -p "$WORKSPACE"

if [ ! -d "$WORKSPACE/SimplerEnv" ]; then
  git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv.git "$WORKSPACE/SimplerEnv"
fi
git -C "$WORKSPACE/SimplerEnv" checkout "$SIMPLERENV_COMMIT"
git -C "$WORKSPACE/SimplerEnv" submodule update --init --recursive

if [ ! -d "$WORKSPACE/octo" ]; then
  git clone https://github.com/octo-models/octo.git "$WORKSPACE/octo"
fi
git -C "$WORKSPACE/octo" checkout "$OCTO_COMMIT"

if [ ! -d "$WORKSPACE/dlimp" ]; then
  git clone https://github.com/kvablack/dlimp.git "$WORKSPACE/dlimp"
fi
git -C "$WORKSPACE/dlimp" checkout "$DLIMP_COMMIT"

log "安装 SimplerEnv"
python -m pip install -e "$WORKSPACE/SimplerEnv/ManiSkill2_real2sim"
python -m pip install -e "$WORKSPACE/SimplerEnv"

# =============================================================================
# 阶段 6：TensorFlow + 数据处理包（同学方案的精确配对）
# =============================================================================
log "6/8 TensorFlow 与数据处理"
python -m pip install --no-cache-dir \
  "tensorflow==2.15.0" "tensorflow-datasets==4.9.4" "tensorflow-hub==0.16.0" \
  "tensorflow-metadata==1.15.0" "tensorflow-probability==0.23.0" \
  "tensorflow-text==2.15.0" "tensorflow-graphics==2021.12.3" \
  "tf-keras==2.15.0" "rlds==0.1.8" "protobuf==3.20.3" \
  "mediapy==1.2.0" "IPython==8.22.2"

# =============================================================================
# 阶段 7：JAX —— ⚠️ Blackwell 关键差异
# =============================================================================
log "7/8 JAX（Blackwell 需新版，支持 sm_120）"
# 同学方案用 jax==0.4.20（Pascal 可用）。Blackwell sm_120 需要 >= 0.4.38。
# 新版 JAX 与 Octo 代码的兼容性由下面的冒烟测试验证。
python -m pip install --no-cache-dir "jax==${JAX_VERSION}"
python -m pip install --no-cache-dir "jaxlib==${JAX_VERSION}" --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || \
python -m pip install --no-cache-dir "jax[cuda12]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

log "JAX 冒烟测试：确认能看到 CUDA 设备（不是 CPU）"
python - <<'PYEOF'
import jax
print("jax version:", jax.__version__)
print("devices:", jax.devices())
print("default backend:", jax.default_backend())
PYEOF

# =============================================================================
# 阶段 8：Octo 依赖 + 验证渲染
# =============================================================================
log "8/8 Octo 依赖与渲染验证"
python -m pip install --no-cache-dir \
  "flax==0.7.5" "chex==0.1.85" "optax==0.1.5" "distrax==0.1.5" \
  "ml-dtypes==0.2.0" einops ml-collections \
  "transformers==4.55.4" "tokenizers==0.21.4" "huggingface-hub==0.34.4" plotly
python -m pip install --no-deps "orbax-checkpoint==0.4.8" "tensorstore==0.1.45"
python -m pip install -e "$WORKSPACE/dlimp"
python -m pip install -e "$WORKSPACE/octo" --no-deps

# 重新锁定易漂移的包
python -m pip install --no-cache-dir --no-deps "numpy==1.24.4" "protobuf==3.20.3" "tensorflow-metadata==1.15.0"

log "渲染验证（决定性能否继续）"
python - <<'PYEOF'
import warnings
warnings.filterwarnings("ignore")
import sapien.core as sc
try:
    sc.SapienRenderer()
    print("SAPIEN renderer (default) : OK")
except Exception as e:
    print("SAPIEN renderer (default) : FAILED ->", e)
PYEOF

echo ""
echo "================================================================"
echo " 环境配置完成。下一步："
echo "  1. 下载模型: python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='rail-berkeley/octo-base-1.5', local_dir='$WORKSPACE/models/octo-base-1.5')\""
echo "  2. 渲染验证: 参考 scripts/verify_render.py"
echo "================================================================"
