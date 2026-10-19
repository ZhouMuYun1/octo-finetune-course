#!/usr/bin/env bash
# =============================================================================
# SimplerEnv 环境安装脚本（Linux / 服务器 / WSL2-Linux）
#
# 用途：在 Linux GPU 服务器上一键装好 SimplerEnv + WidowX 评测环境。
#
# 重要：本脚本固化了一次真实安装过程中踩到的全部坑，请勿随意改动版本号：
#   1) numpy 必须 <2（pyproject/README 明确要求；pinocchio 的 IK 在 numpy 2.x 下报错）
#   2) setuptools 必须 <81，否则 sapien 2.2.2 依赖的 pkg_resources 缺失
#   3) opencv-python 必须 <=4.8.x（5.x 要求 numpy>=2，会与第 1 条冲突）
#   4) 必须带 --recurse-submodules 克隆，否则缺 real_inpainting/ 贴图，视觉匹配无法工作
#   5) 需要 Vulkan GPU 驱动；WSL2 默认只有 llvmpipe（CPU 软件渲染），SAPIEN 无法使用
#
# 用法：
#   bash setup_simplerenv.sh              # 完整安装
#   SKIP_APT=1 bash setup_simplerenv.sh   # 跳过系统包（已有权限受限时）
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 可配置项
# ---------------------------------------------------------------------------
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-simpler_env}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/vla_project}"
SIMPLER_REPO="${SIMPLER_REPO:-https://github.com/simpler-env/SimplerEnv.git}"
PROXY="${PROXY:-}"          # 例如 http://127.0.0.1:7890 ；留空则直连

PY="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"
PIP="$CONDA_ROOT/envs/$CONDA_ENV/bin/pip"
SIMPLER_DIR="$PROJECT_ROOT/SimplerEnv"

# ---------------------------------------------------------------------------
# 代理（若提供）
# ---------------------------------------------------------------------------
if [ -n "$PROXY" ]; then
  export http_proxy="$PROXY" https_proxy="$PROXY"
  export HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
  echo "[env] using proxy: $PROXY"
fi

log() { echo ""; echo "=========== $* ==========="; }

# ---------------------------------------------------------------------------
# 1. 系统依赖（SAPIEN 渲染 + 视频导出所需）
# ---------------------------------------------------------------------------
if [ "${SKIP_APT:-0}" != "1" ]; then
  log "1/7 system packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  # 注意：Ubuntu 24.04 起不再有 libegl1-mesa，改用 libegl1 / libegl-mesa0
  apt-get install -yqq \
    build-essential git curl wget bzip2 ca-certificates \
    libegl1 libegl-mesa0 libgl1 libglvnd-dev libosmesa6-dev \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    ffmpeg libsm6 libxext6 libxrender1
else
  log "1/7 system packages (SKIPPED)"
fi

log "Vulkan device check (SAPIEN requires a GPU Vulkan device)"
if command -v vulkaninfo >/dev/null 2>&1; then
  GPU_DEVS=$(vulkaninfo 2>/dev/null | grep -c "PHYSICAL_DEVICE_TYPE_GPU" || true)
  echo "GPU Vulkan device count: ${GPU_DEVS:-0}"
  vulkaninfo 2>/dev/null | grep -E "deviceName|deviceType" | head -6 || true
  if [ "${GPU_DEVS:-0}" -eq 0 ]; then
    echo "!!! WARNING: no GPU Vulkan device found."
    echo "!!! On a real GPU server this usually means the driver/Vulkan ICD is missing."
    echo "!!! On WSL2 this is expected and SAPIEN will NOT work — use a Linux server."
  fi
else
  echo "vulkaninfo not available; skipping device check"
fi

# ---------------------------------------------------------------------------
# 2. conda 环境（Python 3.10）
# ---------------------------------------------------------------------------
log "2/7 conda env: $CONDA_ENV (python 3.10)"
if [ ! -x "$PY" ]; then
  "$CONDA_ROOT/bin/conda" create -y -n "$CONDA_ENV" python=3.10
else
  echo "already exists: $PY"
fi
"$PY" -V

# ---------------------------------------------------------------------------
# 3. 坑位修正：numpy / setuptools
# ---------------------------------------------------------------------------
log "3/7 pin numpy<2 and setuptools<81"
"$PIP" install -q "numpy==1.24.4" "setuptools<81" wheel
"$PY" -c "import pkg_resources, numpy; print('pkg_resources OK; numpy', numpy.__version__)"

# ---------------------------------------------------------------------------
# 4. 克隆 SimplerEnv（含子模块）
# ---------------------------------------------------------------------------
log "4/7 clone SimplerEnv with submodules"
mkdir -p "$PROJECT_ROOT"
if [ ! -d "$SIMPLER_DIR" ]; then
  git clone --recurse-submodules "$SIMPLER_REPO" "$SIMPLER_DIR"
else
  echo "already cloned: $SIMPLER_DIR"
  git -C "$SIMPLER_DIR" submodule update --init --recursive || true
fi
# 校验视觉匹配贴图是否到位（缺了会导致评测画面与实际不符）
if ! ls "$SIMPLER_DIR"/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png >/dev/null 2>&1; then
  echo "!!! ERROR: real_inpainting assets missing. Re-clone with --recurse-submodules."
  exit 1
fi
echo "real_inpainting assets OK"

# ---------------------------------------------------------------------------
# 5. ManiSkill2_real2sim
# ---------------------------------------------------------------------------
log "5/7 install ManiSkill2_real2sim (editable)"
cd "$SIMPLER_DIR/ManiSkill2_real2sim"
"$PIP" install -e .

# ---------------------------------------------------------------------------
# 6. SimplerEnv 本体 + 坑位修正复核
# ---------------------------------------------------------------------------
log "6/7 install SimplerEnv (editable)"
cd "$SIMPLER_DIR"
"$PIP" install -e .

# 依赖安装可能会把 numpy/opencv 顶上去，这里再钉一次
"$PIP" install -q "numpy==1.24.4" "opencv-python==4.8.1.78" "setuptools<81"
echo "re-pinned numpy/opencv/setuptools"

# 视频导出用（作业要求 2-4 个代表视频）
"$PIP" install -q mediapy decord imageio imageio-ffmpeg

# ---------------------------------------------------------------------------
# 7. 验证
# ---------------------------------------------------------------------------
log "7/7 verification"
"$PY" - <<'PYEOF'
import warnings
warnings.filterwarnings("ignore")
import numpy, sapien
print("numpy :", numpy.__version__)
print("sapien:", sapien.__version__)
import simpler_env
wx = sorted(k for k in simpler_env.ENVIRONMENTS if "widowx" in k)
print("registered environments:", len(simpler_env.ENVIRONMENTS))
print("widowx tasks:", wx)
expected = {
    "widowx_spoon_on_towel",
    "widowx_put_eggplant_in_basket",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
}
missing = expected - set(wx)
assert not missing, f"missing tasks: {missing}"
print("all 4 required WidowX tasks present")
PYEOF

echo ""
echo "============================================================"
echo " Installation finished."
echo " Next: verify rendering with"
echo "   $PY $PROJECT_ROOT/scripts/verify_render.py"
echo "============================================================"
