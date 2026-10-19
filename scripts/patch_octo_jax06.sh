#!/usr/bin/env bash
# =============================================================================
# Octo 代码适配 JAX 0.6.x 的补丁（Blackwell / sm_120 必需）
#
# 背景：Octo 官方代码（commit 241fb35）是为 JAX 0.4.20 写的。
#       用户显卡 RTX 5070 是 sm_120（Blackwell），JAX 0.4.20 不支持，
#       必须用 JAX 0.6.2。JAX 0.6.x 移除了若干旧 API，需打补丁。
#
# 用法（在 Octo 仓库目录下执行）：
#   bash patch_octo_jax06.sh
#
# 已验证的版本组合（在 WSL2 + RTX 5070 上实测通过）：
#   jax==0.6.2  jax[cuda12]  numpy==1.26.4  ml_dtypes==0.5.4
#   tensorflow==2.15.0  tensorflow-probability==0.23.0  distrax 最新
#   transformers==4.55.4（保留 FlaxAutoModel；更新的版本移除了 Flax）
# =============================================================================
set -euo pipefail

OCTO_DIR="${1:-/root/vla_workspace/octo}"
cd "$OCTO_DIR"

echo "=== 补丁 1/4：jax.random.KeyArray -> jax.Array ==="
sed -i 's/PRNGKey = jax.random.KeyArray/PRNGKey = jax.Array/' octo/utils/typing.py

echo "=== 补丁 2/4：jax.tree_map -> jax.tree.map（31 处）==="
grep -rl "jax\.tree_map" --include=*.py . | xargs -r sed -i 's/jax\.tree_map/jax.tree.map/g'

echo "=== 补丁 3/4：jax.tree_leaves -> jax.tree.leaves ==="
grep -rl "jax\.tree_leaves" --include=*.py . | xargs -r sed -i 's/jax\.tree_leaves/jax.tree.leaves/g'

echo "=== 补丁 4/4：确认无残留旧 API ==="
remaining=$(grep -rnE "jax\.tree_(map|leaves|flatten|structure)\b|jax\.random\.KeyArray" --include=*.py . | wc -l)
echo "残留旧 API 数量：$remaining（应为 0）"

echo ""
echo "补丁完成。关键依赖版本（已实测通过）："
echo "  jax[cuda12]==0.6.2  numpy==1.26.4  ml_dtypes==0.5.4"
echo "  tensorflow==2.15.0  tensorflow-probability==0.23.0"
echo "  transformers==4.55.4（Flax 支持，勿升级）"
echo "  distrax（最新，用 TFP 0.23.0 的 jax substrate）"
echo ""
echo "运行时需设置："
echo "  export PYTHONPATH=<dlimp父目录>:<octo父目录>   # 指向内层包"
echo "  export HF_ENDPOINT=https://hf-mirror.com"
