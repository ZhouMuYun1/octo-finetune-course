# 复现步骤

本文档给出从零复现"数据处理 → 微调 → 加载验证"的完整步骤。

---

## 0. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Ubuntu 22.04（原生 Linux）**或** WSL2（Ubuntu 22.04）|
| Python | 3.10 |
| GPU | NVIDIA，支持 CUDA 12（本测试用 RTX 5070 Laptop，sm_120）|
| GPU 显存 | ≥ 8 GB（训练）、≥ 6 GB（推理）|
| 系统内存 | ≥ 12 GB（WSL 需在 `.wslconfig` 中显式分配）|
| 磁盘 | ≥ 30 GB（依赖约 5 GB + 数据 3.3 GB + checkpoint 3 GB）|

> **注意**：训练可以在 WSL2 完成，但**评测（SimplerEnv 渲染）必须用原生 Linux**。WSL 只提供 `llvmpipe` 软件 Vulkan，SAPIEN 会报 `sapien renderer is not supported on WSL`。

---

## 1. 安装 Octo 训练环境

### 1.1 一键安装（推荐）

```bash
bash scripts/setup_native_linux.sh
```

该脚本会依次完成：
1. 检查 NVIDIA 驱动（要求 ≥ 570）
2. 创建 conda 环境 `test_jax`（Python 3.10）
3. 安装 JAX 0.6.2（`jax[cuda12]`）
4. 安装 TensorFlow 2.15.0（仅用于数据加载）
5. 安装 Octo 及其依赖
6. 应用 JAX 0.6.2 兼容补丁

### 1.2 手动安装（关键版本）

若一键脚本失败，按以下版本手动安装：

```bash
conda create -y -n test_jax python=3.10
conda activate test_jax

# JAX（必须用 jax[cuda12]，不能用 jaxlib[cuda12]）
pip install -U "jax[cuda12]"

# 钉版本（关键）
pip install --no-deps "numpy==1.26.4" "ml_dtypes==0.5.4" "protobuf==3.20.3"

# TensorFlow（仅数据加载用）
pip install "tensorflow==2.15.0" "tensorflow-probability==0.23.0"

# Octo 依赖
pip install flax optax einops ml-collections transformers tokenizers
pip install matplotlib plotly gym "tensorstore" "orbax-checkpoint==0.4.8"

# Octo 本体
git clone https://github.com/octo-models/octo.git
cd octo && git checkout 241fb3514b7c40957a86d869fecb7c7fc353f540
pip install -e . --no-deps

# dlimp（Octo 的数据加载依赖）
git clone https://github.com/kvablack/dlimp.git
```

### 1.3 应用兼容补丁

JAX 0.6.2 移除了若干 API，Octo 需要打补丁：

```bash
bash scripts/patch_octo_jax06.sh
```

补丁内容（如脚本不可用，手动执行）：

```bash
# 1. KeyArray → jax.Array
find octo -name "*.py" -exec sed -i 's/PRNGKey = jax.random.KeyArray/PRNGKey = jax.Array/' {} \;

# 2. jax.tree_map → jax.tree.map
find octo -name "*.py" -exec sed -i 's/jax\.tree_map/jax.tree.map/g' {} \;

# 3. jax.tree_leaves → jax.tree.leaves
find octo -name "*.py" -exec sed -i 's/jax\.tree_leaves/jax.tree.leaves/g' {} \;
```

另外需要**手动修改两处**（补丁脚本不覆盖）：

**修改 1：跳过单设备的 `process_allgather`**

文件：`octo/model/octo_model.py`，函数 `from_config`

```python
# 原代码
example_batch = multihost_utils.process_allgather(example_batch)

# 改为
if jax.device_count() > 1:
    example_batch = multihost_utils.process_allgather(example_batch)
```

原因：JAX 0.6.x 的 `process_allgather` 在单设备下给叶子加一个 device 维 `(8,16) → (1,8,16)`，导致 T5 编码器 reshape 失败。

**修改 2：wandb stub**

创建 `wandb_stub/wandb.py`：

```python
class _Config:
    def update(self, *args, **kwargs):
        return None

config = _Config()
def init(*args, **kwargs): return None
def log(*args, **kwargs): return None
def finish(*args, **kwargs): return None
def define_metric(*args, **kwargs): return None
```

并在 `PYTHONPATH` 中把 `wandb_stub` 放在最前面。

原因：真实 wandb 0.31 依赖 protobuf 7.x，与 TF 2.15（要求 protobuf <4.21）冲突。

**修改 3：把 loss 打印到 stdout（保存训练日志所需）**

真实 wandb 被 stub 后，训练指标不会落盘。为满足"提交 loss 记录"的要求，需让 `finetune.py` 在每次 `log_interval` 时打印一行到 stdout。

文件：`octo/scripts/finetune.py`，训练循环中：

```python
# 原代码
if (i + 1) % FLAGS.config.log_interval == 0:
    update_info = jax.device_get(update_info)
    wandb_log(
        {"training": update_info, "timer": timer.get_average_times()}, step=i
    )

# 改为
if (i + 1) % FLAGS.config.log_interval == 0:
    update_info = jax.device_get(update_info)
    _info = flatten_dict(update_info, sep="/")
    _items = " ".join(
        f"{k}={float(v):.6f}" for k, v in sorted(_info.items())
        if hasattr(v, "dtype") or isinstance(v, (int, float))
    )
    print(f"[TRAIN] step={i+1} {_items}", flush=True)
    wandb_log(
        {"training": update_info, "timer": timer.get_average_times()}, step=i
    )
```

输出样例：

```
[TRAIN] step=1000 grad_norm=23.974672 learning_rate=0.000000 loss=1.375373 mse=1.375373 param_norm=185.217194 update_norm=0.000001
```

**修改 4：禁用验证/可视化回调（避免 OOM）**

文件：`octo/scripts/finetune.py`，训练循环中注释掉两个回调调用：

```python
# 原代码
with timer("val"):
    val_metrics = val_callback(train_state, i + 1)
    wandb_log(val_metrics, step=i)

with timer("visualize"):
    viz_metrics = viz_callback(train_state, i + 1)
    wandb_log(viz_metrics, step=i)

# 改为（注释掉）
# with timer("val"):
#     val_metrics = val_callback(train_state, i + 1)
#     wandb_log(val_metrics, step=i)
#
# with timer("visualize"):
#     viz_metrics = viz_callback(train_state, i + 1)
#     wandb_log(viz_metrics, step=i)
```

原因：验证回调需加载 16 个 batch 并编译额外的计算图，在 12 GB 内存的 WSL2 上触发 OOM；可视化回调还会编译 rollout 图，更容易 OOM。

同时把 `train.sh` 中的 `--config.eval_interval` 设为大于训练步数（本项目 5000），作为双保险。

> 说明：禁用验证回调不影响 "Final checkpoint 选择规则" 的合规性——本项目采用"预先确定训练步数"规则，本就不依赖验证集选择。

### 1.4 环境变量

```bash
export PYTHONPATH="/path/to/wandb_stub:/path/to/dlimp:/path/to/octo:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com     # 国内网络必需
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
```

### 1.5 WSL 专项设置

若在 WSL2 运行，**必须先扩充内存与共享内存**，否则训练会在约 500–1000 步时被 OOM 杀死。

Windows 侧 `C:\Users\<用户名>\.wslconfig`：

```ini
[wsl2]
memory=13GB
processors=32
networkingMode=mirrored

[experimental]
hostAddressLoopback=true
```

WSL 侧每次启动后执行：

```bash
mount -o remount,size=10G /dev/shm
```

验证：

```bash
free -h          # 应显示约 12 Gi
df -h /dev/shm   # 应显示 10G
```

---

## 2. 准备数据

### 2.1 下载 BridgeData V2 分片

```bash
export http_proxy=http://127.0.0.1:7890    # 若网络受限
export https_proxy=http://127.0.0.1:7890

BASE="https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0"
RAW="$HOME/vla_workspace/bridge_data/raw"
mkdir -p "$RAW"

for i in $(seq 60 89); do
  f=$(printf "bridge_dataset-train.tfrecord-%05d-of-01024" $i)
  curl -C - --max-time 300 -o "$RAW/$f" "$BASE/$f"
done
```

### 2.2 校验完整性

每个分片固定包含 **52 条轨迹**：

```bash
python -c "
import tensorflow as tf, sys
n = sum(1 for _ in tf.data.TFRecordDataset(sys.argv[1]))
print(n)
sys.exit(0 if n == 52 else 1)
" "$RAW/xxx.tfrecord"
```

### 2.3 划分 train / val

**按完整分片划分**（避免相邻帧泄漏）：

```bash
DEST="$HOME/vla_workspace/bridge_data/bridge_dataset/1.0.0"
mkdir -p "$DEST"

# 27 个分片 → train
i=0
for f in $(ls "$RAW"/*.tfrecord | head -27); do
  cp "$f" "$DEST/$(printf 'bridge_dataset-train.tfrecord-%05d-of-00027' $i)"
  i=$((i+1))
done

# 3 个分片 → val
i=0
for f in $(ls "$RAW"/*.tfrecord | tail -3); do
  cp "$f" "$DEST/$(printf 'bridge_dataset-val.tfrecord-%05d-of-00003' $i)"
  i=$((i+1))
done
```

### 2.4 下载并修改 `dataset_info.json`

```bash
curl -o "$DEST/dataset_info.json" "$BASE/dataset_info.json"
curl -o "$DEST/features.json"     "$BASE/features.json"
```

修改 `dataset_info.json` 中的 split 定义：

```python
import json
p = "dataset_info.json"
d = json.load(open(p))
for s in d["splits"]:
    if s["name"] == "train": s["shardLengths"] = ["52"] * 27
    elif s["name"] == "val": s["shardLengths"] = ["52"] * 3
json.dump(d, open(p, "w"))
```

---

## 3. 训练

### 3.1 配置数据路径

编辑 `train_config.py`（或设置环境变量）：

```python
cfg.dataset_kwargs.data_dir = "/root/vla_workspace/bridge_data"
cfg.dataset_kwargs.filter_functions = (
    ModuleSpec.create("octo.data.oxe.placement_filter:is_placement_task"),
)
```

### 3.2 启动训练

```bash
# 主方案：动作头微调
MODE=head_only NUM_STEPS=1000 BATCH_SIZE=8 bash train.sh

# 对照：完整微调
MODE=full NUM_STEPS=1000 BATCH_SIZE=8 bash train.sh
```

### 3.3 监控

```bash
# 实时查看 loss
tail -f train_log_octo_head_only.txt | grep "\[TRAIN\]"

# 查看 checkpoint
ls checkpoints/octo_finetune/*/
```

### 3.4 预期结果

| 方式 | 耗时 | 产出目录 | 可用的 step |
|---|---|---|---|
| head_only | 约 2.5 分钟 | `checkpoints/octo_finetune/octo_head_only_*` | 250 / 500 / 750 / 1000 |
| full | 约 10 分钟 | `checkpoints/octo_finetune/octo_full_*` | 250 / 500 / 750（1000 可能保存中断）|

> **本项目 Final 采用 step 750**，原因见 `docs/TRAINING_RECORD.md` 第 5 节（为让两个模型步数一致）。

---

## 4. 加载验证

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 国内必需

from octo.model.octo_model import OctoModel
import jax, jax.numpy as jnp

path = "checkpoints/octo_finetune/octo_head_only_xxx"
model = OctoModel.load_pretrained(path, step=750)     # 本项目采用 step 750

obs = {
    "image_primary": jnp.zeros((1, 2, 256, 256, 3), dtype=jnp.uint8),
    "image_wrist":   jnp.zeros((1, 2, 128, 128, 3), dtype=jnp.uint8),
    "timestep_pad_mask": jnp.ones((1, 2), dtype=jnp.bool_),
}
task = model.create_tasks(texts=["put the spoon on the towel"])
actions = model.sample_actions(obs, task, rng=jax.random.PRNGKey(0))
print(actions.shape)   # (1, 4, 7)
```

**预期输出**：`(1, 4, 7)`

---

## 5. 评测（需原生 Linux）

### 5.1 安装 SimplerEnv

```bash
bash scripts/setup_simplerenv.sh
```

### 5.2 渲染自检

```bash
python scripts/verify_render.py
```

若报 `sapien renderer is not supported on WSL`，说明当前环境不支持 Vulkan 硬件渲染，需切换到原生 Linux。

### 5.3 生成评测初始状态清单（已冻结，勿修改）

```bash
python scripts/generate_eval_config.py --output configs/eval_config.json
```

> 本仓库已包含冻结的 `configs/eval_config.json`，**不要重新生成**，否则会破坏 Base/Final 对比的一致性。

### 5.4 运行评测

```bash
# Base T1
python evaluate.py --model-type octo-small \
  --checkpoint "$HOME/vla_workspace/models/octo-small-1.5" \
  --task widowx_spoon_on_towel --episodes 10 \
  --eval-config configs/eval_config.json \
  --output-dir results/base_t1

# Final T1
python evaluate.py --model-type octo-small \
  --checkpoint "$HOME/vla_workspace/checkpoints/octo_finetune/octo_head_only_xxx" \
  --task widowx_spoon_on_towel --episodes 10 \
  --eval-config configs/eval_config.json \
  --output-dir results/final_t1
```

对 `widowx_put_eggplant_in_basket` 重复同样命令。

---

## 6. 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `jaxlib 0.6.2 does not provide the extra 'cuda12'` | 装错了包名 | `pip install -U "jax[cuda12]"`（不是 `jaxlib[cuda12]`）|
| `cannot import name 'float8_e3m4' from 'ml_dtypes'` | ml_dtypes 版本不匹配 | `pip install --no-deps ml_dtypes==0.5.4` |
| `module 'jax.random' has no attribute 'KeyArray'` | 未打补丁 | 执行 `scripts/patch_octo_jax06.sh` |
| `jax.tree_map was removed` | 未打补丁 | 同上 |
| `PrefixGroup tokens must be (batch, n_tokens, d)` | `process_allgather` 加维 | 见 1.3 修改 1 |
| 训练约 500 步被 Killed | 内存不足 | 见 1.5 WSL 内存设置 |
| `sapien renderer is not supported on WSL` | WSL 无 Vulkan 硬件渲染 | 改用原生 Linux |
| 加载模型卡住（下载 t5-base）| HF 直连超时 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `tostring_rgb` AttributeError | matplotlib ≥ 3.10 | 改用 `canvas.buffer_rgba()` |
