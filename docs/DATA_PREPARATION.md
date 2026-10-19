# 数据处理说明

## 1. 数据来源

| 项目 | 内容 |
|---|---|
| 数据集 | **BridgeData V2**（TFDS 版本 `bridge_dataset`）|
| 官方页面 | <https://rail-berkeley.github.io/bridgedata/> |
| 下载地址 | <https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/> |
| 采集平台 | **WidowX 250s**（6-DoF + 平行夹爪）|
| 许可 | 遵循 BridgeData V2 原始许可（见官方页面）|
| 使用分片 | **30 个 TFRecord 分片**（官方共 1,024 个 train 分片）|
| 原始轨迹数 | **1,404 条**（train）+ **156 条**（val）|

### 为什么选 BridgeData V2

1. 作业第 3.2 节明确允许使用 BridgeData V2
2. BridgeData V2 的采集平台就是 WidowX，**动作空间与评测环境天然兼容**
3. 与 Octo-Small-1.5 的预训练数据同源（Bridge 占其训练 batch 的 17%），字段格式已被官方 `bridge_dataset_transform` 适配

---

## 2. 数据下载与完整性校验

### 2.1 下载策略

Berkeley 服务器直连速度极慢（实测约 8 KB/s），通过 Clash 代理加速到约 670 KB/s。

下载脚本的核心逻辑：

```bash
# 伪代码
for shard_id in 00060..00249:
    f = "bridge_dataset-train.tfrecord-%05d-of-01024" % shard_id
    curl -C - --max-time 300 -o "$RAW/$f" "$BASE/$f"
    # 用 TFRecord 结构校验：能解析出 52 条记录才算完整
    if tfrecord_count("$RAW/$f") == 52: keep()
    else: rm()
```

**关键校验方式**：不依赖 HTTP `Content-Length`（服务器返回不稳定），而是**用 TFRecord 结构校验**——每个 Bridge 分片固定包含 **52 条轨迹**，只要能完整解析出 52 条记录就视为完整。这比字节数比对更可靠。

### 2.2 最终数据

| 项目 | 值 |
|---|---|
| 完整分片数 | **30 个** |
| 每个分片轨迹数 | 52 条（固定）|
| 划分 | **27 个分片训练 + 3 个分片验证** |
| 训练轨迹数 | 27 × 52 = **1,404 条** |
| 验证轨迹数 | 3 × 52 = **156 条** |
| 磁盘占用 | 约 3.3 GB |

### 2.3 划分方式（按完整轨迹）

**按分片划分，而非按帧划分**：

```
分片 000..026 → train（1,404 条轨迹）
分片 027..029 → val  （156 条轨迹）
```

这样保证**同一条轨迹的所有帧只出现在一侧**，避免相邻帧泄漏（作业第 3.2 节要求）。

---

## 3. 字段映射

BridgeData V2 的 TFDS 字段经 `octo.data.oxe.oxe_standardization_transforms:bridge_dataset_transform` 转换后：

| 用途 | Bridge 原始字段 | 转换后 | 说明 |
|---|---|---|---|
| 第三人称图像 | `observation.image_0` | `observation.image_primary` | resize 到 256×256 |
| 腕部图像 | `observation.image_1` | `observation.image_wrist` | resize 到 128×128 |
| 语言指令 | `language_instruction` | 经 T5-Base 编码 | `max_length=16`，padding 到 16 |
| 动作 | `action`（7 维）| `action`（7 维）| 见第 4 节 |
| 本体状态 | `observation.state` | `observation.proprio` | 7 维 |

### 数据样本结构（加载后）

```
observation:
  image_0 / image_primary : (T, 256, 256, 3)  uint8
  image_1 / image_wrist   : (T, 128, 128, 3)  uint8
  state                   : (T, 7)            float32
action                      : (T, 7)            float32
language_instruction        : (T,)              string
```

训练时经过 window 切分后：

```
observation.image_primary : (batch, window=1, 256, 256, 3)
observation.image_wrist   : (batch, window=1, 128, 128, 3)
action                    : (batch, window=1, horizon=4, 7)   # 预测未来 4 步
task.language_instruction : (batch,)  →  tokenize 成 (batch, 16)
```

---

## 4. 动作约定与归一化

### 4.1 动作维度

BridgeData V2 的 7 维动作（末端增量控制）：

| 维度 | 含义 | 单位 |
|---|---|---|
| 0 | dx（末端位置增量）| 米 |
| 1 | dy | 米 |
| 2 | dz | 米 |
| 3 | droll（旋转增量，axis-angle）| 弧度 |
| 4 | dpitch | 弧度 |
| 5 | dyaw | 弧度 |
| 6 | **gripper** | 0 = 闭合，1 = 张开 |

### 4.2 归一化

采用 **z-score 归一化**：

```python
x_normalized = (x - mean) / std
```

**掩码**：`action_normalization_mask = [True, True, True, True, True, True, False]`

即前 6 维归一化，**第 7 维（夹爪）保持原始 0/1 值不归一化**。

原因：夹爪是二值开闭语义，归一化会破坏其物理含义。

### 4.3 归一化统计量的来源

训练时使用**本数据集（717 条筛选后轨迹）自身计算**的统计量，保存在训练产出的 `dataset_statistics.json` 中：

| 统计量 | 训练数据（717 条）|
|---|---|
| `mean` | `[0.00038, 0.00007, -0.00021, -0.00021, -0.00038, 0.00124, 0.61764]` |
| `std` | `[0.00973, 0.01428, 0.01288, 0.0273, 0.0312, 0.05302, 0.48593]` |
| `min` | `[-0.05087, -0.06327, -0.05374, -0.21707, -0.29106, -0.30395, 0.0]` |
| `max` | `[0.05502, 0.06301, 0.06775, 0.23341, 0.24861, 0.29119, 1.0]` |

对照 Base checkpoint 的 Bridge 全体统计量（60,064 条轨迹）：

| 统计量 | Base（Bridge 全体）|
|---|---|
| `std` | `[0.00963, 0.0135, 0.01251, 0.02815, 0.03028, 0.07586, 0.48772]` |
| `min` | `[-0.40075, -0.13875, -0.22554, -3.20108, -1.86181, -6.27908, 0.0]` |
| `max` | `[0.41691, 0.25865, 0.21218, 3.1222, 1.86181, 6.28048, 1.0]` |

**观察**：
- `std` 两套非常接近（差异 < 5%）
- `min/max` 差异大：Bridge 全体包含大幅旋转（±6.28 ≈ ±2π），而本数据集的旋转增量集中在 ±0.3 弧度

**说明**：评测时反归一化必须使用**与训练一致的统计量**（即训练产出的 `dataset_statistics.json`），否则动作尺度会错位。

---

## 5. 数据筛选：只保留"放置类"任务

### 5.1 筛选动机

两个目标评测任务都是"放置"语义：

- **T1**：把勺子放到毛巾上
- **T2**：把茄子放进篮子里

BridgeData V2 包含多种任务（折叠、开抽屉、扫地、倒水等）。保留全部任务会引入大量与目标任务语义无关的轨迹。

### 5.2 筛选规则

筛选函数：`octo.data.oxe.placement_filter:is_placement_task`（源码见 [`../scripts/placement_filter.py`](../scripts/placement_filter.py)）

规则：取语言指令的**首个词**，若属于以下集合则保留：

```python
{"put", "place", "move", "moved", "pick", "take",
 "picking", "pickup", "slide", "remove", "insert"}
```

### 5.3 筛选结果

| 阶段 | 轨迹数 | 比例 |
|---|---:|---:|
| 筛选前 | 1,404 条 | 100% |
| **筛选后** | **717 条** | **51.1%** |
| 被排除 | 687 条 | 48.9% |

被排除的指令示例：空指令、`fold the cloth`、`open the drawer`、`sweep the table`、`turn the knob` 等。

---

## 6. 与两个评测任务的匹配度分析（重要）

筛选后的 717 条数据中，按关键词统计与 T1/T2 的匹配度：

| 任务 | 匹配条件 | 轨迹数 | 占比 |
|---|---|---:|---:|
| **T1** | 同时含 `spoon` + `towel/cloth` | **13 条** | 1.8% |
| | 仅含 `spoon` | 57 条 | 8.0% |
| | 仅含 `towel/cloth` | 167 条 | 23.3% |
| **T2** | 同时含 `eggplant` + `basket` | **0 条** | 0% |
| | 仅含 `eggplant` | 11 条 | 1.5% |
| | 仅含 `basket` | 13 条 | 1.8% |

### 高频词分布（717 条）

| 词 | 次数 | 词 | 次数 |
|---|---:|---|---:|
| put | 360 | spoon | 57 |
| pot | 206 | sink | 54 |
| move | 202 | stove | 51 |
| table | 135 | towel | 44 |
| cloth | 119 | basket | 13 |
| pan | 93 | eggplant | 11 |

### 结论：存在明显的 Domain Gap

| | 训练数据主体 | T1 场景 | T2 场景 |
|---|---|---|---|
| 台面 | Bridge 厨房台面、水槽、灶台 | Bridge 桌子 | Bridge 桌子 + 水槽 |
| 主要物体 | pot / pan / burner / sink | **勺子 + 毛巾** | **茄子 + 篮子** |
| 动作 | 把锅碗放到水槽/灶台/桌上 | 抓勺子→放毛巾 | 抓茄子→放篮子 |

**这说明**：
- T1 有 13 条直接相关数据（1.8%），有一定学习基础
- **T2 有 0 条直接相关数据**，模型完全没见过"茄子→篮子"这一组合

**这一差距应在结果分析中如实说明**，作为"微调后表现变化"的可能解释之一（有证据的观察，而非未验证的解释）。

> 说明：BridgeData V2 本身以厨房场景为主，"茄子 + 篮子"组合在该数据集中确实极少，这不是数据下载不完备导致的，而是数据集固有分布。

---

## 7. 复现数据准备

```bash
# 1. 设置代理（如网络受限）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 2. 下载 30 个分片（脚本见 scripts/download_bridge.sh，如未提供可按第 2.1 节伪代码实现）
#    每个分片约 108-137 MB

# 3. 校验并划分（27 train + 3 val）
#    每个分片固定 52 条轨迹，按 TFRecord 结构校验
```

数据目录结构：

```
bridge_data/
└── bridge_dataset/
    └── 1.0.0/
        ├── dataset_info.json               # 更新为 train 27 shard / val 3 shard
        ├── features.json
        ├── bridge_dataset-train.tfrecord-00000-of-00027
        ├── ...
        ├── bridge_dataset-train.tfrecord-00026-of-00027
        ├── bridge_dataset-val.tfrecord-00000-of-00003
        ├── bridge_dataset-val.tfrecord-00001-of-00003
        └── bridge_dataset-val.tfrecord-00002-of-00003
```
