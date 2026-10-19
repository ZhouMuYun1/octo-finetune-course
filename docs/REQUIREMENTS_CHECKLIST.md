# 作业要求对照表（B 类 · 微调部分）

> 本文档把作业文档中的每一项要求映射到仓库的具体位置，便于自查与评审。
> 标注 `⏳ 组员` 的项由负责评测的组员完成，其余已完成。

---

## 一、作业 9.2「实验报告内容」逐项对照

| # | 报告要求 | 仓库中的对应位置 | 状态 |
|---|---|---|---|
| 1 | **项目概述**：A/B 类、实验目标与主要工作 | [README.md](../README.md) 开头「项目概述」 | ✅ |
| 2 | **模型选择**：名称、checkpoint 版本、WidowX 依据、选择原因 | [docs/MODEL_SELECTION.md](MODEL_SELECTION.md) 全文；[README.md](../README.md) 第 1 节 | ✅ |
| 3 | **数据与处理**：来源、数量、筛选、划分、输入输出格式、归一化 | [docs/DATA_PREPARATION.md](DATA_PREPARATION.md) 全文；[README.md](../README.md) 第 2 节 | ✅ |
| 4 | **训练方法**：模块、可训练参数、超参数、步数、计算资源、checkpoint 规则、loss | [docs/TRAINING_RECORD.md](TRAINING_RECORD.md) 全文；[README.md](../README.md) 第 3 节 | ✅ |
| 5 | **评测设置**：SimplerEnv 版本、任务、episode 数、初始状态、种子、隔离方式、控制频率、最大步数、成功判定、动作适配 | [README.md](../README.md) 第 5 节（含 SimplerEnv commit、终止逻辑、成功记录时机）| ✅ |
| 6 | **实验结果**：Base/Final 各任务成功数、成功率、平均分与变化 | [results.csv](../results.csv) 模板；[README.md](../README.md) 第 6.1 节 | ⏳ 组员 |
| 7 | **案例分析与局限**：至少 2 个代表性 episode + 视频 + 局限 | [README.md](../README.md) 第 6.3 节；[video_links.md](../video_links.md) | ⏳ 组员 |
| 8 | **复现信息与分工**：仓库链接、checkpoint 链接、依赖、视频、分工 | [README.md](../README.md) 第 7–8 节；[checkpoint_link.txt](../checkpoint_link.txt) | ✅（链接待填）|

---

## 二、作业 9.1「提交材料」逐项对照

| # | 提交物 | 仓库中的对应 | 状态 |
|---|---|---|---|
| 1 | **实验报告**（≤10 页，含封面/参考文献/附录，PDF）| 报告由组员汇总；建议明确标注「不超过 10 页，含封面、参考文献和附录」| ⏳ 组员 |
| 2 | **公开代码仓库**（依赖、训练配置与日志、Base/Final 命令、环境版本、初始状态清单、README 复现步骤）| 本仓库全部内容 | ✅ |
| 3 | **Checkpoint 下载链接**（+ 依赖基础模型版本 + 加载方式）| [checkpoint_link.txt](../checkpoint_link.txt) | ✅（链接待填）|
| 4 | **实验结果**（汇总表 + 逐 episode）| [results.csv](../results.csv)、[episodes.csv](../episodes.csv) | ⏳ 组员 |
| 5 | **代表性视频**（2–4 个，覆盖不同任务，含失败样例）| [video_links.md](../video_links.md)、`videos/` | ⏳ 组员 |

> ⚠️ **报告页数限制**：作业要求「不超过 10 页，页数包含封面（如有）、参考文献和附录」。写报告时务必控制篇幅。

---

## 三、作业 3.1–3.4「模型、数据与训练」逐项对照

### 3.1 模型选择

| 要求 | 对应位置 | 状态 |
|---|---|---|
| 注明模型名称 | MODEL_SELECTION.md §1 | ✅ |
| 注明 checkpoint 版本 | MODEL_SELECTION.md §1（step 300000，v1.5）| ✅ |
| **提供 WidowX 数据训练依据**（论文/模型卡/官方仓库）| MODEL_SELECTION.md §2（三条独立证据）| ✅ |
| 若已有 SimplerEnv 结果，附论文或仓库链接 | MODEL_SELECTION.md §2 依据 3；§5 参考链接 | ✅ |
| 提交训练命令与评测命令 | README.md §3.3、§5；train.sh / evaluate.py | ✅ |
| 确认与 WidowX 动作空间兼容 | MODEL_SELECTION.md §4（7 维动作接口对齐）| ✅ |

### 3.2 数据处理

| 要求 | 对应位置 | 状态 |
|---|---|---|
| 数据名称、来源 | README.md §2；DATA_PREPARATION.md §1 | ✅ |
| 许可 | DATA_PREPARATION.md §1（遵循 BridgeData V2 原始许可）| ✅ |
| **轨迹数** | 1,404（原始）/ 717（筛选后）| ✅ |
| **训练样本数** | **23,634**（总帧数）| ✅ |
| 图像处理映射 | DATA_PREPARATION.md §3（resize 尺寸、字段对应）| ✅ |
| 语言字段映射 | DATA_PREPARATION.md §3（T5-Base, max_length=16）| ✅ |
| 动作字段映射 | DATA_PREPARATION.md §3、§4 | ✅ |
| 动作单位 | DATA_PREPARATION.md §4.1（米 / 弧度 / 夹爪 0-1）| ✅ |
| 归一化统计量 | DATA_PREPARATION.md §4.2–4.3（含训练数据与 Base 的统计量对比）| ✅ |
| 夹爪开闭约定 | DATA_PREPARATION.md §4.1（0=闭合，1=张开）| ✅ |
| 训练/验证划分方法 | DATA_PREPARATION.md §2.3（按完整分片，避免相邻帧泄漏）| ✅ |
| 未使用正式评测轨迹 | README.md §5「评测隔离声明」| ✅ |

### 3.3 轻量微调

| 要求 | 对应位置 | 状态 |
|---|---|---|
| 至少更新一种带可学习参数的模块 | TRAINING_RECORD.md §1–2（head_only 更新 27 个叶子）| ✅ |
| **训练配置** | train.sh、train_config.py、configs/checkpoint_metadata_*/finetune_config.json | ✅ |
| **可训练参数说明** | TRAINING_RECORD.md §2（27 个叶子的具体模块 + 339 个冻结模块）| ✅ |
| **训练步数** | 1,000 步（Final 取 step 750）| ✅ |
| **loss 记录** | train_log_head_only.txt、train_log_full.txt（各 20 个记录点）+ TRAINING_RECORD.md §4 | ✅ |
| 训练规模选择依据 | TRAINING_RECORD.md §5（数据规模、环境限制、避免过拟合）| ✅ |

### 3.4 Final checkpoint 选择与评测隔离

| 要求 | 对应位置 | 状态 |
|---|---|---|
| **选择规则** | README.md §4.1；TRAINING_RECORD.md §5（预先确定训练步数 = 750）| ✅ |
| 说明为什么是 750 而非 1000 | 同上（full 的 step 1000 保存中断，为保证两模型步数一致）| ✅ |
| 未用正式评测结果选择 checkpoint | README.md §5「评测隔离声明」| ✅ |
| 开发/调参初始状态与正式评测分开 | README.md §5；eval_config.json 在训练前冻结 | ✅ |

---

## 四、作业 6.2「固定设置」逐项对照

| 要求 | 对应位置 | 状态 |
|---|---|---|
| SimplerEnv 版本或提交号 | **`c701ea271b16b80616d6fed9713c4a50a0adea3b`**（README.md §5；setup_native_linux.sh / setup_simplerenv.sh）| ✅ |
| 环境配置 | scripts/setup_simplerenv.sh、requirements.txt | ✅ |
| **评测初始状态清单** | [configs/eval_config.json](../configs/eval_config.json)（含任务、episode 编号、env_seed、obj_episode_id）| ✅ |
| 初始状态选取方法 | scripts/generate_eval_config.py；README.md §5 | ✅ |
| 不使用评测结果筛状态 | README.md §5「评测隔离声明」| ✅ |
| 控制频率（5 Hz）| README.md §5 | ✅ |
| 最大控制步数（T1=60，T2=120）| README.md §5；task_config.py | ✅ |
| **成功记录时机** | README.md §5（每 episode 结束时记录一次，取最终 done）| ✅ |
| **终止逻辑** | README.md §5（4 种终止情形）；evaluate.py L164–200 | ✅ |
| Base/Final 相同初始状态与种子 | eval_config.json（统一 env_seed=0, model_seed=0）| ✅ |
| **模型种子与环境种子分别记录** | eval_config.json 中 `model_seed` 与 `env_seed` 独立字段；README.md §5 | ✅ |
| 环境原生成功判定 | evaluate.py L192–193（`record["success"] = bool(done)`）| ✅ |
| 不筛除失败 episode | evaluate.py（所有 episode 都写入 episodes.csv）；README.md §5 | ✅ |
| 不人工干预 | README.md §5 | ✅ |
| 异常按失败记录 | evaluate.py L200（`record["success"] = False` + 异常信息）| ✅ |

---

## 五、作业 8「分数计算」说明

| 指标 | 定义 | 对应位置 |
|---|---|---|
| 单任务成功率 | 成功 episode 数 / 规定 episode 数 × 100% | README.md §5 末行 |
| **B-SimplerScore** | (T1 成功率 + T2 成功率) / 2 | README.md §5 末行；results.csv 列 |

---

## 六、汇总：已完成 vs 待组员完成

### ✅ 已完成（本仓库交付）

- 模型选择与 WidowX 依据（三条独立证据）
- 数据下载、校验、筛选、划分（30 分片 → 717 条）
- 训练样本数精确统计（23,634）
- 两种微调训练（head_only / full），各 1000 步
- 完整 loss 记录（各 20 个点）
- Final checkpoint（step 750）打包 + 加载验证
- 训练配置、可训练参数说明
- 环境安装脚本 + 4 处代码补丁说明
- 依赖清单（含关键钉版本原因）
- 完整复现文档
- 评测脚本与冻结的初始状态清单

### ⏳ 待组员完成（评测部分）

| 项 | 文件 | 说明 |
|---|---|---|
| checkpoint 上传链接 | checkpoint_link.txt §2 | 上传两个 tar.gz 后填链接 |
| 40 episodes 结果 | results.csv | 2 模型 × 2 任务 × 10 |
| 逐 episode 记录 | episodes.csv | 含 seed、成功标记、步数、异常 |
| 2–4 个代表性视频 | video_links.md | 覆盖 T1/T2，含失败样例 |
| 案例分析与局限 | 报告 | 至少 2 个 episode 结合视频分析 |
| 组员分工 | README.md §8 | 填写姓名与分工 |
| 实验报告（≤10 页）| 独立 PDF | 汇总以上内容 |

---

## 七、自查结论

**作业 9.2 的 8 项报告内容要求中，6 项已在仓库中提供可直接引用的素材，2 项（实验结果、案例分析）属评测组员职责。**

**作业 3.1–3.4、6.2 的全部硬性要求均已在仓库中记录并可追溯。**

**唯一需要注意的**：实验报告本身有 **10 页上限**（含封面、参考文献、附录），写报告时需控制篇幅。建议正文控制在 6–7 页，参考文献与附录 2–3 页。
