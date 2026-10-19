# 代表性视频链接

作业要求：B 类需 **2–4 个代表性视频**，覆盖不同任务，**存在失败时包含失败样例**。

## 视频清单（待评测组员填写）

| 序号 | 文件名/链接 | 任务 | 模型 | episode | 结果 | 说明 |
|---:|---|---|---|---|---|---|
| 1 | `videos/T1_Base_ep*.mp4` | widowx_spoon_on_towel | Base | | | |
| 2 | `videos/T1_Final_ep*.mp4` | widowx_spoon_on_towel | Final | | | |
| 3 | `videos/T2_Base_ep*.mp4` | widowx_put_eggplant_in_basket | Base | | | |
| 4 | `videos/T2_Final_ep*.mp4` | widowx_put_eggplant_in_basket | Final | | | |

## 录制要求

| 项目 | 要求 |
|---|---|
| 分辨率 | 640×480（SimplerEnv 默认第三人称视角）|
| 帧率 | 5 fps（与 5 Hz 控制频率一致）|
| 时长 | T1 约 12 秒（60 步）；T2 约 24 秒（120 步）|
| 内容 | 完整 episode（从初始状态到终止）|
| 命名 | `T<任务号>_<模型>_ep<编号>_<结果>.mp4` |

## 视频与结果表的对应

每个视频必须能在 `episodes.csv` 中找到对应行，字段对应关系：

| 视频命名 | episodes.csv 字段 |
|---|---|
| `T1_Base_ep3_success.mp4` | 模型=Base, 任务=widowx_spoon_on_towel, episode编号=3, 成功标记=1 |
| `T2_Final_ep7_fail.mp4` | 模型=Final, 任务=widowx_put_eggplant_in_basket, episode编号=7, 成功标记=0 |

## 存放方式

两种方式任选：

1. **直接放入仓库** `videos/` 目录（建议总大小 < 100 MB）
2. **提供外部链接**（网盘/对象存储），在本文件中列出

## 报告引用

报告"案例分析与局限"章节引用视频时，请注明：
- 视频对应任务（T1/T2）
- 模型（Base/Final）
- episode 编号与结果
- 观察到的具体行为（如抓取失败、位置偏差、夹爪时机错误等）

> 若评测结果中出现失败 episode，**必须**至少包含 1 个失败视频，并在报告中对失败原因作出基于视频的观察（区分"有证据的观察"与"尚未验证的解释"）。
