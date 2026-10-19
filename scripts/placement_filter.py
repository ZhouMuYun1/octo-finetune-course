"""按语言指令筛选"放置类"轨迹，用于聚焦 WidowX 的 put-on / put-in 任务。

作业要求：所用数据应与目标任务兼容，优先使用语义相关的轨迹。
目标任务是：
    T1 widowx_spoon_on_towel        = "put the spoon on the towel"
    T2 widowx_put_eggplant_in_basket = "put the eggplant in the basket"

因此保留语言指令首词属于 {put, place, move, moved, pick, take, ...} 的轨迹，
过滤掉 open / close / fold / turn / sweep / 空指令 等无关轨迹。

用法（在 Octo 的 finetune 配置里通过 ModuleSpec 引用）：

    "filter_functions": (
        ModuleSpec.create(
            "octo.data.oxe.placement_filter:is_placement_task",
        ),
    )
"""
from __future__ import annotations

import tensorflow as tf

# 与"放置 / 移动 / 抓取拿取"相关的动词（首词）
_RELATED_VERBS = tf.constant(
    [
        "put",
        "place",
        "move",
        "moved",
        "moveing",
        "moving",
        "pick",
        "take",
        "picking",
        "pickup",
        "slide",
        "remove",
        "insert",
    ]
)


def is_placement_task(traj) -> tf.Tensor:
    """tf.data.filter 的筛选函数：返回 True 表示保留该轨迹。

    参数 traj 是 dlimp 展开后的轨迹 dict，其中 language_instruction 为
    shape (T,) 的字符串张量（每个时间步一条指令）。
    """
    li = traj["language_instruction"]

    # 取第一步的指令，并压成标量
    first_text = tf.reshape(li[0], [])

    # 按空格切分，取第一个词并小写
    words = tf.strings.split(first_text, sep=" ")
    n_words = tf.size(words)

    def _first_word() -> tf.Tensor:
        return tf.strings.lower(tf.reshape(words[0], []))

    first_word = tf.cond(n_words > 0, _first_word, lambda: tf.constant(""))

    # 命中任一相关动词即保留
    return tf.reduce_any(tf.equal(first_word, _RELATED_VERBS))
