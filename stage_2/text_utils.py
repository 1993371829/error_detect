"""
文本距离工具：Levenshtein 编辑距离（带剪枝）。

原位于 stage_1/typo_detect.py（Stage 1 的 typo 检测已由 Stage 2 categorical
检测器取代并删除），保留此纯函数供 vocab_denoise / diagnose_mask 复用。
"""

from __future__ import annotations

from typing import Optional


def levenshtein(a: str, b: str, max_distance: Optional[int] = None) -> int:
    """
    计算两个字符串的 Levenshtein 编辑距离（插入/删除/替换）。

    Args:
        a, b: 待比较字符串
        max_distance: 提前剪枝阈值；若已确定距离必然超过它，提前返回该值+1。

    Returns:
        编辑距离整数。
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if max_distance is not None and abs(la - lb) > max_distance:
        return max_distance + 1

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # 删除
                cur[j - 1] + 1,     # 插入
                prev[j - 1] + cost,  # 替换
            )
            if cur[j] < row_min:
                row_min = cur[j]
        # 整行最小值已超过阈值，后续不可能更小，提前退出
        if max_distance is not None and row_min > max_distance:
            return max_distance + 1
        prev = cur
    return prev[lb]
