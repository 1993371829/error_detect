"""
Typo 候选检测（T）。

规则层只能抓到"破坏格式"的拼写错误（如 ProviderNumber 里的字母）。
对自由文本/类别列里的拼写错误（如 City 'birminghxm' vs 'birmingham',
EmergencyService 'yxs' vs 'yes'），需要基于"列内频次 + 编辑距离"来发现:

    低频值 + 与某个高频"锚点值"编辑距离很小 + 锚点频次远高于它
    => 该低频值大概率是锚点值的拼写错误(Typo)

该方法不调用 LLM，纯统计，零成本。空值由 MV 扫描负责，这里跳过。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd

from stage_1.profiling import is_blank


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


def detect_typos(
    series: pd.Series,
    column: str,
    *,
    max_unique: int = 300,
    min_anchor_count: int = 3,
    rare_max_count: int = 2,
    anchor_ratio_min: float = 5.0,
    max_abs_distance: int = 2,
    max_norm_distance: float = 0.34,
    min_anchor_len: int = 2,
    skip_numeric: bool = True,
    numeric_min_ratio: float = 0.8,
) -> list[dict]:
    """
    在单列内检测 Typo 候选。

    判定逻辑:
        1. 统计非空值频次。
        2. 锚点(anchor): count >= min_anchor_count 的高频值（视为"正确"形态）。
        3. 候选(rare):   count <= rare_max_count 的低频值。
        4. 对每个候选，找最近锚点；若满足
               编辑距离 <= max_abs_distance
               且 距离/锚点长度 <= max_norm_distance
               且 锚点频次/候选频次 >= anchor_ratio_min
           则判为 Typo，suggested_fix = 该锚点。

    高基数列(unique_count > max_unique，如纯文本 id)直接跳过以控成本。
    若列内不存在任何锚点(全是低频/全唯一)，自然不会产生任何标记。

    Args:
        series: 待检测列
        column: 列名（用于输出）
        其余: 见上方阈值说明

    Returns:
        错误记录 dict 列表，每条含 row_id/column/value/error_type(T)/
        violated_rule/reason/suggested_fix/confidence。
    """
    non_blank = series[~series.map(is_blank)].astype(str)
    if non_blank.empty:
        return []

    # 跳过数值列：纯数字值之间编辑距离小不代表拼写错误（如不同的 ID/数值），
    # 这类错误由 FI(正则/范围) 与 VAD(函数依赖) 更可靠地覆盖。
    if skip_numeric:
        nums = pd.to_numeric(non_blank, errors="coerce").dropna()
        if len(nums) >= numeric_min_ratio * len(non_blank):
            return []

    counts = Counter(non_blank)
    if len(counts) > max_unique:
        return []

    anchors = [
        v for v, c in counts.items()
        if c >= min_anchor_count and len(v) >= min_anchor_len
    ]
    if not anchors:
        return []

    rares = [(v, c) for v, c in counts.items() if c <= rare_max_count]
    if not rares:
        return []

    # 候选 -> (最佳锚点, 距离) 缓存，避免对相同值重复计算
    best_match: dict[str, Optional[tuple[str, int]]] = {}
    for value, rare_count in rares:
        if value in best_match:
            match = best_match[value]
        else:
            match = _nearest_anchor(
                value, anchors, counts, rare_count,
                max_abs_distance=max_abs_distance,
                max_norm_distance=max_norm_distance,
                anchor_ratio_min=anchor_ratio_min,
            )
            best_match[value] = match
        if match is None:
            continue

    errors: list[dict] = []
    for row_id, value in series.items():
        if is_blank(value):
            continue
        value = str(value)
        match = best_match.get(value)
        if match is None:
            continue
        anchor, distance = match
        anchor_count = counts[anchor]
        rare_count = counts[value]
        confidence = _typo_confidence(distance, anchor, anchor_count, rare_count)
        errors.append({
            "row_id": row_id,
            "column": column,
            "value": value,
            "error_type": "T",
            "violated_rule": "typo_near_frequent",
            "reason": (
                f"疑似拼写错误: '{value}'(出现{rare_count}次) 与高频值 "
                f"'{anchor}'(出现{anchor_count}次) 编辑距离{distance}"
            ),
            "suggested_fix": anchor,
            "confidence": confidence,
        })
    return errors


def _nearest_anchor(
    value: str,
    anchors: list[str],
    counts: Counter,
    rare_count: int,
    *,
    max_abs_distance: int,
    max_norm_distance: float,
    anchor_ratio_min: float,
) -> Optional[tuple[str, int]]:
    """
    为单个候选值找唯一的近邻锚点。

    关键去误报策略：先统计与候选值编辑距离 <= max_abs_distance 的全部锚点。
    - 若有 >1 个，说明该值处于"互相相似的枚举码"邻域(如 al_scip-inf-1/2/3),
      近邻是常态而非拼写错误，返回 None（不判 typo）。
    - 若恰好 1 个，再校验 norm 距离与频次比，通过才判为 typo。
    真正的拼写错误(如 birminghxm)通常只与唯一一个正确词相近。
    """
    neighbors = []
    for anchor in anchors:
        if anchor == value:
            return None  # 自身即高频值，不是 typo
        dist = levenshtein(value, anchor, max_distance=max_abs_distance)
        if dist <= max_abs_distance:
            neighbors.append((anchor, dist))

    if len(neighbors) != 1:
        return None  # 0 个或多个近邻 -> 不可靠，跳过

    anchor, dist = neighbors[0]
    if counts[anchor] / rare_count < anchor_ratio_min:
        return None
    if dist / max(len(anchor), 1) > max_norm_distance:
        return None
    return (anchor, dist)


def _typo_confidence(
    distance: int,
    anchor: str,
    anchor_count: int,
    rare_count: int,
) -> float:
    """
    依据编辑距离与频次比给出置信度(0~1)。

    距离越小、锚点相对越高频，置信度越高。
    """
    dist_score = 1.0 - (distance - 1) / max(len(anchor), 1)
    ratio = anchor_count / rare_count
    freq_score = min(1.0, ratio / 20.0)
    conf = 0.6 * dist_score + 0.4 * freq_score
    return round(max(0.0, min(1.0, conf)), 3)


def detect_typos_dataframe(df: pd.DataFrame, **kwargs) -> list[dict]:
    """对 DataFrame 每列执行 Typo 检测并汇总结果。"""
    errors: list[dict] = []
    for col in df.columns:
        errors.extend(detect_typos(df[col], str(col), **kwargs))
    return errors
