"""
引用/元数据泄漏检测（针对文献类脏数据的字段串扰）。

某些数据集（如 rayyan）在采集时把 RIS / MEDLINE / PubMed 等文献管理格式的标签
与字段误拼接进了正常字段值，例如:

    article_pagination: "714-9 ST  - [Noninvasive prenatal diagnosis ...]-"
    author_list:        "...,\"T. CN  - PubMed DA  - Aug DP  - NLM ET  - 2013 Roszkowski\""

这类错误的形态是"外来元数据混入"，既不是缺失值、也不违反长度/类型，常被规则层
与分布层漏掉。其特征非常稳定：行内出现 RIS 标签 "XX  - "（两位大写字母/数字 tag +
连续空格 + 短横 + 空格），或出现明确的数据库来源关键词。正常自由文本几乎不含这种
形态，因此误报面极小（经多数据集验证，仅在含此类污染的列触发）。

检测命中的单元格标记为 FI（格式不一致），交由 Stage 3 进一步核验与修复。
"""

from __future__ import annotations

import re

import pandas as pd

from stage_1.profiling import is_blank

# RIS / MEDLINE 标签形态：行内 "XX  - "
#   - 两位大写字母或数字组成的 tag（如 ST/CN/DA/DP/ET/C2/C6）
#   - 紧跟 >=2 个空格 + 短横 + 空格
# 用 >=2 空格而非单空格，避免误伤正常的 "A - B"（如人名缩写、连字短语）。
RIS_TAG = re.compile(r"[A-Z][A-Z0-9]\s{2,}-\s")

# 明确的文献数据库/来源关键词（作为强信号补充）
SOURCE_KEYWORDS = re.compile(r"\b(?:PubMed|MEDLINE|NLM|Ovid|EBSCOhost|EMBASE)\b")


def looks_leaked(value: str) -> bool:
    """判断单个值是否疑似混入了 RIS/元数据标签或文献来源关键词。"""
    return bool(RIS_TAG.search(value) or SOURCE_KEYWORDS.search(value))


def detect_leakage(
    series: pd.Series,
    column: str,
    *,
    skip_numeric: bool = True,
    numeric_min_ratio: float = 0.8,
    min_len_ratio: float = 3.0,
) -> list[dict]:
    """
    在单列内检测引用/元数据泄漏。

    Args:
        series: 待检测列
        column: 列名
        skip_numeric: 跳过数值列（数值列不可能是文献元数据泄漏）
        numeric_min_ratio: 判定数值列的可解析比例
        min_len_ratio: 长度离群门控。仅当某格长度 > 列中位长度 * 该比值时才判为泄漏，
            用于把"短结构化字段被附加外来元数据"（如页码/ISSN，长度异常偏长）与
            "本就是长自由文本、元数据恰好嵌入其中"（如作者列表）区分开，后者基准
            常视为干净，标记反而引入误报。设为 0 关闭该门控。

    Returns:
        错误记录 dict 列表（error_type=FI, violated_rule=metadata_leakage）。
    """
    non_blank = series[~series.map(is_blank)].astype(str)
    if non_blank.empty:
        return []

    if skip_numeric:
        nums = pd.to_numeric(non_blank, errors="coerce").dropna()
        if len(nums) >= numeric_min_ratio * len(non_blank):
            return []

    len_threshold = 0.0
    if min_len_ratio > 0:
        len_threshold = float(non_blank.str.len().median()) * min_len_ratio

    candidates: list[dict] = []
    for row_id, value in series.items():
        if is_blank(value):
            continue
        raw = str(value)
        if not looks_leaked(raw):
            continue
        if len_threshold and len(raw) <= len_threshold:
            continue
        candidates.append({
            "row_id": row_id,
            "column": column,
            "value": raw,
            "error_type": "FI",
            "violated_rule": "metadata_leakage",
            "reason": (
                "疑似混入文献管理元数据（RIS/MEDLINE 标签或数据库来源关键词），"
                "该字段被外来内容污染"
            ),
            "confidence": 0.8,
        })
    return candidates
