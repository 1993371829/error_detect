"""
Step 1: 数据画像（Data Profiling）。

纯统计方法，不调用 LLM。为每列生成结构化画像，供后续 LLM 规则归纳使用。

画像字段:
    - 空值率、唯一值数量、样例值
    - 长度分布、字符模式分布 (observed_patterns)
    - 数值统计 (numeric_stats，仅当 >=80% 值可解析为数字时)
    - charset_info: 全列字符集标志与 special_chars 列表（含脏数据噪声，guard 不直接使用）
    - edge_samples: 含特殊字符 / 含空格多词 / 最长 / 最短的边缘样本
    - pattern_coverage: 模式种类数与 top5 覆盖率
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Union

import pandas as pd


def to_pattern(s: str) -> str:
    """
    将具体值抽象为字符模式，便于发现格式规律。

    映射规则: 数字->d, 大写->L, 小写->l
    示例: '90210' -> 'ddddd', 'TX' -> 'LL', '2021-01-01' -> 'dddd-dd-dd'
    """
    s = re.sub(r"\d", "d", s)
    s = re.sub(r"[A-Z]", "L", s)
    s = re.sub(r"[a-z]", "l", s)
    return s


MISSING_SENTINELS = {"", "empty", "null", "n/a", "na", "none", "-"}


def is_blank(v) -> bool:
    """判断单元格是否为空（NaN、纯空白、或字面量 empty 等缺失哨兵）。"""
    if pd.isna(v):
        return True
    return str(v).strip().lower() in MISSING_SENTINELS


EDGE_SPECIAL_SAMPLES = 5
EDGE_SPACE_SAMPLES = 3


def _has_special_char(s: str) -> bool:
    return any(not c.isalnum() and not c.isspace() for c in s)


def profile_column(series: pd.Series, max_samples: int = 20) -> dict:
    """
    为单列生成紧凑画像 dict。

    Args:
        series: 待画像的 pandas 列
        max_samples: 样例值上限（按频次降序取 top-N）

    Returns:
        包含 name, null_rate, sample_values, length_stats,
        observed_patterns, numeric_stats, charset_info,
        edge_samples, pattern_coverage 等字段的字典
    """
    total = len(series)
    blank_mask = series.map(is_blank)
    null_count = int(blank_mask.sum())

    non_null = series[~blank_mask].astype(str)

    if len(non_null) == 0:
        length_stats = {"min": 0, "max": 0, "most_common_len": []}
        patterns = []
        numeric_stats = None
        charset_info = {}
        edge_samples = []
        pattern_coverage = {}
    else:
        lengths = non_null.str.len()
        length_stats = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "most_common_len": Counter(lengths).most_common(3),
        }
        patterns = Counter(non_null.map(to_pattern)).most_common(5)

        nums = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(nums) >= 0.8 * len(non_null) and len(nums) > 0:
            numeric_stats = {
                "min": float(nums.min()),
                "max": float(nums.max()),
                "mean": round(float(nums.mean()), 2),
            }
        else:
            numeric_stats = None

        all_chars = set()
        for v in non_null:
            all_chars.update(v)
        charset_info = {
            "has_uppercase": any(c.isupper() for c in all_chars),
            "has_lowercase": any(c.islower() for c in all_chars),
            "has_digit": any(c.isdigit() for c in all_chars),
            "has_space": " " in all_chars,
            "special_chars": sorted(
                c for c in all_chars if not c.isalnum() and not c.isspace()
            ),
        }

        unique_vals = list(dict.fromkeys(non_null))
        with_special = [v for v in unique_vals if _has_special_char(v)][:EDGE_SPECIAL_SAMPLES]
        with_space = [v for v in unique_vals if " " in str(v).strip()][:EDGE_SPACE_SAMPLES]
        longest = non_null.loc[non_null.str.len().idxmax()]
        shortest = non_null.loc[non_null.str.len().idxmin()]
        edge_samples = list(dict.fromkeys(
            with_special + with_space + [longest, shortest]
        ))

        all_patterns = Counter(non_null.map(to_pattern))
        top5_cover = sum(c for _, c in all_patterns.most_common(5))
        pattern_coverage = {
            "distinct_pattern_count": len(all_patterns),
            "top5_coverage_rate": round(top5_cover / len(non_null), 3),
        }

    return {
        "name": str(series.name),                         # 列名
        "total_count": total,                             # 总行数
        "null_count": null_count,                         # 空值数
        "null_rate": round(null_count / total, 3) if total else 0,  # 空值率
        "unique_count": int(non_null.nunique()),          # 唯一值个数
        "sample_values": non_null.value_counts().head(max_samples).index.tolist(),  # 样本值（频率最高的部分值）
        "length_stats": length_stats,                     # 字符串长度统计（min/max/常见长度）
        "observed_patterns": patterns,                    # 观察到的模式及出现次数
        "numeric_stats": numeric_stats,                   # 数值统计（仅限纯数字列，含 min/max/mean）
        "charset_info": charset_info,                     # 字符集信息（有无大小写、数字、特殊符号等）
        "edge_samples": edge_samples,                     # 边缘样本（特殊字符/含空格多词/最长/最短）
        "pattern_coverage": pattern_coverage,             # 模式覆盖度（模式总数及Top5覆盖率）
    }


def profile_dataframe(
    df: pd.DataFrame,
    max_samples: int = 20,
) -> list[dict]:
    """为 DataFrame 每列生成画像列表。"""
    return [
        profile_column(df[col], max_samples=max_samples)
        for col in df.columns
    ]


def save_profiles(
    profiles: list[dict],
    path: Union[str, Path],
) -> Path:
    """将列画像写入 JSON 文件。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2, default=str)
    return out
