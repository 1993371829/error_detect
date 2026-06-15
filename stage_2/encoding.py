"""
Stage 2 数据编码：表格 DataFrame -> 数值张量（numpy），并保留"列 -> 张量切片"映射。

角色分工优化后的编码策略（见 stage_2/DESIGN.md）:
    - 数值列：鲁棒归一化（median / IQR）+ is_null 指示位。
    - 低基数类别列：one-hot + __UNK__ 维 + 频率特征 + is_null 位。
        * __UNK__：未知/脏值点亮该位，训练分布几乎不出现 -> 推理时产生明显误差。
        * 频率特征：低频取值 = 潜在拼写错误信号。
    - 高基数文本列（如 HospitalName/Address/MeasureName）：不再直接丢弃，
        改用廉价 surrogate 特征（长度/字符比例/频次/是否在干净集/n-gram hash）。
    - ID 列（整数且高基数，如 ProviderNumber/ZipCode/PhoneNumber）：不当数值处理，
        改走 surrogate 通道，避免把 ID 当成量纲产生伪离群。

每个特征维度带有 role（numeric / binary / softmax），供模型施加混合输出头与按列损失。
编码器保存"列 -> 特征切片"映射，使重构误差可还原回原始列以做单元格级定位。
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# 视为空/缺失的哨兵（hospital 用 "empty" 表示缺失）
_BLANK_TOKENS = {"", "nan", "none", "null", "na", "n/a", "empty", "?", "-", "--"}

# 带单位的数值：整串必须是"数字 + 可选 %/单一单位词"，避免把 "1720 university blvd" 误判为数值
_UNIT_NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*%?\s*[a-zA-Z]*\s*$")
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _extract_number(val: str) -> float:
    """从 '97%' / '33 patients' / '100' 提取数值；纯文本含数字（如地址）返回 NaN。"""
    if not _UNIT_NUM_RE.match(val):
        return float("nan")
    m = _NUM_RE.search(val)
    return float(m.group()) if m else float("nan")


def _parse_numeric_unit(non_null: pd.Series) -> pd.Series:
    """对一列非空值做单位感知数值解析，返回 float Series（无法解析为 NaN）。"""
    return non_null.map(_extract_number)


def _is_blank(value) -> bool:
    """统一空值判定：NaN / 空串 / 常见缺失哨兵。"""
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _BLANK_TOKENS


def _blank_mask(series: pd.Series) -> np.ndarray:
    return series.map(_is_blank).to_numpy()


@dataclass
class FeatureBlock:
    """列内一个特征块（连续若干维）及其在模型中的角色。"""

    role: str          # "numeric" | "binary" | "softmax"
    offset: int        # 相对该列 start 的偏移
    width: int


@dataclass
class FeatureSpec:
    """全特征矩阵的角色索引，供模型施加混合头与按列损失/遮蔽。"""

    n_features: int
    numeric_idx: list                      # 线性输出 + MSE
    binary_idx: list                       # sigmoid 输出 + BCE
    softmax_groups: list                   # [(start, end)]：softmax 输出 + CE
    column_slices: dict                    # col -> (start, end)，按列遮蔽/还原用


@dataclass
class ColumnEncoding:
    """单列的编码元信息，用于 transform 与误差还原。"""

    name: str
    kind: str                              # numeric|categorical|highcard|dropped
    start: int = 0
    width: int = 0
    blocks: list = field(default_factory=list)        # list[FeatureBlock]
    # numeric 参数
    median: float = 0.0
    iqr: float = 1.0
    # categorical 参数
    categories: list = field(default_factory=list)    # 不含 __UNK__
    category_index: dict = field(default_factory=dict)
    # 频率特征：value -> 归一化 log 频次
    freq_map: dict = field(default_factory=dict)
    # 高基数 surrogate 参数
    clean_values: set = field(default_factory=set)
    len_med: float = 0.0
    len_iqr: float = 1.0
    n_hash: int = 0


class TabularEncoder:
    """
    表格编码器：fit 学习每列编码方案，transform 生成特征矩阵。

    Args:
        max_cardinality: 类别列唯一值上限，超过则按高基数列走 surrogate 通道。
        numeric_min_ratio: 列中可解析为数字的比例下限，达到则视为数值列。
        n_hash: 高基数 surrogate 的字符 n-gram 哈希桶数（0=关闭）。
        id_min_len: ID 列启发式：整数中位串长 >= 此值且高基数则按 ID（surrogate）处理。
    """

    def __init__(
        self,
        max_cardinality: int = 50,
        numeric_min_ratio: float = 0.8,
        n_hash: int = 16,
        id_min_len: int = 4,
    ):
        self.max_cardinality = max_cardinality
        self.numeric_min_ratio = numeric_min_ratio
        self.n_hash = n_hash
        self.id_min_len = id_min_len
        self.encodings: dict[str, ColumnEncoding] = {}
        self.n_features: int = 0
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, clean_mask: Optional[pd.DataFrame] = None) -> "TabularEncoder":
        """
        学习编码方案（仅基于干净单元格估参）。

        Args:
            df: 原始表格（建议 dtype=str）。
            clean_mask: 与 df 同形状的布尔表，True=干净。
        """
        cursor = 0
        for col in df.columns:
            series = df[col]
            if clean_mask is not None and col in clean_mask.columns:
                series = series[clean_mask[col].astype(bool)]
            enc = self._fit_column(str(col), series)
            enc.start = cursor
            cursor += enc.width
            self.encodings[str(col)] = enc
        self.n_features = cursor
        self._fitted = True
        return self

    def _fit_column(self, name: str, series: pd.Series) -> ColumnEncoding:
        """决定单列类型并估计编码参数。"""
        non_null = series[~_blank_mask(series)].astype(str)
        if len(non_null) == 0:
            return ColumnEncoding(name=name, kind="dropped", width=0)

        lengths = non_null.str.len()
        nunique = non_null.nunique()

        # ID 列启发式：纯整数 + 串长稳定且较长（如 ProviderNumber/ZipCode/PhoneNumber）。
        # ID 不当数值量纲处理（避免 20018 vs 10018 这类伪离群），改走类别/UNK 或 surrogate。
        plain = pd.to_numeric(non_null, errors="coerce")
        is_id = bool(
            plain.notna().mean() >= self.numeric_min_ratio
            and (plain.dropna() % 1 == 0).all()
            and float(lengths.median()) >= self.id_min_len
            and float(lengths.std() or 0.0) <= 1.0
        )

        # 数值列（单位感知：97% / 33 patients 视为数值；含数字的地址不算）
        if not is_id:
            parsed = _parse_numeric_unit(non_null)
            if parsed.notna().mean() >= self.numeric_min_ratio:
                vals = parsed.dropna()
                median = float(vals.median())
                q1, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
                iqr = (q3 - q1) or (float(vals.std()) or 1.0)
                enc = ColumnEncoding(name=name, kind="numeric", median=median, iqr=iqr)
                enc.blocks = [FeatureBlock("numeric", 0, 1), FeatureBlock("binary", 1, 1)]
                enc.width = 2
                return enc

        # 低基数（含 ID）类别列：one-hot + __UNK__ + freq + is_null
        if nunique <= self.max_cardinality:
            return self._fit_categorical(name, non_null)

        # 高基数文本/ID 列：surrogate 特征
        return self._fit_highcard(name, non_null)

    def _fit_categorical(self, name: str, non_null: pd.Series) -> ColumnEncoding:
        categories = sorted(non_null.unique().tolist())
        index = {cat: i for i, cat in enumerate(categories)}
        n_cat = len(categories)
        enc = ColumnEncoding(
            name=name, kind="categorical",
            categories=categories, category_index=index,
            freq_map=self._freq_map(non_null),
        )
        # 布局: [one-hot(含UNK) softmax] [freq numeric] [is_null binary]
        softmax_w = n_cat + 1  # +1 for __UNK__
        enc.blocks = [
            FeatureBlock("softmax", 0, softmax_w),
            FeatureBlock("numeric", softmax_w, 1),
            FeatureBlock("binary", softmax_w + 1, 1),
        ]
        enc.width = softmax_w + 2
        return enc

    def _fit_highcard(self, name: str, non_null: pd.Series) -> ColumnEncoding:
        lengths = non_null.str.len()
        enc = ColumnEncoding(
            name=name, kind="highcard",
            clean_values=set(non_null.tolist()),
            freq_map=self._freq_map(non_null),
            len_med=float(lengths.median()),
            len_iqr=float((lengths.quantile(0.75) - lengths.quantile(0.25)) or 1.0),
            n_hash=self.n_hash,
        )
        # 布局: 6 numeric(len/digit/alpha/punct/space/freq) + 2 binary(null/in_clean)
        #       + n_hash numeric(字符 3-gram 哈希分布)
        blocks = [
            FeatureBlock("numeric", 0, 6),
            FeatureBlock("binary", 6, 2),
        ]
        width = 8
        if self.n_hash > 0:
            blocks.append(FeatureBlock("numeric", width, self.n_hash))
            width += self.n_hash
        enc.blocks = blocks
        enc.width = width
        return enc

    @staticmethod
    def _freq_map(non_null: pd.Series) -> dict:
        """value -> 归一化 log 频次（[0,1]，越小越罕见）。"""
        counts = non_null.value_counts()
        log_counts = np.log1p(counts.to_numpy(dtype=np.float64))
        denom = float(log_counts.max()) or 1.0
        return {val: float(lc / denom) for val, lc in zip(counts.index, log_counts)}

    # -------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """将表格编码为 (n_rows, n_features) 的 float 矩阵。"""
        if not self._fitted:
            raise RuntimeError("TabularEncoder 未 fit，先调用 fit()。")
        n = len(df)
        matrix = np.zeros((n, self.n_features), dtype=np.float32)
        for col, enc in self.encodings.items():
            if enc.width == 0:
                continue
            block = self._transform_column(df[col], enc)
            matrix[:, enc.start:enc.start + enc.width] = block
        return matrix

    def fit_transform(
        self, df: pd.DataFrame, clean_mask: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        return self.fit(df, clean_mask=clean_mask).transform(df)

    def _transform_column(self, series: pd.Series, enc: ColumnEncoding) -> np.ndarray:
        n = len(series)
        out = np.zeros((n, enc.width), dtype=np.float32)
        blank = _blank_mask(series)
        if enc.kind == "numeric":
            nums = series.astype(str).map(_extract_number).to_numpy(dtype=np.float64)
            filled = np.where(np.isnan(nums), enc.median, nums)
            out[:, 0] = (filled - enc.median) / enc.iqr
            out[:, 1] = blank.astype(np.float32)
        elif enc.kind == "categorical":
            self._fill_categorical(series, enc, out, blank)
        elif enc.kind == "highcard":
            self._fill_highcard(series, enc, out, blank)
        return out

    def _fill_categorical(self, series, enc, out, blank) -> None:
        n_cat = len(enc.categories)
        unk_pos = n_cat  # __UNK__ 在 softmax 段末尾
        freq_pos = n_cat + 1
        null_pos = n_cat + 2
        vals = series.astype(str).tolist()
        for i, val in enumerate(vals):
            if blank[i]:
                out[i, null_pos] = 1.0
                out[i, unk_pos] = 1.0  # 缺失也视作非已知类别
                continue
            j = enc.category_index.get(val)
            if j is not None:
                out[i, j] = 1.0
            else:
                out[i, unk_pos] = 1.0
            out[i, freq_pos] = enc.freq_map.get(val, 0.0)

    def _fill_highcard(self, series, enc, out, blank) -> None:
        null_pos, in_clean_pos = 6, 7
        hash_start = 8
        vals = series.astype(str).tolist()
        for i, val in enumerate(vals):
            if blank[i]:
                out[i, null_pos] = 1.0
                continue
            length = len(val)
            digits = sum(c.isdigit() for c in val)
            alphas = sum(c.isalpha() for c in val)
            puncts = sum((not c.isalnum()) and (not c.isspace()) for c in val)
            spaces = val.count(" ")
            denom = length or 1
            out[i, 0] = (length - enc.len_med) / enc.len_iqr
            out[i, 1] = digits / denom
            out[i, 2] = alphas / denom
            out[i, 3] = puncts / denom
            out[i, 4] = spaces / denom
            out[i, 5] = enc.freq_map.get(val, 0.0)
            out[i, in_clean_pos] = 1.0 if val in enc.clean_values else 0.0
            if enc.n_hash > 0:
                out[i, hash_start:hash_start + enc.n_hash] = _hash_ngrams(val, enc.n_hash)

    # ------------------------------------------------------------- 误差还原
    def feature_slice(self, column: str) -> tuple[int, int]:
        enc = self.encodings[column]
        return enc.start, enc.start + enc.width

    def feature_spec(self) -> FeatureSpec:
        """汇总各特征维度的角色索引，供模型使用。"""
        numeric_idx: list[int] = []
        binary_idx: list[int] = []
        softmax_groups: list[tuple[int, int]] = []
        column_slices: dict[str, tuple[int, int]] = {}
        for col, enc in self.encodings.items():
            if enc.width == 0:
                continue
            column_slices[col] = (enc.start, enc.start + enc.width)
            for blk in enc.blocks:
                s = enc.start + blk.offset
                e = s + blk.width
                if blk.role == "numeric":
                    numeric_idx.extend(range(s, e))
                elif blk.role == "binary":
                    binary_idx.extend(range(s, e))
                elif blk.role == "softmax":
                    softmax_groups.append((s, e))
        return FeatureSpec(
            n_features=self.n_features,
            numeric_idx=numeric_idx,
            binary_idx=binary_idx,
            softmax_groups=softmax_groups,
            column_slices=column_slices,
        )

    def aggregate_feature_errors(self, per_feature_error: np.ndarray) -> dict[str, np.ndarray]:
        """
        把 per-feature 误差(n_rows, n_features) 聚合回每列(n_rows,)。

        逐列求和；不同列量纲差异由 score 阶段的按列鲁棒归一化处理。
        """
        out: dict[str, np.ndarray] = {}
        for col, enc in self.encodings.items():
            if enc.width == 0:
                continue
            start, end = enc.start, enc.start + enc.width
            out[col] = per_feature_error[:, start:end].sum(axis=1)
        return out

    def active_columns(self) -> list[str]:
        return [c for c, e in self.encodings.items() if e.width > 0]


def _hash_ngrams(value: str, n_hash: int, n: int = 3) -> np.ndarray:
    """字符 n-gram 哈希分布（确定性 crc32），返回归一化计数向量。"""
    vec = np.zeros(n_hash, dtype=np.float32)
    grams = [value[i:i + n] for i in range(len(value) - n + 1)] or [value]
    for g in grams:
        bucket = zlib.crc32(g.encode("utf-8")) % n_hash
        vec[bucket] += 1.0
    total = float(vec.sum()) or 1.0
    return vec / total
