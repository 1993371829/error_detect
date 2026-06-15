"""
Stage 2 配置。

集中管理分布异常层（统一条件预测模型）的编码参数、模型超参、阈值与输入输出路径，
与 Stage 1 的 dataclass 风格保持一致。CLI 可覆盖关键字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EncodingConfig:
    """表格编码参数（见 stage_2/encoding.py）。"""

    max_cardinality: int = 500           # 类别列唯一值上限；<= 则 one-hot 身份保留，> 则 surrogate
    numeric_min_ratio: float = 0.8       # 判定数值列的可解析比例（单位感知）
    n_hash: int = 16                     # 高基数 surrogate 的字符 n-gram 哈希桶数（仅输入条件）
    target_max_card: Optional[int] = None  # 类别"目标头"最大类数（None=同 max_cardinality）


@dataclass
class ModelConfig:
    """统一条件预测模型超参（见 stage_2/model.py）。"""

    hidden_dim: int = 256
    epochs: int = 120
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-5
    seed: int = 0


@dataclass
class ScoringConfig:
    """打分与阈值参数。"""

    quantile: float = 0.99           # 干净单元格逐列分数高分位阈值
    margin: float = 0.0              # 类别精度闸门：备选类概率需超观测类至少该值才报
    min_predictability: float = 0.5  # 类别列在干净集上的最低 top-1 可预测性，低于则跳过该列
    max_cells_per_row: int = 0       # 每行最多保留 top-N 高分单元格（0=不限）


@dataclass
class PathsConfig:
    """输入输出路径（默认相对项目根）。"""

    input_csv: str = "data/hospital_dirty.csv"          # 待检测脏表
    clean_csv: str = "data/hospital_clean.csv"          # 评估用 ground truth（可选）
    clean_mask: str = "clean_mask.csv"                  # Stage 1 产出的干净单元格掩码
    stage1_errors: str = "data/hospital_errors.csv"     # Stage 1 错误结果
    candidates_out: str = "data/stage2_candidates.csv"  # Stage 2 DIST 候选
    combined_out: str = "data/combined_candidates.csv"  # Stage1 ∪ Stage2 合并集


@dataclass
class Stage2Config:
    """Stage 2 顶层配置。"""

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def model_kwargs(self) -> dict:
        return {k: v for k, v in vars(self.model).items()}
