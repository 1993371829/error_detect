"""
Stage 2 配置。

集中管理分布异常层（统一条件预测模型）的编码参数、模型超参、阈值与输入输出路径，
与 Stage 1 的 dataclass 风格保持一致。CLI 可覆盖关键字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from paths.layout import (
    DatasetPaths,
    OutputLayout,
    default_dataset_paths,
    rel_path,
    resolve_dataset_paths,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DP = default_dataset_paths()


@dataclass
class EncodingConfig:
    """表格编码参数（见 stage_2/encoding.py）。"""

    max_cardinality: int = 500
    numeric_min_ratio: float = 0.8
    n_hash: int = 16
    target_max_card: Optional[int] = None


@dataclass
class ModelConfig:
    """统一条件预测模型超参（见 stage_2/model.py）。"""

    hidden_dim: int = 256
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-5
    seed: int = 0
    device: str = "auto"  # 计算设备：auto（有 GPU 自动用）/ cuda / cpu


@dataclass
class ScoringConfig:
    """打分与阈值参数。"""

    quantile: float = 0.99
    margin: float = 0.0
    min_predictability: float = 0.5
    abs_prob_floor: float = 0.02
    max_cells_per_row: int = 0


@dataclass
class PathsConfig:
    """输入输出路径（默认 hospital 数据集布局）。"""

    input_csv: str = rel_path(_DEFAULT_DP.dirty_csv)
    clean_csv: str = rel_path(_DEFAULT_DP.clean_csv)
    clean_mask: str = rel_path(_DEFAULT_DP.clean_mask)
    stage1_errors: str = rel_path(_DEFAULT_DP.errors)
    candidates_out: str = rel_path(_DEFAULT_DP.stage2_candidates)
    combined_out: str = rel_path(_DEFAULT_DP.combined_candidates)


@dataclass
class Stage2Config:
    """Stage 2 顶层配置。"""

    layout: OutputLayout = field(default_factory=OutputLayout)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def model_kwargs(self) -> dict:
        return {k: v for k, v in vars(self.model).items()}

    def set_paths_from_dataset(
        self,
        dirty_csv: str | Path,
        *,
        dataset: str | None = None,
        cli_overrides: Optional[dict[str, Any]] = None,
    ) -> DatasetPaths:
        """根据 --input 解析 Stage 2 依赖的全部路径。"""
        dp = resolve_dataset_paths(dirty_csv, self.layout, dataset=dataset)
        ov = cli_overrides or {}
        self.paths.input_csv = rel_path(dp.dirty_csv)
        if ov.get("clean_mask") is None:
            self.paths.clean_mask = rel_path(dp.clean_mask)
        if ov.get("stage1_errors") is None:
            self.paths.stage1_errors = rel_path(dp.errors)
        if ov.get("candidates_out") is None:
            self.paths.candidates_out = rel_path(dp.stage2_candidates)
        if ov.get("combined_out") is None:
            self.paths.combined_out = rel_path(dp.combined_candidates)
        if ov.get("clean_csv") is None:
            self.paths.clean_csv = rel_path(dp.clean_csv)
        return dp
