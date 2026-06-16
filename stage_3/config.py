"""
Stage 3 配置。

LLM 调用与 .env 解析直接复用 Stage 1 的 Stage1Config.resolve()，
本配置只补充 Stage 3 专有的路径与精检参数。
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
from stage_1.config import Stage1Config

_DEFAULT_DP = default_dataset_paths()


@dataclass
class PathsConfig:
    """输入输出路径（默认 hospital 数据集布局）。"""

    input_csv: str = rel_path(_DEFAULT_DP.dirty_csv)
    clean_csv: str = rel_path(_DEFAULT_DP.clean_csv)
    candidates: str = rel_path(_DEFAULT_DP.combined_candidates)
    rules: str = rel_path(_DEFAULT_DP.rules)
    results_out: str = rel_path(_DEFAULT_DP.stage3_results)
    final_errors_out: str = rel_path(_DEFAULT_DP.final_errors)
    cache: str = rel_path(_DEFAULT_DP.stage3_cache)


@dataclass
class Stage3Config:
    """Stage 3 顶层配置。"""

    max_normal_samples: int = 8
    limit: int = 0
    dry_run: bool = False
    layout: OutputLayout = field(default_factory=OutputLayout)
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: Stage1Config = field(default=None)

    @classmethod
    def resolve(cls, config_path: str | Path | None = None) -> "Stage3Config":
        """加载配置：LLM 部分走 Stage1Config.resolve（.env -> yaml -> env）。"""
        cfg = cls()
        cfg.llm = Stage1Config.resolve(config_path=config_path)
        cfg.layout = cfg.llm.layout
        return cfg

    def set_paths_from_dataset(
        self,
        dirty_csv: str | Path,
        *,
        dataset: str | None = None,
        cli_overrides: Optional[dict[str, Any]] = None,
    ) -> DatasetPaths:
        """根据 --input 解析 Stage 3 依赖的全部路径。"""
        dp = resolve_dataset_paths(dirty_csv, self.layout, dataset=dataset)
        ov = cli_overrides or {}
        self.paths.input_csv = rel_path(dp.dirty_csv)
        if ov.get("candidates") is None:
            self.paths.candidates = rel_path(dp.combined_candidates)
        if ov.get("rules") is None:
            self.paths.rules = rel_path(dp.rules)
        if ov.get("results_out") is None:
            self.paths.results_out = rel_path(dp.stage3_results)
        if ov.get("final_out") is None:
            self.paths.final_errors_out = rel_path(dp.final_errors)
        if ov.get("cache") is None:
            self.paths.cache = rel_path(dp.stage3_cache)
        if ov.get("clean_csv") is None:
            self.paths.clean_csv = rel_path(dp.clean_csv)
        return dp
