"""
Stage 3 配置。

LLM 调用与 .env 解析直接复用 Stage 1 的 Stage1Config.resolve()，
本配置只补充 Stage 3 专有的路径与精检参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from paths.layout import (
    DatasetPaths,
    OutputLayout,
    default_dataset_paths,
    rel_path,
    resolve_dataset_paths,
)
from stage_1.config import DEFAULT_CONFIG_PATH, Stage1Config

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
    # 跨行共识证据 / 缺证据保护参数
    min_avg_group: float = 3.0          # 探测 key 列的"平均每取值行数"门槛
    min_dominance: float = 0.5          # 某列被视为"共识型列"的组内主导占比门槛
    min_lift: float = 0.15              # 组内占比相对全局基准占比的最小提升（排除类别不平衡伪共识）
    reject_conf_threshold: float = 0.85  # 共识冲突候选被 LLM 否决所需的最低把握
    protect_stage1_mv: bool = True       # Stage1 确定性缺失值不被 LLM 否决（确定信号，零误报回退）
    min_tier: str = "low"                # 仅精检 confidence_tier >= 该层级的候选（low=全部）
    # 性能优化（见 stage_3/verifier.py）
    max_workers: int = 8                  # LLM 并发调用线程数（受服务商 RPM/TPM 限流约束）
    enable_thinking: Optional[bool] = False  # qwen3 思考模式；False=关(省 completion token)，None=服务商默认
    dedup_context_free: bool = False      # 上下文无关格按(列,值,类型)复用判定（需消融验证召回不掉）
    layout: OutputLayout = field(default_factory=OutputLayout)
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: Stage1Config = field(default=None)

    # 可经 YAML `stage3` 段覆盖的字段（CLI 优先级更高，在 cli.py 中再覆盖）
    _YAML_KEYS = (
        "max_normal_samples", "limit", "min_avg_group", "min_dominance",
        "min_lift", "reject_conf_threshold", "protect_stage1_mv", "min_tier",
        "max_workers", "enable_thinking", "dedup_context_free",
    )

    @classmethod
    def resolve(cls, config_path: str | Path | None = None) -> "Stage3Config":
        """加载配置：LLM 部分走 Stage1Config.resolve（.env -> yaml -> env）；
        Stage 3 专有参数读取 YAML 的 `stage3` 段（缺省则用 dataclass 默认值）。"""
        cfg = cls()
        cfg.llm = Stage1Config.resolve(config_path=config_path)
        cfg.layout = cfg.llm.layout
        cfg._apply_stage3_yaml(config_path)
        return cfg

    def _apply_stage3_yaml(self, config_path: str | Path | None) -> None:
        """从 YAML 的 `stage3` 段覆盖 Stage 3 专有参数（仅覆盖显式给出的键）。"""
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            return
        section = data.get("stage3")
        if not isinstance(section, dict):
            return
        for key in self._YAML_KEYS:
            if key in section:
                setattr(self, key, section[key])

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
