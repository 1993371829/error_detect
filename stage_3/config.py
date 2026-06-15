"""
Stage 3 配置。

LLM 调用与 .env 解析直接复用 Stage 1 的 Stage1Config.resolve()，
本配置只补充 Stage 3 专有的路径与精检参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from stage_1.config import Stage1Config


@dataclass
class PathsConfig:
    """输入输出路径（默认相对项目根）。"""

    input_csv: str = "data/hospital_dirty.csv"             # 原始脏表（取整行上下文）
    clean_csv: str = "data/hospital_clean.csv"             # 评估用 ground truth（可选）
    candidates: str = "data/combined_candidates.csv"       # Stage1∪Stage2 候选
    rules: str = "data/hospital_rules.json"                # Stage1 归纳规则（取 semantic_type）
    results_out: str = "data/stage3_results.csv"           # 逐格判定明细
    final_errors_out: str = "data/final_errors.csv"        # 确认为错误的最终结果
    cache: str = ".stage3_cache.json"                      # LLM 响应缓存


@dataclass
class Stage3Config:
    """Stage 3 顶层配置。"""

    max_normal_samples: int = 8     # 每列展示给 LLM 的正常高频样例数
    limit: int = 0                  # 仅处理前 N 行分组（0=全部），便于小规模试跑
    dry_run: bool = False           # 只构造/打印 prompt，不调 LLM
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: Stage1Config = field(default=None)  # 复用 Stage1 的 LLM/.env 解析

    @classmethod
    def resolve(cls, config_path: str | Path | None = None) -> "Stage3Config":
        """加载配置：LLM 部分走 Stage1Config.resolve（.env -> yaml -> env）。"""
        cfg = cls()
        cfg.llm = Stage1Config.resolve(config_path=config_path)
        return cfg
