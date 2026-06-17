"""
Stage 1 全局配置模块。

配置加载优先级（高 -> 低）:
    1. CLI 参数 (--output, --rules-out, --max-violation-rate 等)
    2. 环境变量 (LLM_API_KEY / LLM_BASE_URL / LLM_MODEL)
    3. YAML 配置文件 (configs/default.yaml)
    4. dataclass 内置默认值

使用 Stage1Config.resolve() 一次性完成上述合并。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from paths.layout import (
    OutputLayout,
    DatasetPaths,
    default_dataset_paths,
    ensure_output_dirs,
    rel_path,
    resolve_dataset_paths,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class LLMConfig:
    """LLM 调用相关配置，兼容 OpenAI 接口（可通过 base_url 切换服务商）。"""

    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0  # 规则归纳需要稳定输出，固定为 0
    max_tokens: int = 4096    # 输出上限，防高基数自由文本列 JSON 被截断（0=不显式传）
    max_retries: int = 1      # JSON 解析/接口调用失败时的额外重试次数


@dataclass
class ProfilingConfig:
    """数据画像阶段的参数。"""

    max_samples: int = 20  # 每列最多保留的样例值数量


@dataclass
class GuardConfig:
    """自由文本列 regex 兜底校验参数（结构化列跳过 guard）。"""

    enabled: bool = True
    top5_coverage_structured_min: float = 0.5
    distinct_pattern_structured_max: int = 30
    dominant_pattern_rate_min: float = 0.7


@dataclass
class TypoConfig:
    """类别/文本列 Typo 候选检测参数（基于频次 + 编辑距离）。"""

    enabled: bool = True
    max_unique: int = 300            # 列唯一值超过此数视为高基数 id，跳过
    min_anchor_count: int = 3        # 锚点(正确形态)的最小出现次数
    rare_max_count: int = 2          # 候选(疑似 typo)的最大出现次数
    anchor_ratio_min: float = 5.0    # 锚点频次 / 候选频次 的最小比值
    max_abs_distance: int = 2        # 最大编辑距离
    max_norm_distance: float = 0.34  # 编辑距离 / 锚点长度 的上限
    min_anchor_len: int = 2          # 锚点最小长度，过短不可靠
    skip_numeric: bool = True        # 跳过数值列（typo 由 FI/VAD 覆盖）
    numeric_min_ratio: float = 0.8   # 判定数值列的可解析比例


@dataclass
class DMVConfig:
    """伪缺失值（Disguised Missing Value）检测参数（借鉴 Cocoon）。"""

    enabled: bool = True
    extra_tokens: list[str] = field(default_factory=list)  # 默认词表外追加的伪缺失 token
    detect_numeric_placeholder: bool = False  # 是否检测占位数字（默认关闭，控误报）
    numeric_placeholders: list[str] = field(default_factory=list)  # 自定义占位数字


@dataclass
class StandardizeConfig:
    """不一致表示 / 标准化检测参数（借鉴 Cocoon String Outliers，捕获多数派格式/单位不一致）。"""

    enabled: bool = True
    sample_n: int = 50               # 发给 LLM 的每列 top-N 高频 distinct 值
    max_distinct_ratio: float = 0.6  # distinct/非空 超过此值的列视为自由文本/标识列，跳过
    max_flag_rate: float = 0.95      # 命中率上限闸门，超过判定规范形态选错，整列跳过


@dataclass
class FDConfig:
    """近似函数依赖挖掘参数（用于检测 VAD）。"""

    enabled: bool = True
    min_confidence: float = 0.9              # FD 全局一致率下限
    min_group_support: int = 5               # 参与违反判定的 A 组最小样本量
    min_group_confidence: float = 0.9        # A 组内主导值占比下限
    pure_threshold: float = 0.9              # "纯组"的主导占比门槛
    min_pure_group_ratio: float = 0.85       # 纯组比例下限（区分真依赖 vs 软相关）
    min_distinct_dependents: int = 2         # 不同主导值数下限（防类别不平衡假依赖）
    max_determinant_unique_ratio: float = 0.5  # 决定列唯一率上限（过高跳过）
    min_dependent_unique: int = 2            # 依赖列最少唯一值（排除常量列）
    semantic_check: bool = True              # 统计候选 FD 后用 LLM 做语义校验（借鉴 Cocoon）


@dataclass
class ExecutionConfig:
    """规则执行阶段的参数。"""

    max_violation_rate: float = 0.3  # 误报控制：违反率超过此阈值的规则会被丢弃
    nullable_columns: list[str] = field(default_factory=list)  # 显式跳过 MV 扫描的列
    infer_nullable: bool = False  # 是否根据 LLM 无 not_null + 高空值率推断可空列
    infer_nullable_min_rate: float = 0.5  # infer_nullable 生效时的空值率下限
    enable_type_check: bool = True  # 启用列逻辑类型一致性校验（bool/int/float/date，借鉴 Cocoon）
    guard: GuardConfig = field(default_factory=GuardConfig)
    typo: TypoConfig = field(default_factory=TypoConfig)
    dmv: DMVConfig = field(default_factory=DMVConfig)
    standardize: StandardizeConfig = field(default_factory=StandardizeConfig)
    fd: FDConfig = field(default_factory=FDConfig)


_DEFAULT_DP = default_dataset_paths()


@dataclass
class PathsConfig:
    """输入输出路径配置（默认按 hospital 数据集布局；CLI 可按 --input 重算）。"""

    rule_cache: str = rel_path(_DEFAULT_DP.rule_cache)
    rules_output: str = rel_path(_DEFAULT_DP.rules)
    errors_output: str = rel_path(_DEFAULT_DP.errors)
    profiles_output: str = rel_path(_DEFAULT_DP.profiles)
    clean_mask_output: str = rel_path(_DEFAULT_DP.clean_mask)


@dataclass
class Stage1Config:
    """Stage 1 顶层配置，聚合各子配置段。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    layout: OutputLayout = field(default_factory=OutputLayout)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage1Config":
        """从 YAML 文件加载配置。"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Stage1Config":
        """将嵌套 dict 映射到 dataclass 字段，忽略未知键。"""
        cfg = cls()
        if llm := data.get("llm"):
            if isinstance(llm, dict):
                for key, val in llm.items():
                    if hasattr(cfg.llm, key):
                        setattr(cfg.llm, key, val)
        if profiling := data.get("profiling"):
            if isinstance(profiling, dict):
                for key, val in profiling.items():
                    if hasattr(cfg.profiling, key):
                        setattr(cfg.profiling, key, val)
        if execution := data.get("execution"):
            if isinstance(execution, dict):
                for key, val in execution.items():
                    if key == "guard" and isinstance(val, dict):
                        for gkey, gval in val.items():
                            if hasattr(cfg.execution.guard, gkey):
                                setattr(cfg.execution.guard, gkey, gval)
                    elif key == "typo" and isinstance(val, dict):
                        for tkey, tval in val.items():
                            if hasattr(cfg.execution.typo, tkey):
                                setattr(cfg.execution.typo, tkey, tval)
                    elif key == "dmv" and isinstance(val, dict):
                        for dkey, dval in val.items():
                            if hasattr(cfg.execution.dmv, dkey):
                                setattr(cfg.execution.dmv, dkey, dval)
                    elif key == "standardize" and isinstance(val, dict):
                        for skey, sval in val.items():
                            if hasattr(cfg.execution.standardize, skey):
                                setattr(cfg.execution.standardize, skey, sval)
                    elif key == "fd" and isinstance(val, dict):
                        for fkey, fval in val.items():
                            if hasattr(cfg.execution.fd, fkey):
                                setattr(cfg.execution.fd, fkey, fval)
                    elif hasattr(cfg.execution, key):
                        setattr(cfg.execution, key, val)
        if paths := data.get("paths"):
            if isinstance(paths, dict):
                for key, val in paths.items():
                    if hasattr(cfg.paths, key):
                        setattr(cfg.paths, key, val)
        if layout := data.get("layout"):
            cfg.layout = OutputLayout.from_dict(layout)
        return cfg

    def apply_env_overrides(self) -> "Stage1Config":
        """用环境变量覆盖 LLM 相关配置（便于本地 .env 管理密钥）。"""
        if api_key := os.environ.get("LLM_API_KEY"):
            self.llm.api_key = api_key
        if base_url := os.environ.get("LLM_BASE_URL"):
            self.llm.base_url = base_url
        if model := os.environ.get("LLM_MODEL"):
            self.llm.model = model
        return self

    def apply_cli_overrides(self, overrides: Dict[str, Any]) -> "Stage1Config":
        """用 CLI 参数覆盖路径和执行参数（None 表示未指定，不覆盖）。"""
        if overrides.get("output") is not None:
            self.paths.errors_output = overrides["output"]
        if overrides.get("rules_out") is not None:
            self.paths.rules_output = overrides["rules_out"]
        if overrides.get("max_violation_rate") is not None:
            self.execution.max_violation_rate = overrides["max_violation_rate"]
        if overrides.get("rule_cache") is not None:
            self.paths.rule_cache = overrides["rule_cache"]
        if overrides.get("profiles_out") is not None:
            self.paths.profiles_output = overrides["profiles_out"]
        if overrides.get("clean_mask_out") is not None:
            self.paths.clean_mask_output = overrides["clean_mask_out"]
        return self

    def set_paths_from_dataset(
        self,
        dirty_csv: str | Path,
        *,
        dataset: str | None = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> DatasetPaths:
        """根据 --input 脏表路径解析并写入标准输出路径（CLI 显式指定的项不覆盖）。"""
        dp = resolve_dataset_paths(dirty_csv, self.layout, dataset=dataset)
        ov = cli_overrides or {}
        if ov.get("output") is None:
            self.paths.errors_output = rel_path(dp.errors)
        if ov.get("rules_out") is None:
            self.paths.rules_output = rel_path(dp.rules)
        if ov.get("profiles_out") is None:
            self.paths.profiles_output = rel_path(dp.profiles)
        if ov.get("clean_mask_out") is None:
            self.paths.clean_mask_output = rel_path(dp.clean_mask)
        if ov.get("rule_cache") is None:
            self.paths.rule_cache = rel_path(dp.rule_cache)
        return dp

    @classmethod
    def resolve(
        cls,
        config_path: Optional[str | Path] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> "Stage1Config":
        """
        统一配置入口：加载 .env -> YAML -> 环境变量 -> CLI 覆盖。

        Args:
            config_path: YAML 配置文件路径，默认 configs/default.yaml
            cli_overrides: 来自 argparse 的覆盖字典
        """
        load_dotenv(_PROJECT_ROOT / ".env")
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if path.exists():
            cfg = cls.from_yaml(path)
        else:
            cfg = cls()
        cfg.apply_env_overrides()
        if cli_overrides:
            cfg.apply_cli_overrides(cli_overrides)
        return cfg
