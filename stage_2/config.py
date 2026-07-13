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
    # 掩码推理：预测时把 Stage 1 确认脏的单元格作上下文中性化（隔离脏上下文传播）
    masked_inference: bool = False
    # 迭代轮数：>1 时每轮把上一轮 Stage 2 候选并入脏集再中性化重打分（松绑同行互相掩护）
    masked_inference_iters: int = 1
    # 词表去污：训练前用编辑距离自过滤剔除混入类别词表的漏报 typo
    vocab_denoise: bool = False
    vocab_denoise_ratio: float = 0.05


@dataclass
class DetectorConfig:
    """Stage 2 多检测器开关与关键阈值（文档 §7-§14）。

    默认启用核心高召回检测器（reconstruction/statistical/categorical/neighbor/
    pattern/numeric_format），与文档"多检测器 + 证据融合"主线对齐。
    历史上的 association/clustering/fd 检测器已移除：前两者误报高且与
    statistical/neighbor/FD 覆盖重叠，fd 已前移至 Stage 1（双轨可信度分档验证）。
    """

    reconstruction: bool = True
    statistical: bool = True
    categorical: bool = True
    neighbor: bool = True
    pattern: bool = True
    numeric_format: bool = True
    value_burst: bool = True
    # 关键阈值
    robust_z: float = 3.0
    iqr_k: float = 1.5
    knn_k: int = 10
    sim_threshold: float = 0.85
    # pattern_outlier：形态/日期格式离群
    pattern_dominant_share: float = 0.8   # 主流形态串占比下限（日期列同样受此闸门约束）
    pattern_rare_max: int = 2             # 罕见形态串的最大计数（<= 视为离群）
    # numeric_format：数值列元数(数值 token 个数)一致性，捕捉"单值->多值/带分隔符"格式违规
    numeric_format_min_share: float = 0.9  # 主导 token 个数占比下限（保精度，放过天然多 token 列）
    # reconstruction 伪干净训练：统一纳入整行干净行 + 恰含 1 个检出错误的行(屏蔽该错误格的
    # 输入与损失)，含 >=2 检出错误的行剔除。6 数据集消融显示其在干净行稀缺时(如 hospital)显著
    # 优于纯干净训练、数据充足时(如 movies)等价，故统一默认开启，无需按数据集分档。
    # 关闭(False)则退回纯严格整行干净训练。
    reconstruction_pseudo_clean: bool = True

    def enabled(self) -> list[str]:
        names = ["reconstruction", "statistical", "categorical",
                 "neighbor", "pattern", "numeric_format", "value_burst"]
        return [n for n in names if getattr(self, n)]


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
    detectors: DetectorConfig = field(default_factory=DetectorConfig)
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

        def _pick(key: str, default: Path) -> str:
            # CLI 显式指定优先；否则按数据集默认路径（修复：此前 override 被静默忽略）
            val = ov.get(key)
            return val if val is not None else rel_path(default)

        self.paths.input_csv = rel_path(dp.dirty_csv)
        self.paths.clean_mask = _pick("clean_mask", dp.clean_mask)
        self.paths.stage1_errors = _pick("stage1_errors", dp.errors)
        self.paths.candidates_out = _pick("candidates_out", dp.stage2_candidates)
        self.paths.combined_out = _pick("combined_out", dp.combined_candidates)
        self.paths.clean_csv = _pick("clean_csv", dp.clean_csv)
        return dp
