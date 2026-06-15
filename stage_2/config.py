"""
Stage 2 配置。

集中管理分布异常层的编码参数、模型超参、阈值与输入输出路径，
与 Stage 1 的 dataclass 风格保持一致。CLI 可覆盖关键字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EncodingConfig:
    """表格编码参数（见 stage_2/encoding.py）。"""

    max_cardinality: int = 50        # 类别列唯一值上限，超过按高基数列走 surrogate
    numeric_min_ratio: float = 0.8   # 判定数值列的可解析比例（单位感知）
    n_hash: int = 16                 # 高基数 surrogate 的字符 n-gram 哈希桶数（0=关闭）


@dataclass
class DAEConfig:
    """去噪自编码器超参（单元格级专家）。"""

    hidden_dim: int = 128
    latent_dim: int = 32
    corruption_rate: float = 0.2     # 训练时每行被遮蔽的"列"比例
    swap_rate: float = 0.5           # 被遮蔽列中改用 swap-noise 的比例，否则置零
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 128
    seed: int = 0


@dataclass
class GANomalyConfig:
    """GANomaly 超参（行级 / 多列联合专家）。"""

    hidden_dim: int = 128
    latent_dim: int = 32
    w_recon: float = 10.0            # 降权（原 50 过高使其退化为 AE）
    w_latent: float = 1.0
    w_adv: float = 1.0
    w_fm: float = 1.0                # feature-matching 权重（稳定 GAN）
    label_smooth: float = 0.9        # 真实标签平滑
    epochs: int = 200
    lr: float = 2e-4
    batch_size: int = 128
    seed: int = 0


@dataclass
class ScoringConfig:
    """打分与阈值参数。"""

    quantile: float = 0.99           # DAE 单元格通道：干净子集逐列误差高分位阈值
    max_cells_per_row: int = 1       # DAE 每行最多保留 top-N 高贡献单元格（抑制误差涂抹，0=不限）
    row_quantile: float = 0.99       # GANomaly 行级：干净子集行异常分高分位阈值
    row_top_k: int = 1               # GANomaly 可疑行内取 top-k 列粗定位（小表上从严，控误报）
    row_min_col_z: float = 1.0       # GANomaly 行内列贡献的最小鲁棒 z 阈（过滤弱列）


@dataclass
class PathsConfig:
    """输入输出路径（默认相对项目根）。"""

    input_csv: str = "data/hospital_dirty.csv"          # 待检测脏表
    clean_csv: str = "data/hospital_clean.csv"          # 评估用 ground truth（可选）
    clean_mask: str = "clean_mask.csv"                  # Stage 1 产出的干净单元格掩码
    stage1_errors: str = "data/hospital_errors.csv"     # Stage 1 错误结果
    candidates_out: str = "data/stage2_candidates.csv"  # Stage 2 DIST 候选（融合）
    dae_out: str = "data/stage2_dae.csv"                # DAE 单独候选（三方对比/调试）
    ganomaly_out: str = "data/stage2_ganomaly.csv"      # GANomaly 单独候选
    combined_out: str = "data/combined_candidates.csv"  # Stage1 ∪ Stage2 合并集


@dataclass
class Stage2Config:
    """Stage 2 顶层配置。"""

    mode: str = "fuse"  # "fuse"(DAE+GANomaly) / "dae" / "ganomaly"
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    dae: DAEConfig = field(default_factory=DAEConfig)
    ganomaly: GANomalyConfig = field(default_factory=GANomalyConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def dae_kwargs(self) -> dict:
        return {k: v for k, v in vars(self.dae).items()}

    def ganomaly_kwargs(self) -> dict:
        return {k: v for k, v in vars(self.ganomaly).items()}
