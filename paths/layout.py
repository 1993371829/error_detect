"""
统一生成物目录与命名规范。

data/ 仅放原始数据集（{dataset}_dirty.csv / {dataset}_clean.csv）；
生成物按数据集名写入 output/mask|stage1|stage2|stage3/ 下。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DIRTY_RE = re.compile(r"^(.+)_dirty$", re.IGNORECASE)


@dataclass
class OutputLayout:
    """生成物根目录布局（相对项目根）。"""

    data_dir: str = "data"
    output_root: str = "output"
    mask_dir: str = "output/mask"
    stage1_dir: str = "output/stage1"
    stage2_dir: str = "output/stage2"
    stage3_dir: str = "output/stage3"
    cache_dir: str = "cache"

    @classmethod
    def from_dict(cls, data: dict | None) -> "OutputLayout":
        if not data:
            return cls()
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass
class DatasetPaths:
    """某一数据集的全部标准路径（相对或绝对均可）。"""

    dataset: str
    dirty_csv: Path
    clean_csv: Path
    clean_mask: Path
    errors: Path
    rules: Path
    profiles: Path
    stage2_candidates: Path
    combined_candidates: Path
    stage3_results: Path
    final_errors: Path
    rule_cache: Path
    stage3_cache: Path


def infer_dataset_name(dirty_path: str | Path, *, dataset: str | None = None) -> str:
    """
    从脏表路径推断数据集名。

    Args:
        dirty_path: 如 data/hospital_dirty.csv -> hospital
        dataset: 显式指定时优先使用（CLI --dataset）

    Raises:
        ValueError: 无法从文件名推断且未提供 dataset
    """
    if dataset:
        return dataset.strip()
    stem = Path(dirty_path).stem
    m = _DIRTY_RE.match(stem)
    if m:
        return m.group(1)
    raise ValueError(
        f"无法从输入文件名 '{stem}' 推断数据集名（期望 {{name}}_dirty.csv）。"
        f"请使用 --dataset 显式指定。"
    )


def resolve_dataset_paths(
    dirty_csv: str | Path,
    layout: OutputLayout | None = None,
    *,
    dataset: str | None = None,
    project_root: Path | None = None,
) -> DatasetPaths:
    """
    根据脏表路径与布局配置，解析该数据集的全部标准路径。

    Args:
        dirty_csv: 脏表 CSV 路径（通常在 data/ 下）
        layout: 目录布局，None 则用默认 OutputLayout
        dataset: 可选，强制数据集名
        project_root: 项目根，默认自动检测
    """
    root = project_root or _PROJECT_ROOT
    lay = layout or OutputLayout()
    dirty = Path(dirty_csv)
    if not dirty.is_absolute():
        dirty = root / dirty

    name = infer_dataset_name(dirty, dataset=dataset)

    def _p(subdir: str, filename: str) -> Path:
        return root / subdir / filename

    data_dir = root / lay.data_dir
    cache_dir = root / lay.cache_dir
    return DatasetPaths(
        dataset=name,
        dirty_csv=dirty,
        clean_csv=data_dir / f"{name}_clean.csv",
        clean_mask=_p(lay.mask_dir, f"{name}_clean_mask.csv"),
        errors=_p(lay.stage1_dir, f"{name}_errors.csv"),
        rules=_p(lay.stage1_dir, f"{name}_rules.json"),
        profiles=_p(lay.stage1_dir, f"{name}_profiles.json"),
        stage2_candidates=_p(lay.stage2_dir, f"{name}_stage2_candidates.csv"),
        combined_candidates=_p(lay.stage2_dir, f"{name}_combined_candidates.csv"),
        stage3_results=_p(lay.stage3_dir, f"{name}_stage3_results.csv"),
        final_errors=_p(lay.stage3_dir, f"{name}_final_errors.csv"),
        rule_cache=cache_dir / "rule_cache.json",
        stage3_cache=cache_dir / "stage3_cache.json",
    )


def rel_path(path: Path, root: Path | None = None) -> str:
    """转为相对项目根的路径字符串（用于配置/CLI 默认值）。"""
    root = root or _PROJECT_ROOT
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def default_dataset_paths(
    dataset_dirty: str = "data/hospital_dirty.csv",
    layout: OutputLayout | None = None,
) -> DatasetPaths:
    """默认数据集（hospital）的标准路径。"""
    return resolve_dataset_paths(dataset_dirty, layout=layout)


def ensure_output_dirs(paths: DatasetPaths, *, include_cache: bool = True) -> None:
    """创建生成物目录（写入前调用）。"""
    dirs = {
        paths.clean_mask.parent,
        paths.errors.parent,
        paths.stage2_candidates.parent,
        paths.stage3_results.parent,
    }
    if include_cache:
        dirs.add(paths.rule_cache.parent)
        dirs.add(paths.stage3_cache.parent)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
