"""项目路径布局：原始数据与生成物目录解析。"""

from paths.layout import (
    DatasetPaths,
    OutputLayout,
    default_dataset_paths,
    ensure_output_dirs,
    infer_dataset_name,
    rel_path,
    resolve_dataset_paths,
)

__all__ = [
    "DatasetPaths",
    "OutputLayout",
    "default_dataset_paths",
    "ensure_output_dirs",
    "infer_dataset_name",
    "rel_path",
    "resolve_dataset_paths",
]
