"""
单轮回灌：用 Stage 3 的高置信判定双向修正 clean_mask，供 Stage 2 重训。

动机:
    当前 clean_mask 仅由 Stage 1 errors 取反构建，Stage 2/3 判定从不回写。
    因此 Stage 1 漏报、Stage 2 发现、Stage 3 确认的脏格仍以 True 留在训练集里污染
    分布（诊断中 flights 19% / beers 9%）。本模块把 Stage 3 判定回写掩码:

      - is_error=True 且 confidence>=tau  -> 置 False（剔除污染，主要命中 S2 发现错误）
      - is_error=False 且 confidence>=tau -> 置 True（找回误报误删），
        但 prior_error_type 属硬锚点(MV)者永不翻回 True，防漂移
      - 其余（低把握/模糊）              -> 维持原状（阻尼，避免抖动）

    并对"每列新增剔除数"设上限（默认 ≤5% 行），保护稀有合法值
    （吸取上一轮 vocab folding 误删稀有值导致召回下降的教训）。

调参实测（flights，LLM-free Stage2 评估，详见 PIPELINE.md）:
    最优配置为 tau≈0.6 + 只剔除不找回(--no-include) + cap 5%（默认即此组合，但找回默认开启）；
    - "找回误报"会把 S3 否决的格加回训练集、引入噪声，每个 tau 下都略逊于只剔除；
    - tau 低于 0.6 无额外增益；cap 放大到 10%/20% 会因过度剔除而回落。

用法（项目根目录，需先跑完一轮 Stage 1->2->3）:
    python -m stage_2.refine_mask --dirty data/flights_dirty.csv --no-include   # 推荐
    python -m stage_2.refine_mask --dirty data/beers_dirty.csv --tau 0.6 --no-include
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paths.layout import resolve_dataset_paths
from stage_2.config import Stage2Config
from stage_2.io_utils import read_clean_mask

HARD_ANCHOR_TYPES = ("MV",)

_TRUE_STR = {"true", "1", "yes"}
_FALSE_STR = {"false", "0", "no", ""}


def _parse_bool(v) -> bool:
    """稳健解析 is_error（CSV 读回为字符串 'True'/'False'）。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in _TRUE_STR:
        return True
    if s in _FALSE_STR:
        return False
    return False


def _parse_conf(v) -> float:
    """稳健解析 confidence；缺失/非法视为 0（低把握，不触发翻转）。"""
    try:
        c = float(v)
    except (TypeError, ValueError):
        return 0.0
    if c != c:  # NaN
        return 0.0
    return max(0.0, min(1.0, c))


def read_stage3_results(path: str | Path) -> pd.DataFrame:
    """读取 Stage 3 逐格判定，按字符串读入避免空值/类型歧义。"""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def refine_clean_mask(
    mask_df: pd.DataFrame,
    stage3_df: pd.DataFrame,
    *,
    tau: float = 0.6,
    max_exclude_ratio: float = 0.05,
    allow_include: bool = True,
    hard_anchor_types: tuple[str, ...] = HARD_ANCHOR_TYPES,
) -> tuple[pd.DataFrame, dict]:
    """
    依据 Stage 3 判定生成 round-1 掩码。

    Args:
        mask_df: round-0 布尔掩码（行为 0..n-1 的 RangeIndex，列为数据列）。
        stage3_df: Stage 3 逐格判定，需含 row_id/column/is_error/confidence/prior_error_type。
        tau: 翻转掩码状态所需的最低把握。
        max_exclude_ratio: 每列"新增剔除格"占总行数的上限比例（0=不剔除，<0=不设限）。
        allow_include: 是否允许"找回误删"（False->True）；False 时只剔除不找回。
        hard_anchor_types: 永不翻回 True 的前序类型（硬错误锚点）。

    Returns:
        (refined_mask, stats)
    """
    refined = mask_df.copy()
    n_rows = len(refined)
    cap = None if max_exclude_ratio < 0 else max(1, int(max_exclude_ratio * n_rows))

    # 候选剔除（True->False）按列收集后按把握择优，受 cap 限制；找回（False->True）直接应用
    exclude_by_col: dict[str, list[tuple[int, float]]] = {}
    include_cells: list[tuple[int, str]] = []
    anchors = {t.upper() for t in hard_anchor_types}

    n_seen = 0
    for _, r in stage3_df.iterrows():
        col = str(r.get("column", ""))
        if col not in refined.columns:
            continue
        try:
            row_id = int(r.get("row_id"))
        except (TypeError, ValueError):
            continue
        if row_id not in refined.index:
            continue
        n_seen += 1
        is_err = _parse_bool(r.get("is_error"))
        conf = _parse_conf(r.get("confidence"))
        if conf < tau:
            continue
        cur = bool(refined.at[row_id, col])
        if is_err:
            if cur:  # 仅"原本干净"的格才算新增剔除
                exclude_by_col.setdefault(col, []).append((row_id, conf))
        elif allow_include:
            prior = str(r.get("prior_error_type", "")).upper()
            if not cur and prior not in anchors:
                include_cells.append((row_id, col))

    for row_id, col in include_cells:
        refined.at[row_id, col] = True
    n_included = len(include_cells)

    n_excluded = 0
    n_capped = 0
    per_col_excluded: dict[str, int] = {}
    for col, cells in exclude_by_col.items():
        cells.sort(key=lambda x: x[1], reverse=True)
        chosen = cells if cap is None else cells[:cap]
        n_capped += len(cells) - len(chosen)
        for row_id, _ in chosen:
            refined.at[row_id, col] = False
        if chosen:
            per_col_excluded[col] = len(chosen)
            n_excluded += len(chosen)

    before_clean = int(mask_df.to_numpy().sum())
    after_clean = int(refined.to_numpy().sum())
    stats = {
        "total_cells": refined.size,
        "judgments_seen": n_seen,
        "tau": tau,
        "cap_per_col": cap,
        "excluded": n_excluded,
        "excluded_capped": n_capped,
        "included": n_included,
        "clean_before": before_clean,
        "clean_after": after_clean,
        "clean_delta": after_clean - before_clean,
        "per_col_excluded": per_col_excluded,
    }
    return refined, stats


def _print_stats(dataset: str, in_mask: Path, out_mask: Path, stats: dict) -> None:
    print("=" * 64)
    print(f"数据集: {dataset}  回灌单轮 clean_mask 修正")
    print(f"  输入掩码: {in_mask}")
    print(f"  输出掩码: {out_mask}")
    print("-" * 64)
    print(f"  Stage3 判定参与: {stats['judgments_seen']} 条  (tau={stats['tau']}, "
          f"每列剔除上限={stats['cap_per_col']})")
    print(f"  剔除污染 (True->False): {stats['excluded']}  "
          f"(因上限放弃 {stats['excluded_capped']})")
    print(f"  找回误删 (False->True): {stats['included']}")
    print(f"  干净格: {stats['clean_before']} -> {stats['clean_after']} "
          f"(净变化 {stats['clean_delta']:+d}) / 共 {stats['total_cells']}")
    if stats["per_col_excluded"]:
        top = sorted(stats["per_col_excluded"].items(),
                     key=lambda kv: kv[1], reverse=True)[:10]
        print("  剔除最多的列 Top:")
        for col, n in top:
            print(f"    {col:24s} {n}")
    print("=" * 64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="单轮回灌：用 Stage 3 判定修正 clean_mask（生成 _clean_mask_r1.csv）"
    )
    p.add_argument("--dirty", required=True, help="脏表 CSV（data/{dataset}_dirty.csv）")
    p.add_argument("--dataset", default=None, help="显式指定数据集名")
    p.add_argument("--clean-mask", default=None, help="round-0 输入掩码（默认自动推导）")
    p.add_argument("--stage3-results", default=None, help="Stage 3 逐格判定 CSV（默认自动推导）")
    p.add_argument("--out", default=None, help="输出修正掩码路径（默认 {dataset}_clean_mask_r1.csv）")
    p.add_argument("--tau", type=float, default=0.6,
                   help="翻转掩码所需最低把握（默认 0.6，flights 调参最优）")
    p.add_argument("--max-exclude-ratio", type=float, default=0.05,
                   help="每列新增剔除格占总行数上限（默认 0.05；<0 不设限）")
    p.add_argument("--no-include", action="store_true",
                   help="只剔除污染、不找回误报（禁用 False->True 翻转）")
    return p


def main(argv: list[str] | None = None) -> None:
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    args = build_parser().parse_args(argv)
    cfg = Stage2Config()
    dp = resolve_dataset_paths(args.dirty, cfg.layout, dataset=args.dataset)

    in_mask = Path(args.clean_mask) if args.clean_mask else dp.clean_mask
    s3_path = Path(args.stage3_results) if args.stage3_results else dp.stage3_results
    out_mask = (Path(args.out) if args.out
                else dp.clean_mask.parent / f"{dp.dataset}_clean_mask_r1.csv")

    if not in_mask.exists():
        raise FileNotFoundError(f"未找到 round-0 掩码 {in_mask}，请先运行 Stage 1。")
    if not s3_path.exists():
        raise FileNotFoundError(f"未找到 Stage 3 判定 {s3_path}，请先运行 Stage 3。")

    mask_df = read_clean_mask(in_mask)
    stage3_df = read_stage3_results(s3_path)

    refined, stats = refine_clean_mask(
        mask_df, stage3_df,
        tau=args.tau, max_exclude_ratio=args.max_exclude_ratio,
        allow_include=not args.no_include,
    )

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    refined.to_csv(out_mask, index=False)
    _print_stats(dp.dataset, in_mask, out_mask, stats)


if __name__ == "__main__":
    main()
