"""
单轮回灌闭环编排（S2 ∩ S3 -> clean_mask）。

按序执行并对比两轮指标，验证"用 Stage 3 高置信判定回灌掩码、重训 Stage 2"是否提升效果:

    round-0:  Stage1 -> Stage2 -> Stage3 -> evaluate(记录基线)
    refine:   stage_2.refine_mask 生成 {dataset}_clean_mask_r1.csv
    round-1:  Stage2(--clean-mask r1) -> Stage3 -> evaluate
    诊断:     diagnose_mask 对比 round-0 与 r1 掩码污染率

各阶段沿用既有 CLI（subprocess 调用），不改动主流程文件。

用法（项目根目录）:
    python run_feedback.py --dirty data/flights_dirty.csv
    python run_feedback.py --dirty data/beers_dirty.csv --tau 0.8 --max-exclude-ratio 0.05
    # 已跑过 round-0、想直接复用其产物时:
    python run_feedback.py --dirty data/flights_dirty.csv --skip-round0
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from paths.layout import resolve_dataset_paths
from stage_2.config import Stage2Config

_F1_LINE = re.compile(r"精检后.*?P=([0-9.]+)\s+R=([0-9.]+)\s+F1=([0-9.]+)")
_LEAK_LINE = re.compile(r"leak_rate\s*=\s*([0-9.]+)%")


def _run(argv: list[str], *, capture: bool = False) -> str:
    """运行子进程，强制 UTF-8 IO；capture=True 时返回 stdout 并同时回显。"""
    full_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    print(f"\n$ {' '.join(argv)}")
    if not capture:
        subprocess.run(argv, check=True, env=full_env)
        return ""
    proc = subprocess.run(
        argv, check=True, env=full_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    print(proc.stdout)
    return proc.stdout


def _parse_eval(text: str) -> dict | None:
    m = _F1_LINE.search(text)
    if not m:
        return None
    return {"precision": float(m.group(1)), "recall": float(m.group(2)),
            "f1": float(m.group(3))}


def _parse_leak(text: str) -> float | None:
    m = _LEAK_LINE.search(text)
    return float(m.group(1)) if m else None


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    p = argparse.ArgumentParser(description="单轮回灌闭环编排与轮次对比")
    p.add_argument("--dirty", required=True, help="脏表 CSV（data/{dataset}_dirty.csv）")
    p.add_argument("--dataset", default=None, help="显式指定数据集名")
    p.add_argument("--tau", type=float, default=0.6, help="回灌翻转最低把握（默认 0.6）")
    p.add_argument("--max-exclude-ratio", type=float, default=0.05,
                   help="每列新增剔除上限比例（默认 0.05）")
    p.add_argument("--no-include", action="store_true",
                   help="只剔除污染、不找回误报（flights 调参显示更优）")
    p.add_argument("--skip-round0", action="store_true",
                   help="跳过 round-0 的 S1/S2/S3，直接复用已有产物（含已有 stage3_results）")
    p.add_argument("--skip-stage1", action="store_true",
                   help="round-0 复用已有 Stage 1 clean_mask（跳过 main.py），仍跑 S2/S3")
    args = p.parse_args(argv)

    py = sys.executable
    dirty = args.dirty
    ds_args = ["--dataset", args.dataset] if args.dataset else []

    cfg = Stage2Config()
    dp = resolve_dataset_paths(dirty, cfg.layout, dataset=args.dataset)
    r1_mask = dp.clean_mask.parent / f"{dp.dataset}_clean_mask_r1.csv"

    # ---- round-0 ----
    if not args.skip_round0:
        print("\n########## ROUND 0: Stage1 -> Stage2 -> Stage3 ##########")
        if not args.skip_stage1:
            _run([py, "main.py", "--input", dirty, *ds_args])
        else:
            print("(--skip-stage1: 复用已有 clean_mask，跳过 main.py)")
        _run([py, "-m", "stage_2.cli", "--input", dirty, *ds_args])
        _run([py, "-m", "stage_3.cli", "--input", dirty, *ds_args])
    else:
        print("\n########## ROUND 0: 跳过，复用已有产物 ##########")

    eval0 = _parse_eval(
        _run([py, "-m", "stage_3.evaluate", "--dirty", dirty, *ds_args], capture=True)
    )
    diag0 = _parse_leak(
        _run([py, "-m", "stage_2.diagnose_mask", "--dirty", dirty, *ds_args], capture=True)
    )

    # ---- refine ----
    print("\n########## REFINE: 回灌 Stage3 判定 -> clean_mask_r1 ##########")
    refine_argv = [py, "-m", "stage_2.refine_mask", "--dirty", dirty, *ds_args,
                   "--tau", str(args.tau), "--max-exclude-ratio", str(args.max_exclude_ratio)]
    if args.no_include:
        refine_argv.append("--no-include")
    _run(refine_argv)
    diag1 = _parse_leak(
        _run([py, "-m", "stage_2.diagnose_mask", "--dirty", dirty, *ds_args,
              "--clean-mask", str(r1_mask)], capture=True)
    )

    # ---- round-1 ----
    print("\n########## ROUND 1: Stage2(r1 mask) -> Stage3 ##########")
    _run([py, "-m", "stage_2.cli", "--input", dirty, *ds_args,
          "--clean-mask", str(r1_mask)])
    _run([py, "-m", "stage_3.cli", "--input", dirty, *ds_args])
    eval1 = _parse_eval(
        _run([py, "-m", "stage_3.evaluate", "--dirty", dirty, *ds_args], capture=True)
    )

    # ---- 对比 ----
    print("\n" + "=" * 64)
    print(f"单轮回灌对比  数据集={dp.dataset}  tau={args.tau}  "
          f"max_exclude_ratio={args.max_exclude_ratio}")
    print("-" * 64)
    if diag0 is not None and diag1 is not None:
        print(f"  掩码污染 leak_rate: {diag0:.2f}% -> {diag1:.2f}% "
              f"({diag1 - diag0:+.2f} pp)")
    if eval0 and eval1:
        print(f"  精检后 Precision: {eval0['precision']:.3f} -> {eval1['precision']:.3f} "
              f"({eval1['precision'] - eval0['precision']:+.3f})")
        print(f"  精检后 Recall:    {eval0['recall']:.3f} -> {eval1['recall']:.3f} "
              f"({eval1['recall'] - eval0['recall']:+.3f})")
        print(f"  精检后 F1:        {eval0['f1']:.3f} -> {eval1['f1']:.3f} "
              f"({eval1['f1'] - eval0['f1']:+.3f})")
    else:
        print("  [warn] 未能解析评估输出，请查看上方 stage_3.evaluate 原始结果。")
    print("=" * 64)


if __name__ == "__main__":
    main()
