"""
一键「基线 + 逐项消融」评测脚本。

针对 Stage 3 的时间/Token 优化开关（thinking、上下文无关去重等）做对照实验：
Stage 1 / Stage 2 只跑一次（可复用已有产物），再对同一份候选逐个变体重跑 Stage 3，
自动记录每个变体的 F1 / token / 耗时，并生成对比表（控制台 + CSV + JSON）。

用法（项目根目录）:
    # 单数据集，默认变体集
    python ablate.py --input data/hospital_dirty.csv

    # 多数据集
    python ablate.py --datasets hospital,flights,beers

    # 仅跑指定变体 + 限行快速冒烟
    python ablate.py --input data/hospital_dirty.csv --variants baseline,no_thinking --limit 20

    # 复用已有 Stage1/2 产物（默认已有则跳过 prep；--force-prep 强制重跑）
    python ablate.py --datasets hospital --force-prep

变体（extra flags 传给 stage_3.cli）:
    baseline           思考ON,无去重（F1 基准）
    no_thinking        思考OFF（B1）
    no_thinking_dedup  思考OFF + 上下文无关去重（B1+B3）

产出:
    output/runs/ablation/{ts}_results.csv     # 长表：每 (dataset,variant) 一行
    output/runs/ablation/{ts}_results.json    # 结构化结果 + 元信息
    控制台：每数据集一张对比表（含 vs 基线 的 ΔF1 / token% / 时间%）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from paths.layout import OutputLayout, rel_path, resolve_dataset_paths
from stage_1.config import Stage1Config
from stage_1.llm_usage import init_usage_file, load_usage

_PROJECT_ROOT = Path(__file__).resolve().parent

# 通用指标行解析（覆盖 _report 打印格式，不限定标签前缀）
_METRIC_RE = re.compile(
    r"检出=\s*(\d+)\s+TP=\s*(\d+)\s+FP=\s*(\d+)\s+"
    r"P=([0-9.]+)\s+R=([0-9.]+)\s+F1=([0-9.]+)"
)
_GT_RE = re.compile(r"Ground-truth 注入错误单元格:\s*(\d+)")

# 变体定义：名称 -> 传给 stage_3.cli 的附加参数
VARIANTS: dict[str, list[str]] = {
    "baseline": ["--thinking"],
    "no_thinking": ["--no-thinking"],
    "no_thinking_dedup": ["--no-thinking", "--dedup-context-free"],
}
_BASELINE = "baseline"


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def _run(argv: list[str], env: dict, *, capture: bool = False) -> str:
    print(f"\n$ {' '.join(argv)}")
    if not capture:
        subprocess.run(argv, check=True, env=env)
        return ""
    proc = subprocess.run(
        argv, check=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    print(proc.stdout)
    return proc.stdout


def _parse_report(text: str, label: str) -> dict | None:
    for line in text.splitlines():
        if line.strip().startswith(label):
            m = _METRIC_RE.search(line)
            if m:
                return {
                    "detected": int(m.group(1)),
                    "tp": int(m.group(2)),
                    "fp": int(m.group(3)),
                    "precision": float(m.group(4)),
                    "recall": float(m.group(5)),
                    "f1": float(m.group(6)),
                }
    return None


def _parse_gt(text: str) -> int | None:
    m = _GT_RE.search(text)
    return int(m.group(1)) if m else None


def _pct(new: float, base: float) -> float | None:
    """相对基线的变化百分比；base<=0 时返回 None。"""
    if base is None or base <= 0:
        return None
    return round((new - base) / base * 100.0, 1)


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "   -  "
    return f"{v:+6.1f}%"


def _fmt_f1(v: float | None) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 3 基线+逐项消融一键评测（F1/token/耗时对比）")
    src = p.add_argument_group("数据源（二选一，可组合）")
    src.add_argument("--input", "--dirty", dest="inputs", action="append", default=None,
                     help="脏表 CSV，可重复传入多次")
    src.add_argument("--datasets", default=None,
                     help="逗号分隔数据集名，映射 data/{name}_dirty.csv")
    p.add_argument("--dataset", default=None,
                   help="单数据集时显式指定名称（配合单个 --input）")

    p.add_argument("--variants", default=None,
                   help=f"逗号分隔变体名，默认全部：{','.join(VARIANTS)}")
    p.add_argument("--max-workers", type=int, default=8,
                   help="Stage 3 并发线程数（各变体统一，默认 8）")
    p.add_argument("--limit", type=int, default=None,
                   help="Stage 3 仅处理前 N 行分组（快速冒烟）")
    p.add_argument("--use-cache", action="store_true",
                   help="变体复用 Stage3 缓存（默认 --no-cache 以干净测量 token/耗时）")
    p.add_argument("--force-prep", action="store_true",
                   help="强制重跑 Stage 1 + Stage 2（默认已有候选则跳过）")
    p.add_argument("--out-dir", default=None,
                   help="结果输出目录（默认 output/runs/ablation）")
    return p


def _resolve_inputs(args) -> list[str]:
    inputs: list[str] = []
    if args.inputs:
        inputs.extend(args.inputs)
    if args.datasets:
        for name in args.datasets.split(","):
            name = name.strip()
            if name:
                inputs.append(f"data/{name}_dirty.csv")
    if not inputs:
        raise SystemExit("需指定 --input 或 --datasets")
    return inputs


def _ensure_prep(dp, dirty: str, ds_args: list[str], env: dict, *, force: bool) -> None:
    """确保 Stage1/2 产物存在；缺失或 force 时重跑。"""
    have = dp.combined_candidates.exists() and dp.rules.exists()
    if have and not force:
        print(f"[prep] 复用已有候选：{rel_path(dp.combined_candidates)}")
        return
    print(f"\n########## PREP (Stage1+Stage2) : {dp.dataset} ##########")
    _run([sys.executable, "main.py", "--input", dirty, *ds_args,
          "--rule-cache", rel_path(dp.rule_cache)], env)
    _run([sys.executable, "-m", "stage_2.cli", "--input", dirty, *ds_args], env)


def _run_variant(
    name: str, extra: list[str], dp, dirty: str, ds_args: list[str],
    *, out_dir: Path, max_workers: int, limit: int | None, use_cache: bool,
) -> dict:
    usage_path = out_dir / f"{dp.dataset}__{name}__usage.json"
    init_usage_file(usage_path)
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "LLM_USAGE_FILE": str(usage_path),
    }

    s3_cmd = [sys.executable, "-m", "stage_3.cli", "--input", dirty, *ds_args,
              "--max-workers", str(max_workers), *extra]
    if use_cache:
        cache = out_dir / f"{dp.dataset}__{name}__stage3_cache.json"
        s3_cmd.extend(["--cache", str(cache)])
    else:
        s3_cmd.append("--no-cache")
    if limit is not None:
        s3_cmd.extend(["--limit", str(limit)])

    print(f"\n########## VARIANT [{dp.dataset} / {name}] ##########")
    t0 = time.perf_counter()
    _run(s3_cmd, env)
    stage3_s = round(time.perf_counter() - t0, 1)

    eval_out = _run(
        [sys.executable, "-m", "stage_3.evaluate", "--dirty", dirty, *ds_args],
        env, capture=True,
    )
    before = _parse_report(eval_out, "精检前(合并候选)")
    after = _parse_report(eval_out, "精检后(最终确认)")
    gt = _parse_gt(eval_out)
    usage = load_usage(usage_path).to_dict()

    return {
        "dataset": dp.dataset,
        "variant": name,
        "flags": " ".join(extra),
        "gt": gt,
        "before": before,
        "after": after,
        "usage": usage,
        "stage3_seconds": stage3_s,
    }


def _print_table(dataset: str, rows: list[dict]) -> None:
    base = next((r for r in rows if r["variant"] == _BASELINE), rows[0])
    b_after = base.get("after") or {}
    b_tok = (base.get("usage") or {}).get("total_tokens", 0)
    b_time = base.get("stage3_seconds", 0.0)
    before = base.get("before") or {}

    print("\n" + "=" * 100)
    gt = base.get("gt")
    print(f"数据集: {dataset}   GT错误格={gt}   "
          f"精检前(合并候选) F1={_fmt_f1(before.get('f1'))}   基准变体={_BASELINE}")
    print("-" * 100)
    header = (f"{'variant':<20}{'P':>7}{'R':>7}{'F1':>7}{'dF1':>8}"
             f"{'calls':>7}{'prompt':>9}{'compl':>9}{'total':>9}"
             f"{'tok%':>8}{'stage3_s':>10}{'time%':>8}")
    print(header)
    print("-" * 100)
    for r in rows:
        a = r.get("after") or {}
        u = r.get("usage") or {}
        f1 = a.get("f1")
        d_f1 = None if f1 is None or b_after.get("f1") is None else round(f1 - b_after["f1"], 3)
        tot = u.get("total_tokens", 0)
        s3 = r.get("stage3_seconds", 0.0)
        d_f1_s = "   -  " if d_f1 is None else f"{d_f1:+.3f}"
        print(f"{r['variant']:<20}"
              f"{_fmt_f1(a.get('precision')):>7}{_fmt_f1(a.get('recall')):>7}"
              f"{_fmt_f1(f1):>7}{d_f1_s:>8}"
              f"{u.get('calls', 0):>7}{u.get('prompt_tokens', 0):>9}"
              f"{u.get('completion_tokens', 0):>9}{tot:>9}"
              f"{_fmt_pct(_pct(tot, b_tok)):>8}"
              f"{s3:>10.1f}{_fmt_pct(_pct(s3, b_time)):>8}")
    print("=" * 100)


def _write_csv(path: Path, results: list[dict]) -> None:
    import csv
    # 预计算每数据集基线，供 Δ/百分比列
    base_by_ds: dict[str, dict] = {}
    for r in results:
        if r["variant"] == _BASELINE:
            base_by_ds[r["dataset"]] = r
    cols = [
        "dataset", "variant", "flags", "gt",
        "before_f1",
        "after_precision", "after_recall", "after_f1", "delta_f1_vs_baseline",
        "calls", "prompt_tokens", "completion_tokens", "total_tokens",
        "token_pct_vs_baseline",
        "stage3_seconds", "time_pct_vs_baseline",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            a = r.get("after") or {}
            before = r.get("before") or {}
            u = r.get("usage") or {}
            base = base_by_ds.get(r["dataset"], {})
            b_after = base.get("after") or {}
            b_tok = (base.get("usage") or {}).get("total_tokens", 0)
            b_time = base.get("stage3_seconds", 0.0)
            f1 = a.get("f1")
            d_f1 = ("" if f1 is None or b_after.get("f1") is None
                    else round(f1 - b_after["f1"], 4))
            tot = u.get("total_tokens", 0)
            s3 = r.get("stage3_seconds", 0.0)
            w.writerow([
                r["dataset"], r["variant"], r["flags"], r.get("gt", ""),
                before.get("f1", ""),
                a.get("precision", ""), a.get("recall", ""), a.get("f1", ""), d_f1,
                u.get("calls", 0), u.get("prompt_tokens", 0),
                u.get("completion_tokens", 0), tot,
                _pct(tot, b_tok) if b_tok else "",
                s3, _pct(s3, b_time) if b_time else "",
            ])


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    args = build_parser().parse_args(argv)

    if args.variants:
        names = [n.strip() for n in args.variants.split(",") if n.strip()]
        unknown = [n for n in names if n not in VARIANTS]
        if unknown:
            raise SystemExit(f"未知变体: {unknown}；可选: {list(VARIANTS)}")
    else:
        names = list(VARIANTS)

    inputs = _resolve_inputs(args)
    layout = OutputLayout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        _PROJECT_ROOT / layout.output_root / "runs" / "ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = Stage1Config.resolve().apply_env_overrides()
    prep_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    results: list[dict] = []
    for inp in inputs:
        single_ds = args.dataset if len(inputs) == 1 else None
        dp = resolve_dataset_paths(inp, layout, dataset=single_ds)
        dirty = rel_path(dp.dirty_csv)
        ds_args = ["--dataset", dp.dataset] if single_ds else []

        _ensure_prep(dp, dirty, ds_args, prep_env, force=args.force_prep)

        ds_rows: list[dict] = []
        for name in names:
            row = _run_variant(
                name, VARIANTS[name], dp, dirty, ds_args,
                out_dir=out_dir, max_workers=args.max_workers,
                limit=args.limit, use_cache=args.use_cache,
            )
            ds_rows.append(row)
            results.append(row)
        _print_table(dp.dataset, ds_rows)

    csv_path = out_dir / f"{ts}_results.csv"
    json_path = out_dir / f"{ts}_results.json"
    _write_csv(csv_path, results)
    meta = {
        "timestamp": ts,
        "model": llm_cfg.llm.model,
        "base_url": llm_cfg.llm.base_url,
        "variants": {n: VARIANTS[n] for n in names},
        "max_workers": args.max_workers,
        "limit": args.limit,
        "use_cache": args.use_cache,
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "#" * 100)
    print(f"评测完成  变体={names}  数据集={[Path(i).stem for i in inputs]}")
    print(f"  CSV : {rel_path(csv_path)}")
    print(f"  JSON: {rel_path(json_path)}")
    print("#" * 100)


if __name__ == "__main__":
    main()
