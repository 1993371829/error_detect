"""
一键跑 Stage 1 -> Stage 2 -> Stage 3，统计运行时间与 LLM token 用量。

用法（项目根目录）:
    python run_pipeline.py --input data/hospital_dirty.csv
    python run_pipeline.py --input data/hospital_dirty.csv --backup --backup-tag deepseek
    python run_pipeline.py --input data/hospital_dirty.csv --fresh-cache

产出:
    output/runs/{dataset}_{run_id}_report.json   # 计时 + token + 评估摘要
    output/runs/{dataset}_{run_id}_llm_usage.json
    --backup 时: output/archive/{dataset}_{tag}_{run_id}/ 保存跑前产物
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from paths.layout import OutputLayout, rel_path, resolve_dataset_paths
from stage_1.config import Stage1Config
from stage_1.llm_usage import init_usage_file, load_usage

_PROJECT_ROOT = Path(__file__).resolve().parent

_F1_LINE = re.compile(
    r"(?:精检后\(最终确认\)|合并\(S1\+S2\)|Stage1\(规则层\))\s+"
    r"检出=\s*\d+\s+TP=\s*\d+\s+FP=\s*\d+\s+P=([0-9.]+)\s+R=([0-9.]+)\s+F1=([0-9.]+)"
)


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


def _parse_metrics(text: str, label: str) -> dict | None:
    for line in text.splitlines():
        if label in line:
            m = _F1_LINE.search(line)
            if m:
                return {
                    "precision": float(m.group(1)),
                    "recall": float(m.group(2)),
                    "f1": float(m.group(3)),
                }
    return None


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def backup_outputs(
    dp,
    archive_dir: Path,
    *,
    py: str,
    env: dict,
) -> list[str]:
    """备份当前数据集已有产物与评估结果，返回已备份相对路径列表。"""
    saved: list[str] = []
    pairs = [
        (dp.clean_mask, archive_dir / "mask" / dp.clean_mask.name),
        (dp.errors, archive_dir / "stage1" / dp.errors.name),
        (dp.rules, archive_dir / "stage1" / dp.rules.name),
        (dp.profiles, archive_dir / "stage1" / dp.profiles.name),
        (dp.stage2_candidates, archive_dir / "stage2" / dp.stage2_candidates.name),
        (dp.combined_candidates, archive_dir / "stage2" / dp.combined_candidates.name),
        (dp.stage3_results, archive_dir / "stage3" / dp.stage3_results.name),
        (dp.final_errors, archive_dir / "stage3" / dp.final_errors.name),
        (dp.rule_cache, archive_dir / "cache" / dp.rule_cache.name),
        (dp.stage3_cache, archive_dir / "cache" / dp.stage3_cache.name),
    ]
    for src, dst in pairs:
        if _copy_if_exists(src, dst):
            saved.append(rel_path(dst))

    eval_dir = archive_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    dirty = rel_path(dp.dirty_csv)
    ds_args = ["--dataset", dp.dataset]
    try:
        s2 = _run(
            [py, "-m", "stage_2.evaluate", "--dirty", dirty, *ds_args],
            env, capture=True,
        )
        p = eval_dir / "stage2_eval.txt"
        p.write_text(s2, encoding="utf-8")
        saved.append(rel_path(p))
    except subprocess.CalledProcessError:
        print("[backup-warn] stage_2.evaluate 失败，跳过备份评估。")
    try:
        s3 = _run(
            [py, "-m", "stage_3.evaluate", "--dirty", dirty, *ds_args],
            env, capture=True,
        )
        p = eval_dir / "stage3_eval.txt"
        p.write_text(s3, encoding="utf-8")
        saved.append(rel_path(p))
    except subprocess.CalledProcessError:
        print("[backup-warn] stage_3.evaluate 失败，跳过备份评估。")
    return saved


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="一键跑 Stage1->Stage2->Stage3 并统计耗时/token")
    p.add_argument("--input", "--dirty", dest="input", required=True,
                   help="脏表 CSV（data/{dataset}_dirty.csv）")
    p.add_argument("--dataset", default=None, help="显式指定数据集名")
    p.add_argument("--backup", action="store_true",
                   help="跑前备份当前数据集产物到 output/archive/")
    p.add_argument("--backup-tag", default="prev",
                   help="备份目录标签，如 deepseek / qwen35（默认 prev）")
    p.add_argument("--fresh-cache", action="store_true",
                   help="本次使用独立 rule/stage3 缓存文件（换模型时推荐）")
    p.add_argument("--skip-stage1", action="store_true", help="跳过 Stage 1")
    p.add_argument("--skip-stage2", action="store_true", help="跳过 Stage 2")
    p.add_argument("--skip-stage3", action="store_true", help="跳过 Stage 3")
    p.add_argument("--skip-eval", action="store_true", help="跳过评估")
    p.add_argument("--stage3-no-cache", action="store_true",
                   help="Stage 3 禁用 LLM 响应缓存（换模型时推荐）")
    p.add_argument("--all-detectors", action="store_true",
                   help="Stage 2 启用全部检测器")
    p.add_argument("--min-tier", default=None, choices=["low", "mid", "high"],
                   help="Stage 3 仅精检 >= 该层级的候选")
    return p


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    args = build_parser().parse_args(argv)

    py = sys.executable
    layout = OutputLayout()
    dp = resolve_dataset_paths(args.input, layout, dataset=args.dataset)
    dirty = rel_path(dp.dirty_csv)
    ds_args = ["--dataset", dp.dataset] if args.dataset else []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = _PROJECT_ROOT / layout.output_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    usage_path = runs_dir / f"{dp.dataset}_{run_id}_llm_usage.json"
    report_path = runs_dir / f"{dp.dataset}_{run_id}_report.json"

    if args.fresh_cache:
        rule_cache = _PROJECT_ROOT / layout.cache_dir / f"{dp.dataset}_{run_id}_rule_cache.json"
        stage3_cache = _PROJECT_ROOT / layout.cache_dir / f"{dp.dataset}_{run_id}_stage3_cache.json"
    else:
        rule_cache = dp.rule_cache
        stage3_cache = _PROJECT_ROOT / layout.cache_dir / f"{dp.dataset}_stage3_cache.json"

    llm_cfg = Stage1Config.resolve().apply_env_overrides()

    init_usage_file(usage_path)
    base_env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "LLM_USAGE_FILE": str(usage_path),
    }

    report: dict = {
        "dataset": dp.dataset,
        "run_id": run_id,
        "input": dirty,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "llm": {
            "model": llm_cfg.llm.model,
            "base_url": llm_cfg.llm.base_url,
        },
        "timings_seconds": {},
        "llm_usage": {},
        "evaluation": {},
        "backup": None,
    }

    if args.backup:
        archive_dir = (
            _PROJECT_ROOT / layout.output_root / "archive"
            / f"{dp.dataset}_{args.backup_tag}_{run_id}"
        )
        print(f"\n########## BACKUP -> {rel_path(archive_dir)} ##########")
        saved = backup_outputs(dp, archive_dir, py=py, env=base_env)
        report["backup"] = {"dir": rel_path(archive_dir), "files": saved}
        print(f"已备份 {len(saved)} 个文件。")

    t_total = time.perf_counter()

    def step(name: str, cmd: list[str]) -> None:
        t0 = time.perf_counter()
        print(f"\n########## {name.upper()} ##########")
        _run(cmd, base_env)
        elapsed = round(time.perf_counter() - t0, 1)
        report["timings_seconds"][name] = elapsed
        print(f"[{name}] {elapsed} s")

    if not args.skip_stage1:
        step("stage1", [
            py, "main.py", "--input", dirty, *ds_args,
            "--rule-cache", rel_path(rule_cache),
        ])

    if not args.skip_stage2:
        s2_cmd = [py, "-m", "stage_2.cli", "--input", dirty, *ds_args]
        if args.all_detectors:
            s2_cmd.append("--all-detectors")
        step("stage2", s2_cmd)

    if not args.skip_stage3:
        s3_cmd = [py, "-m", "stage_3.cli", "--input", dirty, *ds_args,
                  "--cache", rel_path(stage3_cache)]
        if args.stage3_no_cache:
            s3_cmd.append("--no-cache")
        if args.min_tier:
            s3_cmd.extend(["--min-tier", args.min_tier])
        step("stage3", s3_cmd)

    if not args.skip_eval:
        try:
            s2_out = _run(
                [py, "-m", "stage_2.evaluate", "--dirty", dirty, *ds_args],
                base_env, capture=True,
            )
            report["evaluation"]["stage2"] = {
                "combined": _parse_metrics(s2_out, "合并(S1+S2)"),
                "stage1": _parse_metrics(s2_out, "Stage1(规则层)"),
            }
        except subprocess.CalledProcessError:
            report["evaluation"]["stage2"] = {"error": "evaluate failed"}
        try:
            s3_out = _run(
                [py, "-m", "stage_3.evaluate", "--dirty", dirty, *ds_args],
                base_env, capture=True,
            )
            report["evaluation"]["stage3"] = {
                "before": _parse_metrics(s3_out, "精检前(合并候选)"),
                "after": _parse_metrics(s3_out, "精检后(最终确认)"),
            }
        except subprocess.CalledProcessError:
            report["evaluation"]["stage3"] = {"error": "evaluate failed"}

    report["timings_seconds"]["total"] = round(time.perf_counter() - t_total, 1)
    report["llm_usage"] = load_usage(usage_path).to_dict()
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    report["artifacts"] = {
        "usage_file": rel_path(usage_path),
        "report_file": rel_path(report_path),
        "rule_cache": rel_path(rule_cache),
        "stage3_cache": rel_path(stage3_cache),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    u = report["llm_usage"]
    print("\n" + "=" * 64)
    print(f"流水线完成  数据集={dp.dataset}  run_id={run_id}")
    print(f"LLM 模型: {report['llm']['model']}")
    print("-" * 64)
    for k, v in report["timings_seconds"].items():
        print(f"  {k:12s} {v:8.1f} s")
    print("-" * 64)
    print(f"  LLM 调用次数     {u.get('calls', 0)}")
    print(f"  prompt_tokens    {u.get('prompt_tokens', 0)}")
    print(f"  completion_tokens {u.get('completion_tokens', 0)}")
    print(f"  total_tokens     {u.get('total_tokens', 0)}")
    print("-" * 64)
    print(f"  报告: {rel_path(report_path)}")
    if report.get("backup"):
        print(f"  备份: {report['backup']['dir']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
