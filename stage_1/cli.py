"""
Stage 1 命令行入口：LLM 辅助规则层（MV / FI / T / VAD）。

前置条件:
    pip install -r requirements.txt
    配置 LLM：复制 .env.example 为 .env，填写 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL

运行命令（在项目根目录 d:\\study\\error_dect 下执行）:

    # hospital 全流程 — Stage 1（正式运行）
    python main.py --input data/hospital_dirty.csv --output data/hospital_errors.csv --rules-out data/hospital_rules.json

    # 等价入口
    python -m stage_1.cli --input data/hospital_dirty.csv --output data/hospital_errors.csv --rules-out data/hospital_rules.json

    # 仅画像，不调 LLM（零成本验证数据）
    python main.py --input data/hospital_dirty.csv --dry-run

    # 可选：调整规则违反率上限（默认 0.3）
    python main.py --input data/hospital_dirty.csv --output data/hospital_errors.csv --rules-out data/hospital_rules.json --max-violation-rate 0.25

产出:
    data/hospital_errors.csv   规则层检出的错误单元格
    data/hospital_rules.json   归纳规则（供 Stage 3 取 semantic_type）
    clean_mask.csv             干净单元格掩码（Stage 2 训练必需）

下一步:
    python -m stage_2.cli

CSV 读取策略:
    dtype=str, keep_default_na=False, na_values=['']
    避免 pandas 将空串自动转为 NaN，干扰 not_null 检测。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stage_1.config import DEFAULT_CONFIG_PATH, Stage1Config
from stage_1.executor import build_clean_mask, run_rule_layer


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Stage 1: LLM-assisted rule-based table error detection",
    )
    parser.add_argument("--input", required=True, help="输入 CSV 路径")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="配置文件路径 (default: configs/default.yaml)",
    )
    parser.add_argument("--output", default=None, help="错误输出 CSV 路径")
    parser.add_argument("--rules-out", default=None, help="归纳规则输出 JSON 路径")
    parser.add_argument("--rule-cache", default=None, help="规则缓存文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只做画像,不调 LLM")
    parser.add_argument(
        "--max-violation-rate",
        type=float,
        default=None,
        help="规则违反率上限,超过则视为不可信并丢弃",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI 主函数：读 CSV -> 执行流水线 -> 写结果文件。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    cli_overrides = {
        "output": args.output,
        "rules_out": args.rules_out,
        "rule_cache": args.rule_cache,
        "max_violation_rate": args.max_violation_rate,
    }
    config = Stage1Config.resolve(config_path=args.config, cli_overrides=cli_overrides)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, na_values=[""])
    print(f"加载数据: {df.shape[0]} 行, {df.shape[1]} 列")

    errors, rule_report = run_rule_layer(df, config=config, dry_run=args.dry_run)

    if args.dry_run:
        return

    # 写入归纳规则报告
    rules_path = Path(config.paths.rules_output)
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rule_report, f, ensure_ascii=False, indent=2)
    print(f"\n归纳规则已写入: {rules_path}")

    # 写入错误检测结果
    errors_path = Path(config.paths.errors_output)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(errors_path, index=False)
    print(f"检测到 {len(errors)} 个错误单元格,已写入: {errors_path}")
    if len(errors):
        print("\n错误类型分布:")
        print(errors["error_type"].value_counts().to_string())
        print("\n列分布:")
        print(errors["column"].value_counts().to_string())

    # 写入干净单元格掩码（供 Stage 2 训练使用）
    if config.paths.clean_mask_output:
        mask_path = Path(config.paths.clean_mask_output)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        clean_mask = build_clean_mask(df, errors)
        clean_mask.to_csv(mask_path, index=False)
        clean_cells = int(clean_mask.values.sum())
        total_cells = clean_mask.size
        print(
            f"干净单元格掩码已写入: {mask_path} "
            f"({clean_cells}/{total_cells} 单元格干净)"
        )


if __name__ == "__main__":
    main()
