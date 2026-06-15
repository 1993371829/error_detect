"""
Stage 1 包：基于 LLM 归纳规则的表格错误检测。

流水线概览:
    数据画像 -> LLM 规则归纳 -> 规则编译 -> 规则执行 + 错误标记

对外暴露 run_rule_layer 供程序化调用；命令行入口见 stage_1.cli。
"""

from stage_1.executor import run_rule_layer

__all__ = ["run_rule_layer"]
