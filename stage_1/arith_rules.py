"""
LLM 跨列算术/约束规则（文档 §5.5）。

让 LLM 归纳跨列算术关系（如 total == subtotal + tax），用 AST 安全求值在数据上
验证（support / violation_rate），仅保留高支持低违反率的规则，再标记违反行的
result_column 单元格（FI/arithmetic_constraint）。默认关闭。

安全：表达式只允许数字、列名、+ - * / ()、比较、and/or/not 与 abs/min/max/round，
任何其它 AST 节点（属性访问、下标、函数调用等）一律拒绝，杜绝代码注入。
"""

from __future__ import annotations

import ast
import math
from typing import Callable, Optional

import pandas as pd

from stage_1.profiling import is_blank
from stage_2.encoding import _extract_number

_ALLOWED_FUNCS = {"abs": abs, "min": min, "max": max, "round": round}

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.USub, ast.UAdd,
    ast.Not, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.Call, ast.Name, ast.Load, ast.Constant,
)


class UnsafeExpression(ValueError):
    """表达式含不允许的语法节点。"""


def _collect_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpression(f"不允许的语法节点: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                raise UnsafeExpression("只允许 abs/min/max/round 函数")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCS:
            names.add(node.id)
    return names


def compile_arith(expression: str) -> tuple[set[str], Callable[[dict], Optional[bool]]]:
    """编译算术约束 -> (引用的列名集合, 求值函数)；求值函数缺值时返回 None。"""
    tree = ast.parse(expression, mode="eval")
    columns = _collect_names(tree)
    code = compile(tree, "<arith>", "eval")

    def _eval(values: dict) -> Optional[bool]:
        env = dict(_ALLOWED_FUNCS)
        for col in columns:
            v = values.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            env[col] = v
        try:
            return bool(eval(code, {"__builtins__": {}}, env))  # noqa: S307 已 AST 白名单
        except (ZeroDivisionError, TypeError, ValueError):
            return None

    return columns, _eval


def validate_and_apply(
    df: pd.DataFrame,
    rules: list[dict],
    *,
    min_support: int = 30,
    max_violation_rate: float = 0.02,
) -> list[dict]:
    """验证算术规则并标记违反行的 result_column 单元格。"""
    errors: list[dict] = []
    for rule in rules:
        expr = str(rule.get("expression", "")).strip()
        result_col = str(rule.get("result_column", "")).strip()
        if not expr or result_col not in df.columns:
            continue
        try:
            cols, fn = compile_arith(expr)
        except (SyntaxError, UnsafeExpression):
            continue
        if not cols.issubset(set(map(str, df.columns))):
            continue

        support, violations, viol_rows = 0, 0, []
        for pos in range(len(df)):
            values = {}
            ok = True
            for c in cols:
                raw = df.iloc[pos][c]
                if is_blank(raw):
                    ok = False
                    break
                values[c] = _extract_number(str(raw))
            if not ok:
                continue
            res = fn(values)
            if res is None:
                continue
            support += 1
            if not res:
                violations += 1
                viol_rows.append(pos)
        if support < min_support:
            continue
        if violations / support > max_violation_rate:
            continue
        for pos in viol_rows:
            errors.append({
                "row_id": df.index[pos], "column": result_col,
                "value": df.iloc[pos][result_col],
                "error_type": "FI", "violated_rule": "arithmetic_constraint",
                "reason": f"违反算术约束 {expr}（支持度 {support}，违反率 "
                          f"{violations/support:.1%}）",
                "confidence": 0.9,
            })
    return errors


# --------------------------------------------------------------------------- #
# LLM 生成（可选）
# --------------------------------------------------------------------------- #

ARITH_PROMPT = """你是数据质量专家。下面是一张表的列名与各列若干样例值。请判断是否存在
跨列的算术/数值约束关系（例如 total == subtotal + tax、price == unit_price * quantity、
end_year >= start_year）。只在你高度确信该关系应当严格成立时才输出。

列与样例:
{columns}

只输出 JSON（不要解释）。expression 只能用列名、数字、+ - * / ()、比较运算与 abs()，
result_column 为该约束中“被推导/最可能出错”的列:
{{"arithmetic_rules": [{{"expression": "abs(total - (subtotal + tax)) <= 0.01", "result_column": "total", "reason": "简短理由"}}]}}
若无此类关系，输出 {{"arithmetic_rules": []}}。"""


def generate_arithmetic_rules(llm, df: pd.DataFrame, max_samples: int = 5) -> list[dict]:
    """调用 LLM 归纳跨列算术规则（llm 为空或失败返回空表）。"""
    if llm is None:
        return []
    from stage_1.llm_rules import complete_json
    lines = []
    for c in df.columns:
        nb = df[c][~df[c].map(is_blank)].astype(str).head(max_samples).tolist()
        lines.append(f"- {c}: {nb}")
    prompt = ARITH_PROMPT.format(columns="\n".join(lines))
    data = complete_json(llm, prompt, label="arithmetic_rules")
    if isinstance(data, dict) and isinstance(data.get("arithmetic_rules"), list):
        return data["arithmetic_rules"]
    return []
