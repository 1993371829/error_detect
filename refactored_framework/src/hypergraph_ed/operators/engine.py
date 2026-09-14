from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import regex as bounded_regex

from ..data.table import is_blank, parse_number
from ..evidence.schema import Evidence, Status


OPERATOR_FAMILY = {"missing": "missing", "dmv": "missing", "type": "format", "regex": "format", "duplicate": "format", "normalize": "semantic", "range": "domain", "statistical": "statistical", "neighbor": "neighbor", "fd": "relation", "compare": "relation", "temporal": "relation", "arithmetic": "relation"}
PARAMS = {"missing": set(), "dmv": {"tokens"}, "type": {"kind"}, "regex": {"pattern"}, "duplicate": set(), "normalize": {"mapping", "strip", "case"}, "range": {"min", "max", "unit", "semantic_type"}, "statistical": {"z"}, "neighbor": {"min_support"}, "fd": {"lhs", "rhs", "min_support", "confidence"}, "compare": {"left", "right"}, "temporal": {"left", "right"}, "arithmetic": {"expression", "variables", "tolerance"}}
AST_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Constant)


def validate_constraint(edge, columns: dict) -> None:
    if edge.operator not in PARAMS:
        raise ValueError(f"unknown operator: {edge.operator}")
    if set(edge.params) - PARAMS[edge.operator]:
        raise ValueError(f"unknown parameters for {edge.operator}")
    p = edge.params
    if edge.operator == "regex":
        expression = p.get("pattern", "")
        if not expression or len(expression) > 128 or any(x in expression for x in ("(", ")", "|", "\\1")):
            raise ValueError("regex must be a short non-branching pattern without groups")
        try:
            re.compile(expression)
        except re.error as exc:
            raise ValueError("invalid regex") from exc
    if edge.operator == "range":
        if "min" not in p and "max" not in p:
            raise ValueError("range requires a bound")
        bounds = [float(p[k]) for k in ("min", "max") if k in p]
        if not all(math.isfinite(x) for x in bounds) or p.get("min", -math.inf) > p.get("max", math.inf):
            raise ValueError("invalid numeric bounds")
        for col in edge.targets:
            spec = columns[col]
            declared = p.get("semantic_type", spec.semantic_type)
            if spec.semantic_type != "unknown" and declared != spec.semantic_type:
                raise ValueError("semantic type mismatch")
            if spec.semantic_type == "duration" and (p.get("unit") == "percent" or declared == "percentage"):
                raise ValueError("duration cannot use percentage bounds")
            if spec.unit and p.get("unit", spec.unit) != spec.unit:
                raise ValueError("incompatible unit")
    if edge.operator == "type" and p.get("kind") not in {"numeric", "integer", "date", "boolean"}:
        raise ValueError("unknown logical type")
    if edge.operator == "normalize":
        if p.get("case", "") not in {"", "lower", "upper"}:
            raise ValueError("invalid case conversion")
        if not isinstance(p.get("mapping", {}), dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in p.get("mapping", {}).items()):
            raise ValueError("mapping must map strings to strings")
    if edge.operator == "fd":
        lhs, rhs = p.get("lhs", []), p.get("rhs")
        if not lhs or rhs in lhs or not set(lhs + [rhs]) <= set(edge.participants):
            raise ValueError("FD needs disjoint lhs/rhs from participants")
    if edge.operator in {"compare", "temporal"}:
        if p.get("left") == p.get("right") or not {p.get("left"), p.get("right")} <= set(edge.participants):
            raise ValueError("comparison requires two known columns")
    if edge.operator == "arithmetic":
        variables = p.get("variables", {})
        if not variables or not set(variables.values()) <= set(edge.participants):
            raise ValueError("arithmetic variables must refer to participants")
        tree = ast.parse(p.get("expression", ""), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, AST_ALLOWED) or (isinstance(node, ast.Name) and node.id not in variables):
                raise ValueError("unsupported arithmetic expression")
            if isinstance(node, ast.Constant) and (not isinstance(node.value, (float, int)) or not math.isfinite(node.value)):
                raise ValueError("arithmetic constants must be finite numbers")


def _prepare(edge, refs):
    p = edge.params
    prepared = {}
    if edge.operator == "statistical":
        for col in edge.targets:
            buckets = defaultdict(list)
            for v in refs[col]:
                number = parse_number(v)
                if number:
                    buckets[number[1]].append(number[0])
            prepared[col] = {}
            for unit, values in buckets.items():
                vals = np.asarray(values)
                median = float(np.median(vals))
                mad = max(float(np.median(np.abs(vals-median))) * 1.4826, float(np.std(vals)) * 0.1, 1e-6)
                prepared[col][unit] = (median, mad, len(vals))
    if edge.operator in {"fd", "neighbor"}:
        for col in edge.targets:
            keys = p["lhs"] if edge.operator == "fd" else [c for c in edge.participants if c != col]
            target = p["rhs"] if edge.operator == "fd" else col
            groups = defaultdict(Counter)
            for record in refs.to_dict("records"):
                if not is_blank(record[target]):
                    groups[tuple(record[k] for k in keys)][str(record[target])] += 1
            prepared[col] = (keys, target, groups)
    return prepared


def _check(edge, row: dict, col: str, refs: pd.DataFrame, row_id: int, prepared=None) -> tuple:
    op, p = edge.operator, edge.params
    value = str(row[col])
    blank = is_blank(value)
    if not edge.applicable(row):
        return None, "outside condition", None, 0
    if op == "missing":
        return blank, "required field is empty" if blank else "required field present", None, len(refs)
    if blank:
        return None, "empty value not evaluated by this check", None, 0
    if op == "dmv":
        tokens = p.get("tokens", ["unknown", "missing", "?", "not available", "unspecified"])
        return value.strip().lower() in tokens, "placeholder membership", None, len(refs)
    if op == "duplicate":
        parts = [x.strip() for x in value.split(",")]
        bad = len(parts) > 1 and bool(parts[0]) and len(set(parts)) == 1
        return bad, "repeated complete token", parts[0] if bad else None, len(refs)
    if op == "normalize":
        fixed = p.get("mapping", {}).get(value, value)
        if p.get("strip"):
            fixed = fixed.strip()
        if p.get("case"):
            fixed = getattr(fixed, p["case"])()
        return value != fixed, "explicit representation mapping", fixed if value != fixed else None, len(refs)
    if op == "regex":
        if len(value) > 512:
            return None, "value exceeds regex resource limit", None, 0
        try:
            matched=bounded_regex.fullmatch(p["pattern"],value,timeout=0.05)
        except TimeoutError:
            return None,"regex execution time limit",None,0
        return matched is None, "format check", None, len(refs)
    if op == "type":
        kind = p["kind"]
        number = parse_number(value)
        ok = number is not None if kind == "numeric" else number is not None and number[0].is_integer() if kind == "integer" else value.strip().lower() in {"true", "false", "yes", "no", "0", "1"} if kind == "boolean" else not pd.isna(pd.to_datetime(value, errors="coerce"))
        return not ok, f"logical type {kind}", None, len(refs)
    if op in {"range", "statistical"}:
        number = parse_number(value)
        if number is None:
            return None, "number cannot be parsed", None, 0
        x, unit = number
        if op == "range":
            if p.get("unit") and unit != p["unit"]:
                return None, "unit mismatch; no implicit conversion", None, 0
            return x < p.get("min", -math.inf) or x > p.get("max", math.inf), "typed numeric bound", None, len(refs)
        info = (prepared if prepared is not None else _prepare(edge, refs)).get(col, {}).get(unit)
        if info is None or info[2] < 5:
            return None, "insufficient same-unit reference", None, 0
        median, mad, support = info
        return abs(x - median) / mad > p.get("z", 6), "robust same-unit deviation", str(median), support
    if op in {"fd", "neighbor"}:
        keys, target, groups = (prepared if prepared is not None else _prepare(edge, refs))[col]
        if not keys:
            return None, "no neighbor context", None, 0
        key = tuple(row[k] for k in keys)
        counts = groups.get(key, Counter()).copy()
        if row_id in refs.index:
            original = refs.loc[row_id]
            if tuple(original[k] for k in keys) == key:
                counts[str(original[target])] -= 1
        counts = +counts
        support = sum(counts.values())
        if support < p.get("min_support", 3) or not counts:
            return None, "insufficient peer group", None, support
        top, amount = counts.most_common(1)[0]
        if amount / support < p.get("confidence", 0.9):
            return None, "peer group is not stable", None, support
        return str(row[target]) != top, "peer disagreement; attribution initially unresolved", top if col == target else None, support
    if op in {"compare", "temporal"}:
        if op == "temporal":
            left = pd.to_datetime(row[p["left"]], errors="coerce")
            right = pd.to_datetime(row[p["right"]], errors="coerce")
            if pd.isna(left) or pd.isna(right):
                return None, "unparseable temporal relation", None, 0
        else:
            a, b = parse_number(row[p["left"]]), parse_number(row[p["right"]])
            if not a or not b or a[1] != b[1]:
                return None, "incompatible comparison units", None, 0
            left, right = a[0], b[0]
        return left > right, "left must not exceed right", None, len(refs)
    if op == "arithmetic":
        env = {}
        for var, field in p["variables"].items():
            number = parse_number(row[field])
            if not number:
                return None, "unparseable arithmetic input", None, 0
            env[var] = number[0]
        try:
            residual = float(eval(compile(ast.parse(p["expression"], mode="eval"), "<validated arithmetic>", "eval"), {"__builtins__": {}}, env))
        except (ZeroDivisionError, OverflowError, ValueError):
            return None, "undefined arithmetic", None, 0
        return abs(residual) > p.get("tolerance", 1e-6), "arithmetic residual", None, len(refs)
    raise ValueError(op)


def execute(table: pd.DataFrame, program, reference: pd.DataFrame | None = None) -> list[Evidence]:
    reference = table if reference is None else reference
    result = []
    for edge in program.constraints:
        refs = reference
        for condition in edge.conditions:
            refs = refs[refs[condition.column] == condition.value]
        prepared = _prepare(edge, refs)
        for row_id, row in zip(table.index, table.to_dict("records")):
            for col in edge.targets:
                bad, reason, fixed, support = _check(edge, row, col, refs, int(row_id), prepared)
                status = Status.ABSTAIN if bad is None else Status.VIOLATION if bad else Status.PASS
                family = OPERATOR_FAMILY[edge.operator]
                et = "MV" if edge.operator == "missing" else "DMV" if edge.operator == "dmv" else "VAD" if family == "relation" else "OTHER" if family in {"statistical", "neighbor"} else "FI"
                result.append(Evidence(int(row_id), col, edge.id, edge.canonical(), family, status, 0.9 if bad else 0., et, reason, fixed if bad else None, support, min(0.95, 0.5 + support / (support + 20) * 0.45), {"operator": edge.operator, "participants": edge.participants, "source": edge.source}))
    return result
