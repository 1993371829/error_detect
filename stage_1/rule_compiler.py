"""
Step 3: 规则编译（Rule Compilation）。

将 LLM 输出的 JSON 规则编译为可执行的校验函数。

每个编译后的函数签名:
    f(value) -> None | (error_type, rule_dict)

    - 返回 None 表示通过校验
    - 返回 (error_type, rule) 表示违反该规则

空值处理策略:
    - 所有格式规则遇空值一律跳过
    - 缺失值检测(MV)由 executor 的独立扫描负责，不在此处处理
"""

from __future__ import annotations

import re

from stage_1.profiling import is_blank

# bool 逻辑类型可接受的取值（strip + lower 后比对）
_BOOL_VALUES = {"true", "false", "t", "f", "yes", "no", "y", "n", "0", "1"}
# date 逻辑类型的宽松校验：至少两段数字 + 分隔符，可带时间后缀。
# 故意宽松：若列内日期格式与之系统性不符，会被 executor 的 max_violation_rate 兜底丢弃，
# 因此宁可宽松，避免对合法日期产生误报。
_DATE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}([-/.]\d{1,4})?([ T].*)?$")


class RuleCompiler:
    """JSON 规则 -> 校验函数 的编译器（仅格式类规则）。"""

    def compile(self, rule: dict):
        """
        编译单条规则为校验函数。

        Args:
            rule: 含 type, spec, error_type, reason 等字段的 dict

        Returns:
            callable(value) -> None | (error_type, rule)
        """
        rtype = rule.get("type")
        spec = rule.get("spec", {}) or {}
        error_type = rule.get("error_type", "OTHER")

        if rtype == "regex":
            pattern_str = spec.get("pattern")
            if not pattern_str:
                return lambda v: None
            try:
                pattern = re.compile(pattern_str)
            except re.error:
                print(f"[warn] 非法正则,跳过: {pattern_str}")
                return lambda v: None

            def f(v):
                if is_blank(v):
                    return None
                # 使用 match 而非 search，要求从头匹配
                return (error_type, rule) if not pattern.match(str(v)) else None
            return f

        if rtype == "value_set":
            allowed = set(str(x) for x in spec.get("values", []))
            if not allowed:
                return lambda v: None

            def f(v):
                if is_blank(v):
                    return None
                return (error_type, rule) if str(v) not in allowed else None
            return f

        if rtype == "length":
            def f(v):
                if is_blank(v):
                    return None
                length = len(str(v))
                if "fixed" in spec and length != spec["fixed"]:
                    return (error_type, rule)
                if "min" in spec and length < spec["min"]:
                    return (error_type, rule)
                if "max" in spec and length > spec["max"]:
                    return (error_type, rule)
                return None
            return f

        if rtype == "numeric_range":
            def f(v):
                if is_blank(v):
                    return None
                try:
                    num = float(v)
                except (ValueError, TypeError):
                    # 非数值由 regex/length 等规则处理，此处静默跳过
                    return None
                if "min" in spec and num < spec["min"]:
                    return (error_type, rule)
                if "max" in spec and num > spec["max"]:
                    return (error_type, rule)
                return None
            return f

        if rtype == "logical_type":
            return self._compile_logical_type(spec, error_type, rule)

        # not_null 已由 executor 的 MV 扫描处理，此处不支持
        # 未知规则类型，安全跳过
        return lambda v: None

    @staticmethod
    def _compile_logical_type(spec: dict, error_type: str, rule: dict):
        """
        编译"逻辑类型一致性"校验（借鉴 Cocoon 的列类型语义校验）。

        仅校验 bool/int/float/date 四类有明确判定标准的类型；
        categorical/string/未知类型一律放行（返回 no-op）。
        非空值才校验，空值由 MV 扫描负责。
        """
        logical = str(spec.get("logical_type", "") or "").strip().lower()

        if logical in ("bool", "boolean"):
            def f(v):
                if is_blank(v):
                    return None
                return None if str(v).strip().lower() in _BOOL_VALUES else (error_type, rule)
            return f

        if logical in ("int", "integer"):
            def f(v):
                if is_blank(v):
                    return None
                s = str(v).strip()
                try:
                    int(s)
                    return None
                except (ValueError, TypeError):
                    return (error_type, rule)
            return f

        if logical in ("float", "numeric", "number"):
            def f(v):
                if is_blank(v):
                    return None
                try:
                    float(str(v).strip())
                    return None
                except (ValueError, TypeError):
                    return (error_type, rule)
            return f

        if logical == "date":
            def f(v):
                if is_blank(v):
                    return None
                return None if _DATE_RE.match(str(v).strip()) else (error_type, rule)
            return f

        # categorical / string / 未知：不做类型约束
        return lambda v: None
