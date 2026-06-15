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

        # not_null 已由 executor 的 MV 扫描处理，此处不支持
        # 未知规则类型，安全跳过
        return lambda v: None
