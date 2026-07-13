"""
LLM 规则缓存模块。

该模块用于缓存“画像”到“规则”的 LLM 归纳结果，避免重复向 LLM 发起相同画像的规则归纳请求。
缓存的索引（key）为画像 profile 的 MD5 哈希；条目内另存版本标记 `_v`（prompt 版本号 + 模型名）：
改 prompt / 换模型时版本不匹配视为未命中并覆盖写入，无需手动删缓存文件。
无 `_v` 的历史条目视为有效（升级前的缓存均产自 v1 prompt + 当时模型，平滑沿用）。
缓存内容持久化保存为 JSON 文件，默认由配置指定为 `cache/rule_cache.json`。
"""

from __future__ import annotations

import hashlib
import json
import os

# Stage 1 各 LLM prompt（规则归纳/标准化/xcol/FD/CFD/DC 审核）的整体版本号。
# 任一 prompt 模板发生语义性变更时递增，使旧缓存自动失效。
STAGE1_PROMPT_VERSION = "v1"


class RuleCache:
    """
    画像 -> 规则 的本地文件缓存。

    用于将每个表列画像（profile）与其 LLM 归纳的规则（rules）对应缓存起来。
    通过本地 JSON 文件持久化，重复运行时可复用缓存，大幅减少 LLM 调用次数。
    """

    def __init__(self, path: str = ".rule_cache.json", *, model: str = ""):
        """
        初始化缓存对象。

        Args:
            path: 缓存文件路径（默认 .rule_cache.json，和当前脚本同级）。
            model: LLM 模型名，与 prompt 版本号一起构成条目版本标记 `_v`
                   （版本不匹配的条目视为 miss；空串表示不校验版本）。
        """
        self.path = path                                   # 缓存文件路径
        self.cache = {}                                    # 内存中的缓存字典
        self._version = f"{STAGE1_PROMPT_VERSION}|{model}" if model else ""

        # 如缓存文件已存在，则加载进来
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    @staticmethod
    def _key(profile: dict) -> str:
        """
        将画像 profile 序列化为 JSON 字符串（保证排序、避免顺序影响），
        并取其 MD5 哈希，作为缓存的唯一 key。

        Args:
            profile: 字典类型的列画像

        Returns:
            画像的 MD5 字符串（32位）。
        """
        blob = json.dumps(profile, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    def get(self, profile: dict):
        """
        查询缓存：命中且版本匹配（或为无版本的历史条目）返回规则 dict，否则 None。

        Args:
            profile: 列画像 dict

        Returns:
            对应的规则 dict（不含内部版本标记 `_v`），未命中时返回 None
        """
        entry = self.cache.get(self._key(profile))
        if entry is None:
            return None
        if isinstance(entry, dict):
            v = entry.get("_v")
            if v is not None and self._version and v != self._version:
                return None  # prompt 版本或模型已变，旧结论失效
            return {k: val for k, val in entry.items() if k != "_v"}
        return entry

    def set(self, profile: dict, rules: dict):
        """
        新增或更新指定画像的规则到缓存，并同步写入本地 JSON 文件。

        Args:
            profile: 列画像 dict
            rules:   规则 dict（LLM 归纳结果）
        """
        entry = dict(rules) if isinstance(rules, dict) else rules
        if isinstance(entry, dict) and self._version:
            entry["_v"] = self._version
        self.cache[self._key(profile)] = entry
        # 立即持久化保存，避免丢失
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
