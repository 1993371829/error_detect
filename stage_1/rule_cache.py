"""
LLM 规则缓存模块。

该模块用于缓存“画像”到“规则”的 LLM 归纳结果，避免重复向 LLM 发起相同画像的规则归纳请求。
缓存的索引（key）为画像 profile 的 MD5 哈希，只要画像完全相同，无论顺序如何，得到的 key 都一致。
缓存内容持久化保存为 JSON 文件，默认由配置指定为 `cache/rule_cache.json`。

注意事项：
- 如果 prompt、规则类型或归纳逻辑发生变动时，需手动删除缓存文件，避免旧结果被误复用！
"""

from __future__ import annotations

import hashlib
import json
import os


class RuleCache:
    """
    画像 -> 规则 的本地文件缓存。

    用于将每个表列画像（profile）与其 LLM 归纳的规则（rules）对应缓存起来。
    通过本地 JSON 文件持久化，重复运行时可复用缓存，大幅减少 LLM 调用次数。
    """

    def __init__(self, path: str = ".rule_cache.json"):
        """
        初始化缓存对象。

        Args:
            path: 缓存文件路径（默认 .rule_cache.json，和当前脚本同级）。
        """
        self.path = path                                   # 缓存文件路径
        self.cache = {}                                    # 内存中的缓存字典

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
        查询缓存：如果该画像已有规则，则直接返回规则 dict，否则返回 None。

        Args:
            profile: 列画像 dict

        Returns:
            对应的规则 dict（若命中），未命中时返回 None
        """
        return self.cache.get(self._key(profile))

    def set(self, profile: dict, rules: dict):
        """
        新增或更新指定画像的规则到缓存，并同步写入本地 JSON 文件。

        Args:
            profile: 列画像 dict
            rules:   规则 dict（LLM 归纳结果）
        """
        self.cache[self._key(profile)] = rules
        # 立即持久化保存，避免丢失
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
