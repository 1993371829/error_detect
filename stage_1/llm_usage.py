"""
LLM token 用量统计。

通过环境变量 LLM_USAGE_FILE 指定 JSON 路径，各阶段子进程在每次 API 调用后
累加 prompt/completion/total tokens。run_pipeline.py 在流水线结束后读取汇总。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class LLMUsageStats:
    """跨进程可累加的 LLM 用量计数器。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Any) -> None:
        """累加单次 completion 的 usage 对象（OpenAI 兼容）。"""
        if usage is None:
            return
        self.calls += 1
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        total = getattr(usage, "total_tokens", None)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        if total is not None:
            self.total_tokens += int(total or 0)
        else:
            self.total_tokens += pt + ct

    @classmethod
    def from_dict(cls, data: dict | None) -> "LLMUsageStats":
        if not data:
            return cls()
        return cls(
            calls=int(data.get("calls", 0) or 0),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
        )

    @classmethod
    def load(cls, path: Path) -> "LLMUsageStats":
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def to_dict(self) -> dict:
        return asdict(self)


def usage_file_path() -> Optional[Path]:
    raw = os.environ.get("LLM_USAGE_FILE")
    return Path(raw) if raw else None


def init_usage_file(path: Path) -> None:
    """初始化（或重置）用量文件。"""
    LLMUsageStats().save(path)


def record_llm_usage(usage: Any) -> None:
    """每次 LLM API 调用后累加 token（无 LLM_USAGE_FILE 时静默跳过）。"""
    path = usage_file_path()
    if path is None:
        return
    stats = LLMUsageStats.load(path)
    stats.add(usage)
    stats.save(path)


def load_usage(path: Path) -> LLMUsageStats:
    return LLMUsageStats.load(path)
