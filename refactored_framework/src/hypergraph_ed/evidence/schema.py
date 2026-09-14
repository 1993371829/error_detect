from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Status(str, Enum):
    VIOLATION = "violation"
    PASS = "check_passed"
    ABSTAIN = "abstain"


@dataclass
class Evidence:
    row_id: int
    column: str
    constraint_id: str
    source_group: str
    family: str
    status: Status
    score: float
    error_type: str
    reason: str
    suggested_fix: str | None = None
    support: int = 0
    reliability: float = 0.5
    details: dict = field(default_factory=dict)

    def record(self) -> dict:
        record = asdict(self)
        record["status"] = self.status.value
        return record
