from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from ..artifacts import fingerprint
from ..config import StrictModel


class ColumnSpec(StrictModel):
    name: str
    kind: Literal["numeric", "categorical", "text"] = "text"
    semantic_type: str = "unknown"
    unit: str = ""
    nullable: Optional[bool] = None


class Condition(StrictModel):
    column: str
    value: str


class Constraint(StrictModel):
    id: str
    operator: str
    participants: list[str]
    targets: list[str]
    conditions: list[Condition] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    source: str = "local"
    description: str = ""
    version: int = 1

    @model_validator(mode="after")
    def valid_scope(self):
        if not 1 <= len(self.participants) <= 4 or len(set(self.participants)) != len(self.participants):
            raise ValueError("one to four distinct participant columns required")
        if not self.targets or not set(self.targets) <= set(self.participants):
            raise ValueError("targets must belong to participants")
        if len(self.conditions) > 2 or len({x.column for x in self.conditions}) != len(self.conditions):
            raise ValueError("at most two distinct condition columns")
        return self

    def canonical(self) -> str:
        return fingerprint({"op": self.operator, "participants": self.participants, "targets": sorted(self.targets), "conditions": sorted((x.column, x.value) for x in self.conditions), "params": self.params})

    def applicable(self, row: dict) -> bool:
        return all(str(row.get(c.column, "")) == c.value for c in self.conditions)


class ConstraintProgram(StrictModel):
    version: int = 1
    columns: list[ColumnSpec]
    constraints: list[Constraint]

    @model_validator(mode="after")
    def validate_program(self):
        from ..operators import validate_constraint
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("duplicate column definitions")
        ids = set()
        for edge in self.constraints:
            if edge.id in ids:
                raise ValueError("duplicate constraint id")
            ids.add(edge.id)
            if not (set(edge.participants) | {x.column for x in edge.conditions}) <= set(names):
                raise ValueError(f"unknown column in {edge.id}")
            validate_constraint(edge, {c.name: c for c in self.columns})
        return self


def merge_programs(programs: list[ConstraintProgram]) -> ConstraintProgram:
    if not programs:
        raise ValueError("at least one program required")
    columns = {c.name: c.model_copy(deep=True) for c in programs[0].columns}
    edges = {}
    for program in programs:
        if set(c.name for c in program.columns) != set(columns):
            raise ValueError("program schema does not match table")
        for c in program.columns:
            old = columns[c.name]
            if c.semantic_type != "unknown" and old.semantic_type == "unknown":
                columns[c.name] = c.model_copy(deep=True)
        for edge in program.constraints:
            edges.setdefault(edge.canonical(), edge)
    count = {c: 0 for c in columns}
    kept = []
    # Keep explicit semantic checks before statistical fallbacks, deterministically.
    order = {"missing": 0, "range": 1, "normalize": 2, "regex": 3, "fd": 4, "arithmetic": 4, "compare": 4, "temporal": 4, "type": 5, "dmv": 6, "duplicate": 7, "statistical": 8, "neighbor": 9}
    for key, edge in sorted(edges.items(), key=lambda kv: (order.get(kv[1].operator, 9), kv[0])):
        if len(kept) >= 96 or any(count[c] >= 6 for c in edge.targets):
            continue
        kept.append(edge.model_copy(update={"id": key[:16]}))
        for c in edge.targets:
            count[c] += 1
    return ConstraintProgram(columns=list(columns.values()), constraints=kept)
