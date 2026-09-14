from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from ..data.table import profile_table, row_keys
from ..operators import execute
from ..operators.engine import PARAMS
from ..evidence.schema import Status
from .schema import ColumnSpec, Constraint, ConstraintProgram, merge_programs


def local_program(table: pd.DataFrame) -> ConstraintProgram:
    profiles = profile_table(table)
    columns, edges = [], []
    for profile in profiles:
        name = profile["name"]
        units = profile["units"]
        kind = profile["kind"]
        semantic = profile["semantic_type"]
        unit = units[0] if len(units) == 1 else ""
        columns.append(ColumnSpec(name=name, kind=kind, semantic_type=semantic, unit=unit))
        def add(op: str, params: dict | None = None):
            edges.append(Constraint(id=f"local_{len(edges)}", operator=op, participants=[name], targets=[name], params=params or {}))
        # Missingness is a weak task hypothesis, not a hard truth label.
        if profile["missing_ratio"] < 0.5:
            add("missing")
        if kind == "numeric":
            add("type", {"kind": "numeric"})
            bounds = {"age": (0, 120), "year": (1000, 2100), "latitude": (-90, 90), "longitude": (-180, 180), "percentage": (0, 100)}
            if semantic in bounds:
                lo, hi = bounds[semantic]
                add("range", {"min": lo, "max": hi, "unit": unit, "semantic_type": semantic})
            add("statistical", {"z": 6})
        else:
            add("dmv")
            add("duplicate")
    # Purely statistical proposals: LLM may qualify these in its program.
    for lhs, rhs in combinations(table.columns, 2):
        for a, b in [(lhs, rhs), (rhs, lhs)]:
            if not 2 <= table[a].nunique() <= max(2, len(table) // 3):
                continue
            groups = table.groupby(a, dropna=False)[b]
            good = [g for _, g in groups if len(g) >= 3]
            if good and sum(g.value_counts().iloc[0] for g in good) / sum(map(len, good)) >= 0.9:
                edges.append(Constraint(id=f"local_{len(edges)}", operator="fd", participants=[a, b], targets=[a, b], params={"lhs": [a], "rhs": b, "min_support": 3, "confidence": 0.9}))
    return merge_programs([ConstraintProgram(columns=columns, constraints=edges)])


def sample_views(table, base, seed):
    frequency = np.zeros(len(table))
    for col in table:
        counts = table[col].value_counts()
        frequency += table[col].map(counts).to_numpy() / len(table)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0,1e-6,len(table))
    typical = np.argsort(-(frequency+jitter)).tolist()
    rare = np.argsort(frequency+jitter).tolist()
    pool = sorted(set(typical[:96]+rare[:96]+rng.permutation(len(table))[:64].tolist()))
    signals = execute(table.iloc[pool],base,reference=table)
    cell_signals = defaultdict(set)
    severity = defaultdict(float)
    for e in signals:
        if e.status != Status.ABSTAIN:
            cell_signals[(e.row_id,e.column)].add(e.status)
        if e.status == Status.VIOLATION:
            severity[e.row_id] += e.score
    for (i,_),states in cell_signals.items():
        if len(states)>1:
            severity[i] += 1
    conflict = sorted(pool,key=lambda i:(-severity[i],frequency[i]))
    keys = row_keys(table)
    def unique_rows(order):
        selected,seen = [],set()
        for i in order:
            if keys[i] not in seen:
                selected.append(i); seen.add(keys[i])
            if len(selected)==6:
                break
        return selected
    return [unique_rows(order) for order in (typical,rare,conflict)]


def induce(table: pd.DataFrame, client, base: ConstraintProgram, seed: int = 42) -> tuple[ConstraintProgram, list[str]]:
    programs, issues = [base], []
    profiles = profile_table(table)
    views = sample_views(table,base,seed)
    schema = ConstraintProgram.model_json_schema()
    for name, indices in zip(["typical", "rare", "conflict"], views):
        prompt = {"task": "Design a constrained executable semantic hypergraph. Do not label all cells. Only use registered operators. Empty constraints are allowed. Every column must be defined; never invent column names. Different units must not be compared implicitly.", "view": name, "profiles": profiles, "rows": table.loc[indices].to_dict("records"), "schema": schema, "operator_params": {k:sorted(v) for k,v in PARAMS.items()}}
        answer = client.ask("structure", json.dumps(prompt, ensure_ascii=False, default=list))
        if answer is None:
            issues.append(f"structure {name}: unavailable")
            continue
        if answer.get("columns") == [] and answer.get("constraints") == []:
            continue  # Explicit fixture/no-additional-knowledge response.
        for repair in range(2):
            try:
                proposed = ConstraintProgram.model_validate(answer)
                if [c.name for c in proposed.columns] != list(table.columns):
                    raise ValueError("column order/schema mismatch")
                for e in proposed.constraints:
                    e.source = f"llm:{name}"
                merge_programs(programs + [proposed])
                programs.append(proposed)
                break
            except (ValueError, TypeError) as exc:
                issues.append(f"invalid structure {name}: {exc}")
                if repair==0:
                    answer=client.ask("structure",json.dumps({"task":"Repair this structure to match the original contract. Return only the corrected program.","contract":prompt,"invalid_program":answer,"validation_error":str(exc)[:1000]},ensure_ascii=False))
                    if answer is None:
                        break
    return merge_programs(programs), issues
