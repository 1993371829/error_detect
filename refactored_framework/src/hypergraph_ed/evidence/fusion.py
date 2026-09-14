from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schema import Status


FAMILIES = ["missing", "format", "semantic", "domain", "statistical", "neighbor", "relation"]


def evidence_features(evidence, row_count: int, columns: list[str]) -> np.ndarray:
    """[violation, passed, reliability, support, coverage, disagreement] per family.

    Check passes remain scoped inputs, never a global clean vote. Repeated source
    entries are deduplicated before normalized pooling.
    """
    result = np.zeros((row_count, len(columns), len(FAMILIES), 6), dtype=np.float32)
    col_ids = {c: i for i, c in enumerate(columns)}
    groups = defaultdict(dict)
    for item in evidence:
        if item.status == Status.ABSTAIN:
            continue
        key = (item.row_id, col_ids[item.column], FAMILIES.index(item.family))
        groups[key].setdefault(item.source_group, item)
    for key, sources in groups.items():
        values = list(sources.values())
        violations = [e.score if e.status == Status.VIOLATION else 0 for e in values]
        passed = [e.status == Status.PASS for e in values]
        result[key] = [np.mean(violations), np.mean(passed), np.mean([e.reliability for e in values]), np.mean([min(e.support / 50, 1) for e in values]), 1, float(any(violations) and any(passed))]
    return result


def relation_quality(program, evidence) -> dict[str, list[float]]:
    grouped = defaultdict(dict)
    for e in evidence:
        grouped[e.constraint_id].setdefault((e.row_id, e.column), e)
    result = {}
    for edge in program.constraints:
        vals = list(grouped[edge.id].values())
        active = [e for e in vals if e.status != Status.ABSTAIN]
        if not active:
            result[edge.id] = [0, 0, 0, 0, 0]
            continue
        support = np.mean([min(e.support / 50, 1) for e in active])
        # Independent bootstrap dispersion of violation rates, not semantic truth.
        labels = np.array([e.status == Status.VIOLATION for e in active], dtype=float)
        rng = np.random.default_rng(42)
        rates = [np.mean(rng.choice(labels, size=len(labels), replace=True)) for _ in range(10)]
        stability = max(0., 1 - 4 * np.std(rates))
        result[edge.id] = [len(active) / max(1, len(vals)), float(support), float(stability), float(np.mean([e.details.get("counterevidence", False) for e in active])), 0.0]
    return result
