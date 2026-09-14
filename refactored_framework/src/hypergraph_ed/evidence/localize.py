from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations

from ..operators.engine import _check, _prepare
from .schema import Status


def localize(table, program, evidence, fold_ids, alternatives, config, reference_ids=None):
    """Single-cell counterfactuals with source-excluded context.

    No majority across conflicting independent contexts. Multiple differing
    participants stay unresolved. Counterfactuals never mutate input data.
    """
    edge_map = {e.id: e for e in program.constraints}
    grouped = defaultdict(list)
    for e in evidence:
        if e.family == "relation" and (e.status == Status.VIOLATION or (e.status == Status.ABSTAIN and e.reason == "insufficient peer group")):
            grouped[(e.row_id, e.constraint_id)].append(e)
    records = {}
    processed_rows = set()
    reference_cache = {}
    for (row_id, edge_id), items in sorted(grouped.items()):
        if row_id not in processed_rows and len(processed_rows) >= config.max_counterfactual_rows:
            for e in items:
                e.details["localization"] = "skipped_resource_cap"
            continue
        processed_rows.add(row_id)
        edge = edge_map[edge_id]
        row = table.iloc[row_id].to_dict()
        reference_key = -1 if reference_ids is not None else int(fold_ids[row_id])
        if reference_key not in reference_cache:
            reference = table.iloc[reference_ids] if reference_ids is not None else table.loc[fold_ids != fold_ids[row_id]]
            indices = {c: {v:set(g.index) for v,g in reference.groupby(c)} for c in table.columns}
            reference_cache[reference_key] = (reference,indices)
        reference,indices = reference_cache[reference_key]
        outside = [c for c in table.columns if c not in edge.participants and c not in [x.column for x in edge.conditions]]
        votes, seen, candidate_fixes = [], set(), defaultdict(list)
        # Capped context width; high-cardinality/empty matches automatically abstain.
        for width in (1,2):
            for anchors in combinations(outside[:12], width):
                peer_ids = None
                for col in anchors:
                    matching = indices[col].get(row[col],set())
                    peer_ids = matching.copy() if peer_ids is None else peer_ids & matching
                peer_ids.discard(row_id)
                if len(peer_ids) < config.min_peer_support or len(peer_ids) > max(config.min_peer_support, len(reference)*0.5):
                    continue
                key = tuple(sorted(peer_ids))
                if key in seen:
                    continue
                seen.add(key)
                peers = reference.loc[list(key)]
                patterns = Counter(tuple(x) for x in peers[edge.participants].itertuples(index=False, name=None))
                expected, support = patterns.most_common(1)[0]
                if support / len(peers) < 0.9:
                    continue
                changed = [c for c,v in zip(edge.participants, expected) if str(row[c]) != str(v)]
                target = changed[0] if len(changed) == 1 else None if not changed else "__multiple__"
                votes.append({"target": target, "anchors": list(anchors), "support": support})
                if len(changed) == 1:
                    candidate_fixes[target].append(str(expected[edge.participants.index(target)]))
        targets = {v["target"] for v in votes}
        decision = "unresolved"
        target = None
        if len(targets) == 1:
            target = next(iter(targets))
            decision = "context_supports_original" if target is None else "multiple_participants_differ" if target == "__multiple__" else "localized"
            if target == "__multiple__":
                target = None
        elif len(targets) > 1:
            decision = "conflicting_context"
        checks = []
        independent_edges = [h for h in program.constraints if h.canonical() != edge.canonical() and not (h.source == edge.source and h.operator == edge.operator and set(h.participants) == set(edge.participants))]
        for col in edge.participants:
            candidates = list(dict.fromkeys(candidate_fixes[col] + alternatives.get((row_id,col), [])))[:config.max_counterfactual_candidates]
            for value in candidates:
                modified = dict(row)
                modified[col] = value
                before, after, coverage = 0, 0, 0
                for h in independent_edges:
                    if col not in h.participants:
                        continue
                    prepared = _prepare(h, reference)
                    for checked in h.targets:
                        a = _check(h,row,checked,reference,row_id,prepared)[0]
                        b = _check(h,modified,checked,reference,row_id,prepared)[0]
                        if a is not None and b is not None:
                            before += int(a); after += int(b); coverage += 1
                checks.append({"column": col, "candidate": value, "independent_checks": coverage, "violation_reduction": before-after})
        info = {"status": decision, "target": target, "contexts": votes, "counterfactuals": checks}
        if decision == "localized" and target not in {e.column for e in items}:
            relocated = replace(items[0],column=target,details=dict(items[0].details))
            evidence.append(relocated)
            items.append(relocated)
        for e in items:
            e.details["localization"] = info
            if decision == "context_supports_original":
                if not e.details.get("counterevidence"):
                    e.reliability *= 0.25
                e.details["counterevidence"] = True
            elif decision == "localized":
                if e.column == target:
                    e.status = Status.VIOLATION
                    e.score = 0.9
                    e.suggested_fix = candidate_fixes[target][0]
                    e.details["localized_target"] = True
                else:
                    # Keep disagreement in the trace, but it does not indict this participant.
                    if not e.details.get("counterevidence"):
                        e.reliability *= 0.25
                    e.details["counterevidence"] = True
            records.setdefault((row_id,e.column), []).append(info)
    return records
