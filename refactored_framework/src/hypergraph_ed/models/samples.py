from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..data.table import parse_number, is_blank
from ..operators.engine import OPERATOR_FAMILY


class SampleBuilder:
    def __init__(self, table, reference_ids, encoder, program, evidence, quality, cfg, pseudo=None):
        self.table, self.encoder, self.cfg = table, encoder, cfg
        self.reference_ids = set(map(int, reference_ids))
        self.features, self.categories, self.targets = encoder.transform(table)
        self.evidence, self.quality = evidence, quality
        self.context_trust = np.clip(1 - np.max(evidence[:,:,:,0] * evidence[:,:,:,2],axis=2),0.1,1).astype(np.float32)
        self.rows = table.to_dict("records")
        self.cols = list(table.columns)
        self.pseudo = pseudo or {}
        self.edges = defaultdict(list)
        self.constraint_index = {edge.id: i for i,edge in enumerate(program.constraints)}
        self.peer_indices = {}
        for edge in program.constraints:
            for col in edge.targets:
                self.edges[col].append(edge)
                keys = [c for c in edge.participants if c != col]
                groups = defaultdict(list)
                for i in sorted(self.reference_ids):
                    if edge.applicable(self.rows[i]):
                        groups[tuple(self.rows[i][c] for c in keys)].append(i)
                self.peer_indices[(edge.id, col)] = (keys, groups)

    def sample(self, row_id: int, column: int, synthetic: bool = False):
        row, col = self.rows[row_id], self.cols[column]
        contexts, columns, categories, relations, quality, constraint_ids, trust = [], [], [], [], [], [], []
        for edge in self.edges[col][:self.cfg.max_relations]:
            if not edge.applicable(row):
                continue
            rid = list(OPERATOR_FAMILY).index(edge.operator) + 1
            q = self.quality.get(edge.id, [0.5, 0.5, 0.5, 0, 0])
            for other in edge.participants:
                if other == col:
                    continue
                j = self.cols.index(other)
                contexts.append(self.features[row_id,j]); columns.append(j); categories.append(self.categories[row_id,j]); relations.append(rid); quality.append(q)
                constraint_ids.append(self.constraint_index[edge.id])
                trust.append(self.context_trust[row_id,j])
            keys, groups = self.peer_indices[(edge.id, col)]
            peers = [i for i in groups.get(tuple(row[c] for c in keys), []) if i != row_id]
            if peers:
                selected = np.linspace(0, len(peers)-1, min(self.cfg.max_peers, len(peers))).astype(int)
                for position in selected:
                    i = peers[position]
                    contexts.append(self.features[i,column]); columns.append(column); categories.append(self.categories[i,column]); relations.append(rid); quality.append(q)
                    constraint_ids.append(self.constraint_index[edge.id])
                    trust.append(self.context_trust[i,column])
        if not contexts:
            # Explicit background relation for uncovered columns, target still masked.
            for j in range(len(self.cols)):
                if j != column:
                    contexts.append(self.features[row_id,j]); columns.append(j); categories.append(self.categories[row_id,j]); relations.append(0); quality.append([0.1, 0, 0, 0, 0]); constraint_ids.append(-1)
                    trust.append(self.context_trust[row_id,j])
                if len(contexts) >= 4:
                    break
        if not contexts:
            contexts, columns, categories, relations, quality, constraint_ids = [np.zeros(self.encoder.dim)], [column], [0], [0], [[0,0,0,0,0]], [-1]
            trust = [0.]
        observed = self.features[row_id,column].copy()
        cat, target_cat = self.categories[row_id,column], self.targets[row_id,column]
        if synthetic:
            value = str(row[col])
            number = parse_number(value)
            corrupt = str(number[0] + 10 * self.encoder.specs[column]["scale"]) + (" " + number[1] if number[1] else "") if number else value + "#?~"
            observed, cat, target_cat = self.encoder.value(column, corrupt)
            if self.encoder.semantic.dim:
                # Avoid teaching the critic that a missing semantic vector means error.
                observed[48:] = self.features[row_id,column,48:]
        pseudo = self.pseudo.get((row_id, col), -1) if not synthetic else -1
        return {"context": np.asarray(contexts, dtype=np.float32), "context_columns": np.array(columns), "context_categories": np.array(categories), "relation_types": np.array(relations), "quality": np.asarray(quality, dtype=np.float32), "constraint_ids": np.array(constraint_ids), "context_trust": np.asarray(trust,dtype=np.float32), "column": column, "target": self.features[row_id,column], "target_category": self.targets[row_id,column], "observed": observed, "observed_category": cat, "observed_target_category": target_cat, "evidence": self.evidence[row_id,column], "pseudo": pseudo, "synthetic": float(synthetic), "target_valid": float(not is_blank(row[col]) and not (self.evidence[row_id,column,1,0] >= 0.9)), "row_id": row_id}

    def batch(self, cells, device, synthetic=False):
        samples = [self.sample(int(i), int(j), synthetic and (k % 2 == 1)) for k, (i,j) in enumerate(cells)]
        length = max(len(s["context"]) for s in samples)
        arrays = {}
        seq = {"context", "context_columns", "context_categories", "relation_types", "quality", "constraint_ids", "context_trust"}
        integer = {"context_columns", "context_categories", "relation_types", "column", "target_category", "observed_category", "observed_target_category", "row_id", "constraint_ids"}
        for key in samples[0]:
            if key in seq:
                values = [np.pad(s[key], [(0, length-len(s[key]))] + [(0,0)]*(s[key].ndim-1)) for s in samples]
            else:
                values = [s[key] for s in samples]
            arrays[key] = torch.as_tensor(np.asarray(values), dtype=torch.long if key in integer else torch.float32, device=device)
        arrays["padding"] = torch.tensor([[k >= len(s["context"]) for k in range(length)] for s in samples], dtype=torch.bool, device=device)
        return arrays
