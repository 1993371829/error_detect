from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from . import register_model


@register_model("relation_attention")
class RelationAttention(nn.Module):
    """Target-masked relation attention and shared, provenance-pooled critic."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.project = nn.Linear(cfg.input_dim, h)
        self.column = nn.Embedding(cfg.column_count, h)
        self.category = nn.Embedding(cfg.category_count, h, padding_idx=0)
        self.relation = nn.Embedding(16, h)
        self.quality_gate = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.attention = nn.ModuleList([nn.MultiheadAttention(h, cfg.heads, batch_first=True, dropout=0) for _ in range(cfg.layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(h) for _ in range(cfg.layers)])
        self.feed = nn.ModuleList([nn.Sequential(nn.Linear(h, h*2), nn.GELU(), nn.Linear(h*2, h)) for _ in range(cfg.layers)])
        self.heads = nn.ModuleList([nn.Linear(h, size) for size in cfg.target_sizes])
        self.evidence_project = nn.Sequential(nn.Linear(6, h), nn.ReLU())
        self.evidence_gate = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))
        self.critic = nn.Sequential(nn.Linear(h*3+2, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, batch):
        context = self.project(batch["context"]) + self.column(batch["context_columns"]) + self.category(batch["context_categories"]) + self.relation(batch["relation_types"])
        reliability = self.quality_gate(batch["quality"])
        context = context * reliability * batch["context_trust"].unsqueeze(-1)
        query = self.column(batch["column"]).unsqueeze(1)
        for attention, norm, feed in zip(self.attention, self.norm, self.feed):
            attended, _ = attention(query, context, context, key_padding_mask=batch["padding"], need_weights=False)
            query = norm(query + attended)
            query = norm(query + feed(query))
        hidden = query[:,0]
        losses = torch.zeros(hidden.shape[0], device=hidden.device)
        residual = torch.zeros_like(losses)
        predictions = {}
        for j, head in enumerate(self.heads):
            selected = batch["column"] == j
            if not bool(selected.any()):
                continue
            pred = head(hidden[selected])
            predictions[j] = (selected, pred)
            if self.cfg.target_kinds[j] == "categorical":
                losses[selected] = F.cross_entropy(pred, batch["target_category"][selected], reduction="none")
                residual[selected] = F.cross_entropy(pred, batch["observed_target_category"][selected], reduction="none")
            elif self.cfg.target_kinds[j] == "numeric":
                losses[selected] = F.huber_loss(pred[:,0], batch["target"][selected,0], reduction="none")
                residual[selected] = torch.abs(pred[:,0] - batch["observed"][selected,0])
            else:
                def distance(target):
                    value = F.mse_loss(pred[:,:48],target[:,:48],reduction="none").mean(1)
                    if pred.shape[1] > 48:
                        value = value + 1 - F.cosine_similarity(pred[:,48:],target[:,48:],dim=1)
                    return value
                losses[selected] = distance(batch["target"][selected])
                residual[selected] = distance(batch["observed"][selected])
        evidence = batch["evidence"]
        coverage = evidence[:,:,4]
        scores = self.evidence_gate(evidence).squeeze(-1).masked_fill(coverage == 0, -1e4)
        weights = torch.softmax(scores, dim=1) * coverage
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = (weights.unsqueeze(-1) * self.evidence_project(evidence)).sum(1)
        observed = self.project(batch["observed"]) + self.category(batch["observed_category"])
        critic_input = torch.cat([hidden, observed, pooled, torch.log1p(residual).unsqueeze(1), coverage.mean(1, keepdim=True)], dim=1)
        logits = self.critic(critic_input).squeeze(1)
        return {"loss": losses, "labels": batch.get("pseudo"), "logits": logits, "predictions": predictions, "residual": residual, "reliability": reliability, "evidence_weights": weights}
