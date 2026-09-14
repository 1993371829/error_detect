from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..artifacts import atomic_json, fingerprint
from ..data.table import assign_folds
from ..data.encoding import CellEncoder
from ..operators import execute
from ..evidence.fusion import evidence_features, relation_quality
from ..evidence.localize import localize
from . import build_model
from .samples import SampleBuilder


def resolve_device(preference: str) -> torch.device:
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose local_smoke.yaml for CPU")
    return torch.device("cuda" if preference == "auto" and torch.cuda.is_available() else "cpu" if preference == "auto" else preference)


def objective(result, batch, epoch):
    weights = batch["target_valid"].clone()
    if epoch >= 5:
        for col in batch["column"].unique():
            selected = (batch["column"] == col) & (weights > 0)
            if bool(selected.any()):
                threshold = torch.quantile(result["loss"][selected].detach(), 0.9)
                weights[selected & (result["loss"].detach() > threshold)] *= 0.1
    masked = (result["loss"] * weights).sum() / weights.sum().clamp_min(1)
    targets = batch["synthetic"]
    weak = torch.where(targets > 0, 1., 0.25)
    synthetic = (F.binary_cross_entropy_with_logits(result["logits"], targets, reduction="none") * weak).mean()
    valid = batch["pseudo"] >= 0
    teacher = F.binary_cross_entropy_with_logits(result["logits"][valid], batch["pseudo"][valid]) if bool(valid.any()) else result["logits"].sum() * 0
    return masked + 0.5 * synthetic + 0.1 * teacher


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _save_checkpoint(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def estimate_prediction_gain(model, builder, cells, device):
    """Node-ablation gain on outer-training validation cells, never test labels."""
    if not len(cells):
        return
    model.eval()
    with torch.no_grad():
        batch = builder.batch(cells[:64],device)
        baseline = model(batch)["loss"]
        for edge_id,index in builder.constraint_index.items():
            removed = (batch["constraint_ids"] == index) & ~batch["padding"]
            affected = removed.any(1) & (batch["target_valid"] > 0)
            if not bool(affected.any()):
                continue
            ablated = {k:v.clone() for k,v in batch.items()}
            ablated["padding"] |= removed
            empty = ablated["padding"].all(1)
            if bool(empty.any()):
                ablated["padding"][empty,0] = False
                for key in ("context","quality","context_categories","relation_types"):
                    ablated[key][empty,0] = 0
            gain = (model(ablated)["loss"][affected] - baseline[affected]).mean().item()
            builder.quality[edge_id][4] = float(np.clip(gain,-1,1))


def train_fold(model, builder, train_ids, cfg, checkpoint: Path, identity: str, seed: int):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(train_ids))
    rng.shuffle(ids)
    n_val = max(1, len(ids)//6) if len(ids) > 3 else 0
    validation_ids, fitting_ids = ids[:n_val], ids[n_val:]
    def cells(rows):
        return np.array([(i,j) for i in rows for j in range(len(builder.cols))], dtype=np.int64).reshape(-1,2)
    training = cells(fitting_ids)
    validation = cells(validation_ids)
    if len(training) > cfg.max_train_cells:
        training = training[rng.choice(len(training), cfg.max_train_cells, replace=False)]
    if len(validation) > cfg.max_train_cells // 4 + 1:
        validation = validation[:cfg.max_train_cells//4+1]
    start, best_loss, bad_epochs, history, best = 0, float("inf"), 0, [], _snapshot(model)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["identity"] != identity:
            raise ValueError("checkpoint configuration/input mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best_loss, bad_epochs, history, best = state["epoch"]+1, state["best_loss"], state["bad_epochs"], state["history"], state["best"]
        builder.quality.update(state["quality"])
        if state["complete"]:
            model.load_state_dict(best)
            return history
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, cfg.epochs):
        torch.manual_seed(seed + epoch)
        order = np.random.default_rng(seed+epoch).permutation(len(training))
        model.train()
        total = 0.
        for at in range(0, len(training), cfg.batch_size):
            current = training[order[at:at+cfg.batch_size]]
            batch = builder.batch(np.repeat(current, 2, axis=0), device, synthetic=True)
            optimizer.zero_grad()
            result = model(batch)
            loss = objective(result, batch, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item() * len(current)
        model.eval()
        val_loss, val_count = 0., 0
        with torch.no_grad():
            for at in range(0, len(validation), cfg.batch_size):
                batch = builder.batch(validation[at:at+cfg.batch_size], device)
                result = model(batch)
                weight = batch["target_valid"]
                val_loss += float((result["loss"] * weight).sum())
                val_count += int(weight.sum())
        measured = val_loss / val_count if val_count else total/max(1,len(training))
        if measured < best_loss - 1e-4:
            best_loss, bad_epochs, best = measured, 0, _snapshot(model)
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_loss": total/max(1,len(training)), "validation_loss": measured})
        if epoch == min(4,max(0,cfg.epochs//2-1)):
            estimate_prediction_gain(model,builder,validation if len(validation) else training[:64],device)
        done = bad_epochs >= cfg.patience or epoch+1 == cfg.epochs
        _save_checkpoint(checkpoint, {"identity": identity, "epoch": epoch, "model": _snapshot(model), "optimizer": optimizer.state_dict(), "best": best, "best_loss": best_loss, "bad_epochs": bad_epochs, "history": history, "quality": builder.quality, "complete": done})
        if done:
            break
    model.load_state_dict(best)
    return history


def cross_fit(table, program, config, semantic, output: Path, pseudo=None):
    cfg = config.model
    torch.set_num_threads(cfg.threads)
    device = resolve_device(cfg.device)
    fold_ids = assign_folds(table, cfg.folds, config.seed)
    predictions = np.zeros(table.shape, dtype=np.float32)
    residuals = np.zeros_like(predictions)
    evidence_all, histories, alternative_values = [], {}, {}
    for fold in range(cfg.folds):
        torch.manual_seed(config.seed+fold)
        train_ids, test_ids = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        train_set, test_set = set(train_ids), set(test_ids)
        reference = table.iloc[train_ids]
        encoder = CellEncoder(reference, semantic, cfg.max_categories)
        evidence = execute(table, program, reference=reference)
        localize(table,program,evidence,fold_ids,{},config,reference_ids=train_ids)
        features = evidence_features(evidence, len(table), list(table.columns))
        quality = relation_quality(program, [e for e in evidence if e.row_id in train_set])
        local_cfg = cfg.model_copy(update={"input_dim": encoder.dim, "column_count": len(table.columns), "category_count": encoder.category_count, "target_sizes": [s["size"] for s in encoder.specs], "target_kinds": [s["kind"] for s in encoder.specs]})
        local_pseudo = {key: value for key, value in (pseudo or {}).items() if key[0] in train_set}
        builder = SampleBuilder(table, train_ids, encoder, program, features, quality, cfg, local_pseudo)
        model = build_model(local_cfg).to(device)
        identity = fingerprint([table.to_dict("split"), program.model_dump(), local_cfg.model_dump(), sorted((i,c,v) for (i,c),v in local_pseudo.items()), config.seed, fold])
        histories[str(fold)] = train_fold(model, builder, train_ids, cfg, output/f"fold_{fold}.pt", identity, config.seed+fold)
        cells = [(i,j) for i in test_ids for j in range(len(table.columns))]
        model.eval()
        with torch.no_grad():
            for at in range(0,len(cells),cfg.batch_size):
                chunk = cells[at:at+cfg.batch_size]
                batch = builder.batch(chunk, device)
                result = model(batch)
                scores = torch.sigmoid(result["logits"]).cpu().numpy()
                for k,(i,j) in enumerate(chunk):
                    predictions[i,j] = scores[k]
                    residuals[i,j] = result["residual"][k].item()
                for j,(selected,pred) in result["predictions"].items():
                    positions = selected.nonzero().flatten().tolist()
                    spec = encoder.specs[j]
                    if spec["kind"] == "categorical":
                        reverse = {v:k for k,v in spec["vocab"].items()}
                        choices = pred.topk(min(3,pred.shape[1]),dim=1).indices.tolist()
                        for pos, choice in zip(positions,choices):
                            i,_ = chunk[pos]
                            alternative_values[(int(i),table.columns[j])] = [reverse[x] for x in choice if x in reverse]
                    elif spec["kind"] == "numeric":
                        for pos,val in zip(positions,pred[:,0].tolist()):
                            i,_ = chunk[pos]
                            alternative_values[(int(i),table.columns[j])] = [str(val*spec["scale"]+spec["median"])]
        evidence_all.extend(e for e in evidence if e.row_id in test_set)
        atomic_json(output/f"fold_{fold}_metadata.json", {"training_rows": train_ids.tolist(), "scoring_rows": test_ids.tolist(), "encoder_specs": encoder.specs, "model_config": local_cfg.model_dump(), "quality": quality, "identity": identity})
        del model, builder, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_json(output/"training_history.json", histories)
    return predictions, residuals, evidence_all, fold_ids, alternative_values
