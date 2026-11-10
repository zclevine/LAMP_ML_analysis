from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (LampDataset, LengthBucketBatchSampler,
                   MutationOrderedBatchSampler, Standardizer, collate,
                   isolate_inner_fold, isolate_outer_fold,
                   restrict_train_chemistry,
                   tabular_frame_for_ablation, target_for_task)
from .metrics import cluster_bootstrap, selection_value, stratified_metrics, task_metrics
from .models import build_model
from .sampling import deterministic_group_fraction


REGRESSION_TASKS = {"positive", "tp30", "tp60", "tp60_censored", "tp120"}
TOKEN_PHYSICS_MODELS = {"lyra_physics_", "transformer_physics_", "mamba2_physics_"}


@dataclass
class TrainConfig:
    model: str
    task: str = "positive"
    d_model: int = 64
    dropout: float = 0.2
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 80
    patience: int = 15
    seed: int = 0
    loss: str = "huber"
    ablation: str = "none"
    train_fraction: float = 1.0
    sampling_seed: int = 0
    num_workers: int = int(os.environ.get("LAMPBENCH_NUM_WORKERS", "4"))
    train_sampler: str = "length_bucket"
    no_refit: bool = False
    spearman_weight: float = 1.0
    huber_weight: float = 0.3
    train_chemistry: str | None = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_batch_sampler(dataset, cfg, shuffle: bool):
    kind = getattr(cfg, "train_sampler", "length_bucket")
    if kind == "mutation_ordered":
        return MutationOrderedBatchSampler(dataset, cfg.batch_size)
    if kind == "length_bucket":
        return LengthBucketBatchSampler(dataset, cfg.batch_size, shuffle, cfg.seed)
    raise ValueError(f"Unknown train_sampler: {kind}")


def _loaders(train_df, train_y, eval_df, eval_y, std, cfg, shuffle=True):
    token_physics = any(cfg.model.startswith(prefix) for prefix in TOKEN_PHYSICS_MODELS)
    train_ds = LampDataset(train_df, train_y, std, cfg.ablation, token_physics)
    kwargs = dict(collate_fn=collate, num_workers=cfg.num_workers,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds, batch_sampler=_train_batch_sampler(train_ds, cfg, shuffle), **kwargs)
    if eval_df is None:
        return train_loader, None
    eval_ds = LampDataset(eval_df, eval_y, std, cfg.ablation, token_physics)
    return train_loader, DataLoader(eval_ds, batch_sampler=LengthBucketBatchSampler(
        eval_ds, cfg.batch_size, False), **kwargs)


def _use_amp(model_name: str, device: torch.device) -> bool:
    # S4D uses complex FFT kernels and should remain in fp32.
    # Mamba Triton kernels segfault under AMP + forked DataLoader workers.
    return (device.type == "cuda" and not model_name.startswith("lyra")
            and model_name != "s4d" and "mamba" not in model_name)


def _forward(model, batch, device):
    args = [batch[k].to(device, non_blocking=True) for k in ("seq", "role", "mask", "lengths", "tab")]
    return model(*args)


def _target_stats(task, y):
    if task in REGRESSION_TASKS:
        return float(np.mean(y)), float(np.std(y) or 1.0)
    if task == "hurdle":
        vals = y[y[:, 2].astype(bool), 1]
        return float(np.mean(vals)), float(np.std(vals) or 1.0)
    return 0.0, 1.0


def _soft_rank(x, regularization_strength=1.0):
    """Differentiable ranks; larger regularization_strength is smoother. O(N^2)."""
    x = x.reshape(-1)
    pairwise = (x[:, None] - x[None, :]) / max(float(regularization_strength), 1e-8)
    return torch.sigmoid(pairwise).sum(dim=1)


def _spearman_loss(y_pred, y_true, regularization_strength=1.0):
    if y_pred.numel() < 2:
        return y_pred.new_tensor(0.0)
    pred_rank = _soft_rank(y_pred, regularization_strength)
    true_rank = _soft_rank(y_true, regularization_strength)
    pred_rank = pred_rank - pred_rank.mean()
    true_rank = true_rank - true_rank.mean()
    cov = (pred_rank * true_rank).mean()
    scale = pred_rank.std() * true_rank.std()
    spearman = cov / (scale + 1e-8)
    return 1 - spearman


def _loss(task, output, y, mean, std, kind, spearman_weight=1.0, huber_weight=0.3):
    if task in REGRESSION_TASKS:
        target = (y - mean) / std
        pred = output.squeeze(-1)
        if kind == "mse":
            return F.mse_loss(pred, target)
        if kind == "rank_mse":
            target = torch.argsort(torch.argsort(target)).to(target.dtype) / max(1, len(target) - 1)
            return F.mse_loss(pred, target)
        if kind == "rank_huber":
            base = F.smooth_l1_loss(pred, target)
            delta_y = target[:, None] - target[None, :]
            delta_p = pred[:, None] - pred[None, :]
            keep = delta_y.abs() > 1e-6
            pairwise = (F.softplus(-delta_y.sign()[keep] * delta_p[keep]).mean()
                        if keep.any() else pred.new_tensor(0.0))
            return base + 0.2 * pairwise
        if kind == "spearman_huber":
            base = F.smooth_l1_loss(pred, target)
            return spearman_weight * _spearman_loss(pred, target) + huber_weight * base
        return F.smooth_l1_loss(pred, target)
    if task == "classification":
        return F.binary_cross_entropy_with_logits(output.squeeze(-1), y)
    if task == "hurdle":
        amp, tp, mask = y[:, 0], y[:, 1], y[:, 2].bool()
        amp_loss = F.binary_cross_entropy_with_logits(output[:, 0], amp)
        tp_loss = F.smooth_l1_loss(output[mask, 1], (tp[mask] - mean) / std) if mask.any() else 0.0
        return amp_loss + tp_loss
    raise ValueError(task)


def predict(model, loader, task, device, mean, std, model_name=""):
    model.eval()
    preds, ys, indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=_use_amp(model_name, device)):
                out = _forward(model, batch, device)
            if task in REGRESSION_TASKS:
                pred = out.squeeze(-1).float().cpu().numpy() * std + mean
            elif task == "classification":
                pred = out.squeeze(-1).float().sigmoid().cpu().numpy()
            else:
                pred = out.float().cpu().numpy()
                pred[:, 0] = torch.from_numpy(pred[:, 0]).sigmoid().numpy()
                pred[:, 1] = pred[:, 1] * std + mean
            preds.append(pred)
            ys.append(batch["target"].numpy())
            indices.append(batch["index"].numpy())
    idx = np.concatenate(indices)
    order = np.argsort(idx)
    return np.concatenate(ys)[order], np.concatenate(preds)[order]


def fit_epochs(train_df, train_y, eval_df, eval_y, cfg: TrainConfig, fixed_epochs=None):
    set_seed(cfg.seed)
    device = _device()
    std = Standardizer.fit(tabular_frame_for_ablation(train_df, cfg.ablation))
    train_loader, eval_loader = _loaders(train_df, train_y, eval_df, eval_y, std, cfg)
    model = build_model(cfg.model, cfg.task, cfg.d_model, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mean, target_std = _target_stats(cfg.task, train_y)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(cfg.model, device))
    best_value, best_epoch, best_state, stale = -np.inf, 0, None, 0
    epochs = fixed_epochs or cfg.max_epochs
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            y = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=_use_amp(cfg.model, device)):
                out = _forward(model, batch, device)
                loss = _loss(cfg.task, out, y, mean, target_std, cfg.loss,
                             spearman_weight=cfg.spearman_weight,
                             huber_weight=cfg.huber_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        if fixed_epochs is not None:
            continue
        y_eval, pred = predict(model, eval_loader, cfg.task, device, mean, target_std, cfg.model)
        metrics = task_metrics(cfg.task, y_eval, pred)
        value = selection_value(cfg.task, metrics)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "selection": value})
        if np.isfinite(value) and value > best_value:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("No finite inner-validation metric")
        model.load_state_dict(best_state)
    return model, std, mean, target_std, best_epoch or epochs, history


def select_inner_block(outer_block: str) -> str:
    """Choose the adjacent development fold within the same source dataset."""
    outer_idx = int(outer_block.replace("block", ""))
    if 0 <= outer_idx <= 9:
        return f"block{(outer_idx + 1) % 10}"
    if 10 <= outer_idx <= 18:
        return f"block{10 + ((outer_idx - 10 + 1) % 9)}"
    raise ValueError(f"Outer development block must be block0--block18: {outer_block}")


def nested_fold(units, outer_block: str, cfg: TrainConfig, output_dir: Path):
    inner_block = select_inner_block(outer_block)
    candidate, outer, outer_isolation = isolate_outer_fold(units, outer_block)
    fit, inner, inner_isolation = isolate_inner_fold(candidate, inner_block)
    fit = restrict_train_chemistry(fit, cfg.train_chemistry)
    inner = restrict_train_chemistry(inner, cfg.train_chemistry)
    candidate = restrict_train_chemistry(candidate, cfg.train_chemistry)
    fit, fit_y = target_for_task(fit, cfg.task)
    inner, inner_y = target_for_task(inner, cfg.task)
    outer, outer_y = target_for_task(outer, cfg.task)
    fit, fit_sampling = deterministic_group_fraction(
        fit, cfg.train_fraction, cfg.sampling_seed)
    fit, fit_y = target_for_task(fit, cfg.task)
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    es_model, std, mean, target_std, best_epoch, history = fit_epochs(
        fit, fit_y, inner, inner_y, cfg)
    if cfg.no_refit:
        final = es_model
        final_sampling = {"mode": "no_refit", "n_fit": len(fit)}
    else:
        full, full_y = target_for_task(candidate, cfg.task)
        full, final_sampling = deterministic_group_fraction(
            full, cfg.train_fraction, cfg.sampling_seed)
        full, full_y = target_for_task(full, cfg.task)
        final, std, mean, target_std, _, _ = fit_epochs(
            full, full_y, None, None,
            cfg, fixed_epochs=best_epoch,
        )
    eval_ds = LampDataset(outer, outer_y, std, cfg.ablation,
                          any(cfg.model.startswith(prefix) for prefix in TOKEN_PHYSICS_MODELS))
    eval_loader = DataLoader(eval_ds, batch_sampler=LengthBucketBatchSampler(
        eval_ds, cfg.batch_size, False), collate_fn=collate, num_workers=cfg.num_workers)
    y, pred = predict(final, eval_loader, cfg.task, _device(), mean, target_std, cfg.model)
    outer = outer.reset_index(drop=True)
    result = {
        "config": asdict(cfg), "outer_block": outer_block, "inner_block": inner_block,
        "best_epoch": best_epoch, "elapsed_sec": time.time() - started,
        "n_parameters": int(sum(p.numel() for p in final.parameters())),
        "peak_cuda_memory_bytes": (int(torch.cuda.max_memory_allocated())
                                   if torch.cuda.is_available() else 0),
        "n_fit": len(fit), "n_inner": len(inner), "n_outer": len(outer),
        "training_fraction": float(cfg.train_fraction),
        "sampling_seed": int(cfg.sampling_seed),
        "no_refit": bool(cfg.no_refit),
        "train_sampler": cfg.train_sampler,
        "training_subsamples": {
            "inner_fit": fit_sampling,
            "final_fit": final_sampling,
        },
        "sequence_identity_isolation": {
            "outer": outer_isolation, "inner": inner_isolation,
        },
        "metrics": stratified_metrics(cfg.task, outer, y, pred),
        "bootstrap": cluster_bootstrap(cfg.task, outer, y, pred, n_boot=200),
        "history": history,
    }
    fold_dir = output_dir / cfg.task / cfg.model / f"seed_{cfg.seed}" / outer_block
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final.state_dict(), "standardizer": std.to_json(),
                "target_mean": mean, "target_std": target_std, "config": asdict(cfg)},
               fold_dir / "model.pt")
    np.savez_compressed(fold_dir / "predictions.npz", y=y, pred=pred,
                        group_id=outer.group_id.astype(str).to_numpy(dtype=str),
                        primer_set=outer.primer_set.astype(str).to_numpy(dtype=str),
                        canonical_mm=outer.canonical_mm.astype(str).to_numpy(dtype=str),
                        target_conc=outer.target_conc.to_numpy(),
                        n_replicates=outer.n_replicates.to_numpy(),
                        amp_fraction=outer.amp_fraction.to_numpy())
    (fold_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
