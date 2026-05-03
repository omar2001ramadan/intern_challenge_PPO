"""Outcome-only teacher distillation for the placement policy.

This module trains from saved offline teacher outcomes.  It does not import or
call any teacher solver; the input must already be a serialized dataset from
``teacher_data.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from constraints import density_bin_constraints, density_pressure_per_cell, outline_from_cells
from env import wirelength_loss, write_positions
from ordering_policy import (
    CONTROL_NAMES,
    OrderingPolicy,
    active_branch_weights_from_scores,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from teacher_data import load_teacher_dataset_payload


def default_device_arg():
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


@dataclass
class DistillConfig:
    epochs: int = 5
    batch_size: int = 1
    lr: float = 1e-4
    temperature: float = 1.0
    relaxation: str = "sigmoid"
    soft_tau: float = 1.0
    max_branch_pairs_per_sample: int = 65_536
    branch_coef: float = 1.0
    pair_branch_coef: float = 0.50
    flow_coef: float = 2.0
    pair_coef: float = 0.10
    stop_coef: float = 0.50
    value_coef: float = 0.25
    equivariance_coef: float = 0.001
    grad_clip: float = 1.0
    teacher_aux_lr_scale: float = 0.10
    teacher_aux_loss_cap: float = 256.0
    teacher_aux_weight_cap: float = 0.25
    seed: int = 1234


def teacher_lambda_at(update: int, lambda0: float = 1.0, anneal_updates: int = 1) -> float:
    """Exponential teacher coefficient with exact zero after the anneal window."""
    if anneal_updates <= 0 or update >= anneal_updates:
        return 0.0
    return float(lambda0) * math.exp(-5.0 * float(update) / max(float(anneal_updates), 1.0))


def _to_device_sample(sample: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in sample.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _wirelength_gradient(cell_features, pin_features, edge_list):
    positions = cell_features[:, 2:4].clone().detach()
    positions.requires_grad_(True)
    current = write_positions(cell_features, positions)
    loss = wirelength_loss(current, pin_features, edge_list)
    grad = torch.autograd.grad(loss, positions, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(positions)
    return grad.detach()


def _build_graph(sample: dict, *, final: bool = False, active_pairs=None):
    from ordering_policy import build_graph_state

    cell_features = sample["final_cell_features"] if final else sample["cell_features"]
    pin_features = sample["pin_features"]
    edge_list = sample["edge_list"]
    if active_pairs is None:
        active_pairs = sample["branch_pairs"]
    active_pairs = active_pairs.to(device=cell_features.device, dtype=torch.long)
    branch_duals = torch.zeros((active_pairs.shape[0], 4), dtype=cell_features.dtype, device=cell_features.device)
    boundary_duals = torch.zeros((cell_features.shape[0], 4), dtype=cell_features.dtype, device=cell_features.device)
    bounds = outline_from_cells(cell_features)
    density_g, assignment = density_bin_constraints(cell_features[:, 2:4], cell_features, bounds, bins=8, rho_max=0.85)
    density_duals = torch.zeros_like(density_g)
    exact_overlap = sample.get("metrics", {}).get("overlap_ratio", 0.0) if final else 1.0
    return build_graph_state(
        cell_features,
        pin_features,
        edge_list,
        active_pairs,
        branch_duals,
        boundary_duals,
        density_duals,
        _wirelength_gradient(cell_features, pin_features, edge_list),
        density_pressure_per_cell(density_g, assignment, density_duals),
        exact_overlap_ratio=float(exact_overlap),
        stop_logit_bias=0.0,
    )


def _teacher_seq_plus(sample: dict, n: int, device: torch.device):
    seq_plus = sample.get("teacher_seq_plus")
    if torch.is_tensor(seq_plus) and seq_plus.numel() == n:
        return seq_plus.to(device=device, dtype=torch.long)
    centers = sample["final_cell_features"][:, 2:4]
    key = centers[:, 0] + centers[:, 1]
    return torch.argsort(key).to(device=device)


def _policy_outputs(policy: OrderingPolicy, graph: dict, sample: dict, cfg: DistillConfig):
    h = policy.encode(graph)
    pooled, _base_context = policy._global_features(h)
    _memory, context = policy._memory_pair(h, graph)
    value = policy.value_head(pooled).squeeze(-1)

    temperature = max(float(cfg.temperature), 1e-4)
    plus_scores = policy.plus_head(h).squeeze(-1) / temperature
    seq_plus = _teacher_seq_plus(sample, int(h.shape[0]), h.device)
    minus_scores = policy._conditioned_minus_scores(
        h,
        policy.minus_base_head(h).squeeze(-1),
        seq_plus,
    )
    minus_scores = minus_scores / temperature

    node_context = context.unsqueeze(0).expand_as(h)
    residual_params = policy.residual_head(torch.cat([h, node_context], dim=1))
    residual_mean = residual_params[:, :2]

    control_params = policy.control_head(context)
    control_mean = control_params[: len(CONTROL_NAMES)]
    controls = policy._transform_controls(control_mean)

    stop_logits = policy.stop_head(context).squeeze(-1)
    _dag_axis, _dag_axis_logits, _pair_branch_choices, pair_branch_logits = policy._sample_pair_discrete_actions(
        h,
        graph,
        deterministic=True,
    )
    return {
        "plus_scores": plus_scores,
        "minus_scores": minus_scores,
        "residual_mean": residual_mean,
        "pair_emphasis": controls["pair_emphasis"],
        "stop_logits": stop_logits,
        "value": value,
        "pair_branch_logits": pair_branch_logits,
    }


def _limit_pairs(pairs: torch.Tensor, labels: torch.Tensor, max_pairs: int):
    if pairs.shape[0] <= int(max_pairs):
        return pairs, labels
    return pairs[: int(max_pairs)], labels[: int(max_pairs)]


def distillation_loss(policy: OrderingPolicy, sample: dict, cfg: DistillConfig) -> tuple[torch.Tensor, dict]:
    device = next(policy.parameters()).device
    sample = _to_device_sample(sample, device)
    branch_pairs, branch_labels = _limit_pairs(
        sample["branch_pairs"].long(),
        sample["branch_labels"].long(),
        cfg.max_branch_pairs_per_sample,
    )
    active_pairs = branch_pairs
    graph = _build_graph(sample, final=False, active_pairs=active_pairs)
    outputs = _policy_outputs(policy, graph, sample, cfg)
    weight = sample.get("weight", torch.tensor(1.0, device=device)).to(device=device, dtype=outputs["value"].dtype)

    zero = outputs["value"] * 0.0
    branch_loss = zero
    pair_branch_loss = zero
    branch_accuracy = 0.0
    if branch_pairs.numel() > 0:
        q = active_branch_weights_from_scores(
            outputs["plus_scores"],
            outputs["minus_scores"],
            branch_pairs,
            relaxation=cfg.relaxation,
            tau=cfg.soft_tau,
        )
        log_q = torch.log(torch.clamp(q, min=1e-8))
        branch_loss = F.nll_loss(log_q, branch_labels, reduction="mean")
        branch_accuracy = float((q.detach().argmax(dim=1) == branch_labels).float().mean().item())
        if outputs["pair_branch_logits"].shape[0] == branch_labels.shape[0]:
            pair_branch_loss = F.cross_entropy(outputs["pair_branch_logits"], branch_labels, reduction="mean")

    flow_target = sample["flow_target"].to(device=device, dtype=outputs["residual_mean"].dtype)
    flow_loss = F.mse_loss(outputs["residual_mean"], flow_target)

    active_pair_target = torch.tensor(
        1.0 if sample.get("active_pair_labels", branch_pairs).numel() > 0 else 0.0,
        dtype=outputs["pair_emphasis"].dtype,
        device=device,
    )
    pair_loss = F.binary_cross_entropy(
        torch.clamp(outputs["pair_emphasis"], min=1e-6, max=1.0 - 1e-6),
        active_pair_target,
    )

    initial_stop_target = sample.get("initial_stop_label", torch.tensor(0.0, device=device)).to(
        device=device,
        dtype=outputs["stop_logits"].dtype,
    )
    initial_stop_loss = F.binary_cross_entropy_with_logits(outputs["stop_logits"], initial_stop_target)
    final_graph = _build_graph(sample, final=True, active_pairs=active_pairs)
    final_outputs = _policy_outputs(policy, final_graph, sample, cfg)
    stop_target = sample["stop_label"].to(device=device, dtype=final_outputs["stop_logits"].dtype)
    final_stop_loss = F.binary_cross_entropy_with_logits(final_outputs["stop_logits"], stop_target)
    stop_loss = 0.5 * (initial_stop_loss + final_stop_loss)

    value_target = sample["value_target"].to(device=device, dtype=outputs["value"].dtype)
    value_loss = F.mse_loss(outputs["value"], value_target)
    equivariance_loss = policy.equivariance_loss(graph) if cfg.equivariance_coef > 0.0 else zero

    total = weight * (
        cfg.branch_coef * branch_loss
        + cfg.pair_branch_coef * pair_branch_loss
        + cfg.flow_coef * flow_loss
        + cfg.pair_coef * pair_loss
        + cfg.stop_coef * stop_loss
        + cfg.value_coef * value_loss
        + cfg.equivariance_coef * equivariance_loss
    )
    stats = {
        "loss": float(total.detach().item()),
        "branch_loss": float(branch_loss.detach().item()),
        "pair_branch_loss": float(pair_branch_loss.detach().item()),
        "flow_loss": float(flow_loss.detach().item()),
        "pair_loss": float(pair_loss.detach().item()),
        "stop_loss": float(stop_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "equivariance_loss": float(equivariance_loss.detach().item()),
        "branch_accuracy": branch_accuracy,
        "stop_false_positive": float(torch.sigmoid(outputs["stop_logits"]).detach().item() > 0.5),
        "weight": float(weight.detach().item()),
    }
    return total, stats


def _mean_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}


def outcome_distill(
    policy: OrderingPolicy,
    dataset: list[dict],
    cfg: DistillConfig | None = None,
    *,
    device: torch.device | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    cfg = cfg or DistillConfig()
    if not dataset:
        return {
            "teacher_samples": 0,
            "teacher_lambda_final": 0.0,
            "dagger_correction_count": 0,
        }
    if device is None:
        device = next(policy.parameters()).device
    policy.to(device)
    policy.train()
    owns_optimizer = optimizer is None
    optimizer = optimizer or torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    rng = random.Random(int(cfg.seed))
    epoch_summaries = []

    for epoch in range(int(cfg.epochs)):
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        pending = 0
        rows = []
        optimizer.zero_grad(set_to_none=True)
        for index in indices:
            loss, stats = distillation_loss(policy, dataset[index], cfg)
            (loss / max(int(cfg.batch_size), 1)).backward()
            pending += 1
            rows.append(stats)
            if pending >= max(int(cfg.batch_size), 1):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(cfg.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(cfg.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        summary = _mean_stats(rows)
        summary["epoch"] = epoch
        epoch_summaries.append(summary)

    final = dict(epoch_summaries[-1])
    dagger_count = sum(1 for sample in dataset if bool(sample.get("dagger_source", False)))
    final.update(
        {
            "teacher_samples": len(dataset),
            "teacher_epochs": int(cfg.epochs),
            "teacher_lambda_initial": 1.0,
            "teacher_lambda_final": 0.0,
            "dagger_correction_count": dagger_count,
            "distill_optimizer_owned": owns_optimizer,
        }
    )
    return final


def teacher_auxiliary_update(
    policy: OrderingPolicy,
    dataset: list[dict],
    cfg: DistillConfig | None = None,
    *,
    optimizer: torch.optim.Optimizer,
    lambda_teacher: float,
    batch_size: int | None = None,
    device: torch.device | None = None,
    seed: int | None = None,
) -> dict:
    """Run one annealed teacher rehearsal step from the offline dataset."""
    cfg = cfg or DistillConfig()
    if not dataset or float(lambda_teacher) <= 0.0:
        return {
            "teacher_lambda": float(lambda_teacher),
            "teacher_aux_loss": 0.0,
            "teacher_aux_weighted_loss": 0.0,
            "teacher_aux_batch_size": 0,
        }
    if device is None:
        device = next(policy.parameters()).device
    policy.to(device)
    policy.train()

    effective_batch = max(int(batch_size or cfg.batch_size), 1)
    if effective_batch >= len(dataset):
        batch = list(dataset)
    else:
        rng = random.Random(int(cfg.seed if seed is None else seed))
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        batch = [dataset[index] for index in indices[:effective_batch]]

    optimizer.zero_grad(set_to_none=True)
    losses = []
    rows = []
    for sample in batch:
        loss, stats = distillation_loss(policy, sample, cfg)
        losses.append(loss)
        rows.append(stats)
    mean_loss = torch.stack(losses).mean()
    mean_loss_value = float(mean_loss.detach().item())
    effective_lambda = min(float(lambda_teacher), float(cfg.teacher_aux_weight_cap))
    summary = _mean_stats(rows)
    summary.update(
        {
            "teacher_lambda": float(lambda_teacher),
            "teacher_aux_effective_lambda": float(effective_lambda),
            "teacher_aux_loss": mean_loss_value,
            "teacher_aux_batch_size": len(batch),
        }
    )
    if (not math.isfinite(mean_loss_value)) or mean_loss_value > float(cfg.teacher_aux_loss_cap):
        optimizer.zero_grad(set_to_none=True)
        summary.update(
            {
                "teacher_aux_weighted_loss": 0.0,
                "teacher_aux_skipped": 1.0,
                "teacher_aux_lr_scale": float(cfg.teacher_aux_lr_scale),
            }
        )
        return summary

    original_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    lr_scale = max(min(float(cfg.teacher_aux_lr_scale), 1.0), 0.0)
    for group, original_lr in zip(optimizer.param_groups, original_lrs):
        group["lr"] = original_lr * lr_scale
    weighted_loss = mean_loss * effective_lambda
    weighted_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(cfg.grad_clip))
    optimizer.step()
    for group, original_lr in zip(optimizer.param_groups, original_lrs):
        group["lr"] = original_lr
    optimizer.zero_grad(set_to_none=True)

    summary.update(
        {
            "teacher_aux_weighted_loss": float(weighted_loss.detach().item()),
            "teacher_aux_skipped": 0.0,
            "teacher_aux_lr_scale": lr_scale,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run outcome-only teacher distillation.")
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--device", default=default_device_arg())
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-passes", type=int, default=2)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--global-flow-rank", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.resume_checkpoint:
        policy, _checkpoint = load_policy_checkpoint(args.resume_checkpoint, device)
    else:
        policy = OrderingPolicy(
            hidden_dim=args.hidden_dim,
            message_passes=args.message_passes,
            num_clusters=args.num_clusters,
            global_flow_rank=args.global_flow_rank,
        ).to(device)
    cfg = DistillConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    dataset_payload = load_teacher_dataset_payload(args.teacher_dataset)
    dataset = list(dataset_payload["samples"])
    stats = outcome_distill(policy, dataset, cfg, device=device)
    save_policy_checkpoint(
        policy,
        args.output,
        config={
            **asdict(cfg),
            "teacher_dataset_path": args.teacher_dataset,
            "teacher_dataset_version": int(dataset_payload.get("version", 0)),
            "teacher_metadata": dataset_payload.get("metadata", {}),
        },
        stats=stats,
    )
    print(json.dumps(stats, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
