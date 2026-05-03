"""Classical teacher solver used for offline distillation datasets.

This teacher is intentionally simple and deterministic: it repeatedly audits
exact overlaps and applies the minimum separating translation for each
colliding pair. Distillation only needs strong final outcomes, so a direct
repair heuristic is a better fit here than depending on a PPO checkpoint.
"""

from __future__ import annotations

import os

import torch

from constraints import exact_overlap_pairs
from induce_branches import branch_antisymmetry_error, sequence_pair_from_centers
from placement import _candidate_key, _score_candidate, _write_positions


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    return float(value)


def _separation_update(
    centers: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    overlaps: torch.Tensor,
    *,
    step_scale: float,
    margin: float,
) -> torch.Tensor:
    if overlaps.numel() == 0:
        return torch.zeros_like(centers)

    i = overlaps[:, 0].long()
    j = overlaps[:, 1].long()
    dx = centers[j, 0] - centers[i, 0]
    dy = centers[j, 1] - centers[i, 1]

    sep_x = 0.5 * (widths[i] + widths[j]) - dx.abs()
    sep_y = 0.5 * (heights[i] + heights[j]) - dy.abs()
    move_x = sep_x <= sep_y

    sign_x = torch.where(dx >= 0, torch.ones_like(dx), -torch.ones_like(dx))
    sign_y = torch.where(dy >= 0, torch.ones_like(dy), -torch.ones_like(dy))
    sign_x = torch.where(dx.abs() < 1e-6, torch.ones_like(sign_x), sign_x)
    sign_y = torch.where(dy.abs() < 1e-6, torch.ones_like(sign_y), sign_y)

    tx = (0.5 * sep_x + float(margin)) * sign_x * float(step_scale)
    ty = (0.5 * sep_y + float(margin)) * sign_y * float(step_scale)
    half = 0.5 * torch.ones_like(tx)

    update = torch.zeros_like(centers)
    update.index_add_(
        0,
        i,
        torch.stack(
            [
                torch.where(move_x, -tx * half, torch.zeros_like(tx)),
                torch.where(move_x, torch.zeros_like(ty), -ty * half),
            ],
            dim=1,
        ),
    )
    update.index_add_(
        0,
        j,
        torch.stack(
            [
                torch.where(move_x, tx * half, torch.zeros_like(tx)),
                torch.where(move_x, torch.zeros_like(ty), ty * half),
            ],
            dim=1,
        ),
    )
    return update


def train_teacher_placement(
    cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
    *,
    verbose: bool = False,
) -> dict:
    """Run a zero-overlap classical teacher for offline demonstrations."""

    input_device = cell_features.device
    base_cell_features = cell_features.clone()
    pin_features_device = pin_features.clone()
    edge_list_device = edge_list.clone()
    initial_cell_features = base_cell_features.clone()

    n = int(base_cell_features.shape[0])
    if n <= 1:
        return {
            "final_cell_features": base_cell_features.to(input_device),
            "initial_cell_features": initial_cell_features.to(input_device),
            "loss_history": {"total_loss": [], "wirelength_loss": [], "overlap_loss": []},
        }

    max_iters = max(_env_int("PLACEMENT_TEACHER_MAX_ITERS", 256), 1)
    step_scale = _env_float("PLACEMENT_TEACHER_STEP_SCALE", 1.2)
    margin = _env_float("PLACEMENT_TEACHER_MARGIN", 0.5)

    centers = base_cell_features[:, 2:4].clone()
    widths = base_cell_features[:, 4]
    heights = base_cell_features[:, 5]

    best_score = _score_candidate(
        _write_positions(base_cell_features, centers),
        pin_features_device,
        edge_list_device,
    )
    best_centers = centers.clone()
    best_overlap_pairs = int(best_score["num_overlap_pairs"])

    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
        "official_overlap_ratio": [best_score["overlap_ratio"]],
        "official_normalized_wl": [best_score["normalized_wl"]],
        "num_overlap_pairs": [best_overlap_pairs],
    }

    for step_idx in range(max_iters):
        candidate = _write_positions(base_cell_features, centers)
        overlaps = exact_overlap_pairs(candidate)
        candidate_score = _score_candidate(candidate, pin_features_device, edge_list_device)

        if _candidate_key((candidate_score, centers)) < _candidate_key((best_score, best_centers)):
            best_score = candidate_score
            best_centers = centers.clone()
            best_overlap_pairs = int(overlaps.shape[0])

        loss_history["official_overlap_ratio"].append(candidate_score["overlap_ratio"])
        loss_history["official_normalized_wl"].append(candidate_score["normalized_wl"])
        loss_history["num_overlap_pairs"].append(int(overlaps.shape[0]))

        if verbose:
            print(
                f"Teacher iter {step_idx + 1}/{max_iters}: "
                f"overlap={candidate_score['overlap_ratio']:.4f} "
                f"wl={candidate_score['normalized_wl']:.4f} "
                f"pairs={overlaps.shape[0]}"
            )

        if overlaps.numel() == 0:
            break

        centers = centers + _separation_update(
            centers,
            widths,
            heights,
            overlaps,
            step_scale=step_scale,
            margin=margin,
        )
        # Keep the cloud centered so repeated pair repairs do not drift.
        centers = centers - centers.mean(dim=0, keepdim=True)

    final_features = _write_positions(base_cell_features, best_centers)
    seq_plus, seq_minus = sequence_pair_from_centers(final_features)
    return {
        "final_cell_features": final_features.to(input_device),
        "initial_cell_features": initial_cell_features.to(input_device),
        "loss_history": loss_history,
        "teacher_stats": {
            "best_overlap_ratio": best_score["overlap_ratio"],
            "best_normalized_wl": best_score["normalized_wl"],
            "best_num_overlap_pairs": best_overlap_pairs,
            "teacher_iters": max_iters,
            "branch_antisymmetry_error": branch_antisymmetry_error(seq_plus, seq_minus),
        },
    }


train_prior_solver_placement = train_teacher_placement
