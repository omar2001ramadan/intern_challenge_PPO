"""Comprehensive step-by-step visual and structured trace for policy rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from constraints import (
    boundary_signed_constraints,
    branch_signed_constraints,
    exact_overlap_pairs,
    overlap_ratio_from_pairs,
)
from env import EnvConfig, PlacementOrderingEnv, normalized_wirelength, write_positions
from ordering_policy import (
    apply_rollout_memory_policy,
    hierarchical_active_branch_weights,
    load_policy_checkpoint,
    rollout_memory_config_from_checkpoint,
)
from placement import generate_placement_input
from validate_policy import default_device_arg, initialize_random_spread, make_case, parse_cases


ACTIVE_PAIR_FIELDNAMES = [
    "rollout",
    "step",
    "phase",
    "pair_index",
    "i",
    "j",
    "age",
    "active_branch",
    "dual_L",
    "dual_R",
    "dual_B",
    "dual_A",
    "active_dual",
    "signed_g_raw",
    "signed_g_norm",
    "positive_violation_raw",
    "positive_violation_norm",
    "branch_pressure_value",
    "soft_w_L",
    "soft_w_R",
    "soft_w_B",
    "soft_w_A",
]

BOUNDARY_DUAL_FIELDNAMES = [
    "rollout",
    "step",
    "phase",
    "cell_id",
    "side",
    "dual",
    "signed_g_raw",
    "signed_g_norm",
    "positive_violation_raw",
    "positive_violation_norm",
    "pressure_value",
]

DENSITY_DUAL_FIELDNAMES = [
    "rollout",
    "step",
    "phase",
    "bin_id",
    "dual",
    "signed_g",
    "positive_violation",
    "pressure_value",
]

ORDERING_SCORE_FIELDNAMES = [
    "rollout",
    "step",
    "level",
    "item_id",
    "source_cell_id",
    "plus_score",
    "minus_score",
    "plus_rank",
    "minus_rank",
]

CLUSTER_LOGIT_FIELDNAMES = [
    "rollout",
    "step",
    "cell_id",
    "cluster_label",
    "logit",
    "sampled_cluster",
    "is_sampled",
]

PAIR_BRANCH_LOGIT_FIELDNAMES = [
    "rollout",
    "step",
    "pair_index",
    "i",
    "j",
    "dag_axis_logit_x",
    "dag_axis_logit_y",
    "sampled_axis",
    "branch_logit_L",
    "branch_logit_R",
    "branch_logit_B",
    "branch_logit_A",
    "sampled_branch",
    "soft_w_L",
    "soft_w_R",
    "soft_w_B",
    "soft_w_A",
]

PHR_STEP_FIELDNAMES = [
    "rollout",
    "step",
    "inner_step",
    "lagrangian",
    "wirelength",
    "branch_violation",
    "boundary_violation",
    "density_overflow",
    "grad_norm",
    "grad_norm_clipped",
    "delta_mean",
    "delta_max",
]

PHR_COORDINATE_FIELDNAMES = [
    "rollout",
    "step",
    "inner_step",
    "cell_id",
    "x",
    "y",
]

ENSEMBLE_CANDIDATE_FIELDNAMES = [
    "case_label",
    "rollout",
    "policy_seed",
    "steps_recorded",
    "best_source_step",
    "best_overlap_ratio",
    "best_overlap_cells",
    "best_num_overlap_pairs",
    "best_normalized_wl",
]

MEMORY_SUMMARY_FIELDNAMES = [
    "rollout",
    "step",
    "pre_norm",
    "post_norm",
    "delta_norm",
    "pre_mean_abs",
    "post_mean_abs",
    "delta_mean_abs",
]

MEMORY_VECTOR_FIELDNAMES = [
    "rollout",
    "step",
    "phase",
    "index",
    "value",
]

PAIR_LIFECYCLE_FIELDNAMES = [
    "rollout",
    "step",
    "i",
    "j",
    "event",
    "was_active",
    "is_active",
    "seen_before",
    "last_active_step_before",
    "current_age",
    "active_branch_pre",
    "active_branch_post",
    "exact_overlap_post",
]

FULL_PAIR_MATRIX_FIELDNAMES = [
    "rollout",
    "step",
    "stage",
    "i",
    "j",
    "dx",
    "dy",
    "ox",
    "oy",
    "overlap_area",
    "exact_overlap",
    "active_pair",
    "geometry_branch",
    "active_branch",
    "soft_w_L",
    "soft_w_R",
    "soft_w_B",
    "soft_w_A",
]

CASE_CANDIDATE_FIELDNAMES = [
    "case_label",
    "case_source",
    "rollout",
    "policy_seed",
    "steps_recorded",
    "best_source_step",
    "best_overlap_ratio",
    "best_overlap_cells",
    "best_num_overlap_pairs",
    "best_normalized_wl",
]


def parse_size_spec(value: str) -> tuple[int, int]:
    macros, std_cells = value.split(":", 1)
    return int(macros), int(std_cells)


def parse_int_list(value: str | None) -> list[int]:
    if value is None:
        return []
    items = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            items.append(int(token))
    return items


def sanitize_label(value: str) -> str:
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def build_case_specs(args):
    if args.size is not None:
        num_macros, num_std_cells = parse_size_spec(args.size)
        batch_seeds = parse_int_list(args.batch_seeds)
        if not batch_seeds:
            batch_seeds = [int(args.seed)]
        specs = []
        for seed in batch_seeds:
            specs.append(
                {
                    "kind": "synthetic",
                    "num_macros": int(num_macros),
                    "num_std_cells": int(num_std_cells),
                    "seed": int(seed),
                    "label": f"synthetic_{num_macros}m_{num_std_cells}s_seed{seed}",
                }
            )
        return specs

    selected = parse_cases(str(args.case))
    specs = []
    for case in selected:
        specs.append(
            {
                "kind": "test_case",
                "case": case,
                "label": f"test_case_{int(case[0])}",
            }
        )
    return specs


def build_case_from_spec(case_spec, device):
    if case_spec["kind"] == "synthetic":
        num_macros = int(case_spec["num_macros"])
        num_std_cells = int(case_spec["num_std_cells"])
        seed = int(case_spec["seed"])
        torch.manual_seed(seed)
        cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)
        cell_features = initialize_random_spread(cell_features)
        metadata = {
            "label": str(case_spec["label"]),
            "source": "synthetic",
            "seed": seed,
            "num_macros": num_macros,
            "num_std_cells": num_std_cells,
        }
        return metadata, cell_features.to(device), pin_features.to(device), edge_list.to(device)

    case = case_spec["case"]
    test_id, cell_features, pin_features, edge_list = make_case(case, device)
    metadata = {
        "label": str(case_spec["label"]),
        "source": "test_case",
        "test_case": {
            "test_id": int(case[0]),
            "num_macros": int(case[1]),
            "num_std_cells": int(case[2]),
            "seed": int(case[3]),
        },
    }
    return metadata, cell_features, pin_features, edge_list


def _coalesce(override, fallback):
    return fallback if override is None else override


def env_config_from_checkpoint(checkpoint, args) -> EnvConfig:
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    stats = checkpoint.get("stats", {}) if isinstance(checkpoint, dict) else {}

    soft_relaxation = _coalesce(args.soft_relaxation, not bool(config.get("no_soft_relax", False)))
    residual_flow = _coalesce(args.residual_flow, not bool(config.get("no_residual_flow", False)))
    phr_layer = _coalesce(args.phr_layer, not bool(config.get("no_phr_layer", False)))
    exact_audit = _coalesce(args.exact_audit, not bool(config.get("no_exact_audit", False)))
    density = _coalesce(args.density, not bool(config.get("no_density", False)))
    clusters = _coalesce(args.clusters, not bool(config.get("disable_clusters", False)))
    stop = _coalesce(args.stop, not bool(config.get("disable_stop", False)))
    fixed_pd_controls = _coalesce(args.fixed_pd_controls, bool(config.get("fixed_pd_controls", False)))
    soft_tau_value = stats.get("soft_tau", None)
    if soft_tau_value is None:
        soft_tau_value = config.get("soft_tau_start", 1.0)
    if soft_tau_value is None:
        soft_tau_value = 1.0

    return EnvConfig(
        horizon=int(_coalesce(args.steps, config.get("horizon", 32))),
        coordinate_steps=int(config.get("coordinate_steps", 8)),
        coordinate_lr=float(config.get("coordinate_lr", 0.01)),
        rho=float(config.get("rho", 8.0)),
        lambda_wirelength=float(config.get("lambda_wirelength", 1.0)),
        lambda_overlap=float(config.get("lambda_overlap", 16.0)),
        lambda_density=float(config.get("lambda_density", 4.0)),
        entropy_reward_coef=float(config.get("entropy_reward_coef", 0.005)),
        terminal_feasible_bonus=float(config.get("terminal_feasible_bonus", 2.0)),
        terminal_wirelength_coef=float(config.get("terminal_wirelength_coef", 0.25)),
        active_pair_limit=int(config.get("active_pair_limit", 500_000)),
        active_pair_retention=int(config.get("active_pair_retention", 4)),
        density_bins=int(config.get("density_bins", 8)),
        density_rho_max=float(config.get("density_rho_max", 0.85)),
        soft_relaxation=bool(soft_relaxation),
        soft_tau=float(soft_tau_value),
        exact_overlap_reward_coef=float(config.get("overlap_reward_coef", 8.0)),
        exact_wirelength_reward_coef=float(config.get("wirelength_reward_coef", 0.10)),
        exact_overlap_pairs_reward_coef=float(config.get("overlap_pairs_reward_coef", 1.0)),
        current_overlap_penalty_coef=float(config.get("current_overlap_penalty", 1.0)),
        current_overlap_pairs_penalty_coef=float(config.get("current_overlap_pairs_penalty", 0.25)),
        lag_reward_coef=float(config.get("lag_reward_coef", 0.10)),
        lag_reward_tanh_scale=float(config.get("lag_reward_tanh_scale", 25.0)),
        exact_overlap_regression_coef=float(config.get("overlap_regression_coef", 16.0)),
        branch_violation_penalty_coef=float(config.get("branch_violation_penalty", 2.0)),
        missed_pair_penalty_coef=float(config.get("missed_pair_penalty", 0.01)),
        stop_no_progress_penalty=float(config.get("stop_no_progress_penalty", 2.0)),
        stop_gate_overlap_threshold=float(config.get("stop_gate_overlap", 0.02)),
        stop_gate_penalty=float(config.get("stop_gate_penalty", 4.0)),
        soft_branch_epsilon=float(config.get("soft_branch_epsilon", 1e-4)),
        audit_missed_target=float(config.get("audit_missed_target", 64.0)),
        audit_pressure_gamma=float(config.get("audit_pressure_gamma", 1.0)),
        audit_pressure_max=float(config.get("audit_pressure_max", 4.0)),
        enable_residual_flow=bool(residual_flow),
        enable_phr_layer=bool(phr_layer),
        enable_exact_audit=bool(exact_audit),
        enable_density=bool(density),
        enable_clusters=bool(clusters),
        enable_stop=bool(stop),
        fixed_pd_controls=bool(fixed_pd_controls),
        ordering_representation=str(
            _coalesce(args.ordering_representation, config.get("ordering_representation", "sequence_pair"))
        ),
        branch_mode=str(_coalesce(args.branch_mode, config.get("branch_mode", "ordering"))),
        al_mode=str(_coalesce(args.al_mode, config.get("al_mode", "signed_phr"))),
    )


def branch_name(value: int) -> str:
    names = {0: "L", 1: "R", 2: "B", 3: "A"}
    return names.get(int(value), f"unknown_{value}")


def boundary_name(value: int) -> str:
    names = {0: "L", 1: "R", 2: "B", 3: "T"}
    return names.get(int(value), f"unknown_{value}")


def sequence_ranks(sequence: torch.Tensor, size: int) -> torch.Tensor:
    ranks = torch.full((size,), -1, dtype=torch.long, device=sequence.device if sequence.numel() > 0 else "cpu")
    if sequence.numel() > 0:
        ranks[sequence.long()] = torch.arange(sequence.numel(), dtype=torch.long, device=sequence.device)
    return ranks


def json_ready(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def branch_lookup_from_pairs(active_pairs, branches):
    lookup = {}
    if active_pairs is None or branches is None or active_pairs.numel() == 0 or branches.numel() == 0:
        return lookup
    for idx, pair in enumerate(active_pairs.detach().cpu().tolist()):
        lookup[tuple(pair)] = branch_name(int(branches[idx].detach().item()))
    return lookup


def age_lookup_from_pairs(active_pairs, ages):
    lookup = {}
    if active_pairs is None or ages is None or active_pairs.numel() == 0 or ages.numel() == 0:
        return lookup
    for idx, pair in enumerate(active_pairs.detach().cpu().tolist()):
        lookup[tuple(pair)] = int(ages[idx].detach().item())
    return lookup


def geometric_branch_for_pair(centers, cell_features, i, j):
    xi = float(centers[i, 0].detach().item())
    yi = float(centers[i, 1].detach().item())
    xj = float(centers[j, 0].detach().item())
    yj = float(centers[j, 1].detach().item())
    wi = float(cell_features[i, 4].detach().item())
    hi = float(cell_features[i, 5].detach().item())
    wj = float(cell_features[j, 4].detach().item())
    hj = float(cell_features[j, 5].detach().item())
    values = {
        "L": xi + wi / 2.0 - xj + wj / 2.0,
        "R": xj + wj / 2.0 - xi + wi / 2.0,
        "B": yi + hi / 2.0 - yj + hj / 2.0,
        "A": yj + hj / 2.0 - yi + hi / 2.0,
    }
    return min(values, key=values.get)


def pair_soft_weights_from_action(action, i, j):
    p_plus = torch.sigmoid(action.plus_scores[i] - action.plus_scores[j])
    p_minus = torch.sigmoid(action.minus_scores[i] - action.minus_scores[j])
    return torch.stack(
        [
            p_plus * p_minus,
            (1.0 - p_plus) * (1.0 - p_minus),
            p_plus * (1.0 - p_minus),
            (1.0 - p_plus) * p_minus,
        ]
    ).detach().cpu()


def overlap_pair_rows(cell_features, centers, active_pairs=None, branches=None):
    current = write_positions(cell_features, centers)
    pairs = exact_overlap_pairs(current)
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    branch_lookup = {}
    active_lookup = set()
    if active_pairs is not None and active_pairs.numel() > 0:
        active_list = active_pairs.detach().cpu().tolist()
        active_lookup = {tuple(pair) for pair in active_list}
        if branches is not None and branches.numel() == active_pairs.shape[0]:
            for idx, pair in enumerate(active_list):
                branch_lookup[tuple(pair)] = branch_name(int(branches[idx].detach().item()))

    pair_rows = []
    overlap_cells = set()
    for i, j in pairs.detach().cpu().tolist():
        dx = abs(float(centers[i, 0].detach().item() - centers[j, 0].detach().item()))
        dy = abs(float(centers[i, 1].detach().item() - centers[j, 1].detach().item()))
        ox = max(0.0, 0.5 * float(widths[i].detach().item() + widths[j].detach().item()) - dx)
        oy = max(0.0, 0.5 * float(heights[i].detach().item() + heights[j].detach().item()) - dy)
        area = ox * oy
        overlap_cells.add(i)
        overlap_cells.add(j)
        pair_rows.append(
            {
                "i": int(i),
                "j": int(j),
                "ox": float(ox),
                "oy": float(oy),
                "area": float(area),
                "active_pair": tuple((i, j)) in active_lookup,
                "branch": branch_lookup.get(tuple((i, j)), ""),
            }
        )
    pair_rows.sort(key=lambda row: (-row["area"], row["i"], row["j"]))
    return overlap_cells, pair_rows


def stage_snapshot(cell_features, pin_features, edge_list, centers, active_pairs=None, branches=None):
    current = write_positions(cell_features, centers)
    pairs = exact_overlap_pairs(current)
    overlap_ratio, overlap_cells = overlap_ratio_from_pairs(int(current.shape[0]), pairs)
    overlap_cell_ids, pair_rows = overlap_pair_rows(
        cell_features,
        centers,
        active_pairs=active_pairs,
        branches=branches,
    )
    return {
        "score": {
            "overlap_ratio": float(overlap_ratio),
            "overlap_cells": int(overlap_cells),
            "num_overlap_pairs": int(pairs.shape[0]),
            "normalized_wl": float(normalized_wirelength(current, pin_features, edge_list)),
        },
        "overlap_cells": overlap_cell_ids,
        "pairs": pair_rows,
    }


def candidate_key(score):
    return (
        int(score["overlap_cells"]),
        float(score["overlap_ratio"]),
        float(score["normalized_wl"]),
        int(score["num_overlap_pairs"]),
    )


def compute_base_centers(env, action):
    step_scale = float(action.step_scale.detach().item())
    residual = torch.tanh(action.residual_flow.detach()) * (step_scale * env.length_scale)
    if not env.config.enable_residual_flow:
        residual = torch.zeros_like(residual)
    return env.centers.detach().clone() + residual


def top_move_indices(from_centers, to_centers, limit):
    deltas = (to_centers - from_centers).detach()
    norms = torch.norm(deltas, dim=1)
    if norms.numel() == 0:
        return []
    k = min(int(limit), int(norms.numel()))
    if k <= 0:
        return []
    values, indices = torch.topk(norms, k)
    return [int(idx.item()) for idx, value in zip(indices, values) if float(value.item()) > 1e-6]


def render_stage_panel(ax, cell_features, centers, overlap_cells, pair_rows, title, move_from=None, arrow_color="black", arrow_limit=12):
    from matplotlib.patches import FancyArrowPatch, Rectangle

    positions = centers.detach().cpu()
    widths = cell_features[:, 4].detach().cpu()
    heights = cell_features[:, 5].detach().cpu()

    for cell_idx in range(int(cell_features.shape[0])):
        x = float(positions[cell_idx, 0].item() - widths[cell_idx].item() / 2.0)
        y = float(positions[cell_idx, 1].item() - heights[cell_idx].item() / 2.0)
        is_overlap = cell_idx in overlap_cells
        rect = Rectangle(
            (x, y),
            float(widths[cell_idx].item()),
            float(heights[cell_idx].item()),
            facecolor="#fca5a5" if is_overlap else "#93c5fd",
            edgecolor="#991b1b" if is_overlap else "#1d4ed8",
            linewidth=0.7,
            alpha=0.80,
        )
        ax.add_patch(rect)

    for row in pair_rows[: min(len(pair_rows), 10)]:
        i = row["i"]
        j = row["j"]
        xi = float(positions[i, 0].item())
        yi = float(positions[i, 1].item())
        xj = float(positions[j, 0].item())
        yj = float(positions[j, 1].item())
        ax.plot([xi, xj], [yi, yj], color="#dc2626", linewidth=0.7 + min(row["area"], 3.0), alpha=0.35)

    if move_from is not None:
        move_from = move_from.detach().cpu()
        for idx in top_move_indices(move_from, positions, arrow_limit):
            start = (float(move_from[idx, 0].item()), float(move_from[idx, 1].item()))
            end = (float(positions[idx, 0].item()), float(positions[idx, 1].item()))
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=8,
                linewidth=0.7,
                color=arrow_color,
                alpha=0.8,
            )
            ax.add_patch(arrow)

    ax.set_title(f"{title}\npairs={len(pair_rows)}", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def save_frame(path, cell_features, frame_record, arrow_limit=12, dpi=150):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    stages = ["pre", "base", "post", "best"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.reshape(-1)

    all_centers = [frame_record[stage]["centers"] for stage in stages]
    widths = cell_features[:, 4].detach().cpu()
    heights = cell_features[:, 5].detach().cpu()
    all_x = torch.cat([centers[:, 0].detach().cpu() for centers in all_centers])
    all_y = torch.cat([centers[:, 1].detach().cpu() for centers in all_centers])
    margin = max(float(widths.max().item()), float(heights.max().item()), 1.0) * 1.5
    x_min = float(all_x.min().item() - margin)
    x_max = float(all_x.max().item() + margin)
    y_min = float(all_y.min().item() - margin)
    y_max = float(all_y.max().item() + margin)

    descriptors = {
        "pre": ("Current", None, None),
        "base": ("Policy Base", frame_record["pre"]["centers"], "#059669"),
        "post": ("Post PHR", frame_record["base"]["centers"], "#d97706"),
        "best": (f"Best So Far (step {frame_record['best_source_step']})", None, None),
    }

    for ax, stage in zip(axes, stages):
        title, move_from, arrow_color = descriptors[stage]
        render_stage_panel(
            ax,
            cell_features,
            frame_record[stage]["centers"],
            frame_record[stage]["snapshot"]["overlap_cells"],
            frame_record[stage]["snapshot"]["pairs"],
            title,
            move_from=move_from,
            arrow_color=arrow_color or "black",
            arrow_limit=arrow_limit,
        )
        score = frame_record[stage]["snapshot"]["score"]
        active_pairs = frame_record[stage].get("active_pairs_count", "")
        note = (
            f"overlap={score['overlap_ratio']:.4f}\n"
            f"pairs={score['num_overlap_pairs']}\n"
            f"wl={score['normalized_wl']:.4f}"
        )
        if active_pairs != "":
            note += f"\nactive={active_pairs}"
        ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    info = frame_record["info"]
    fig.suptitle(
        (
            f"Rollout {frame_record['rollout']}  Step {frame_record['step']}  "
            f"reward={info['reward']:.3f}  step_scale={info['step_scale']:.3f}  "
            f"rho={info['rho']:.3f}  eta={info['eta']:.3f}  alpha={info['alpha']:.3f}  "
            f"pd_steps={info['pd_steps']}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_timeline_plot(path, step_rows, dpi=150):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    steps = [row["step"] for row in step_rows]
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    for stage, color in [("pre", "#1d4ed8"), ("base", "#059669"), ("post", "#d97706"), ("best", "#7c3aed")]:
        axes[0].plot(steps, [row[f"{stage}_overlap_ratio"] for row in step_rows], label=stage, color=color)
        axes[1].plot(steps, [row[f"{stage}_num_overlap_pairs"] for row in step_rows], label=stage, color=color)
        axes[2].plot(steps, [row[f"{stage}_normalized_wl"] for row in step_rows], label=stage, color=color)
    axes[0].set_ylabel("Overlap Ratio")
    axes[1].set_ylabel("Overlap Pairs")
    axes[2].set_ylabel("Normalized WL")
    axes[0].legend(loc="best", ncol=4, fontsize=8)

    axes[3].plot(steps, [row["reward"] for row in step_rows], label="reward", color="#111827")
    axes[3].plot(steps, [row["audit_pressure_scale"] for row in step_rows], label="audit_pressure", color="#dc2626")
    axes[3].plot(steps, [row["step_scale"] for row in step_rows], label="step_scale", color="#2563eb")
    axes[3].plot(steps, [row["residual_norm"] for row in step_rows], label="residual_norm", color="#16a34a")
    axes[3].set_ylabel("Controls")
    axes[3].set_xlabel("Step")
    axes[3].legend(loc="best", ncol=4, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.20)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_ordering_score_rows(rows, rollout_idx, step, action):
    plus_ranks = sequence_ranks(action.seq_plus, int(action.plus_scores.numel())).detach().cpu()
    minus_ranks = sequence_ranks(action.seq_minus, int(action.minus_scores.numel())).detach().cpu()
    for item_id in range(int(action.plus_scores.numel())):
        rows.append(
            {
                "rollout": int(rollout_idx),
                "step": int(step),
                "level": "global_cell",
                "item_id": int(item_id),
                "source_cell_id": int(item_id),
                "plus_score": float(action.plus_scores[item_id].detach().item()),
                "minus_score": float(action.minus_scores[item_id].detach().item()),
                "plus_rank": int(plus_ranks[item_id].item()),
                "minus_rank": int(minus_ranks[item_id].item()),
            }
        )

    if action.macro_seq_plus.numel() > 0:
        macro_plus_ranks = sequence_ranks(action.macro_seq_plus, int(action.macro_plus_scores.numel())).detach().cpu()
        macro_minus_ranks = sequence_ranks(action.macro_seq_minus, int(action.macro_minus_scores.numel())).detach().cpu()
        macro_cell_indices = action.macro_cell_indices.detach().cpu()
        for item_id in range(int(action.macro_plus_scores.numel())):
            rows.append(
                {
                    "rollout": int(rollout_idx),
                    "step": int(step),
                    "level": "macro",
                    "item_id": int(item_id),
                    "source_cell_id": int(macro_cell_indices[item_id].item()),
                    "plus_score": float(action.macro_plus_scores[item_id].detach().item()),
                    "minus_score": float(action.macro_minus_scores[item_id].detach().item()),
                    "plus_rank": int(macro_plus_ranks[item_id].item()),
                    "minus_rank": int(macro_minus_ranks[item_id].item()),
                }
            )

    if action.cluster_seq_plus.numel() > 0:
        cluster_plus_ranks = sequence_ranks(action.cluster_seq_plus, int(action.cluster_plus_scores.numel())).detach().cpu()
        cluster_minus_ranks = sequence_ranks(action.cluster_seq_minus, int(action.cluster_minus_scores.numel())).detach().cpu()
        for item_id in range(int(action.cluster_plus_scores.numel())):
            rows.append(
                {
                    "rollout": int(rollout_idx),
                    "step": int(step),
                    "level": "cluster",
                    "item_id": int(item_id),
                    "source_cell_id": -1,
                    "plus_score": float(action.cluster_plus_scores[item_id].detach().item()),
                    "minus_score": float(action.cluster_minus_scores[item_id].detach().item()),
                    "plus_rank": int(cluster_plus_ranks[item_id].item()),
                    "minus_rank": int(cluster_minus_ranks[item_id].item()),
                }
            )


def append_cluster_logit_rows(rows, rollout_idx, step, action, std_mask):
    if not hasattr(action, "cluster_logits") or action.cluster_logits.numel() == 0:
        return
    cluster_logits = action.cluster_logits.detach().cpu()
    cluster_ids = action.cluster_ids.detach().cpu()
    std_mask = std_mask.detach().cpu()
    for cell_id in torch.where(std_mask)[0].tolist():
        sampled_cluster = int(cluster_ids[cell_id].item())
        for cluster_label in range(int(cluster_logits.shape[1])):
            rows.append(
                {
                    "rollout": int(rollout_idx),
                    "step": int(step),
                    "cell_id": int(cell_id),
                    "cluster_label": int(cluster_label),
                    "logit": float(cluster_logits[cell_id, cluster_label].item()),
                    "sampled_cluster": int(sampled_cluster),
                    "is_sampled": int(cluster_label == sampled_cluster),
                }
            )


def append_pair_branch_rows(rows, rollout_idx, step, active_pairs, action, soft_weights):
    if active_pairs.numel() == 0:
        return
    pairs_cpu = active_pairs.detach().cpu()
    dag_logits = action.dag_axis_logits.detach().cpu()
    dag_axis = action.dag_axis.detach().cpu() if action.dag_axis.numel() > 0 else torch.empty(0, dtype=torch.long)
    pair_logits = action.pair_branch_logits.detach().cpu()
    pair_choices = action.pair_branch_choices.detach().cpu() if action.pair_branch_choices.numel() > 0 else torch.empty(0, dtype=torch.long)
    weights_cpu = None if soft_weights is None else soft_weights.detach().cpu()
    for pair_index, pair in enumerate(pairs_cpu.tolist()):
        dag_x = dag_y = ""
        axis = ""
        if dag_logits.numel() > 0 and pair_index < dag_logits.shape[0]:
            dag_x = float(dag_logits[pair_index, 0].item())
            dag_y = float(dag_logits[pair_index, 1].item())
            axis = int(dag_axis[pair_index].item())
        logit_l = logit_r = logit_b = logit_a = ""
        sampled_branch = ""
        if pair_logits.numel() > 0 and pair_index < pair_logits.shape[0]:
            logit_l = float(pair_logits[pair_index, 0].item())
            logit_r = float(pair_logits[pair_index, 1].item())
            logit_b = float(pair_logits[pair_index, 2].item())
            logit_a = float(pair_logits[pair_index, 3].item())
            sampled_branch = int(pair_choices[pair_index].item())
        row = {
            "rollout": int(rollout_idx),
            "step": int(step),
            "pair_index": int(pair_index),
            "i": int(pair[0]),
            "j": int(pair[1]),
            "dag_axis_logit_x": dag_x,
            "dag_axis_logit_y": dag_y,
            "sampled_axis": axis,
            "branch_logit_L": logit_l,
            "branch_logit_R": logit_r,
            "branch_logit_B": logit_b,
            "branch_logit_A": logit_a,
            "sampled_branch": sampled_branch,
            "soft_w_L": "",
            "soft_w_R": "",
            "soft_w_B": "",
            "soft_w_A": "",
        }
        if weights_cpu is not None and pair_index < weights_cpu.shape[0]:
            row["soft_w_L"] = float(weights_cpu[pair_index, 0].item())
            row["soft_w_R"] = float(weights_cpu[pair_index, 1].item())
            row["soft_w_B"] = float(weights_cpu[pair_index, 2].item())
            row["soft_w_A"] = float(weights_cpu[pair_index, 3].item())
        rows.append(row)


def append_policy_trace_row(rows, rollout_idx, step, action, info):
    rows.append(
        {
            "rollout": int(rollout_idx),
            "step": int(step),
            "phase_name": str(getattr(action, "phase_name", "")),
            "phase_request": int(getattr(action, "phase_request", action.stop.new_zeros(())).detach().item()) if torch.is_tensor(getattr(action, "phase_request", None)) else 0,
            "enable_unlock": bool(getattr(action, "enable_unlock", False)),
            "unlock_source_index": int(getattr(action, "unlock_source_index", action.stop.new_zeros((), dtype=torch.long)).detach().item()) if torch.is_tensor(getattr(action, "unlock_source_index", None)) else 0,
            "unlock_radius_index": int(getattr(action, "unlock_radius_index", action.stop.new_zeros((), dtype=torch.long)).detach().item()) if torch.is_tensor(getattr(action, "unlock_radius_index", None)) else 0,
            "ordering_representation": str(action.ordering_representation),
            "branch_mode": str(action.branch_mode),
            "enable_clusters": bool(action.enable_clusters),
            "enable_stop": bool(action.enable_stop),
            "seq_plus": json_ready(action.seq_plus),
            "seq_minus": json_ready(action.seq_minus),
            "macro_seq_plus": json_ready(action.macro_seq_plus),
            "macro_seq_minus": json_ready(action.macro_seq_minus),
            "macro_cell_indices": json_ready(action.macro_cell_indices),
            "cluster_seq_plus": json_ready(action.cluster_seq_plus),
            "cluster_seq_minus": json_ready(action.cluster_seq_minus),
            "cluster_ids": json_ready(action.cluster_ids),
            "controls": {
                "step_scale": float(action.step_scale.detach().item()),
                "rho": float(action.rho.detach().item()),
                "eta": float(action.eta.detach().item()),
                "alpha": float(action.alpha.detach().item()),
                "branch_pressure": float(action.branch_pressure.detach().item()),
                "density_pressure": float(action.density_pressure.detach().item()),
                "boundary_pressure": float(action.boundary_pressure.detach().item()),
                "pair_emphasis": float(action.pair_emphasis.detach().item()),
                "tau": float(action.tau.detach().item()),
                "incumbent_mix": float(action.incumbent_mix.detach().item()),
                "pd_steps": int(action.pd_steps),
                "stop": float(action.stop.detach().item()),
                "stop_probability": float(action.stop_probability.detach().item()),
            },
            "memory": {
                "pre": json_ready(action.memory),
                "post": json_ready(action.next_memory),
            },
            "group_logprobs": {key: float(value.detach().item()) for key, value in action.group_logprobs.items()},
            "group_entropies": {key: float(value.detach().item()) for key, value in action.group_entropies.items()},
            "group_token_counts": {key: int(value.detach().item()) for key, value in action.group_token_counts.items()},
            "step_info": dict(info),
        }
    )


def append_memory_rows(summary_rows, vector_rows, jsonl_rows, rollout_idx, step, pre_memory, post_memory):
    pre_memory = pre_memory.detach().cpu()
    post_memory = post_memory.detach().cpu()
    delta = post_memory - pre_memory
    summary_rows.append(
        {
            "rollout": int(rollout_idx),
            "step": int(step),
            "pre_norm": float(pre_memory.norm().item()),
            "post_norm": float(post_memory.norm().item()),
            "delta_norm": float(delta.norm().item()),
            "pre_mean_abs": float(pre_memory.abs().mean().item()) if pre_memory.numel() > 0 else 0.0,
            "post_mean_abs": float(post_memory.abs().mean().item()) if post_memory.numel() > 0 else 0.0,
            "delta_mean_abs": float(delta.abs().mean().item()) if delta.numel() > 0 else 0.0,
        }
    )
    for phase_name, vector in [("pre", pre_memory), ("post", post_memory), ("delta", delta)]:
        for index in range(int(vector.numel())):
            vector_rows.append(
                {
                    "rollout": int(rollout_idx),
                    "step": int(step),
                    "phase": phase_name,
                    "index": int(index),
                    "value": float(vector[index].item()),
                }
            )
    jsonl_rows.append(
        {
            "rollout": int(rollout_idx),
            "step": int(step),
            "pre_memory": json_ready(pre_memory),
            "post_memory": json_ready(post_memory),
            "delta_memory": json_ready(delta),
        }
    )


def append_pair_lifecycle_rows(
    rows,
    pair_history,
    rollout_idx,
    step,
    pre_active_pairs,
    post_active_pairs,
    pre_branch_lookup,
    post_branch_lookup,
    post_age_lookup,
    post_overlap_pairs,
):
    pre_set = {tuple(pair) for pair in pre_active_pairs.detach().cpu().tolist()} if pre_active_pairs.numel() > 0 else set()
    post_set = {tuple(pair) for pair in post_active_pairs.detach().cpu().tolist()} if post_active_pairs.numel() > 0 else set()
    union_pairs = sorted(pre_set | post_set)
    for pair in union_pairs:
        seen_before = pair in pair_history
        last_active_step_before = pair_history.get(pair, {}).get("last_active_step")
        was_active = pair in pre_set
        is_active = pair in post_set
        if was_active and is_active:
            event = "remain"
        elif was_active and not is_active:
            event = "drop"
        elif (not was_active) and is_active:
            event = "reenter" if seen_before else "enter"
        else:
            continue
        rows.append(
            {
                "rollout": int(rollout_idx),
                "step": int(step),
                "i": int(pair[0]),
                "j": int(pair[1]),
                "event": event,
                "was_active": int(was_active),
                "is_active": int(is_active),
                "seen_before": int(seen_before),
                "last_active_step_before": "" if last_active_step_before is None else int(last_active_step_before),
                "current_age": "" if pair not in post_age_lookup else int(post_age_lookup[pair]),
                "active_branch_pre": pre_branch_lookup.get(pair, ""),
                "active_branch_post": post_branch_lookup.get(pair, ""),
                "exact_overlap_post": int(pair in post_overlap_pairs),
            }
        )
        if is_active:
            pair_history[pair] = {"last_active_step": int(step)}
        elif seen_before:
            pair_history[pair]["last_inactive_step"] = int(step)


def append_full_pair_matrix_rows(
    rows,
    rollout_idx,
    step,
    stage,
    cell_features,
    centers,
    active_pairs,
    branches,
    action=None,
):
    n = int(cell_features.shape[0])
    active_set = set()
    active_branch_lookup = {}
    if active_pairs is not None and active_pairs.numel() > 0:
        active_list = active_pairs.detach().cpu().tolist()
        active_set = {tuple(pair) for pair in active_list}
        if branches is not None and branches.numel() == active_pairs.shape[0]:
            active_branch_lookup = branch_lookup_from_pairs(active_pairs, branches)
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(centers[j, 0].detach().item() - centers[i, 0].detach().item())
            dy = float(centers[j, 1].detach().item() - centers[i, 1].detach().item())
            wi = float(cell_features[i, 4].detach().item())
            hi = float(cell_features[i, 5].detach().item())
            wj = float(cell_features[j, 4].detach().item())
            hj = float(cell_features[j, 5].detach().item())
            ox = max(0.0, 0.5 * (wi + wj) - abs(dx))
            oy = max(0.0, 0.5 * (hi + hj) - abs(dy))
            weights = None
            if action is not None:
                weights = pair_soft_weights_from_action(action, i, j)
            rows.append(
                {
                    "rollout": int(rollout_idx),
                    "step": int(step),
                    "stage": stage,
                    "i": int(i),
                    "j": int(j),
                    "dx": float(dx),
                    "dy": float(dy),
                    "ox": float(ox),
                    "oy": float(oy),
                    "overlap_area": float(ox * oy),
                    "exact_overlap": int(ox > 0.0 and oy > 0.0),
                    "active_pair": int((i, j) in active_set),
                    "geometry_branch": geometric_branch_for_pair(centers, cell_features, i, j),
                    "active_branch": active_branch_lookup.get((i, j), ""),
                    "soft_w_L": "" if weights is None else float(weights[0].item()),
                    "soft_w_R": "" if weights is None else float(weights[1].item()),
                    "soft_w_B": "" if weights is None else float(weights[2].item()),
                    "soft_w_A": "" if weights is None else float(weights[3].item()),
                }
            )


def collect_active_pair_dual_rows(
    rows,
    rollout_idx,
    step,
    phase,
    cell_features,
    length_scale,
    centers,
    active_pairs,
    branches,
    branch_duals,
    ages=None,
    soft_weights=None,
    branch_pressure_values=None,
):
    if active_pairs is None or active_pairs.numel() == 0 or branches is None or branches.numel() == 0:
        return
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    signed_raw = branch_signed_constraints(centers, widths, heights, active_pairs, branches)
    signed_norm = signed_raw / length_scale
    active_pairs_cpu = active_pairs.detach().cpu()
    branches_cpu = branches.detach().cpu()
    duals_cpu = branch_duals.detach().cpu()
    signed_raw_cpu = signed_raw.detach().cpu()
    signed_norm_cpu = signed_norm.detach().cpu()
    ages_cpu = None if ages is None else ages.detach().cpu()
    weights_cpu = None if soft_weights is None else soft_weights.detach().cpu()
    pressure_cpu = None if branch_pressure_values is None else branch_pressure_values.detach().cpu()
    for pair_index, pair in enumerate(active_pairs_cpu.tolist()):
        branch_idx = int(branches_cpu[pair_index].item())
        rows.append(
            {
                "rollout": int(rollout_idx),
                "step": int(step),
                "phase": phase,
                "pair_index": int(pair_index),
                "i": int(pair[0]),
                "j": int(pair[1]),
                "age": "" if ages_cpu is None or pair_index >= ages_cpu.numel() else int(ages_cpu[pair_index].item()),
                "active_branch": branch_name(branch_idx),
                "dual_L": float(duals_cpu[pair_index, 0].item()),
                "dual_R": float(duals_cpu[pair_index, 1].item()),
                "dual_B": float(duals_cpu[pair_index, 2].item()),
                "dual_A": float(duals_cpu[pair_index, 3].item()),
                "active_dual": float(duals_cpu[pair_index, branch_idx].item()),
                "signed_g_raw": float(signed_raw_cpu[pair_index].item()),
                "signed_g_norm": float(signed_norm_cpu[pair_index].item()),
                "positive_violation_raw": float(max(signed_raw_cpu[pair_index].item(), 0.0)),
                "positive_violation_norm": float(max(signed_norm_cpu[pair_index].item(), 0.0)),
                "branch_pressure_value": ""
                if pressure_cpu is None or pair_index >= pressure_cpu.numel()
                else float(pressure_cpu[pair_index].item()),
                "soft_w_L": ""
                if weights_cpu is None or pair_index >= weights_cpu.shape[0]
                else float(weights_cpu[pair_index, 0].item()),
                "soft_w_R": ""
                if weights_cpu is None or pair_index >= weights_cpu.shape[0]
                else float(weights_cpu[pair_index, 1].item()),
                "soft_w_B": ""
                if weights_cpu is None or pair_index >= weights_cpu.shape[0]
                else float(weights_cpu[pair_index, 2].item()),
                "soft_w_A": ""
                if weights_cpu is None or pair_index >= weights_cpu.shape[0]
                else float(weights_cpu[pair_index, 3].item()),
            }
        )


def collect_boundary_dual_rows(
    rows,
    rollout_idx,
    step,
    phase,
    env,
    centers,
    boundary_duals,
    boundary_pressure_values=None,
):
    widths = env.cell_features[:, 4]
    heights = env.cell_features[:, 5]
    boundary_raw = boundary_signed_constraints(centers, widths, heights, env.bounds)
    boundary_norm = boundary_raw / env.length_scale
    duals_cpu = boundary_duals.detach().cpu()
    raw_cpu = boundary_raw.detach().cpu()
    norm_cpu = boundary_norm.detach().cpu()
    pressure_cpu = None if boundary_pressure_values is None else boundary_pressure_values.detach().cpu()
    for cell_id in range(int(boundary_duals.shape[0])):
        for side_idx in range(4):
            rows.append(
                {
                    "rollout": int(rollout_idx),
                    "step": int(step),
                    "phase": phase,
                    "cell_id": int(cell_id),
                    "side": boundary_name(side_idx),
                    "dual": float(duals_cpu[cell_id, side_idx].item()),
                    "signed_g_raw": float(raw_cpu[cell_id, side_idx].item()),
                    "signed_g_norm": float(norm_cpu[cell_id, side_idx].item()),
                    "positive_violation_raw": float(max(raw_cpu[cell_id, side_idx].item(), 0.0)),
                    "positive_violation_norm": float(max(norm_cpu[cell_id, side_idx].item(), 0.0)),
                    "pressure_value": ""
                    if pressure_cpu is None
                    else float(pressure_cpu[cell_id, side_idx].item()),
                }
            )


def collect_density_dual_rows(
    rows,
    rollout_idx,
    step,
    phase,
    env,
    centers,
    density_duals,
    density_pressure_values=None,
):
    density_g, _assignment = env._density_constraints(centers)
    if density_duals.numel() == 0:
        return
    duals_cpu = density_duals.detach().cpu()
    density_cpu = density_g.detach().cpu()
    pressure_cpu = None if density_pressure_values is None else density_pressure_values.detach().cpu()
    for bin_id in range(int(density_duals.numel())):
        rows.append(
            {
                "rollout": int(rollout_idx),
                "step": int(step),
                "phase": phase,
                "bin_id": int(bin_id),
                "dual": float(duals_cpu[bin_id].item()),
                "signed_g": float(density_cpu[bin_id].item()),
                "positive_violation": float(max(density_cpu[bin_id].item(), 0.0)),
                "pressure_value": ""
                if pressure_cpu is None or bin_id >= pressure_cpu.numel()
                else float(pressure_cpu[bin_id].item()),
            }
        )


def append_phr_trace_rows(rows, coordinate_rows, rollout_idx, step, phr_steps):
    for record in phr_steps:
        rows.append(
            {
                "rollout": int(rollout_idx),
                "step": int(step),
                "inner_step": int(record["inner_step"]),
                "lagrangian": float(record["lagrangian"]),
                "wirelength": float(record["wirelength"]),
                "branch_violation": float(record["branch_violation"]),
                "boundary_violation": float(record["boundary_violation"]),
                "density_overflow": float(record["density_overflow"]),
                "grad_norm": float(record["grad_norm"]),
                "grad_norm_clipped": float(record["grad_norm_clipped"]),
                "delta_mean": float(record["delta_mean"]),
                "delta_max": float(record["delta_max"]),
            }
        )
        positions = record["positions"]
        for cell_id in range(int(positions.shape[0])):
            coordinate_rows.append(
                {
                    "rollout": int(rollout_idx),
                    "step": int(step),
                    "inner_step": int(record["inner_step"]),
                    "cell_id": int(cell_id),
                    "x": float(positions[cell_id, 0].detach().item()),
                    "y": float(positions[cell_id, 1].detach().item()),
                }
            )


def trace_single_rollout(
    *,
    case_meta,
    rollout_idx,
    policy_seed,
    policy,
    cell_features,
    pin_features,
    edge_list,
    env_config,
    temperature,
    rollout_memory_cfg,
    args,
    output_dir,
):
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(policy_seed))
    env = PlacementOrderingEnv(cell_features, pin_features, edge_list, env_config)
    memory = policy.initial_memory(cell_features.device)
    initial_memory = memory.detach().clone()

    step_rows = []
    coordinate_rows = []
    overlap_rows = []
    phr_step_rows = []
    phr_coordinate_rows = []
    active_pair_rows = []
    boundary_dual_rows = []
    density_dual_rows = []
    ordering_score_rows = []
    cluster_logit_rows = []
    pair_branch_rows = []
    policy_trace_rows = []
    memory_summary_rows = []
    memory_vector_rows = []
    memory_trace_rows = []
    pair_lifecycle_rows = []
    full_pair_matrix_rows = []
    frame_records = []
    pair_history = {
        tuple(pair): {"last_active_step": -1}
        for pair in env.active_pairs.detach().cpu().tolist()
    }

    best_score, best_centers = env.best_candidate()
    best_source_step = -1
    std_mask = cell_features[:, 0] <= 3.0 + 1e-6
    dump_full_pair_matrix = bool(args.full_pair_matrix) and int(cell_features.shape[0]) <= int(args.full_pair_matrix_threshold)

    for step in range(env_config.horizon):
        pre_memory = memory.detach().clone()
        pre_centers = env.centers.detach().clone()
        pre_active_pairs = env.active_pairs.detach().clone()
        graph = env.graph_state(memory=memory)
        with torch.no_grad():
            action = policy.sample_action(
                graph,
                temperature=temperature,
                deterministic=bool(args.deterministic),
            )
            soft_weights = None
            if env_config.soft_relaxation:
                soft_weights = hierarchical_active_branch_weights(
                    action,
                    graph["active_pairs"],
                    relaxation=args.relaxation,
                    tau=float(action.tau.detach().item()),
                ).detach()

        append_ordering_score_rows(ordering_score_rows, rollout_idx, step, action)
        append_cluster_logit_rows(cluster_logit_rows, rollout_idx, step, action, std_mask=std_mask)
        append_pair_branch_rows(pair_branch_rows, rollout_idx, step, graph["active_pairs"], action, soft_weights)

        branches = env._induce_action_branches(action)
        base_centers = compute_base_centers(env, action)

        pre_snapshot = stage_snapshot(
            cell_features,
            pin_features,
            edge_list,
            pre_centers,
            active_pairs=pre_active_pairs,
            branches=branches,
        )
        base_snapshot = stage_snapshot(
            cell_features,
            pin_features,
            edge_list,
            base_centers,
            active_pairs=pre_active_pairs,
            branches=branches,
        )

        reward, done, info = env.step_action(
            action,
            entropy=action.entropy,
            soft_branch_weights=soft_weights,
            soft_tau=float(action.tau.detach().item()),
            trace_transition=True,
        )
        memory, memory_reset_info = apply_rollout_memory_policy(
            action.next_memory,
            initial_memory,
            reset_mode=rollout_memory_cfg["memory_reset_mode"],
            reset_retain=rollout_memory_cfg["memory_reset_retain"],
            incumbent_improved=bool(info.get("incumbent_improved", False)),
            best_overlap_delta=float(info.get("best_overlap_delta", 0.0)),
            best_pair_delta_count=float(info.get("best_pair_delta_count", 0.0)),
            steps_since_best_before=int(info.get("steps_since_best_before", 0)),
            phase_transition=bool(info.get("phase_transition", False)),
            phase_transition_reason=str(info.get("phase_transition_reason", "")),
            phase_reset_retain=float(info.get("phase_reset_retain", rollout_memory_cfg["memory_reset_retain"])),
            event_reset=bool(info.get("refine_rejected", False) or info.get("rollback_to_incumbent", False)),
            event_reset_reason=(
                str(info.get("rollback_reason", "rollback_to_incumbent"))
                if bool(info.get("rollback_to_incumbent", False))
                else str(info.get("refine_reject_reason", "refine_rejected"))
            ),
            event_reset_retain=(
                0.10
                if bool(info.get("rollback_to_incumbent", False))
                else (0.25 if bool(info.get("refine_rejected", False)) else None)
            ),
            min_overlap_gain=rollout_memory_cfg["memory_reset_min_overlap_gain"],
            min_pair_gain_count=rollout_memory_cfg["memory_reset_min_pair_gain_count"],
            min_steps_since_best=rollout_memory_cfg["memory_reset_min_steps_since_best"],
        )
        append_memory_rows(
            memory_summary_rows,
            memory_vector_rows,
            memory_trace_rows,
            rollout_idx,
            step,
            pre_memory,
            memory.detach(),
        )
        info = {
            **info,
            **memory_reset_info,
        }
        transition = env.last_transition_trace or {}
        post_centers = env.centers.detach().clone()
        post_active_pairs = env.active_pairs.detach().clone()
        post_branches = transition.get("post_audit_branches")
        post_snapshot = stage_snapshot(
            cell_features,
            pin_features,
            edge_list,
            post_centers,
            active_pairs=post_active_pairs,
            branches=post_branches,
        )

        prior_best_key = candidate_key(best_score)
        best_score, best_centers = env.best_candidate()
        best_snapshot = stage_snapshot(cell_features, pin_features, edge_list, best_centers, active_pairs=None, branches=None)
        if candidate_key(best_score) != prior_best_key:
            best_source_step = step

        append_policy_trace_row(policy_trace_rows, rollout_idx, step, action, info)
        pre_branch_lookup = branch_lookup_from_pairs(pre_active_pairs, branches)
        post_branch_lookup = branch_lookup_from_pairs(post_active_pairs, post_branches)
        post_age_lookup = age_lookup_from_pairs(
            transition.get("post_audit_active_pairs", torch.empty((0, 2), dtype=torch.long, device=cell_features.device)),
            transition.get("post_audit_active_pair_ages", torch.empty(0, dtype=torch.long, device=cell_features.device)),
        )
        post_overlap_pairs = {tuple((row["i"], row["j"])) for row in post_snapshot["pairs"]}
        append_pair_lifecycle_rows(
            pair_lifecycle_rows,
            pair_history,
            rollout_idx,
            step,
            pre_active_pairs,
            post_active_pairs,
            pre_branch_lookup,
            post_branch_lookup,
            post_age_lookup,
            post_overlap_pairs,
        )

        if transition:
            append_phr_trace_rows(
                phr_step_rows,
                phr_coordinate_rows,
                rollout_idx,
                step,
                transition.get("phr_steps", []),
            )
            collect_active_pair_dual_rows(
                active_pair_rows,
                rollout_idx,
                step,
                "pre",
                cell_features,
                env.length_scale,
                transition["pre_centers"],
                transition["pre_active_pairs"],
                transition["branches_pre"],
                transition["pre_branch_duals"],
                ages=transition["pre_active_pair_ages"],
                soft_weights=transition.get("soft_branch_weights_pre"),
                branch_pressure_values=transition.get("branch_pressure_values"),
            )
            collect_active_pair_dual_rows(
                active_pair_rows,
                rollout_idx,
                step,
                "post_update",
                cell_features,
                env.length_scale,
                transition["post_centers"],
                transition["pre_active_pairs"],
                transition["branches_pre"],
                transition["post_update_branch_duals"],
                ages=transition["pre_active_pair_ages"],
                soft_weights=transition.get("soft_branch_weights_pre"),
                branch_pressure_values=transition.get("branch_pressure_values"),
            )
            post_audit_pairs = transition["post_audit_active_pairs"]
            post_audit_soft_weights = None
            if env_config.soft_relaxation and post_audit_pairs.numel() > 0:
                post_audit_soft_weights = hierarchical_active_branch_weights(
                    action,
                    post_audit_pairs,
                    relaxation=args.relaxation,
                    tau=float(action.tau.detach().item()),
                ).detach()
            collect_active_pair_dual_rows(
                active_pair_rows,
                rollout_idx,
                step,
                "post_audit",
                cell_features,
                env.length_scale,
                transition["post_centers"],
                post_audit_pairs,
                transition["post_audit_branches"],
                transition["post_audit_branch_duals"],
                ages=transition["post_audit_active_pair_ages"],
                soft_weights=post_audit_soft_weights,
                branch_pressure_values=None,
            )

            collect_boundary_dual_rows(
                boundary_dual_rows,
                rollout_idx,
                step,
                "pre",
                env,
                transition["pre_centers"],
                transition["pre_boundary_duals"],
                boundary_pressure_values=transition.get("boundary_pressure_values"),
            )
            collect_boundary_dual_rows(
                boundary_dual_rows,
                rollout_idx,
                step,
                "post_update",
                env,
                transition["post_centers"],
                transition["post_update_boundary_duals"],
                boundary_pressure_values=transition.get("boundary_pressure_values"),
            )
            collect_boundary_dual_rows(
                boundary_dual_rows,
                rollout_idx,
                step,
                "post_audit",
                env,
                transition["post_centers"],
                transition["post_audit_boundary_duals"],
                boundary_pressure_values=None,
            )

            collect_density_dual_rows(
                density_dual_rows,
                rollout_idx,
                step,
                "pre",
                env,
                transition["pre_centers"],
                transition["pre_density_duals"],
                density_pressure_values=transition.get("density_pressure_values"),
            )
            collect_density_dual_rows(
                density_dual_rows,
                rollout_idx,
                step,
                "post_update",
                env,
                transition["post_centers"],
                transition["post_update_density_duals"],
                density_pressure_values=transition.get("density_pressure_values"),
            )
            collect_density_dual_rows(
                density_dual_rows,
                rollout_idx,
                step,
                "post_audit",
                env,
                transition["post_centers"],
                transition["post_audit_density_duals"],
                density_pressure_values=None,
            )

        if dump_full_pair_matrix:
            append_full_pair_matrix_rows(
                full_pair_matrix_rows,
                rollout_idx,
                step,
                "pre",
                cell_features,
                pre_centers,
                pre_active_pairs,
                branches,
                action=action,
            )
            append_full_pair_matrix_rows(
                full_pair_matrix_rows,
                rollout_idx,
                step,
                "base",
                cell_features,
                base_centers,
                pre_active_pairs,
                branches,
                action=action,
            )
            append_full_pair_matrix_rows(
                full_pair_matrix_rows,
                rollout_idx,
                step,
                "post",
                cell_features,
                post_centers,
                post_active_pairs,
                post_branches,
                action=action,
            )
            append_full_pair_matrix_rows(
                full_pair_matrix_rows,
                rollout_idx,
                step,
                "best",
                cell_features,
                best_centers,
                None,
                None,
                action=None,
            )

        frame_record = {
            "rollout": int(rollout_idx),
            "step": int(step),
            "info": info,
            "best_source_step": best_source_step,
            "pre": {
                "centers": pre_centers,
                "snapshot": pre_snapshot,
                "active_pairs_count": int(pre_active_pairs.shape[0]),
            },
            "base": {
                "centers": base_centers,
                "snapshot": base_snapshot,
                "active_pairs_count": int(pre_active_pairs.shape[0]),
            },
            "post": {
                "centers": post_centers,
                "snapshot": post_snapshot,
                "active_pairs_count": int(post_active_pairs.shape[0]),
            },
            "best": {"centers": best_centers, "snapshot": best_snapshot, "active_pairs_count": ""},
        }
        frame_records.append(frame_record)

        step_row = {
            "rollout": int(rollout_idx),
            "step": int(step),
            "phase": str(info.get("phase", "")),
            "phase_before": str(info.get("phase_before", "")),
            "phase_request": str(info.get("phase_request", "")),
            "phase_transition": int(bool(info.get("phase_transition", False))),
            "phase_transition_reason": str(info.get("phase_transition_reason", "")),
            "phase_step": int(info.get("phase_step", 0)),
            "unlock_window_size": int(info.get("unlock_window_size", 0)),
            "phase_entry_best_overlap": float(info.get("phase_entry_best_overlap", 0.0)),
            "phase_entry_best_wirelength": float(info.get("phase_entry_best_wirelength", 0.0)),
            "best_source_step": int(best_source_step),
            "best_updated": int(candidate_key(best_score) != prior_best_key),
            "reward": float(reward),
            "lag_before": float(info["lag_before"]),
            "lag_after": float(info["lag_after"]),
            "active_pairs_before": int(pre_active_pairs.shape[0]),
            "active_pairs_after": int(post_active_pairs.shape[0]),
            "missed_pairs": float(info.get("missed_pairs", 0.0)),
            "inactive_missed_pairs": float(info.get("inactive_missed_pairs", 0.0)),
            "exact_overlap_pairs_after": float(info.get("exact_overlap_pairs", 0.0)),
            "audit_pressure_scale": float(info.get("audit_pressure_scale", 1.0)),
            "audit_pressure_target": float(info.get("audit_pressure_target", 0.0)),
            "step_scale": float(info["step_scale"]),
            "pair_emphasis": float(info["pair_emphasis"]),
            "tau": float(info["tau"]),
            "rho": float(info["rho"]),
            "eta": float(info["eta"]),
            "alpha": float(info["alpha"]),
            "pd_steps": int(info["pd_steps"]),
            "residual_norm": float(info["residual_norm"]),
            "branch_violation": float(info["branch_violation"]),
            "boundary_violation": float(info["boundary_violation"]),
            "density_overflow": float(info["density_overflow"]),
            "memory_reset_applied": int(bool(info.get("memory_reset_applied", False))),
            "memory_reset_retain": float(info.get("memory_reset_retain", 1.0)),
        }
        for stage_name, snapshot in [("pre", pre_snapshot), ("base", base_snapshot), ("post", post_snapshot), ("best", best_snapshot)]:
            score = snapshot["score"]
            step_row[f"{stage_name}_overlap_ratio"] = float(score["overlap_ratio"])
            step_row[f"{stage_name}_overlap_cells"] = int(score["overlap_cells"])
            step_row[f"{stage_name}_num_overlap_pairs"] = int(score["num_overlap_pairs"])
            step_row[f"{stage_name}_normalized_wl"] = float(score["normalized_wl"])
        step_rows.append(step_row)

        for stage_name, centers_tensor, snapshot in [
            ("pre", pre_centers, pre_snapshot),
            ("base", base_centers, base_snapshot),
            ("post", post_centers, post_snapshot),
            ("best", best_centers, best_snapshot),
        ]:
            overlap_cells = snapshot["overlap_cells"]
            for cell_idx in range(int(cell_features.shape[0])):
                coordinate_rows.append(
                    {
                        "rollout": int(rollout_idx),
                        "step": int(step),
                        "stage": stage_name,
                        "cell_id": int(cell_idx),
                        "x": float(centers_tensor[cell_idx, 0].detach().item()),
                        "y": float(centers_tensor[cell_idx, 1].detach().item()),
                        "width": float(cell_features[cell_idx, 4].detach().item()),
                        "height": float(cell_features[cell_idx, 5].detach().item()),
                        "area": float(cell_features[cell_idx, 0].detach().item()),
                        "num_pins": float(cell_features[cell_idx, 1].detach().item()),
                        "overlap_flag": int(cell_idx in overlap_cells),
                    }
                )
            pair_limit = None if int(args.max_overlap_pairs_per_stage) < 0 else int(args.max_overlap_pairs_per_stage)
            selected_pairs = snapshot["pairs"] if pair_limit is None else snapshot["pairs"][:pair_limit]
            for pair_row in selected_pairs:
                overlap_rows.append(
                    {
                        "rollout": int(rollout_idx),
                        "step": int(step),
                        "stage": stage_name,
                        **pair_row,
                    }
                )

        save_frame(
            frames_dir / f"step_{step:03d}.png",
            cell_features,
            frame_record,
            arrow_limit=args.arrow_count,
            dpi=args.dpi,
        )
        if done:
            break

    step_fieldnames = list(step_rows[0].keys()) if step_rows else ["rollout", "step"]
    coordinate_fieldnames = list(coordinate_rows[0].keys()) if coordinate_rows else [
        "rollout",
        "step",
        "stage",
        "cell_id",
        "x",
        "y",
        "width",
        "height",
        "area",
        "num_pins",
        "overlap_flag",
    ]
    overlap_fieldnames = list(overlap_rows[0].keys()) if overlap_rows else [
        "rollout",
        "step",
        "stage",
        "i",
        "j",
        "ox",
        "oy",
        "area",
        "active_pair",
        "branch",
    ]

    write_csv(output_dir / "steps.csv", step_rows, step_fieldnames)
    write_csv(output_dir / "coordinates.csv", coordinate_rows, coordinate_fieldnames)
    write_csv(output_dir / "overlap_pairs.csv", overlap_rows, overlap_fieldnames)
    write_csv(output_dir / "phr_steps.csv", phr_step_rows, PHR_STEP_FIELDNAMES)
    write_csv(output_dir / "phr_coordinates.csv", phr_coordinate_rows, PHR_COORDINATE_FIELDNAMES)
    write_csv(output_dir / "active_pair_duals.csv", active_pair_rows, ACTIVE_PAIR_FIELDNAMES)
    write_csv(output_dir / "boundary_duals.csv", boundary_dual_rows, BOUNDARY_DUAL_FIELDNAMES)
    write_csv(output_dir / "density_duals.csv", density_dual_rows, DENSITY_DUAL_FIELDNAMES)
    write_csv(output_dir / "ordering_scores.csv", ordering_score_rows, ORDERING_SCORE_FIELDNAMES)
    write_csv(output_dir / "cluster_logits.csv", cluster_logit_rows, CLUSTER_LOGIT_FIELDNAMES)
    write_csv(output_dir / "pair_branch_logits.csv", pair_branch_rows, PAIR_BRANCH_LOGIT_FIELDNAMES)
    write_csv(output_dir / "memory_summary.csv", memory_summary_rows, MEMORY_SUMMARY_FIELDNAMES)
    write_csv(output_dir / "memory_vectors.csv", memory_vector_rows, MEMORY_VECTOR_FIELDNAMES)
    write_csv(output_dir / "pair_lifecycle.csv", pair_lifecycle_rows, PAIR_LIFECYCLE_FIELDNAMES)
    if dump_full_pair_matrix:
        write_csv(output_dir / "pair_matrix.csv", full_pair_matrix_rows, FULL_PAIR_MATRIX_FIELDNAMES)
    write_jsonl(output_dir / "steps.jsonl", step_rows)
    write_jsonl(output_dir / "policy_trace.jsonl", policy_trace_rows)
    write_jsonl(output_dir / "memory_trace.jsonl", memory_trace_rows)
    save_timeline_plot(output_dir / "timeline.png", step_rows, dpi=args.dpi)

    summary = {
        "case_label": str(case_meta["label"]),
        "rollout": int(rollout_idx),
        "policy_seed": int(policy_seed),
        "steps_recorded": len(step_rows),
        "best_source_step": int(best_source_step),
        "best_final_score": frame_records[-1]["best"]["snapshot"]["score"] if frame_records else None,
        "output_files": {
            "steps_csv": str(output_dir / "steps.csv"),
            "steps_jsonl": str(output_dir / "steps.jsonl"),
            "coordinates_csv": str(output_dir / "coordinates.csv"),
            "overlap_pairs_csv": str(output_dir / "overlap_pairs.csv"),
            "phr_steps_csv": str(output_dir / "phr_steps.csv"),
            "phr_coordinates_csv": str(output_dir / "phr_coordinates.csv"),
            "active_pair_duals_csv": str(output_dir / "active_pair_duals.csv"),
            "boundary_duals_csv": str(output_dir / "boundary_duals.csv"),
            "density_duals_csv": str(output_dir / "density_duals.csv"),
            "ordering_scores_csv": str(output_dir / "ordering_scores.csv"),
            "cluster_logits_csv": str(output_dir / "cluster_logits.csv"),
            "pair_branch_logits_csv": str(output_dir / "pair_branch_logits.csv"),
            "memory_summary_csv": str(output_dir / "memory_summary.csv"),
            "memory_vectors_csv": str(output_dir / "memory_vectors.csv"),
            "pair_lifecycle_csv": str(output_dir / "pair_lifecycle.csv"),
            "policy_trace_jsonl": str(output_dir / "policy_trace.jsonl"),
            "memory_trace_jsonl": str(output_dir / "memory_trace.jsonl"),
            "timeline_png": str(output_dir / "timeline.png"),
            "frames_dir": str(frames_dir),
        },
    }
    if dump_full_pair_matrix:
        summary["output_files"]["pair_matrix_csv"] = str(output_dir / "pair_matrix.csv")
    (output_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=default_device_arg())
    parser.add_argument("--case", default="1")
    parser.add_argument("--size", default=None)
    parser.add_argument("--seed", type=int, default=1001234)
    parser.add_argument("--batch-seeds", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--soft-relaxation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--residual-flow", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--phr-layer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exact-audit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--density", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--clusters", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fixed-pd-controls", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ordering-representation", choices=["sequence_pair", "dag"], default=None)
    parser.add_argument("--branch-mode", choices=["ordering", "independent_pair"], default=None)
    parser.add_argument("--al-mode", choices=["signed_phr", "positive_only"], default=None)
    parser.add_argument("--relaxation", choices=["sigmoid", "neuralsort", "gumbel_sinkhorn"], default="sigmoid")
    parser.add_argument("--max-overlap-pairs-per-stage", type=int, default=128)
    parser.add_argument("--arrow-count", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--policy-seed-base", type=int, default=314159)
    parser.add_argument("--full-pair-matrix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-pair-matrix-threshold", type=int, default=48)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, checkpoint = load_policy_checkpoint(args.checkpoint, device)
    rollout_memory_cfg = rollout_memory_config_from_checkpoint(checkpoint)
    env_config = env_config_from_checkpoint(checkpoint, args)
    temperature = float(
        _coalesce(
            args.temperature,
            checkpoint.get("stats", {}).get("temperature", checkpoint.get("config", {}).get("temperature_start", 0.35) or 0.35),
        )
    )
    case_specs = build_case_specs(args)
    if not case_specs:
        raise ValueError("No trace cases resolved from the provided arguments.")

    if args.output_dir is None:
        checkpoint_name = Path(args.checkpoint).stem
        if len(case_specs) == 1:
            output_root = Path("/Users/omarramadan/Desktop/compression/intern_challenge_PPO/traces") / f"{case_specs[0]['label']}_{checkpoint_name}"
        else:
            output_root = Path("/Users/omarramadan/Desktop/compression/intern_challenge_PPO/traces") / f"batch_{checkpoint_name}"
    else:
        output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    case_summaries = []
    all_candidate_rows = []
    for case_idx, case_spec in enumerate(case_specs):
        case_meta, cell_features, pin_features, edge_list = build_case_from_spec(case_spec, device)
        if len(case_specs) == 1:
            case_output_root = output_root
        else:
            case_output_root = output_root / f"case_{case_idx:03d}_{sanitize_label(case_meta['label'])}"
        case_output_root.mkdir(parents=True, exist_ok=True)

        rollout_summaries = []
        for rollout_idx in range(max(int(args.num_rollouts), 1)):
            policy_seed = int(args.policy_seed_base) + int(rollout_idx)
            if int(args.num_rollouts) == 1:
                rollout_dir = case_output_root
            else:
                rollout_dir = case_output_root / f"rollout_{rollout_idx:03d}"
            rollout_summary = trace_single_rollout(
                case_meta=case_meta,
                rollout_idx=rollout_idx,
                policy_seed=policy_seed,
                policy=policy,
                cell_features=cell_features.clone(),
                pin_features=pin_features.clone(),
                edge_list=edge_list.clone(),
                env_config=env_config,
                temperature=temperature,
                rollout_memory_cfg=rollout_memory_cfg,
                args=args,
                output_dir=rollout_dir,
            )
            rollout_summaries.append(rollout_summary)

        case_summary = {
            "case": case_meta,
            "rollouts": rollout_summaries,
        }
        if rollout_summaries:
            best_rollout = min(rollout_summaries, key=lambda item: candidate_key(item["best_final_score"]))
            case_summary["best_rollout"] = {
                "rollout": int(best_rollout["rollout"]),
                "policy_seed": int(best_rollout["policy_seed"]),
                "best_final_score": best_rollout["best_final_score"],
                "best_source_step": int(best_rollout["best_source_step"]),
            }
            ensemble_rows = []
            for item in rollout_summaries:
                score = item["best_final_score"]
                row = {
                    "case_label": str(case_meta["label"]),
                    "rollout": int(item["rollout"]),
                    "policy_seed": int(item["policy_seed"]),
                    "steps_recorded": int(item["steps_recorded"]),
                    "best_source_step": int(item["best_source_step"]),
                    "best_overlap_ratio": float(score["overlap_ratio"]),
                    "best_overlap_cells": int(score["overlap_cells"]),
                    "best_num_overlap_pairs": int(score["num_overlap_pairs"]),
                    "best_normalized_wl": float(score["normalized_wl"]),
                }
                ensemble_rows.append(row)
                all_candidate_rows.append(
                    {
                        "case_label": str(case_meta["label"]),
                        "case_source": str(case_meta["source"]),
                        **{key: value for key, value in row.items() if key != "case_label"},
                    }
                )
            write_csv(case_output_root / "ensemble_candidates.csv", ensemble_rows, ENSEMBLE_CANDIDATE_FIELDNAMES)
            case_summary["ensemble_candidates_csv"] = str(case_output_root / "ensemble_candidates.csv")
        case_summary_path = case_output_root / (
            "trace_summary.json"
            if len(rollout_summaries) == 1
            else "ensemble_summary.json"
        )
        case_summary_path.write_text(json.dumps(case_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        case_summaries.append(case_summary)

    summary = {
        "checkpoint": args.checkpoint,
        "env_config": env_config.__dict__,
        "temperature": temperature,
        "deterministic": bool(args.deterministic),
        "num_rollouts": int(args.num_rollouts),
        "policy_seed_base": int(args.policy_seed_base),
        "num_cases": int(len(case_summaries)),
        "cases": case_summaries,
    }

    if case_summaries:
        best_case_summary = min(
            (case_summary for case_summary in case_summaries if "best_rollout" in case_summary),
            key=lambda case_summary: candidate_key(case_summary["best_rollout"]["best_final_score"]),
        )
        summary["best_case"] = {
            "case": best_case_summary["case"],
            "best_rollout": best_case_summary["best_rollout"],
        }
    if all_candidate_rows:
        write_csv(output_root / "case_rollout_candidates.csv", all_candidate_rows, CASE_CANDIDATE_FIELDNAMES)
        summary["case_rollout_candidates_csv"] = str(output_root / "case_rollout_candidates.csv")

    if len(case_summaries) == 1 and int(args.num_rollouts) == 1:
        summary_path = output_root / "trace_summary.json"
    elif len(case_summaries) == 1:
        summary_path = output_root / "ensemble_summary.json"
    else:
        summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
