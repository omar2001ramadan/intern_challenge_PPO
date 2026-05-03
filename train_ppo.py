"""Train the sequence-pair ordering policy with PPO."""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from env import EnvConfig, PlacementOrderingEnv
from ordering_policy import (
    DISCOVER_MODE_NAMES,
    DISCOVER_MODE_TO_INDEX,
    OrderingPolicy,
    REFINE_VARIANT_NAMES,
    REFINE_VARIANT_TO_INDEX,
    apply_rollout_memory_policy,
    hierarchical_active_branch_weights,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from placement import generate_placement_input
from ppo import Transition, detach_action, detach_graph, ppo_update


def default_device_arg():
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def parse_sizes(value):
    sizes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        macros, std_cells = item.split(":")
        sizes.append((int(macros), int(std_cells)))
    if not sizes:
        raise ValueError("At least one training size is required.")
    return sizes


def initialize_random_spread(cell_features, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6
    angles = torch.rand(total_cells, device=cell_features.device) * 2 * 3.14159
    radii = torch.rand(total_cells, device=cell_features.device) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)
    return cell_features


def initialize_random_spread_for_mode(cell_features, discover_mode="balanced", seed=None):
    discover_mode = str(discover_mode).lower()
    if discover_mode == "balanced":
        return initialize_random_spread(cell_features, seed=seed)
    if seed is not None:
        torch.manual_seed(seed)
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    length_scale = total_area ** 0.5
    macro_mask = cell_features[:, 0] > 3.0 + 1e-6
    std_mask = ~macro_mask
    centers = torch.zeros((total_cells, 2), dtype=cell_features.dtype, device=cell_features.device)

    def place_ring(mask, radius_min, radius_max, angle_offset=0.0):
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return
        count = int(idx.numel())
        if count == 1:
            angles = torch.tensor([angle_offset], dtype=cell_features.dtype, device=cell_features.device)
        else:
            angles = torch.linspace(0.0, 2.0 * 3.14159, steps=count + 1, device=cell_features.device, dtype=cell_features.dtype)[:-1]
            angles = angles + angle_offset
        angle_noise = (torch.rand(count, device=cell_features.device, dtype=cell_features.dtype) - 0.5) * 0.35
        radii = radius_min + (radius_max - radius_min) * torch.rand(count, device=cell_features.device, dtype=cell_features.dtype)
        centers[idx, 0] = radii * torch.cos(angles + angle_noise)
        centers[idx, 1] = radii * torch.sin(angles + angle_noise)

    def place_disk(mask, radius_scale):
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return
        count = int(idx.numel())
        angles = torch.rand(count, device=cell_features.device, dtype=cell_features.dtype) * 2.0 * 3.14159
        radii = torch.sqrt(torch.rand(count, device=cell_features.device, dtype=cell_features.dtype)) * radius_scale
        centers[idx, 0] = radii * torch.cos(angles)
        centers[idx, 1] = radii * torch.sin(angles)

    if discover_mode == "spread_first":
        place_ring(macro_mask, 0.30 * length_scale, 0.55 * length_scale, angle_offset=0.2)
        place_ring(std_mask, 0.75 * length_scale, 1.20 * length_scale, angle_offset=0.6)
    elif discover_mode == "wire_first":
        place_ring(macro_mask, 0.18 * length_scale, 0.32 * length_scale, angle_offset=0.1)
        place_disk(std_mask, 0.22 * length_scale)
    elif discover_mode == "macro_clearance":
        place_ring(macro_mask, 0.75 * length_scale, 1.05 * length_scale, angle_offset=0.0)
        place_ring(std_mask, 0.35 * length_scale, 0.70 * length_scale, angle_offset=0.4)
    else:
        spread_radius = 0.75 * length_scale
        angles = torch.rand(total_cells, device=cell_features.device) * 2 * 3.14159
        radii = torch.rand(total_cells, device=cell_features.device) * spread_radius
        centers[:, 0] = radii * torch.cos(angles)
        centers[:, 1] = radii * torch.sin(angles)
    cell_features[:, 2:4] = centers
    return cell_features


def linear_schedule(update_idx, total_updates, start, end):
    if total_updates <= 1:
        return end
    mix = min(max(update_idx / (total_updates - 1), 0.0), 1.0)
    return start * (1.0 - mix) + end * mix


def temperature_at(update_idx, total_updates, start=1.4, end=0.55):
    return linear_schedule(update_idx, total_updates, start, end)


def soft_tau_at(update_idx, total_updates, start=2.0, end=0.10):
    return linear_schedule(update_idx, total_updates, start, end)


def mean_numeric_dicts(rows):
    if not rows:
        return {}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, bool))})
    return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}


def lexicographic_candidate_key(row):
    return (
        float(row["candidate_overlap"]),
        int(row["candidate_pairs"]),
        float(row["candidate_wirelength"]),
    )


def candidate_generation_index(row, fallback=0):
    return int(row.get("candidate_generation_index", fallback))


def lexicographic_candidate_generation_key(row):
    return (*lexicographic_candidate_key(row), candidate_generation_index(row))


def candidate_source_name(row):
    return str(row.get("candidate_source") or row.get("source") or "")


def is_swap_or_reassign_candidate(row):
    source = candidate_source_name(row)
    variant = str(row.get("variant_name", ""))
    return source == "refine_variant:swap_or_reassign_local" or variant == "swap_or_reassign_local"


def chooser_pairwise_weight(
    better_row,
    worse_row,
    *,
    overlap_weight=8.0,
    pair_weight=4.0,
    wire_scale=8.0,
    wire_cap=3.0,
    swap_legality_multiplier=2.0,
    rollout_teacher_multiplier=1.5,
):
    better_overlap, better_pairs, better_wire = lexicographic_candidate_key(better_row)
    worse_overlap, worse_pairs, worse_wire = lexicographic_candidate_key(worse_row)
    overlap_gap = float(worse_overlap) - float(better_overlap)
    if overlap_gap > 0.0:
        weight = float(overlap_weight) * max(overlap_gap, 1.0)
        if is_swap_or_reassign_candidate(worse_row):
            weight *= float(swap_legality_multiplier)
        if candidate_source_name(better_row) == "rollout_best":
            weight *= float(rollout_teacher_multiplier)
        return weight
    pair_gap = int(worse_pairs) - int(better_pairs)
    if pair_gap > 0:
        weight = float(pair_weight) * float(pair_gap)
        if is_swap_or_reassign_candidate(worse_row):
            weight *= float(swap_legality_multiplier)
        if candidate_source_name(better_row) == "rollout_best":
            weight *= float(rollout_teacher_multiplier)
        return weight
    wire_gap = float(worse_wire) - float(better_wire)
    return 1.0 + min(max(wire_gap, 0.0) * float(wire_scale), float(wire_cap))


def continuation_margin_regressed(
    best_row,
    final_row,
    *,
    overlap_margin=0.03,
    pair_margin=2,
    wire_margin=0.08,
):
    overlap_gap = float(final_row["candidate_overlap"]) - float(best_row["candidate_overlap"])
    if overlap_gap >= float(overlap_margin):
        return True
    if abs(overlap_gap) <= float(overlap_margin):
        pair_gap = int(final_row["candidate_pairs"]) - int(best_row["candidate_pairs"])
        if pair_gap >= int(pair_margin):
            return True
        if abs(pair_gap) < int(pair_margin):
            wire_gap = float(final_row["candidate_wirelength"]) - float(best_row["candidate_wirelength"])
            if wire_gap >= float(wire_margin):
                return True
    return False


def continuation_supervision_eligible(
    row,
    *,
    required_phase="REFINE",
    max_overlap=0.10,
    max_pairs=4,
):
    if str(row.get("best_candidate_phase", "")).upper() != str(required_phase).upper():
        return False
    if float(row.get("best_overlap", 1.0)) > float(max_overlap):
        return False
    if int(row.get("best_exact_overlap_pairs", 10**9)) > int(max_pairs):
        return False
    return True


def enrich_refine_supervision_rows(variant_rows):
    enriched = []
    sorted_rows = sorted(
        [dict(row) for row in variant_rows],
        key=lambda row: (
            float(row["overlap_ratio"]),
            int(row["num_overlap_pairs"]),
            float(row["normalized_wl"]),
        ),
    )
    rank_lookup = {str(row["variant_name"]): rank for rank, row in enumerate(sorted_rows)}
    for row in variant_rows:
        enriched.append(
            {
                "candidate_source": f"refine_variant:{row['variant_name']}",
                "variant_name": str(row["variant_name"]),
                "candidate_overlap": float(row["overlap_ratio"]),
                "candidate_pairs": int(row["num_overlap_pairs"]),
                "candidate_wirelength": float(row["normalized_wl"]),
                "candidate_repair_legal": bool(row["repair_legal"]),
                "candidate_accepted": bool(row["accepted"]),
                "candidate_lexi_rank": int(rank_lookup[str(row["variant_name"])]),
                "overlap_delta": float(row["overlap_delta"]),
                "pair_delta": int(row["pair_delta"]),
                "wire_delta": float(row["wire_delta"]),
                "operator_metadata": dict(row.get("operator_metadata", {})),
            }
        )
    return enriched


def candidate_record(
    *,
    source,
    overlap,
    pairs,
    wirelength,
    repair_legal=True,
    accepted=False,
    live_input=False,
    diagnostic_only=False,
    chooser_selected=False,
    external_selected=False,
    generation_index=None,
    variant_name="",
    origin_source="",
    operator_metadata=None,
):
    return {
        "candidate_source": str(source),
        "candidate_overlap": float(overlap),
        "candidate_pairs": int(pairs),
        "candidate_wirelength": float(wirelength),
        "candidate_repair_legal": bool(repair_legal),
        "candidate_accepted": bool(accepted),
        "candidate_live_input": bool(live_input),
        "candidate_diagnostic_only": bool(diagnostic_only),
        "candidate_chooser_selected": bool(chooser_selected),
        "candidate_external_selected": bool(external_selected),
        "candidate_generation_index": 0 if generation_index is None else int(generation_index),
        "candidate_variant_name": str(variant_name),
        "candidate_origin_source": str(origin_source or source),
        "candidate_operator_metadata": dict(operator_metadata or {}),
    }


def finalize_candidate_records(records):
    normalized = []
    for idx, row in enumerate(records):
        generation_index = int(row.get("candidate_generation_index", idx))
        normalized.append({**row, "candidate_generation_index": generation_index})
    ranked = sorted(normalized, key=lexicographic_candidate_generation_key)
    rank_lookup = {
        int(row["candidate_generation_index"]): idx
        for idx, row in enumerate(ranked)
    }
    finalized = []
    for row in normalized:
        finalized.append(
            {
                **row,
                "candidate_lexi_rank": int(rank_lookup[int(row["candidate_generation_index"])]),
            }
        )
    return finalized


def candidate_feature_tensor(record, *, device, dtype):
    source = str(record.get("candidate_source", "unknown"))
    if source.startswith("refine_variant:"):
        source_value = 0.75
    elif source == "final_selected":
        source_value = 1.00
    elif source == "rollout_best":
        source_value = 0.50
    else:
        source_value = 0.25
    return torch.tensor(
        [
            float(record.get("candidate_overlap", 0.0)),
            float(record.get("candidate_pairs", 0.0)),
            float(record.get("candidate_wirelength", 0.0)),
            1.0 if bool(record.get("candidate_repair_legal", False)) else 0.0,
            1.0 if bool(record.get("candidate_accepted", False)) else 0.0,
            float(source_value),
        ],
        dtype=dtype,
        device=device,
    )


def chooser_source_value(source):
    source = str(source)
    if source == "rollout_best":
        return 0.0
    if source.startswith("refine_variant:"):
        return 0.5
    if source == "final_selected":
        return 1.0
    return 0.25


def chooser_variant_value(record):
    variant_name = str(record.get("candidate_variant_name", ""))
    if not variant_name:
        return 0.0
    index = REFINE_VARIANT_TO_INDEX.get(variant_name)
    if index is None:
        return 0.0
    return float(index + 1) / float(max(len(REFINE_VARIANT_NAMES), 1))


def chooser_large_case_flag(graph):
    if graph is None:
        return 0.0
    cell_features = graph.get("cell_features")
    if cell_features is not None and hasattr(cell_features, "shape"):
        return 1.0 if int(cell_features.shape[0]) >= 100 else 0.0
    case_descriptor = graph.get("case_descriptor")
    if case_descriptor is None:
        return 0.0
    if torch.is_tensor(case_descriptor):
        values = case_descriptor.detach().reshape(-1)
        if values.numel() >= 2:
            return 1.0 if float(values[1].item()) >= 100.0 else 0.0
        return 0.0
    values = list(case_descriptor)
    if len(values) >= 2:
        return 1.0 if float(values[1]) >= 100.0 else 0.0
    return 0.0


def chooser_candidate_feature_tensor(record, *, rollout_best_record, large_case_flag, device, dtype):
    baseline = rollout_best_record if rollout_best_record is not None else record
    overlap_margin = float(baseline.get("candidate_overlap", 0.0)) - float(record.get("candidate_overlap", 0.0))
    pair_margin = float(baseline.get("candidate_pairs", 0.0)) - float(record.get("candidate_pairs", 0.0))
    wire_margin = float(baseline.get("candidate_wirelength", 0.0)) - float(record.get("candidate_wirelength", 0.0))
    is_rollout = 1.0 if str(record.get("candidate_source", "")) == "rollout_best" else 0.0
    is_swap = 1.0 if str(record.get("candidate_source", "")) == "refine_variant:swap_or_reassign_local" else 0.0
    large_swap_conflict = 0.0
    if (
        float(large_case_flag) > 0.0
        and is_swap > 0.0
        and wire_margin > 0.0
        and (overlap_margin < 0.0 or pair_margin < 0.0)
    ):
        large_swap_conflict = 1.0
    return torch.tensor(
        [
            float(record.get("candidate_overlap", 0.0)),
            float(record.get("candidate_pairs", 0.0)),
            float(record.get("candidate_wirelength", 0.0)),
            1.0 if bool(record.get("candidate_repair_legal", False)) else 0.0,
            1.0 if bool(record.get("candidate_accepted", False)) else 0.0,
            float(record.get("candidate_overlap", 0.0)) - float(baseline.get("candidate_overlap", 0.0)),
            float(record.get("candidate_pairs", 0.0)) - float(baseline.get("candidate_pairs", 0.0)),
            float(record.get("candidate_wirelength", 0.0)) - float(baseline.get("candidate_wirelength", 0.0)),
            chooser_source_value(record.get("candidate_source", "unknown")),
            chooser_variant_value(record),
            overlap_margin,
            pair_margin,
            wire_margin,
            large_swap_conflict,
            float(large_case_flag),
            is_rollout,
        ],
        dtype=dtype,
        device=device,
    )


def chooser_input_records(candidate_records):
    return [dict(row) for row in candidate_records if bool(row.get("candidate_live_input", False))]


def chooser_rollout_best_record(candidate_records):
    for row in candidate_records:
        if str(row.get("candidate_source", "")) == "rollout_best":
            return dict(row)
    return None


def external_candidate_teacher_winner(candidate_records):
    if not candidate_records:
        return None
    return min(candidate_records, key=lexicographic_candidate_generation_key)


def chooser_target_index(candidate_records):
    if not candidate_records:
        return None
    winner = external_candidate_teacher_winner(candidate_records)
    if winner is None:
        return None
    winner_generation_index = int(winner["candidate_generation_index"])
    for idx, row in enumerate(candidate_records):
        if int(row["candidate_generation_index"]) == winner_generation_index:
            return idx
    return None


def chooser_candidate_features(candidate_records, *, device, dtype, graph=None):
    live_records = chooser_input_records(candidate_records)
    rollout_best = chooser_rollout_best_record(live_records)
    if not live_records or rollout_best is None:
        return None
    large_case_flag = chooser_large_case_flag(graph)
    chooser_features = torch.stack(
        [
            chooser_candidate_feature_tensor(
                row,
                rollout_best_record=rollout_best,
                large_case_flag=large_case_flag,
                device=device,
                dtype=dtype,
            )
            for row in live_records
        ],
        dim=0,
    )
    legacy_features = torch.stack(
        [candidate_feature_tensor(row, device=device, dtype=dtype) for row in live_records],
        dim=0,
    )
    return {
        "records": live_records,
        "rollout_best": rollout_best,
        "chooser_features": chooser_features,
        "legacy_features": legacy_features,
        "target_index": chooser_target_index(live_records),
    }


def build_validation_suite(sizes, episodes, seed_base):
    count = max(int(episodes), 1)
    suite = []
    for episode_idx in range(count):
        suite.append(
            {
                "size": tuple(sizes[episode_idx % len(sizes)]),
                "seed": int(seed_base) + episode_idx,
            }
        )
    return suite


def compare_episode_infos(lhs, rhs):
    lhs_key = (
        float(lhs["best_overlap"]),
        float(lhs["best_exact_overlap_pairs"]),
        float(lhs["best_wl"]),
    )
    rhs_key = (
        float(rhs["best_overlap"]),
        float(rhs["best_exact_overlap_pairs"]),
        float(rhs["best_wl"]),
    )
    return lhs_key < rhs_key


def build_hard_replay_suite(
    policy,
    sizes,
    env_config,
    device,
    *,
    seed_base,
    temperature,
    soft_tau,
    relaxation,
    pool_size,
    suite_size,
    memory_reset_mode,
    memory_reset_retain,
    memory_reset_min_overlap_gain,
    memory_reset_min_pair_gain_count,
    memory_reset_min_steps_since_best,
):
    suite_size = max(min(int(suite_size), int(pool_size)), 0)
    if suite_size <= 0 or int(pool_size) <= 0:
        return []

    candidate_specs = build_validation_suite(sizes, int(pool_size), int(seed_base))
    ranked = []
    was_training = policy.training
    policy.eval()
    for spec in candidate_specs:
        forced_size = tuple(spec["size"])
        episode_seed = int(spec["seed"])
        _transitions, info = collect_episode(
            policy,
            sizes,
            env_config,
            device,
            episode_seed,
            temperature,
            soft_tau=soft_tau,
            relaxation=relaxation,
            forced_size=forced_size,
            deterministic=True,
            discover_mode=str(spec.get("discover_mode", "balanced")),
            memory_reset_mode=memory_reset_mode,
            memory_reset_retain=memory_reset_retain,
            memory_reset_min_overlap_gain=memory_reset_min_overlap_gain,
            memory_reset_min_pair_gain_count=memory_reset_min_pair_gain_count,
            memory_reset_min_steps_since_best=memory_reset_min_steps_since_best,
        )
        ranked.append(
            {
                "size": forced_size,
                "seed": episode_seed,
                "discover_mode": str(info.get("discover_mode", spec.get("discover_mode", "balanced"))),
                "best_overlap": float(info["best_overlap"]),
                "best_exact_overlap_pairs": float(info["best_exact_overlap_pairs"]),
                "best_wl": float(info["best_wl"]),
                "phase_failure_score": float(info.get("phase_failure_score", 0.0)),
            }
        )
    if was_training:
        policy.train()

    ranked.sort(
        key=lambda item: (
            -item["phase_failure_score"],
            -item["best_overlap"],
            -item["best_exact_overlap_pairs"],
            -item["best_wl"],
        )
    )
    return ranked[:suite_size]


def build_validation_replay_suite(validation_rows, suite_size):
    suite_size = max(int(suite_size), 0)
    if suite_size <= 0:
        return []
    ranked = []
    for row in validation_rows:
        ranked.append(
            {
                "size": tuple(row["size"]),
                "seed": int(row["seed"]),
                "discover_mode": str(row.get("winning_discover_mode", row.get("discover_mode", "balanced"))),
                "best_overlap": float(row["best_overlap"]),
                "best_exact_overlap_pairs": float(row["best_exact_overlap_pairs"]),
                "best_wl": float(row["best_wl"]),
                "phase_failure_score": float(row.get("phase_failure_score", 0.0)),
            }
        )
    ranked.sort(
        key=lambda item: (
            -item["phase_failure_score"],
            -item["best_overlap"],
            -item["best_exact_overlap_pairs"],
            -item["best_wl"],
        )
    )
    return ranked[:suite_size]


def metric_gated_temperature(current_tau, start=1.4, end=0.55):
    return max(float(end), min(float(start), float(current_tau)))


def optimizer_lr(optimizer):
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def set_optimizer_lr(optimizer, lr_value):
    for group in optimizer.param_groups:
        group["lr"] = float(lr_value)


def decay_optimizer_lr(optimizer, decay, min_lr):
    current_lr = optimizer_lr(optimizer)
    new_lr = max(float(min_lr), current_lr * float(decay))
    set_optimizer_lr(optimizer, new_lr)
    return current_lr, new_lr


def snapshot_training_state(policy, optimizer, hardening_state, soft_tau_state):
    return {
        "model_state": copy.deepcopy(policy.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "hardening_state": copy.deepcopy(hardening_state),
        "soft_tau_state": float(soft_tau_state),
    }


def restore_training_state(policy, optimizer, snapshot):
    policy.load_state_dict(snapshot["model_state"])
    optimizer.load_state_dict(snapshot["optimizer_state"])
    policy.train()
    return copy.deepcopy(snapshot["hardening_state"]), float(snapshot["soft_tau_state"])


def should_apply_validation_rewind(
    metric_overlap,
    metric_pairs,
    best_overlap,
    best_pairs,
    *,
    overlap_epsilon,
    pair_epsilon,
):
    if metric_overlap is None or best_overlap is None:
        return False
    if float(metric_overlap) > float(best_overlap) + float(overlap_epsilon):
        return True
    if (
        best_pairs is not None
        and metric_pairs is not None
        and abs(float(metric_overlap) - float(best_overlap)) <= float(overlap_epsilon)
        and float(metric_pairs) > float(best_pairs) + float(pair_epsilon)
    ):
        return True
    return False


def make_problem(sizes, device, seed, forced_size=None):
    return make_problem_for_mode(sizes, device, seed, forced_size=forced_size, discover_mode="balanced")


def make_problem_for_mode(sizes, device, seed, forced_size=None, discover_mode="balanced"):
    num_macros, num_std_cells = forced_size if forced_size is not None else random.choice(sizes)
    torch.manual_seed(seed)
    cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)
    cell_features = initialize_random_spread_for_mode(cell_features, discover_mode=discover_mode, seed=seed + 17)
    return cell_features.to(device), pin_features.to(device), edge_list.to(device), (num_macros, num_std_cells)


def collect_episode(
    policy,
    sizes,
    env_config,
    device,
    seed,
    temperature,
    soft_tau=None,
    relaxation="sigmoid",
    forced_size=None,
    deterministic=False,
    discover_mode="balanced",
    memory_reset_mode="none",
    memory_reset_retain=1.0,
    memory_reset_min_overlap_gain=0.03,
    memory_reset_min_pair_gain_count=2.0,
    memory_reset_min_steps_since_best=2,
    post_legal_refine_portfolio=False,
):
    cell_features, pin_features, edge_list, size = make_problem_for_mode(
        sizes,
        device,
        seed,
        forced_size=forced_size,
        discover_mode=discover_mode,
    )
    env = PlacementOrderingEnv(cell_features, pin_features, edge_list, env_config, discover_mode=discover_mode)
    transitions = []
    infos = []
    memory = policy.initial_memory(device)
    initial_memory = memory.detach().clone()
    memory_reset_count = 0
    phase_counts = {}
    phase_improvement_counts = {}
    phase_transition_counts = {}
    phase_failure_score = 0.0

    for step_idx in range(env_config.horizon):
        graph = env.graph_state(memory=memory)
        action = policy.sample_action(graph, temperature=temperature, deterministic=deterministic)
        soft_weights = None
        if soft_tau is not None:
            soft_weights = hierarchical_active_branch_weights(
                action,
                graph["active_pairs"],
                relaxation=relaxation,
                tau=soft_tau,
            ).detach()
        reward, done, info = env.step_action(
            action,
            entropy=action.entropy,
            soft_branch_weights=soft_weights,
            soft_tau=soft_tau,
        )
        memory, memory_reset_info = apply_rollout_memory_policy(
            action.next_memory,
            initial_memory,
            reset_mode=memory_reset_mode,
            reset_retain=memory_reset_retain,
            incumbent_improved=bool(info.get("incumbent_improved", False)),
            best_overlap_delta=float(info.get("best_overlap_delta", 0.0)),
            best_pair_delta_count=float(info.get("best_pair_delta_count", 0.0)),
            steps_since_best_before=int(info.get("steps_since_best_before", 0)),
            phase_transition=bool(info.get("phase_transition", False)),
            phase_transition_reason=str(info.get("phase_transition_reason", "")),
            phase_reset_retain=float(info.get("phase_reset_retain", memory_reset_retain)),
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
            min_overlap_gain=memory_reset_min_overlap_gain,
            min_pair_gain_count=memory_reset_min_pair_gain_count,
            min_steps_since_best=memory_reset_min_steps_since_best,
        )
        info = {
            **info,
            **memory_reset_info,
        }
        step_phase = str(info.get("phase_before", info.get("phase", "DISCOVER")))
        phase_counts[step_phase] = phase_counts.get(step_phase, 0) + 1
        if bool(info.get("incumbent_improved", False)):
            phase_improvement_counts[step_phase] = phase_improvement_counts.get(step_phase, 0) + 1
        if bool(info.get("phase_transition", False)):
            transition_key = f"{info.get('phase_before', step_phase)}->{info.get('phase', step_phase)}"
            phase_transition_counts[transition_key] = phase_transition_counts.get(transition_key, 0) + 1
        if str(info.get("phase_transition_reason", "")) in {"refine_regression", "unlock_request_after_stagnation"}:
            phase_failure_score += 1.0
        if bool(info.get("refine_regressed", False)):
            phase_failure_score += 1.0
        if bool(info.get("refine_rejected", False)):
            phase_failure_score += 0.5
        if bool(info.get("rollback_to_incumbent", False)):
            phase_failure_score += 0.25
        memory_reset_count += int(memory_reset_info["memory_reset_applied"])
        transitions.append(
            Transition(
                graph=detach_graph(graph),
                action=detach_action(action),
                old_group_logprobs={
                    key: value.detach().clone() for key, value in action.group_logprobs.items()
                },
                group_token_counts={
                    key: value.detach().clone() for key, value in action.group_token_counts.items()
                },
                value=action.value.detach().clone(),
                reward=reward,
                done=done,
                temperature=temperature,
            )
        )
        infos.append(info)
        if done:
            break

    best_score, best_centers = env.best_candidate()
    rollout_best_score = dict(best_score)
    rollout_best_centers = best_centers.detach().clone()
    winning_refine_variant = "rollout_best"
    refine_variant_rows = []
    refine_supervision_rows = []
    live_candidate_records = [
        candidate_record(
            source="rollout_best",
            overlap=rollout_best_score["overlap_ratio"],
            pairs=rollout_best_score["num_overlap_pairs"],
            wirelength=rollout_best_score["normalized_wl"],
            repair_legal=int(rollout_best_score["overlap_cells"]) == 0,
            accepted=True,
            live_input=True,
            generation_index=0,
        )
    ]
    external_teacher_winner = dict(live_candidate_records[0])
    chooser_selected_record = dict(live_candidate_records[0])
    chooser_logits = torch.zeros((1,), dtype=env.centers.dtype, device=env.centers.device)
    refine_window_size = 0
    if post_legal_refine_portfolio:
        refine_result = env.run_post_legal_refine_portfolio(best_centers, policy=policy)
        refine_variant_rows = list(refine_result["variant_rows"])
        refine_supervision_rows = enrich_refine_supervision_rows(refine_variant_rows)
        refine_window_size = int(refine_result["window"].numel())
        for option in refine_result.get("candidate_options", []):
            if not bool(option.get("evaluated", False)):
                continue
            score = dict(option.get("score", {}))
            live_candidate_records.append(
                candidate_record(
                    source=str(option.get("source", f"refine_variant:{option.get('variant_name', '')}")),
                    overlap=score.get("overlap_ratio", rollout_best_score["overlap_ratio"]),
                    pairs=score.get("num_overlap_pairs", rollout_best_score["num_overlap_pairs"]),
                    wirelength=score.get("normalized_wl", rollout_best_score["normalized_wl"]),
                    repair_legal=bool(option.get("repair_legal", False)),
                    accepted=bool(option.get("accepted", False)),
                    live_input=bool(option.get("live_input", True)),
                    generation_index=int(option.get("generation_index", len(live_candidate_records))),
                    variant_name=str(option.get("variant_name", "")),
                    operator_metadata=option.get("operator_metadata", {}),
                )
            )
        chooser_graph = env.auxiliary_graph_for_candidate(rollout_best_centers, phase_name="REFINE")
        chooser_bundle = chooser_candidate_features(
            live_candidate_records,
            device=device,
            dtype=env.centers.dtype,
            graph=chooser_graph,
        )
        if chooser_bundle is not None:
            with torch.no_grad():
                chooser_logits, chooser_selected_index = policy.choose_repaired_candidate(
                    chooser_graph,
                    chooser_bundle["chooser_features"],
                    legacy_candidate_features=chooser_bundle["legacy_features"],
                )
            chooser_selected_record = dict(chooser_bundle["records"][chooser_selected_index])
            external_teacher_winner = dict(
                chooser_bundle["records"][int(chooser_bundle["target_index"])]
                if chooser_bundle["target_index"] is not None
                else chooser_bundle["records"][0]
            )
        else:
            chooser_selected_record = dict(live_candidate_records[0])
            external_teacher_winner = dict(live_candidate_records[0])

    chooser_source = str(chooser_selected_record.get("candidate_source", "rollout_best"))
    chooser_variant_name = str(chooser_selected_record.get("candidate_variant_name", ""))
    if chooser_source.startswith("refine_variant:"):
        winning_refine_variant = chooser_variant_name or chooser_source.split(":", 1)[-1]
    elif chooser_source == "rollout_best":
        winning_refine_variant = "rollout_best"
    else:
        winning_refine_variant = chooser_source

    if chooser_source == "rollout_best":
        best_score = dict(rollout_best_score)
        best_centers = rollout_best_centers.detach().clone()
    else:
        selected_option = next(
            (
                option
                for option in (refine_result.get("candidate_options", []) if post_legal_refine_portfolio else [])
                if str(option.get("source", "")) == chooser_source
            ),
            None,
        )
        if selected_option is not None:
            best_score = dict(selected_option["score"])
            best_centers = selected_option["centers"].detach().clone()
        else:
            best_score = dict(rollout_best_score)
            best_centers = rollout_best_centers.detach().clone()
            winning_refine_variant = "rollout_best"

    external_source = str(external_teacher_winner.get("candidate_source", "rollout_best"))
    candidate_records = []
    chooser_generation_index = int(chooser_selected_record.get("candidate_generation_index", 0))
    external_generation_index = int(external_teacher_winner.get("candidate_generation_index", 0))
    for row in live_candidate_records:
        generation_index = int(row.get("candidate_generation_index", 0))
        candidate_records.append(
            {
                **row,
                "candidate_chooser_selected": bool(generation_index == chooser_generation_index),
                "candidate_external_selected": bool(generation_index == external_generation_index),
            }
        )
    candidate_records.append(
        candidate_record(
            source="final_selected",
            overlap=external_teacher_winner["candidate_overlap"],
            pairs=external_teacher_winner["candidate_pairs"],
            wirelength=external_teacher_winner["candidate_wirelength"],
            repair_legal=bool(external_teacher_winner.get("candidate_repair_legal", False)),
            accepted=bool(external_teacher_winner.get("candidate_accepted", False)),
            live_input=False,
            diagnostic_only=True,
            chooser_selected=False,
            external_selected=True,
            generation_index=len(candidate_records),
            variant_name=str(external_teacher_winner.get("candidate_variant_name", "")),
            origin_source=str(external_teacher_winner.get("candidate_source", "rollout_best")),
        )
    )
    candidate_records = finalize_candidate_records(candidate_records)
    by_source = {}
    for row in candidate_records:
        by_source.setdefault(str(row["candidate_source"]), row)
    best_vs_final_regressed = continuation_margin_regressed(
        by_source["rollout_best"],
        chooser_selected_record,
    ) if "rollout_best" in by_source else False
    best_so_far_returned = not bool(best_vs_final_regressed)
    selected_candidate_phase = "REFINE" if post_legal_refine_portfolio else str(env.best_candidate_phase)
    aux_graph = detach_graph(
        env.auxiliary_graph_for_candidate(
            rollout_best_centers if post_legal_refine_portfolio else best_centers,
            phase_name=selected_candidate_phase,
        )
    )
    total_steps = max(len(transitions), 1)
    phase_summary = {}
    for phase_name in ("DISCOVER", "LEGALIZE", "REFINE", "UNLOCK"):
        phase_summary[f"phase_fraction_{phase_name}"] = float(phase_counts.get(phase_name, 0)) / float(total_steps)
        phase_summary[f"phase_improvements_{phase_name}"] = int(phase_improvement_counts.get(phase_name, 0))
    for transition_name, count in phase_transition_counts.items():
        safe_name = transition_name.replace("->", "_to_")
        phase_summary[f"phase_transition_{safe_name}"] = int(count)
    phase_summary["phase_failure_score"] = float(phase_failure_score)
    return transitions, {
        "size": size,
        "seed": int(seed),
        "discover_mode": str(discover_mode),
        "best_centers": best_centers.detach().clone(),
        "steps": len(transitions),
        "best_overlap": best_score["overlap_ratio"],
        "best_overlap_cells": best_score["overlap_cells"],
        "best_exact_overlap_pairs": best_score["num_overlap_pairs"],
        "best_wl": best_score["normalized_wl"],
        "winning_refine_variant": str(winning_refine_variant),
        "chooser_selected_source": str(chooser_selected_record.get("candidate_source", "rollout_best")),
        "chooser_selected_overlap": float(chooser_selected_record.get("candidate_overlap", rollout_best_score["overlap_ratio"])),
        "chooser_selected_pairs": int(chooser_selected_record.get("candidate_pairs", rollout_best_score["num_overlap_pairs"])),
        "chooser_selected_wirelength": float(chooser_selected_record.get("candidate_wirelength", rollout_best_score["normalized_wl"])),
        "external_selected_source": str(external_teacher_winner.get("candidate_source", "rollout_best")),
        "external_selected_overlap": float(external_teacher_winner.get("candidate_overlap", rollout_best_score["overlap_ratio"])),
        "external_selected_pairs": int(external_teacher_winner.get("candidate_pairs", rollout_best_score["num_overlap_pairs"])),
        "external_selected_wirelength": float(external_teacher_winner.get("candidate_wirelength", rollout_best_score["normalized_wl"])),
        "chooser_match": bool(
            lexicographic_candidate_key(chooser_selected_record) == lexicographic_candidate_key(external_teacher_winner)
        ),
        "chooser_regret_overlap": float(chooser_selected_record.get("candidate_overlap", rollout_best_score["overlap_ratio"]))
        - float(external_teacher_winner.get("candidate_overlap", rollout_best_score["overlap_ratio"])),
        "chooser_regret_pairs": int(chooser_selected_record.get("candidate_pairs", rollout_best_score["num_overlap_pairs"]))
        - int(external_teacher_winner.get("candidate_pairs", rollout_best_score["num_overlap_pairs"])),
        "chooser_regret_wirelength": float(chooser_selected_record.get("candidate_wirelength", rollout_best_score["normalized_wl"]))
        - float(external_teacher_winner.get("candidate_wirelength", rollout_best_score["normalized_wl"])),
        "refine_variant_rows": [dict(row) for row in refine_variant_rows],
        "refine_supervision_rows": [dict(row) for row in refine_supervision_rows],
        "cleanup_supervision_available": bool(len(refine_supervision_rows) > 0),
        "refine_window_size": int(refine_window_size),
        "refine_window_metadata": dict(refine_result.get("window_metadata", {})) if post_legal_refine_portfolio else {},
        "best_candidate_phase": str(selected_candidate_phase),
        "best_candidate_discover_mode": str(env.best_candidate_discover_mode),
        "best_candidate_refine_variant": str(winning_refine_variant),
        "candidate_records": [dict(row) for row in candidate_records],
        "chooser_graph": aux_graph,
        "best_so_far_returned": bool(best_so_far_returned),
        "best_vs_final_regressed": bool(best_vs_final_regressed),
        "aux_graph": aux_graph,
        "last_reward": infos[-1]["reward"] if infos else 0.0,
        "active_pairs": infos[-1]["active_pairs"] if infos else 0,
        "density_overflow": infos[-1].get("density_overflow", 0.0) if infos else 0.0,
        "branch_violation": infos[-1].get("branch_violation", 0.0) if infos else 0.0,
        "boundary_violation": infos[-1].get("boundary_violation", 0.0) if infos else 0.0,
        "lag_before": infos[-1].get("lag_before", 0.0) if infos else 0.0,
        "lag_after": infos[-1].get("lag_after", 0.0) if infos else 0.0,
        "pd_steps": infos[-1].get("pd_steps", 0) if infos else 0,
        "rho": infos[-1].get("rho", 0.0) if infos else 0.0,
        "eta": infos[-1].get("eta", 0.0) if infos else 0.0,
        "alpha": infos[-1].get("alpha", 0.0) if infos else 0.0,
        "step_scale": infos[-1].get("step_scale", 0.0) if infos else 0.0,
        "incumbent_mix": infos[-1].get("incumbent_mix", 0.0) if infos else 0.0,
        "steps_since_best": infos[-1].get("steps_since_best", 0) if infos else 0,
        "best_candidate_step": infos[-1].get("best_candidate_step", 0) if infos else 0,
        "continuation_risk": infos[-1].get("continuation_risk", 0.0) if infos else 0.0,
        "pair_emphasis": infos[-1].get("pair_emphasis", 0.0) if infos else 0.0,
        "tau": infos[-1].get("tau", 0.0) if infos else 0.0,
        "branch_pressure": infos[-1].get("branch_pressure", 0.0) if infos else 0.0,
        "density_pressure": infos[-1].get("density_pressure", 0.0) if infos else 0.0,
        "boundary_pressure": infos[-1].get("boundary_pressure", 0.0) if infos else 0.0,
        "missed_pairs": infos[-1].get("missed_pairs", 0) if infos else 0,
        "inactive_missed_pairs": infos[-1].get("inactive_missed_pairs", 0) if infos else 0,
        "exact_overlap_pairs": infos[-1].get("exact_overlap_pairs", 0) if infos else 0,
        "sampled_pairs": infos[-1].get("sampled_pairs", 0) if infos else 0,
        "cluster_pairs": infos[-1].get("cluster_pairs", 0) if infos else 0,
        "uncertain_pairs": infos[-1].get("uncertain_pairs", 0) if infos else 0,
        "new_active_pairs": infos[-1].get("new_active_pairs", 0) if infos else 0,
        "retained_pairs": infos[-1].get("retained_pairs", 0) if infos else 0,
        "hard_pair_age_mean": infos[-1].get("hard_pair_age_mean", 0.0) if infos else 0.0,
        "hard_pair_age_max": infos[-1].get("hard_pair_age_max", 0.0) if infos else 0.0,
        "hard_pair_age_min": infos[-1].get("hard_pair_age_min", 0.0) if infos else 0.0,
        "audit_pressure_scale": infos[-1].get("audit_pressure_scale", 1.0) if infos else 1.0,
        "audit_pressure_target": infos[-1].get("audit_pressure_target", 0.0) if infos else 0.0,
        "retention_horizon": infos[-1].get("retention_horizon", 0) if infos else 0,
        "stop": infos[-1].get("stop", False) if infos else False,
        "stop_probability": infos[-1].get("stop_probability", 0.0) if infos else 0.0,
        "stop_logit_bias": infos[-1].get("stop_logit_bias", 0.0) if infos else 0.0,
        "stop_gated": infos[-1].get("stop_gated", False) if infos else False,
        "stop_overlap": infos[-1].get("stop_overlap", 0.0) if infos else 0.0,
        "false_stop": infos[-1].get("false_stop", False) if infos else False,
        "residual_norm": infos[-1].get("residual_norm", 0.0) if infos else 0.0,
        "dual_clamp_fraction": infos[-1].get("dual_clamp_fraction", 0.0) if infos else 0.0,
        "overlap_delta": infos[-1].get("overlap_delta", 0.0) if infos else 0.0,
        "branch_violation_penalty": infos[-1].get("branch_violation_penalty", 0.0) if infos else 0.0,
        "missed_pair_penalty": infos[-1].get("missed_pair_penalty", 0.0) if infos else 0.0,
        "memory_reset_count": int(memory_reset_count),
        "memory_reset_applied": bool(memory_reset_count > 0),
        "memory_reset_retain": infos[-1].get("memory_reset_retain", 1.0) if infos else 1.0,
        "adaptive_pd_steps_applied": infos[-1].get("adaptive_pd_steps_applied", False) if infos else False,
        "adaptive_pd_steps_stage": infos[-1].get("adaptive_pd_steps_stage", "") if infos else "",
        "adaptive_pd_steps_case_descriptor_bucket": infos[-1].get("adaptive_pd_steps_case_descriptor_bucket", "") if infos else "",
        "adaptive_pd_extra_steps": infos[-1].get("adaptive_pd_extra_steps", 0) if infos else 0,
        "final_phase": infos[-1].get("phase", "DISCOVER") if infos else "DISCOVER",
        **phase_summary,
    }


def validate_policy(
    policy,
    sizes,
    env_config,
    device,
    seed,
    temperature,
    soft_tau=None,
    relaxation="sigmoid",
    episodes=4,
    validation_suite=None,
    memory_reset_mode="none",
    memory_reset_retain=1.0,
    memory_reset_min_overlap_gain=0.03,
    memory_reset_min_pair_gain_count=2.0,
    memory_reset_min_steps_since_best=2,
    return_rows=False,
):
    policy.eval()
    rows = []
    suite = validation_suite or build_validation_suite(sizes, episodes, seed)
    for episode_idx, spec in enumerate(suite):
        forced_size = tuple(spec["size"])
        episode_seed = int(spec["seed"])
        mode_rows = []
        for discover_mode in DISCOVER_MODE_NAMES:
            _transitions, info = collect_episode(
                policy,
                sizes,
                env_config,
                device,
                episode_seed,
                temperature,
                soft_tau=soft_tau,
                relaxation=relaxation,
                forced_size=forced_size,
                deterministic=True,
                discover_mode=discover_mode,
                memory_reset_mode=memory_reset_mode,
                memory_reset_retain=memory_reset_retain,
                memory_reset_min_overlap_gain=memory_reset_min_overlap_gain,
                memory_reset_min_pair_gain_count=memory_reset_min_pair_gain_count,
                memory_reset_min_steps_since_best=memory_reset_min_steps_since_best,
                post_legal_refine_portfolio=True,
            )
            mode_rows.append(info)
        winner = min(mode_rows, key=lambda row: (float(row["best_overlap"]), float(row["best_exact_overlap_pairs"]), float(row["best_wl"])))
        external_mode_rows = [
            {
                "discover_mode": str(row["discover_mode"]),
                "candidate_source": str(row.get("external_selected_source", row.get("chooser_selected_source", "rollout_best"))),
                "candidate_overlap": float(row.get("external_selected_overlap", row["best_overlap"])),
                "candidate_pairs": int(row.get("external_selected_pairs", row["best_exact_overlap_pairs"])),
                "candidate_wirelength": float(row.get("external_selected_wirelength", row["best_wl"])),
                "candidate_generation_index": idx,
            }
            for idx, row in enumerate(mode_rows)
        ]
        external_case_winner = external_candidate_teacher_winner(external_mode_rows)
        winner = {
            **winner,
            "winning_discover_mode": str(winner["discover_mode"]),
            "winning_refine_variant": str(winner.get("winning_refine_variant", "incumbent_hold")),
            "cleanup_supervision_available": bool(winner.get("cleanup_supervision_available", False)),
            "external_winning_mode": str(external_case_winner["discover_mode"]),
            "external_selected_source": str(external_case_winner["candidate_source"]),
            "external_selected_overlap": float(external_case_winner["candidate_overlap"]),
            "external_selected_pairs": int(external_case_winner["candidate_pairs"]),
            "external_selected_wirelength": float(external_case_winner["candidate_wirelength"]),
            "chooser_case_match": bool(
                (
                    float(winner["best_overlap"]),
                    int(winner["best_exact_overlap_pairs"]),
                    float(winner["best_wl"]),
                )
                == (
                    float(external_case_winner["candidate_overlap"]),
                    int(external_case_winner["candidate_pairs"]),
                    float(external_case_winner["candidate_wirelength"]),
                )
            ),
            "chooser_case_regret_overlap": float(winner["best_overlap"]) - float(external_case_winner["candidate_overlap"]),
            "chooser_case_regret_pairs": int(winner["best_exact_overlap_pairs"]) - int(external_case_winner["candidate_pairs"]),
            "chooser_case_regret_wirelength": float(winner["best_wl"]) - float(external_case_winner["candidate_wirelength"]),
            "per_mode_validation": [
                {
                    "discover_mode": str(row["discover_mode"]),
                    "best_overlap": float(row["best_overlap"]),
                    "best_exact_overlap_pairs": float(row["best_exact_overlap_pairs"]),
                    "best_wl": float(row["best_wl"]),
                    "winning_refine_variant": str(row.get("winning_refine_variant", "incumbent_hold")),
                    "cleanup_supervision_available": bool(row.get("cleanup_supervision_available", False)),
                    "chooser_selected_source": str(row.get("chooser_selected_source", "rollout_best")),
                    "external_selected_source": str(row.get("external_selected_source", "rollout_best")),
                    "chooser_match": bool(row.get("chooser_match", True)),
                }
                for row in mode_rows
            ],
            "per_mode_info_rows": mode_rows,
        }
        rows.append(winner)
    policy.train()

    def mean(key):
        return sum(float(row.get(key, 0.0)) for row in rows) / max(len(rows), 1)

    summary = {
        "validation_episodes": len(rows),
        "validation_overlap": mean("best_overlap"),
        "validation_wirelength": mean("best_wl"),
        "validation_branch_violation": mean("branch_violation"),
        "validation_missed_pairs": mean("missed_pairs"),
        "validation_exact_overlap_pairs": mean("best_exact_overlap_pairs"),
        "validation_hard_pair_age_mean": mean("hard_pair_age_mean"),
        "validation_audit_pressure_scale": mean("audit_pressure_scale"),
        "validation_audit_pressure_target": mean("audit_pressure_target"),
        "validation_stop_probability": mean("stop_probability"),
        "validation_stop_gated_rate": mean("stop_gated"),
        "validation_stop_overlap": mean("stop_overlap"),
        "validation_false_stop_rate": mean("false_stop"),
        "validation_stop_rate": sum(1.0 if row.get("stop", False) else 0.0 for row in rows) / max(len(rows), 1),
        "validation_memory_reset_count": mean("memory_reset_count"),
        "validation_memory_reset_rate": mean("memory_reset_applied"),
        "validation_fixed_suite": True,
        "validation_best_vs_final_regression_rate": sum(1.0 if row.get("best_vs_final_regressed", False) else 0.0 for row in rows) / max(len(rows), 1),
        "validation_best_so_far_returned_rate": sum(1.0 if row.get("best_so_far_returned", False) else 0.0 for row in rows) / max(len(rows), 1),
        "chooser_top1_match_rate_case": sum(1.0 if row.get("chooser_case_match", False) else 0.0 for row in rows) / max(len(rows), 1),
        "chooser_mean_overlap_regret_case": mean("chooser_case_regret_overlap"),
        "chooser_mean_pair_regret_case": mean("chooser_case_regret_pairs"),
        "chooser_mean_wire_regret_case": mean("chooser_case_regret_wirelength"),
    }
    mode_win_counts = {mode: 0 for mode in DISCOVER_MODE_NAMES}
    refine_variant_win_counts = {}
    refine_variant_break_counts = {name: 0 for name in REFINE_VARIANT_NAMES}
    refine_variant_eval_counts = {name: 0 for name in REFINE_VARIANT_NAMES}
    chooser_mode_match_hits = 0
    chooser_mode_total = 0
    chooser_mode_regret_overlap = 0.0
    chooser_mode_regret_pairs = 0.0
    chooser_mode_regret_wire = 0.0
    chooser_confusion_counts = {}
    for row in rows:
        mode_win_counts[str(row.get("winning_discover_mode", row.get("discover_mode", "balanced")))] += 1
        refine_name = str(row.get("winning_refine_variant", "incumbent_hold"))
        refine_variant_win_counts[refine_name] = refine_variant_win_counts.get(refine_name, 0) + 1
        for mode_row in row.get("per_mode_info_rows", []):
            chooser_mode_match_hits += int(bool(mode_row.get("chooser_match", False)))
            chooser_mode_total += 1
            chooser_mode_regret_overlap += float(mode_row.get("chooser_regret_overlap", 0.0))
            chooser_mode_regret_pairs += float(mode_row.get("chooser_regret_pairs", 0.0))
            chooser_mode_regret_wire += float(mode_row.get("chooser_regret_wirelength", 0.0))
            confusion_key = (
                str(mode_row.get("chooser_selected_source", "rollout_best")),
                str(mode_row.get("external_selected_source", "rollout_best")),
            )
            chooser_confusion_counts[confusion_key] = chooser_confusion_counts.get(confusion_key, 0) + 1
            for variant_row in mode_row.get("refine_variant_rows", []):
                name = str(variant_row.get("variant_name", "incumbent_hold"))
                if name not in refine_variant_eval_counts:
                    continue
                if bool(variant_row.get("evaluated", True)):
                    refine_variant_eval_counts[name] += 1
                    if not bool(variant_row.get("repair_legal", False)):
                        refine_variant_break_counts[name] += 1
    for mode_name, count in mode_win_counts.items():
        summary[f"validation_mode_wins_{mode_name}"] = int(count)
    for refine_name, count in refine_variant_win_counts.items():
        summary[f"validation_refine_variant_wins_{refine_name}"] = int(count)
    for refine_name in REFINE_VARIANT_NAMES:
        eval_count = max(refine_variant_eval_counts.get(refine_name, 0), 1)
        summary[f"validation_refine_variant_legality_break_rate_{refine_name}"] = float(refine_variant_break_counts.get(refine_name, 0)) / float(eval_count)
    summary["chooser_top1_match_rate_mode"] = float(chooser_mode_match_hits) / max(chooser_mode_total, 1)
    summary["chooser_mean_overlap_regret_mode"] = chooser_mode_regret_overlap / max(chooser_mode_total, 1)
    summary["chooser_mean_pair_regret_mode"] = chooser_mode_regret_pairs / max(chooser_mode_total, 1)
    summary["chooser_mean_wire_regret_mode"] = chooser_mode_regret_wire / max(chooser_mode_total, 1)
    for (chooser_source, external_source), count in chooser_confusion_counts.items():
        safe_chooser = chooser_source.replace(":", "_")
        safe_external = external_source.replace(":", "_")
        summary[f"validation_chooser_confusion_{safe_chooser}_to_{safe_external}"] = int(count)
    for phase_name in ("DISCOVER", "LEGALIZE", "REFINE", "UNLOCK"):
        summary[f"validation_phase_fraction_{phase_name}"] = mean(f"phase_fraction_{phase_name}")
        summary[f"validation_phase_improvements_{phase_name}"] = mean(f"phase_improvements_{phase_name}")
    summary["validation_phase_failure_score"] = mean("phase_failure_score")
    if return_rows:
        return summary, rows
    return summary


def auxiliary_supervision_update(
    policy,
    optimizer,
    validation_rows,
    *,
    cleanup_weight=1.0,
    chooser_weight=1.0,
    ranking_weight=0.5,
    mode_weight=0.5,
    continuation_weight=0.5,
):
    if not validation_rows:
        return {}
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    cleanup_variant_losses = []
    cleanup_accept_losses = []
    cleanup_rank_losses = []
    cleanup_variant_hits = 0
    cleanup_accept_correct = 0
    cleanup_accept_total = 0
    cleanup_rank_hits = 0
    cleanup_samples = 0
    chooser_losses = []
    chooser_pairwise_losses = []
    chooser_hits = 0
    chooser_total = 0
    chooser_regret_overlap = 0.0
    chooser_regret_pairs = 0.0
    chooser_regret_wire = 0.0
    ranking_losses = []
    ranking_best_final_hits = 0
    ranking_best_final_total = 0
    repair_authority_hits = 0
    repair_authority_total = 0
    mode_losses = []
    mode_top1_hits = 0
    mode_top2_hits = 0
    mode_total = 0
    regret_overlap = 0.0
    regret_pairs = 0.0
    regret_wire = 0.0
    continuation_losses = []
    continuation_hits = 0
    continuation_total = 0
    refine_gate_top1_hits = 0
    refine_gate_top2_hits = 0
    refine_gate_total = 0
    refine_gate_regret_overlap = 0.0
    refine_gate_regret_pairs = 0.0
    refine_gate_regret_wire = 0.0

    for row in validation_rows:
        aux_graph = row.get("aux_graph")
        aux = policy.auxiliary_predictions(aux_graph) if aux_graph is not None else None
        if aux is not None and continuation_supervision_eligible(row):
            continue_target = torch.tensor(
                [0.0 if bool(row.get("best_so_far_returned", False)) else 1.0],
                dtype=dtype,
                device=device,
            )
            continuation_losses.append(
                F.binary_cross_entropy_with_logits(
                    aux["continuation_preserve_logit"].reshape(1),
                    continue_target,
                )
            )
            continuation_pred = torch.sigmoid(aux["continuation_preserve_logit"]).reshape(1) > 0.5
            continuation_hits += int(int(continuation_pred.item()) == int(continue_target.item()))
            continuation_total += 1

        if aux is not None and row.get("winning_discover_mode") in DISCOVER_MODE_TO_INDEX:
            mode_target = torch.tensor(
                DISCOVER_MODE_TO_INDEX[str(row["winning_discover_mode"])],
                dtype=torch.long,
                device=device,
            )
            mode_losses.append(F.cross_entropy(aux["mode_selector_logits"].unsqueeze(0), mode_target.unsqueeze(0)))
            topk = torch.topk(aux["mode_selector_logits"], k=min(2, aux["mode_selector_logits"].numel())).indices.tolist()
            target_idx = int(mode_target.item())
            mode_top1_hits += int(topk[0] == target_idx)
            mode_top2_hits += int(target_idx in topk)
            mode_total += 1
            per_mode = {str(item["discover_mode"]): item for item in row.get("per_mode_info_rows", [])}
            predicted_mode = DISCOVER_MODE_NAMES[int(torch.argmax(aux["mode_selector_logits"]).item())]
            if predicted_mode in per_mode and str(row["winning_discover_mode"]) in per_mode:
                predicted_row = per_mode[predicted_mode]
                winner_row = per_mode[str(row["winning_discover_mode"])]
                regret_overlap += float(predicted_row["best_overlap"]) - float(winner_row["best_overlap"])
                regret_pairs += float(predicted_row["best_exact_overlap_pairs"]) - float(winner_row["best_exact_overlap_pairs"])
                regret_wire += float(predicted_row["best_wl"]) - float(winner_row["best_wl"])

        for mode_row in row.get("per_mode_info_rows", []):
            mode_graph = mode_row.get("chooser_graph") or mode_row.get("aux_graph")
            if mode_graph is None:
                continue
            aux = policy.auxiliary_predictions(mode_graph)
            cleanup_rows = list(mode_row.get("refine_supervision_rows", []))
            if cleanup_rows:
                rank_sorted = sorted(cleanup_rows, key=lambda item: int(item["candidate_lexi_rank"]))
                best_variant = str(rank_sorted[0]["variant_name"])
                target_idx = REFINE_VARIANT_TO_INDEX[best_variant]
                cleanup_variant_losses.append(
                    F.cross_entropy(
                        aux["cleanup_variant_logits"].unsqueeze(0),
                        torch.tensor([target_idx], dtype=torch.long, device=device),
                    )
                )
                cleanup_variant_pred = int(torch.argmax(aux["cleanup_variant_logits"]).item())
                cleanup_variant_hits += int(cleanup_variant_pred == target_idx)
                topk_variants = torch.topk(
                    aux["cleanup_variant_logits"],
                    k=min(2, aux["cleanup_variant_logits"].numel()),
                ).indices.tolist()
                refine_gate_top1_hits += int(int(topk_variants[0]) == target_idx)
                refine_gate_top2_hits += int(int(target_idx) in [int(idx) for idx in topk_variants])
                refine_gate_total += 1
                accept_target = torch.zeros(
                    (len(REFINE_VARIANT_NAMES),),
                    dtype=dtype,
                    device=device,
                )
                for item in cleanup_rows:
                    accept_target[REFINE_VARIANT_TO_INDEX[str(item["variant_name"])]] = (
                        1.0 if bool(item["candidate_accepted"]) else 0.0
                    )
                cleanup_accept_losses.append(
                    F.binary_cross_entropy_with_logits(aux["cleanup_accept_logit"], accept_target)
                )
                accept_pred = (torch.sigmoid(aux["cleanup_accept_logit"]) > 0.5).to(dtype)
                cleanup_accept_correct += int((accept_pred == accept_target).sum().item())
                cleanup_accept_total += int(accept_target.numel())
                pair_terms = []
                for better in cleanup_rows:
                    for worse in cleanup_rows:
                        if int(better["candidate_lexi_rank"]) >= int(worse["candidate_lexi_rank"]):
                            continue
                        i = REFINE_VARIANT_TO_INDEX[str(better["variant_name"])]
                        j = REFINE_VARIANT_TO_INDEX[str(worse["variant_name"])]
                        pair_terms.append(F.softplus(-(aux["cleanup_rank_value"][i] - aux["cleanup_rank_value"][j])))
                if pair_terms:
                    cleanup_rank_losses.append(torch.stack(pair_terms).mean())
                cleanup_rank_hits += int(int(torch.argmax(aux["cleanup_rank_value"]).item()) == target_idx)
                cleanup_samples += 1
                per_variant = {str(item["variant_name"]): item for item in cleanup_rows}
                predicted_variant = REFINE_VARIANT_NAMES[cleanup_variant_pred]
                if predicted_variant in per_variant and best_variant in per_variant:
                    predicted_row = per_variant[predicted_variant]
                    winner_row = per_variant[best_variant]
                    refine_gate_regret_overlap += float(predicted_row["candidate_overlap"]) - float(winner_row["candidate_overlap"])
                    refine_gate_regret_pairs += float(predicted_row["candidate_pairs"]) - float(winner_row["candidate_pairs"])
                    refine_gate_regret_wire += float(predicted_row["candidate_wirelength"]) - float(winner_row["candidate_wirelength"])

            chooser_bundle = chooser_candidate_features(
                list(mode_row.get("candidate_records", [])),
                device=device,
                dtype=dtype,
                graph=mode_graph,
            )
            if chooser_bundle is not None and chooser_bundle.get("records"):
                chooser_records = chooser_bundle["records"]
                chooser_target = chooser_bundle.get("target_index")
                chooser_logits, chooser_selected_index = policy.choose_repaired_candidate(
                    mode_graph,
                    chooser_bundle["chooser_features"],
                    legacy_candidate_features=chooser_bundle["legacy_features"],
                )
                if chooser_target is not None:
                    chooser_losses.append(
                        F.cross_entropy(
                            chooser_logits.unsqueeze(0),
                            torch.tensor([int(chooser_target)], dtype=torch.long, device=device),
                        )
                    )
                    pair_terms = []
                    for i, better in enumerate(chooser_records):
                        for j, worse in enumerate(chooser_records):
                            if lexicographic_candidate_generation_key(better) >= lexicographic_candidate_generation_key(worse):
                                continue
                            pair_terms.append(
                                chooser_pairwise_weight(better, worse)
                                * F.softplus(-(chooser_logits[i] - chooser_logits[j]))
                            )
                    if pair_terms:
                        pairwise_loss = torch.stack(pair_terms).mean()
                        chooser_pairwise_losses.append(pairwise_loss)
                        ranking_losses.append(pairwise_loss)
                    chooser_hits += int(int(chooser_selected_index) == int(chooser_target))
                    chooser_total += 1
                    predicted_row = chooser_records[int(chooser_selected_index)]
                    teacher_row = chooser_records[int(chooser_target)]
                    chooser_regret_overlap += float(predicted_row["candidate_overlap"]) - float(teacher_row["candidate_overlap"])
                    chooser_regret_pairs += float(predicted_row["candidate_pairs"]) - float(teacher_row["candidate_pairs"])
                    chooser_regret_wire += float(predicted_row["candidate_wirelength"]) - float(teacher_row["candidate_wirelength"])
                    by_source = {str(item["candidate_source"]): (idx, item) for idx, item in enumerate(chooser_records)}
                    if "rollout_best" in by_source:
                        rollout_idx, _rollout_row = by_source["rollout_best"]
                        better_is_teacher = int(chooser_target) != int(rollout_idx)
                        predicted_prefers_teacher = (
                            float(chooser_logits[int(chooser_target)].item()) > float(chooser_logits[int(rollout_idx)].item())
                            if int(chooser_target) != int(rollout_idx)
                            else False
                        )
                        ranking_best_final_hits += int(predicted_prefers_teacher == better_is_teacher)
                        ranking_best_final_total += 1
                    repair_authority_hits += int(int(chooser_selected_index) == int(chooser_target))
                    repair_authority_total += 1

    total_loss = None
    if cleanup_variant_losses:
        cleanup_loss = (
            torch.stack(cleanup_variant_losses).mean()
            + torch.stack(cleanup_accept_losses).mean()
            + (torch.stack(cleanup_rank_losses).mean() if cleanup_rank_losses else 0.0)
        )
        total_loss = cleanup_weight * cleanup_loss if total_loss is None else total_loss + cleanup_weight * cleanup_loss
    if chooser_losses:
        chooser_loss = torch.stack(chooser_losses).mean()
        total_loss = chooser_weight * chooser_loss if total_loss is None else total_loss + chooser_weight * chooser_loss
        if chooser_pairwise_losses:
            chooser_pairwise_loss = torch.stack(chooser_pairwise_losses).mean()
            total_loss = (
                ranking_weight * chooser_pairwise_loss
                if total_loss is None
                else total_loss + ranking_weight * chooser_pairwise_loss
            )
        else:
            chooser_pairwise_loss = None
    else:
        chooser_loss = None
        chooser_pairwise_loss = None
    if ranking_losses:
        ranking_loss = torch.stack(ranking_losses).mean()
    else:
        ranking_loss = None
    if mode_losses:
        mode_loss = torch.stack(mode_losses).mean()
        total_loss = mode_weight * mode_loss if total_loss is None else total_loss + mode_weight * mode_loss
    else:
        mode_loss = None
    if continuation_losses:
        continuation_loss = torch.stack(continuation_losses).mean()
        total_loss = (
            continuation_weight * continuation_loss
            if total_loss is None
            else total_loss + continuation_weight * continuation_loss
        )
    else:
        continuation_loss = None

    if total_loss is not None:
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        "cleanup_aux_loss": float(torch.stack(cleanup_variant_losses).mean().item()) if cleanup_variant_losses else 0.0,
        "chooser_aux_loss": float(chooser_loss.item()) if chooser_loss is not None else 0.0,
        "chooser_pairwise_loss": float(chooser_pairwise_loss.item()) if chooser_pairwise_loss is not None else 0.0,
        "chooser_top1_match_rate_mode": float(chooser_hits) / max(chooser_total, 1),
        "chooser_mean_overlap_regret_mode": chooser_regret_overlap / max(chooser_total, 1),
        "chooser_mean_pair_regret_mode": chooser_regret_pairs / max(chooser_total, 1),
        "chooser_mean_wire_regret_mode": chooser_regret_wire / max(chooser_total, 1),
        "cleanup_accept_accuracy": float(cleanup_accept_correct) / max(cleanup_accept_total, 1),
        "cleanup_variant_top1": float(cleanup_variant_hits) / max(cleanup_samples, 1),
        "cleanup_rank_agreement": float(cleanup_rank_hits) / max(cleanup_samples, 1),
        "ranking_aux_loss": float(ranking_loss.item()) if ranking_loss is not None else 0.0,
        "ranking_agreement_best_vs_final": float(ranking_best_final_hits) / max(ranking_best_final_total, 1),
        "repair_authority_agreement": float(repair_authority_hits) / max(repair_authority_total, 1),
        "mode_selector_top1": float(mode_top1_hits) / max(mode_total, 1),
        "mode_selector_top2": float(mode_top2_hits) / max(mode_total, 1),
        "mode_selector_regret_overlap": regret_overlap / max(mode_total, 1),
        "mode_selector_regret_pairs": regret_pairs / max(mode_total, 1),
        "mode_selector_regret_wirelength": regret_wire / max(mode_total, 1),
        "continuation_aux_loss": float(continuation_loss.item()) if continuation_loss is not None else 0.0,
        "continuation_preserve_accuracy": float(continuation_hits) / max(continuation_total, 1),
        "refine_gate_top1": float(refine_gate_top1_hits) / max(refine_gate_total, 1),
        "refine_gate_top2": float(refine_gate_top2_hits) / max(refine_gate_total, 1),
        "refine_gate_regret_overlap": refine_gate_regret_overlap / max(refine_gate_total, 1),
        "refine_gate_regret_pairs": refine_gate_regret_pairs / max(refine_gate_total, 1),
        "refine_gate_regret_wirelength": refine_gate_regret_wire / max(refine_gate_total, 1),
        "cleanup_supervision_rows": int(sum(len(r.get("refine_supervision_rows", [])) for row in validation_rows for r in row.get("per_mode_info_rows", []))),
    }


def build_teacher_transfer_record(
    args,
    distill_stats,
    teacher_metadata,
    *,
    dataset_path="",
    dataset_version=0,
    teacher_lambda=None,
):
    quality_cfg = teacher_metadata.get("quality_cfg", {}) if isinstance(teacher_metadata, dict) else {}
    dataset_summary = teacher_metadata.get("dataset_summary", {}) if isinstance(teacher_metadata, dict) else {}
    teacher_samples = int(distill_stats.get("teacher_samples", dataset_summary.get("samples", 0)))
    if args.teacher_dataset:
        source_phase = "integrated_distill"
    elif dataset_path:
        source_phase = "resume_dataset"
    else:
        source_phase = "resume_checkpoint" if teacher_samples > 0 else "none"
    return {
        "teacher_dataset": bool(dataset_path),
        "teacher_dataset_path": dataset_path,
        "teacher_dataset_version": int(dataset_version),
        "teacher_source_phase": source_phase,
        "teacher_solver": teacher_metadata.get("teacher_solver", "") if isinstance(teacher_metadata, dict) else "",
        "teacher_samples": teacher_samples,
        "teacher_samples_requested": int(teacher_metadata.get("num_cases_requested", 0)) if isinstance(teacher_metadata, dict) else 0,
        "teacher_samples_accepted": int(
            teacher_metadata.get("num_cases_accepted", dataset_summary.get("samples", teacher_samples))
        ) if isinstance(teacher_metadata, dict) else teacher_samples,
        "teacher_lambda": float(
            distill_stats.get("teacher_lambda_final", 0.0) if teacher_lambda is None else teacher_lambda
        ),
        "teacher_lambda_initial": float(distill_stats.get("teacher_lambda_initial", 0.0)),
        "teacher_lambda_final": float(distill_stats.get("teacher_lambda_final", 0.0)),
        "teacher_anneal_updates": int(getattr(args, "teacher_anneal_updates", 0)),
        "dagger_correction_count": int(distill_stats.get("dagger_correction_count", 0)),
        "distill_branch_accuracy": float(distill_stats.get("branch_accuracy", 0.0)),
        "distill_flow_loss": float(distill_stats.get("flow_loss", 0.0)),
        "distill_stop_false_positive": float(distill_stats.get("stop_false_positive", 0.0)),
        "teacher_demo_avg_overlap": float(dataset_summary.get("avg_overlap_ratio", 0.0)),
        "teacher_demo_avg_wirelength": float(dataset_summary.get("avg_normalized_wl", 0.0)),
        "teacher_demo_avg_weight": float(dataset_summary.get("avg_weight", 0.0)),
        "teacher_demo_min_weight": float(dataset_summary.get("min_weight", 0.0)),
        "teacher_demo_max_weight": float(dataset_summary.get("max_weight", 0.0)),
        "teacher_zero_overlap_fraction": float(dataset_summary.get("zero_overlap_fraction", 0.0)),
        "teacher_dagger_fraction": float(dataset_summary.get("dagger_fraction", 0.0)),
        "teacher_max_demo_overlap": float(quality_cfg.get("max_demo_overlap", 0.0)),
        "teacher_alpha_o": float(quality_cfg.get("alpha_o", 0.0)),
        "teacher_alpha_w": float(quality_cfg.get("alpha_w", 0.0)),
    }


def update_metric_gated_tau(
    current_tau,
    overlap,
    branch_violation,
    missed_pairs,
    exact_overlap_pairs,
    state,
    *,
    tau_min,
    tau_max,
    gamma_down,
    gamma_up,
    overlap_epsilon,
    branch_violation_max,
    missed_pairs_max,
    exact_overlap_pairs_max,
    patience,
):
    if state["best_overlap"] is None:
        state["best_overlap"] = overlap
        state["bad_windows"] = 0
        state["has_baseline"] = True
        return float(current_tau)

    improved = overlap < state["best_overlap"] - overlap_epsilon
    stable = (
        branch_violation <= branch_violation_max
        and missed_pairs <= missed_pairs_max
        and exact_overlap_pairs <= exact_overlap_pairs_max
    )
    if improved and stable:
        state["best_overlap"] = overlap
        state["bad_windows"] = 0
        return max(float(tau_min), float(current_tau) * float(gamma_down))
    if improved:
        state["best_overlap"] = overlap
        state["bad_windows"] = 0
        return float(current_tau)
    state["bad_windows"] += 1
    if state["bad_windows"] >= int(patience):
        state["bad_windows"] = 0
        return min(float(tau_max), float(current_tau) * float(gamma_up))
    return float(current_tau)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=default_device_arg())
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--coordinate-steps", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-passes", type=int, default=2)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--global-flow-rank", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature-start", type=float, default=1.4)
    parser.add_argument("--temperature-end", type=float, default=0.55)
    parser.add_argument("--equivariance-coef", type=float, default=0.001)
    parser.add_argument("--soft-relax-frac", type=float, default=0.40)
    parser.add_argument("--soft-tau-start", type=float, default=2.0)
    parser.add_argument("--soft-tau-end", type=float, default=0.10)
    parser.add_argument("--metric-gated-hardening", action="store_true")
    parser.add_argument("--tau-max", type=float, default=2.5)
    parser.add_argument("--tau-down", type=float, default=0.96)
    parser.add_argument("--tau-up", type=float, default=1.08)
    parser.add_argument("--hardening-patience", type=int, default=5)
    parser.add_argument("--hardening-overlap-eps", type=float, default=0.005)
    parser.add_argument("--hardening-branch-vmax", type=float, default=0.01)
    parser.add_argument("--hardening-missed-max", type=float, default=64.0)
    parser.add_argument("--hardening-exact-pairs-max", type=float, default=8.0)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--disable-validation-rewind", action="store_true")
    parser.add_argument("--validation-rewind-overlap-eps", type=float, default=0.02)
    parser.add_argument("--validation-rewind-pair-eps", type=float, default=2.0)
    parser.add_argument("--validation-rewind-patience", type=int, default=1)
    parser.add_argument("--validation-rewind-lr-decay", type=float, default=0.5)
    parser.add_argument("--validation-rewind-min-lr", type=float, default=1e-5)
    parser.add_argument("--validation-rewind-cooldown-validations", type=int, default=1)
    parser.add_argument("--validation-rewind-max-count", type=int, default=8)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--teacher-dataset", default="")
    parser.add_argument("--distill-epochs", type=int, default=0)
    parser.add_argument("--distill-batch-size", type=int, default=1)
    parser.add_argument("--distill-lr", type=float, default=1e-4)
    parser.add_argument("--distill-max-branch-pairs", type=int, default=65_536)
    parser.add_argument("--teacher-lambda0", type=float, default=1.0)
    parser.add_argument("--teacher-anneal-updates", type=int, default=50)
    parser.add_argument("--teacher-aux-batch-size", type=int, default=1)
    parser.add_argument("--teacher-aux-steps-per-update", type=int, default=1)
    parser.add_argument("--teacher-aux-lr-scale", type=float, default=0.10)
    parser.add_argument("--teacher-aux-loss-cap", type=float, default=256.0)
    parser.add_argument("--teacher-aux-weight-cap", type=float, default=0.25)
    parser.add_argument("--cleanup-aux-weight", type=float, default=1.0)
    parser.add_argument("--chooser-aux-weight", type=float, default=1.0)
    parser.add_argument("--ranking-aux-weight", type=float, default=0.5)
    parser.add_argument("--mode-selector-aux-weight", type=float, default=0.5)
    parser.add_argument("--lag-reward-coef", type=float, default=0.10)
    parser.add_argument("--lag-reward-tanh-scale", type=float, default=25.0)
    parser.add_argument("--overlap-reward-coef", type=float, default=8.0)
    parser.add_argument("--overlap-pairs-reward-coef", type=float, default=1.0)
    parser.add_argument("--overlap-regression-coef", type=float, default=16.0)
    parser.add_argument("--wirelength-reward-coef", type=float, default=0.10)
    parser.add_argument("--wirelength-gate-overlap-threshold", type=float, default=0.25)
    parser.add_argument("--wirelength-gate-pairs-threshold", type=float, default=8.0)
    parser.add_argument("--current-overlap-penalty", type=float, default=1.0)
    parser.add_argument("--current-overlap-pairs-penalty", type=float, default=0.25)
    parser.add_argument("--incumbent-overlap-gap-penalty", type=float, default=4.0)
    parser.add_argument("--incumbent-pair-gap-penalty", type=float, default=1.0)
    parser.add_argument("--incumbent-position-gap-penalty", type=float, default=0.25)
    parser.add_argument("--branch-violation-penalty", type=float, default=2.0)
    parser.add_argument("--missed-pair-penalty", type=float, default=0.01)
    parser.add_argument("--stop-gate-penalty", type=float, default=5.0)
    parser.add_argument("--stop-gate-overlap", type=float, default=0.02)
    parser.add_argument("--stop-no-progress-penalty", type=float, default=4.0)
    parser.add_argument("--soft-branch-epsilon", type=float, default=1e-4)
    parser.add_argument("--audit-missed-target", type=float, default=64.0)
    parser.add_argument("--audit-pressure-gamma", type=float, default=1.0)
    parser.add_argument("--audit-pressure-max", type=float, default=4.0)
    parser.add_argument("--hard-replay-fraction", type=float, default=0.5)
    parser.add_argument("--hard-replay-pool-size", type=int, default=32)
    parser.add_argument("--hard-replay-suite-size", type=int, default=8)
    parser.add_argument("--validation-replay-fraction", type=float, default=0.5)
    parser.add_argument("--validation-replay-topk", type=int, default=4)
    parser.add_argument("--wire-overlap-threshold", type=float, default=0.05)
    parser.add_argument("--no-soft-relax", action="store_true")
    parser.add_argument("--no-residual-flow", action="store_true")
    parser.add_argument("--no-phr-layer", action="store_true")
    parser.add_argument("--no-exact-audit", action="store_true")
    parser.add_argument("--no-density", action="store_true")
    parser.add_argument("--disable-clusters", action="store_true")
    parser.add_argument("--disable-stop", action="store_true")
    parser.add_argument("--disable-incumbent-state", action="store_true")
    parser.add_argument("--disable-incumbent-action", action="store_true")
    parser.add_argument(
        "--memory-reset-mode",
        choices=[
            "none",
            "incumbent_improve",
            "incumbent_improve_material",
            "incumbent_improve_stale",
            "incumbent_improve_material_or_stale",
        ],
        default="incumbent_improve",
    )
    parser.add_argument("--memory-reset-retain", type=float, default=0.25)
    parser.add_argument("--memory-reset-min-overlap-gain", type=float, default=0.03)
    parser.add_argument("--memory-reset-min-pair-gain-count", type=float, default=2.0)
    parser.add_argument("--memory-reset-min-steps-since-best", type=int, default=2)
    parser.add_argument("--fixed-pd-controls", action="store_true")
    parser.add_argument("--ordering-representation", choices=["sequence_pair", "dag"], default="sequence_pair")
    parser.add_argument("--branch-mode", choices=["ordering", "independent_pair"], default="ordering")
    parser.add_argument("--al-mode", choices=["signed_phr", "positive_only"], default="signed_phr")
    parser.add_argument("--relaxation", choices=["sigmoid", "neuralsort", "gumbel_sinkhorn"], default="sigmoid")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sizes", default="2:20,3:25,2:30,3:50,4:75,5:100,5:150")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log", default="training_logs/ppo_train.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.memory_reset_retain = max(0.0, min(float(args.memory_reset_retain), 1.0))
    args.memory_reset_min_overlap_gain = max(float(args.memory_reset_min_overlap_gain), 0.0)
    args.memory_reset_min_pair_gain_count = max(float(args.memory_reset_min_pair_gain_count), 0.0)
    args.memory_reset_min_steps_since_best = max(int(args.memory_reset_min_steps_since_best), 0)
    device = torch.device(args.device)
    sizes = parse_sizes(args.sizes)

    env_config = EnvConfig(
        horizon=args.horizon,
        coordinate_steps=args.coordinate_steps,
        soft_relaxation=not args.no_soft_relax,
        enable_residual_flow=not args.no_residual_flow,
        enable_phr_layer=not args.no_phr_layer,
        enable_exact_audit=not args.no_exact_audit,
        enable_density=not args.no_density,
        enable_clusters=not args.disable_clusters,
        enable_stop=not args.disable_stop,
        enable_incumbent_state=not args.disable_incumbent_state,
        enable_incumbent_action=not args.disable_incumbent_action,
        fixed_pd_controls=args.fixed_pd_controls,
        ordering_representation=args.ordering_representation,
        branch_mode=args.branch_mode,
        al_mode=args.al_mode,
        lag_reward_coef=args.lag_reward_coef,
        lag_reward_tanh_scale=args.lag_reward_tanh_scale,
        exact_overlap_reward_coef=args.overlap_reward_coef,
        exact_overlap_pairs_reward_coef=args.overlap_pairs_reward_coef,
        exact_overlap_regression_coef=args.overlap_regression_coef,
        exact_wirelength_reward_coef=args.wirelength_reward_coef,
        exact_wirelength_reward_overlap_threshold=args.wirelength_gate_overlap_threshold,
        exact_wirelength_reward_pairs_threshold=args.wirelength_gate_pairs_threshold,
        current_overlap_penalty_coef=args.current_overlap_penalty,
        current_overlap_pairs_penalty_coef=args.current_overlap_pairs_penalty,
        incumbent_overlap_gap_penalty_coef=args.incumbent_overlap_gap_penalty,
        incumbent_pair_gap_penalty_coef=args.incumbent_pair_gap_penalty,
        incumbent_position_gap_penalty_coef=args.incumbent_position_gap_penalty,
        branch_violation_penalty_coef=args.branch_violation_penalty,
        missed_pair_penalty_coef=args.missed_pair_penalty,
        stop_gate_penalty=args.stop_gate_penalty,
        stop_gate_overlap_threshold=args.stop_gate_overlap,
        stop_no_progress_penalty=args.stop_no_progress_penalty,
        soft_branch_epsilon=args.soft_branch_epsilon,
        audit_missed_target=args.audit_missed_target,
        audit_pressure_gamma=args.audit_pressure_gamma,
        audit_pressure_max=args.audit_pressure_max,
    )
    _checkpoint = {}
    if args.resume_checkpoint:
        policy, _checkpoint = load_policy_checkpoint(args.resume_checkpoint, device)
    else:
        policy = OrderingPolicy(
            hidden_dim=args.hidden_dim,
            message_passes=args.message_passes,
            num_clusters=args.num_clusters,
            global_flow_rank=args.global_flow_rank,
            enable_incumbent_controls=not args.disable_incumbent_action,
        ).to(device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resume_update_offset = int(_checkpoint.get("stats", {}).get("update", -1)) + 1 if isinstance(_checkpoint, dict) else 0

    teacher_dataset_path = args.teacher_dataset
    allow_resumed_teacher_dataset = max(int(args.teacher_aux_steps_per_update), 0) > 0
    if not teacher_dataset_path and allow_resumed_teacher_dataset and isinstance(_checkpoint, dict):
        resumed_teacher_path = _checkpoint.get("config", {}).get("teacher_dataset_path", "")
        if resumed_teacher_path and os.path.exists(resumed_teacher_path):
            teacher_dataset_path = resumed_teacher_path
    teacher_metadata = {}
    teacher_dataset_version = 0
    teacher_dataset = []
    teacher_distill_cfg = None
    distill_stats = {
        "teacher_samples": 0,
        "teacher_lambda_initial": 0.0,
        "teacher_lambda_final": 0.0,
        "dagger_correction_count": 0,
    }
    if teacher_dataset_path:
        from distill import DistillConfig, outcome_distill, teacher_auxiliary_update, teacher_lambda_at
        from teacher_data import load_teacher_dataset_payload

        dataset_payload = load_teacher_dataset_payload(teacher_dataset_path)
        teacher_dataset = list(dataset_payload["samples"])
        teacher_metadata = dict(dataset_payload.get("metadata", {}))
        teacher_dataset_version = int(dataset_payload.get("version", 0))
        teacher_distill_cfg = DistillConfig(
            epochs=max(int(args.distill_epochs), 1),
            batch_size=args.distill_batch_size,
            lr=args.distill_lr,
            max_branch_pairs_per_sample=args.distill_max_branch_pairs,
            relaxation=args.relaxation,
            soft_tau=args.soft_tau_start,
            teacher_aux_lr_scale=args.teacher_aux_lr_scale,
            teacher_aux_loss_cap=args.teacher_aux_loss_cap,
            teacher_aux_weight_cap=args.teacher_aux_weight_cap,
            seed=args.seed,
        )
        if args.teacher_dataset:
            distill_stats = outcome_distill(policy, teacher_dataset, teacher_distill_cfg, device=device)
            save_policy_checkpoint(
                policy,
                checkpoint_dir / "outcome_distilled_warmstart.pt",
                config={
                    "distill": teacher_distill_cfg.__dict__,
                    "teacher_dataset_path": teacher_dataset_path,
                    "teacher_dataset_version": teacher_dataset_version,
                    "teacher_metadata": teacher_metadata,
                    "train": vars(args),
                },
                stats=distill_stats,
            )
            with open(checkpoint_dir / "outcome_distill_stats.json", "w", encoding="utf-8") as handle:
                json.dump(distill_stats, handle, sort_keys=True, indent=2)
            with open(checkpoint_dir / "teacher_transfer_manifest.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "teacher_dataset_path": teacher_dataset_path,
                        "teacher_dataset_version": teacher_dataset_version,
                        "teacher_metadata": teacher_metadata,
                        "distill_stats": distill_stats,
                    },
                    handle,
                    sort_keys=True,
                    indent=2,
                )
        elif isinstance(_checkpoint, dict):
            resume_stats = _checkpoint.get("stats", {})
            for key in ("teacher_samples", "teacher_lambda_initial", "teacher_lambda_final", "dagger_correction_count"):
                if key in resume_stats:
                    distill_stats[key] = resume_stats[key]
    elif isinstance(_checkpoint, dict):
        resume_stats = _checkpoint.get("stats", {})
        for key in ("teacher_samples", "teacher_lambda_initial", "teacher_lambda_final", "dagger_correction_count"):
            if key in resume_stats:
                distill_stats[key] = resume_stats[key]
        teacher_metadata = _checkpoint.get("config", {}).get("teacher_metadata", {})
        teacher_dataset_version = int(_checkpoint.get("config", {}).get("teacher_dataset_version", 0) or 0)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    skipped_shape_keys = list(_checkpoint.get("_load_skipped_shape_keys", [])) if isinstance(_checkpoint, dict) else []
    if isinstance(_checkpoint, dict) and "optimizer_state" in _checkpoint and not skipped_shape_keys:
        try:
            optimizer.load_state_dict(_checkpoint["optimizer_state"])
        except (ValueError, RuntimeError):
            pass
        set_optimizer_lr(optimizer, args.lr)
    checkpoint_config = {
        **vars(args),
        "teacher_dataset_path": teacher_dataset_path,
        "teacher_dataset_version": teacher_dataset_version,
        "teacher_metadata": teacher_metadata,
    }

    best_exact_overlap = float("inf")
    best_exact_overlap_pairs = float("inf")
    best_exact_wirelength = float("inf")
    best_lex_overlap = float("inf")
    best_lex_wl = float("inf")
    best_wire_under_threshold = float("inf")
    best_reward = -float("inf")
    hardening_state = {"best_overlap": None, "bad_windows": 0, "has_baseline": False}
    soft_tau_state = float(args.soft_tau_start)
    best_training_snapshot = None
    rewind_state = {"bad_windows": 0, "rewind_count": 0, "cooldown_until_update": -1}
    validation_suite_seed = args.seed + 1_000_000
    validation_suite = build_validation_suite(sizes, args.validation_episodes, validation_suite_seed)
    initial_temperature = (
        metric_gated_temperature(
            soft_tau_state,
            start=args.temperature_start,
            end=args.temperature_end,
        )
        if args.metric_gated_hardening
        else temperature_at(
            0,
            args.updates,
            start=args.temperature_start,
            end=args.temperature_end,
        )
    )
    initial_soft_tau = soft_tau_state if not args.no_soft_relax else None
    hard_replay_seed_base = args.seed + 2_000_000
    hard_replay_suite = build_hard_replay_suite(
        policy,
        sizes,
        env_config,
        device,
        seed_base=hard_replay_seed_base,
        temperature=initial_temperature,
        soft_tau=initial_soft_tau,
        relaxation=args.relaxation,
        pool_size=args.hard_replay_pool_size,
        suite_size=args.hard_replay_suite_size,
        memory_reset_mode=args.memory_reset_mode,
        memory_reset_retain=args.memory_reset_retain,
        memory_reset_min_overlap_gain=args.memory_reset_min_overlap_gain,
        memory_reset_min_pair_gain_count=args.memory_reset_min_pair_gain_count,
        memory_reset_min_steps_since_best=args.memory_reset_min_steps_since_best,
    )
    validation_replay_suite = []
    started = time.time()

    for update_idx in range(args.updates):
        if args.metric_gated_hardening:
            use_soft = not args.no_soft_relax
            soft_tau = soft_tau_state if use_soft else None
            temperature = metric_gated_temperature(
                soft_tau_state,
                start=args.temperature_start,
                end=args.temperature_end,
            )
        else:
            temperature = temperature_at(
                update_idx,
                args.updates,
                start=args.temperature_start,
                end=args.temperature_end,
            )
            soft_cutoff = int(args.updates * args.soft_relax_frac)
            use_soft = (not args.no_soft_relax) and update_idx < soft_cutoff
            soft_tau = soft_tau_at(update_idx, max(soft_cutoff, 1), args.soft_tau_start, args.soft_tau_end) if use_soft else None
        transitions = []
        episode_infos = []
        validation_replay_episodes = 0
        if validation_replay_suite:
            validation_replay_episodes = min(
                int(round(args.episodes_per_update * max(float(args.validation_replay_fraction), 0.0))),
                int(args.episodes_per_update),
            )
        hard_replay_episodes = 0
        if hard_replay_suite:
            hard_replay_episodes = min(
                int(round(args.episodes_per_update * max(float(args.hard_replay_fraction), 0.0))),
                max(int(args.episodes_per_update) - validation_replay_episodes, 0),
            )
        for episode_idx in range(args.episodes_per_update):
            forced_size = None
            if episode_idx < validation_replay_episodes:
                val_spec = validation_replay_suite[
                    (update_idx * max(validation_replay_episodes, 1) + episode_idx) % len(validation_replay_suite)
                ]
                seed = int(val_spec["seed"])
                forced_size = tuple(val_spec["size"])
                discover_mode = str(val_spec.get("discover_mode", "balanced"))
            elif episode_idx < validation_replay_episodes + hard_replay_episodes:
                hard_idx = episode_idx - validation_replay_episodes
                hard_spec = hard_replay_suite[
                    (update_idx * max(hard_replay_episodes, 1) + hard_idx) % len(hard_replay_suite)
                ]
                seed = int(hard_spec["seed"])
                forced_size = tuple(hard_spec["size"])
                discover_mode = str(hard_spec.get("discover_mode", "balanced"))
            else:
                seed = args.seed + update_idx * 10_000 + episode_idx
                discover_mode = DISCOVER_MODE_NAMES[
                    (update_idx * max(int(args.episodes_per_update), 1) + episode_idx) % len(DISCOVER_MODE_NAMES)
                ]
            episode_transitions, info = collect_episode(
                policy,
                sizes,
                env_config,
                device,
                seed,
                temperature,
                soft_tau=soft_tau,
                relaxation=args.relaxation,
                forced_size=forced_size,
                discover_mode=discover_mode,
                memory_reset_mode=args.memory_reset_mode,
                memory_reset_retain=args.memory_reset_retain,
                memory_reset_min_overlap_gain=args.memory_reset_min_overlap_gain,
                memory_reset_min_pair_gain_count=args.memory_reset_min_pair_gain_count,
                memory_reset_min_steps_since_best=args.memory_reset_min_steps_since_best,
            )
            transitions.extend(episode_transitions)
            episode_infos.append(info)

        metrics = ppo_update(
            policy,
            optimizer,
            transitions,
            update_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            equivariance_coef=args.equivariance_coef,
        )
        teacher_lambda_current = 0.0
        teacher_aux_stats = {}
        if teacher_dataset and teacher_distill_cfg is not None:
            teacher_lambda_current = teacher_lambda_at(
                resume_update_offset + update_idx,
                lambda0=args.teacher_lambda0,
                anneal_updates=args.teacher_anneal_updates,
            )
            aux_rows = []
            for aux_step in range(max(int(args.teacher_aux_steps_per_update), 0)):
                aux_rows.append(
                    teacher_auxiliary_update(
                        policy,
                        teacher_dataset,
                        teacher_distill_cfg,
                        optimizer=optimizer,
                        lambda_teacher=teacher_lambda_current,
                        batch_size=args.teacher_aux_batch_size,
                        device=device,
                        seed=args.seed + (resume_update_offset + update_idx) * 10_000 + aux_step,
                    )
                )
            teacher_aux_stats = mean_numeric_dicts(aux_rows)

        avg_overlap = sum(info["best_overlap"] for info in episode_infos) / len(episode_infos)
        avg_wl = sum(info["best_wl"] for info in episode_infos) / len(episode_infos)
        avg_reward = sum(info["last_reward"] for info in episode_infos) / len(episode_infos)
        avg_density = sum(info["density_overflow"] for info in episode_infos) / len(episode_infos)
        avg_branch_violation = sum(info["branch_violation"] for info in episode_infos) / len(episode_infos)
        avg_boundary_violation = sum(info["boundary_violation"] for info in episode_infos) / len(episode_infos)
        avg_lag_before = sum(info["lag_before"] for info in episode_infos) / len(episode_infos)
        avg_lag_after = sum(info["lag_after"] for info in episode_infos) / len(episode_infos)
        avg_pd_steps = sum(info["pd_steps"] for info in episode_infos) / len(episode_infos)
        avg_rho = sum(info["rho"] for info in episode_infos) / len(episode_infos)
        avg_eta = sum(info["eta"] for info in episode_infos) / len(episode_infos)
        avg_alpha = sum(info["alpha"] for info in episode_infos) / len(episode_infos)
        avg_step_scale = sum(info["step_scale"] for info in episode_infos) / len(episode_infos)
        avg_pair_emphasis = sum(info["pair_emphasis"] for info in episode_infos) / len(episode_infos)
        avg_tau = sum(info["tau"] for info in episode_infos) / len(episode_infos)
        avg_branch_pressure = sum(info["branch_pressure"] for info in episode_infos) / len(episode_infos)
        avg_density_pressure = sum(info["density_pressure"] for info in episode_infos) / len(episode_infos)
        avg_boundary_pressure = sum(info["boundary_pressure"] for info in episode_infos) / len(episode_infos)
        avg_missed_pairs = sum(info["missed_pairs"] for info in episode_infos) / len(episode_infos)
        avg_exact_overlap_pairs = sum(info.get("exact_overlap_pairs", 0) for info in episode_infos) / len(episode_infos)
        avg_best_exact_overlap_pairs = sum(info.get("best_exact_overlap_pairs", 0) for info in episode_infos) / len(episode_infos)
        avg_sampled_pairs = sum(info["sampled_pairs"] for info in episode_infos) / len(episode_infos)
        avg_cluster_pairs = sum(info["cluster_pairs"] for info in episode_infos) / len(episode_infos)
        avg_uncertain_pairs = sum(info["uncertain_pairs"] for info in episode_infos) / len(episode_infos)
        avg_new_active_pairs = sum(info["new_active_pairs"] for info in episode_infos) / len(episode_infos)
        avg_retained_pairs = sum(info["retained_pairs"] for info in episode_infos) / len(episode_infos)
        avg_hard_pair_age_mean = sum(info.get("hard_pair_age_mean", 0.0) for info in episode_infos) / len(episode_infos)
        avg_hard_pair_age_max = sum(info.get("hard_pair_age_max", 0.0) for info in episode_infos) / len(episode_infos)
        avg_hard_pair_age_min = sum(info.get("hard_pair_age_min", 0.0) for info in episode_infos) / len(episode_infos)
        avg_audit_pressure_scale = sum(info.get("audit_pressure_scale", 1.0) for info in episode_infos) / len(episode_infos)
        avg_audit_pressure_target = sum(info.get("audit_pressure_target", 0.0) for info in episode_infos) / len(episode_infos)
        avg_retention_horizon = sum(info.get("retention_horizon", 0.0) for info in episode_infos) / len(episode_infos)
        stop_rate = sum(1.0 if info["stop"] else 0.0 for info in episode_infos) / len(episode_infos)
        avg_stop_probability = sum(info.get("stop_probability", 0.0) for info in episode_infos) / len(episode_infos)
        stop_gated_rate = sum(1.0 if info.get("stop_gated", False) else 0.0 for info in episode_infos) / len(episode_infos)
        false_stop_rate = sum(1.0 if info.get("false_stop", False) else 0.0 for info in episode_infos) / len(episode_infos)
        avg_stop_overlap = sum(info.get("stop_overlap", 0.0) for info in episode_infos) / len(episode_infos)
        avg_residual_norm = sum(info["residual_norm"] for info in episode_infos) / len(episode_infos)
        avg_memory_reset_count = sum(info.get("memory_reset_count", 0.0) for info in episode_infos) / len(episode_infos)
        memory_reset_rate = sum(1.0 if info.get("memory_reset_applied", False) else 0.0 for info in episode_infos) / len(episode_infos)
        phase_rollup = mean_numeric_dicts(
            [
                {key: value for key, value in info.items() if key.startswith("phase_")}
                for info in episode_infos
            ]
        )
        record = {
            "update": update_idx,
            "temperature": temperature,
            "soft_tau": soft_tau,
            "soft_relaxation": use_soft,
            "relaxation": args.relaxation,
            "ordering_representation": args.ordering_representation,
            "branch_mode": args.branch_mode,
            "al_mode": args.al_mode,
            "enable_clusters": env_config.enable_clusters,
            "enable_stop": env_config.enable_stop,
            "episodes": len(episode_infos),
            "transitions": len(transitions),
            "avg_overlap": avg_overlap,
            "avg_wirelength": avg_wl,
            "avg_reward": avg_reward,
            "avg_density_overflow": avg_density,
            "avg_branch_violation": avg_branch_violation,
            "avg_boundary_violation": avg_boundary_violation,
            "avg_lag_before": avg_lag_before,
            "avg_lag_after": avg_lag_after,
            "avg_pd_steps": avg_pd_steps,
            "avg_rho": avg_rho,
            "avg_eta": avg_eta,
            "avg_alpha": avg_alpha,
            "avg_step_scale": avg_step_scale,
            "avg_pair_emphasis": avg_pair_emphasis,
            "avg_tau": avg_tau,
            "avg_branch_pressure": avg_branch_pressure,
            "avg_density_pressure": avg_density_pressure,
            "avg_boundary_pressure": avg_boundary_pressure,
            "avg_missed_pairs": avg_missed_pairs,
            "avg_exact_overlap_pairs": avg_exact_overlap_pairs,
            "avg_best_exact_overlap_pairs": avg_best_exact_overlap_pairs,
            "current_lr": optimizer_lr(optimizer),
            "avg_sampled_pairs": avg_sampled_pairs,
            "avg_cluster_pairs": avg_cluster_pairs,
            "avg_uncertain_pairs": avg_uncertain_pairs,
            "avg_new_active_pairs": avg_new_active_pairs,
            "avg_retained_pairs": avg_retained_pairs,
            "avg_hard_pair_age_mean": avg_hard_pair_age_mean,
            "avg_hard_pair_age_max": avg_hard_pair_age_max,
            "avg_hard_pair_age_min": avg_hard_pair_age_min,
            "avg_audit_pressure_scale": avg_audit_pressure_scale,
            "avg_audit_pressure_target": avg_audit_pressure_target,
            "avg_retention_horizon": avg_retention_horizon,
            "stop_rate": stop_rate,
            "avg_stop_probability": avg_stop_probability,
            "stop_gated_rate": stop_gated_rate,
            "false_stop_rate": false_stop_rate,
            "avg_stop_overlap": avg_stop_overlap,
            "avg_residual_norm": avg_residual_norm,
            "avg_memory_reset_count": avg_memory_reset_count,
            "memory_reset_rate": memory_reset_rate,
            "memory_reset_mode": args.memory_reset_mode,
            "memory_reset_retain": args.memory_reset_retain,
            "memory_reset_min_overlap_gain": args.memory_reset_min_overlap_gain,
            "memory_reset_min_pair_gain_count": args.memory_reset_min_pair_gain_count,
            "memory_reset_min_steps_since_best": args.memory_reset_min_steps_since_best,
            "validation_rewind_enabled": not args.disable_validation_rewind,
            "validation_rewind_overlap_eps": args.validation_rewind_overlap_eps,
            "validation_rewind_pair_eps": args.validation_rewind_pair_eps,
            "validation_rewind_patience": args.validation_rewind_patience,
            "validation_rewind_lr_decay": args.validation_rewind_lr_decay,
            "validation_rewind_min_lr": args.validation_rewind_min_lr,
            "validation_rewind_cooldown_validations": args.validation_rewind_cooldown_validations,
            "validation_rewind_max_count": args.validation_rewind_max_count,
            "validation_suite_seed": validation_suite_seed,
            "validation_suite_episodes": len(validation_suite),
            "validation_fixed_suite": True,
            "validation_replay_fraction": max(float(args.validation_replay_fraction), 0.0) if validation_replay_suite else 0.0,
            "validation_replay_topk": int(args.validation_replay_topk),
            "validation_replay_suite_size": len(validation_replay_suite),
            "validation_replay_episodes": validation_replay_episodes,
            "hard_replay_fraction": max(float(args.hard_replay_fraction), 0.0) if hard_replay_suite else 0.0,
            "hard_replay_pool_size": int(args.hard_replay_pool_size),
            "hard_replay_suite_size": len(hard_replay_suite),
            "hard_replay_episodes": hard_replay_episodes,
            "elapsed": time.time() - started,
            **build_teacher_transfer_record(
                args,
                distill_stats,
                teacher_metadata,
                dataset_path=teacher_dataset_path,
                dataset_version=teacher_dataset_version,
                teacher_lambda=teacher_lambda_current,
            ),
            **metrics,
        }
        record.update(teacher_aux_stats)
        record.update(phase_rollup)
        record["teacher_update_index"] = resume_update_offset + update_idx
        validation = None
        validation_rows = None
        aux_stats = {}
        if args.validation_interval > 0 and update_idx % args.validation_interval == 0:
            validation, validation_rows = validate_policy(
                policy,
                sizes,
                env_config,
                device,
                validation_suite_seed,
                temperature,
                soft_tau=soft_tau,
                relaxation=args.relaxation,
                episodes=args.validation_episodes,
                validation_suite=validation_suite,
                memory_reset_mode=args.memory_reset_mode,
                memory_reset_retain=args.memory_reset_retain,
                memory_reset_min_overlap_gain=args.memory_reset_min_overlap_gain,
                memory_reset_min_pair_gain_count=args.memory_reset_min_pair_gain_count,
                memory_reset_min_steps_since_best=args.memory_reset_min_steps_since_best,
                return_rows=True,
            )
            record.update(validation)
            aux_stats = auxiliary_supervision_update(
                policy,
                optimizer,
                validation_rows,
                cleanup_weight=float(args.cleanup_aux_weight),
                chooser_weight=float(args.chooser_aux_weight),
                ranking_weight=float(args.ranking_aux_weight),
                mode_weight=float(args.mode_selector_aux_weight),
            )
            validation_replay_suite = build_validation_replay_suite(
                validation_rows,
                suite_size=args.validation_replay_topk,
            )
        record.update(aux_stats)
        if args.metric_gated_hardening:
            hardening_source = "hold"
            hardening_overlap_improved = False
            hardening_stable = False
            hardening_exact_pairs_stable = False
            if validation is not None:
                gate_overlap = record["validation_overlap"]
                gate_branch = record["validation_branch_violation"]
                gate_missed = record["validation_missed_pairs"]
                gate_exact_pairs = record["validation_exact_overlap_pairs"]
                prior_best_overlap = hardening_state["best_overlap"]
                prior_has_baseline = bool(hardening_state.get("has_baseline", False))
                hardening_overlap_improved = (
                    prior_has_baseline
                    and prior_best_overlap is not None
                    and gate_overlap < prior_best_overlap - args.hardening_overlap_eps
                )
                hardening_exact_pairs_stable = gate_exact_pairs <= args.hardening_exact_pairs_max
                hardening_stable = (
                    gate_branch <= args.hardening_branch_vmax
                    and gate_missed <= args.hardening_missed_max
                    and hardening_exact_pairs_stable
                )
                hardening_source = "validation" if prior_has_baseline else "baseline"
                soft_tau_state = update_metric_gated_tau(
                    soft_tau_state,
                    gate_overlap,
                    gate_branch,
                    gate_missed,
                    gate_exact_pairs,
                    hardening_state,
                    tau_min=args.soft_tau_end,
                    tau_max=args.tau_max,
                    gamma_down=args.tau_down,
                    gamma_up=args.tau_up,
                    overlap_epsilon=args.hardening_overlap_eps,
                    branch_violation_max=args.hardening_branch_vmax,
                    missed_pairs_max=args.hardening_missed_max,
                    exact_overlap_pairs_max=args.hardening_exact_pairs_max,
                    patience=args.hardening_patience,
                )
            record["next_soft_tau"] = soft_tau_state
            record["hardening_best_overlap"] = hardening_state["best_overlap"]
            record["hardening_bad_windows"] = hardening_state["bad_windows"]
            record["hardening_has_baseline"] = bool(hardening_state.get("has_baseline", False))
            record["hardening_source"] = hardening_source
            record["hardening_overlap_improved"] = hardening_overlap_improved
            record["hardening_stable"] = hardening_stable
            record["hardening_exact_pairs_stable"] = hardening_exact_pairs_stable
        authority_metrics_available = validation is not None
        record["checkpoint_metric_source"] = "validation" if validation else "held"
        record["checkpoint_metric_overlap"] = record.get("validation_overlap")
        record["checkpoint_metric_wirelength"] = record.get("validation_wirelength")
        record["checkpoint_metric_exact_overlap_pairs"] = record.get("validation_exact_overlap_pairs")
        record["reward_misaligned"] = bool(
            authority_metrics_available
            and avg_reward > best_reward
            and record["checkpoint_metric_overlap"] is not None
            and record["checkpoint_metric_overlap"] > best_exact_overlap
        )
        metric_overlap = record["checkpoint_metric_overlap"]
        metric_wl = record["checkpoint_metric_wirelength"]
        metric_pairs = record["checkpoint_metric_exact_overlap_pairs"]

        is_new_best_exact = bool(
            authority_metrics_available
            and (metric_overlap, metric_pairs, metric_wl) < (best_exact_overlap, best_exact_overlap_pairs, best_exact_wirelength)
        )
        if authority_metrics_available and not is_new_best_exact:
            rewind_state["bad_windows"] = (
                rewind_state["bad_windows"] + 1
                if should_apply_validation_rewind(
                    metric_overlap,
                    metric_pairs,
                    best_exact_overlap,
                    best_exact_overlap_pairs,
                    overlap_epsilon=args.validation_rewind_overlap_eps,
                    pair_epsilon=args.validation_rewind_pair_eps,
                )
                else 0
            )

        rewind_applied = False
        rewind_lr_before = optimizer_lr(optimizer)
        rewind_lr_after = rewind_lr_before
        rewind_rebuilt_hard_replay = False
        rewind_reason = ""
        rewind_target_overlap = best_exact_overlap
        rewind_target_pairs = best_exact_overlap_pairs
        record["validation_regressed_vs_best"] = bool(
            authority_metrics_available
            and should_apply_validation_rewind(
                metric_overlap,
                metric_pairs,
                best_exact_overlap,
                best_exact_overlap_pairs,
                overlap_epsilon=args.validation_rewind_overlap_eps,
                pair_epsilon=args.validation_rewind_pair_eps,
            )
        )
        record["validation_rewind_bad_windows"] = rewind_state["bad_windows"]
        record["validation_rewind_count"] = rewind_state["rewind_count"]
        record["validation_rewind_cooldown_until_update"] = rewind_state["cooldown_until_update"]
        record["validation_rewind_applied"] = False
        record["validation_rewind_reason"] = ""
        record["validation_rewind_lr_before"] = rewind_lr_before
        record["validation_rewind_lr_after"] = rewind_lr_after
        record["validation_rewind_target_overlap"] = rewind_target_overlap
        record["validation_rewind_target_pairs"] = rewind_target_pairs
        record["validation_rewind_rebuilt_hard_replay"] = False

        if avg_reward > best_reward:
            best_reward = avg_reward
            save_policy_checkpoint(policy, checkpoint_dir / "shaped_reward_debug.pt", config=checkpoint_config, stats=record)
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_best_reward.pt", config=checkpoint_config, stats=record)
        if is_new_best_exact:
            best_exact_overlap = metric_overlap
            best_exact_overlap_pairs = metric_pairs
            best_exact_wirelength = metric_wl
            best_training_snapshot = snapshot_training_state(
                policy,
                optimizer,
                hardening_state,
                soft_tau_state,
            )
            rewind_state["bad_windows"] = 0
            save_policy_checkpoint(
                policy,
                checkpoint_dir / "best_exact_overlap.pt",
                config=checkpoint_config,
                stats=record,
                extra={"optimizer_state": optimizer.state_dict()},
            )
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_best_overlap.pt", config=checkpoint_config, stats=record)
        if authority_metrics_available and (metric_overlap, metric_wl) < (best_lex_overlap, best_lex_wl):
            best_lex_overlap, best_lex_wl = metric_overlap, metric_wl
            save_policy_checkpoint(policy, checkpoint_dir / "best_lexicographic.pt", config=checkpoint_config, stats=record)
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_best_validation.pt", config=checkpoint_config, stats=record)
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy.pt", config=checkpoint_config, stats=record)
        if authority_metrics_available and metric_overlap <= args.wire_overlap_threshold and metric_wl < best_wire_under_threshold:
            best_wire_under_threshold = metric_wl
            save_policy_checkpoint(policy, checkpoint_dir / "best_wire_given_overlap_threshold.pt", config=checkpoint_config, stats=record)

        rewind_allowed = (
            authority_metrics_available
            and not args.disable_validation_rewind
            and best_training_snapshot is not None
            and not is_new_best_exact
            and rewind_state["rewind_count"] < int(args.validation_rewind_max_count)
            and update_idx >= int(rewind_state["cooldown_until_update"])
            and rewind_state["bad_windows"] >= int(args.validation_rewind_patience)
            and record["validation_regressed_vs_best"]
        )
        if rewind_allowed:
            save_policy_checkpoint(
                policy,
                checkpoint_dir / "latest_pre_rewind.pt",
                config=checkpoint_config,
                stats=record,
                extra={"optimizer_state": optimizer.state_dict()},
            )
            hardening_state, soft_tau_state = restore_training_state(
                policy,
                optimizer,
                best_training_snapshot,
            )
            rewind_lr_before, rewind_lr_after = decay_optimizer_lr(
                optimizer,
                args.validation_rewind_lr_decay,
                args.validation_rewind_min_lr,
            )
            rewind_state["rewind_count"] += 1
            rewind_state["bad_windows"] = 0
            rewind_state["cooldown_until_update"] = update_idx + max(
                int(args.validation_interval),
                1,
            ) * max(int(args.validation_rewind_cooldown_validations), 1)
            rewind_applied = True
            rewind_reason = "validation_regression"
            rewind_target_overlap = best_exact_overlap
            rewind_target_pairs = best_exact_overlap_pairs
            if hard_replay_suite:
                rewind_temperature = (
                    metric_gated_temperature(
                        soft_tau_state,
                        start=args.temperature_start,
                        end=args.temperature_end,
                    )
                    if args.metric_gated_hardening
                    else temperature
                )
                rewind_soft_tau = None if args.no_soft_relax else soft_tau_state
                hard_replay_suite = build_hard_replay_suite(
                    policy,
                    sizes,
                    env_config,
                    device,
                    seed_base=hard_replay_seed_base,
                    temperature=rewind_temperature,
                    soft_tau=rewind_soft_tau,
                    relaxation=args.relaxation,
                    pool_size=args.hard_replay_pool_size,
                    suite_size=args.hard_replay_suite_size,
                    memory_reset_mode=args.memory_reset_mode,
                    memory_reset_retain=args.memory_reset_retain,
                    memory_reset_min_overlap_gain=args.memory_reset_min_overlap_gain,
                    memory_reset_min_pair_gain_count=args.memory_reset_min_pair_gain_count,
                    memory_reset_min_steps_since_best=args.memory_reset_min_steps_since_best,
                )
                rewind_rebuilt_hard_replay = True

        record["validation_rewind_applied"] = rewind_applied
        record["validation_rewind_reason"] = rewind_reason
        record["validation_rewind_lr_before"] = rewind_lr_before
        record["validation_rewind_lr_after"] = rewind_lr_after
        record["validation_rewind_target_overlap"] = rewind_target_overlap
        record["validation_rewind_target_pairs"] = rewind_target_pairs
        record["validation_rewind_rebuilt_hard_replay"] = rewind_rebuilt_hard_replay
        record["validation_rewind_count"] = rewind_state["rewind_count"]
        record["validation_rewind_bad_windows"] = rewind_state["bad_windows"]
        record["validation_rewind_cooldown_until_update"] = rewind_state["cooldown_until_update"]
        record["current_lr"] = optimizer_lr(optimizer)
        if args.metric_gated_hardening:
            record["next_soft_tau"] = soft_tau_state
            record["hardening_best_overlap"] = hardening_state["best_overlap"]
            record["hardening_bad_windows"] = hardening_state["bad_windows"]
            record["hardening_has_baseline"] = bool(hardening_state.get("has_baseline", False))

        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        latest_path = checkpoint_dir / "latest.pt"
        save_policy_checkpoint(
            policy,
            latest_path,
            config=checkpoint_config,
            stats=record,
            extra={"optimizer_state": optimizer.state_dict()},
        )
        save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_latest.pt", config=checkpoint_config, stats=record)


if __name__ == "__main__":
    main()
