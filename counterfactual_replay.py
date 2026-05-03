"""Counterfactual replay analysis for same-case, same-checkpoint placement rollouts.

This tool isolates one factor at a time by separating two causal regimes:

1. Action-locked transition replays:
   - same case
   - same checkpoint
   - same sampled action sequence
   - vary only transition-side factors:
     * PHR budget
     * active-set completeness
     * horizon cutoff

2. Seed-locked policy resampling:
   - same case
   - same checkpoint
   - same per-step sampling seeds
   - vary only memory handling

Memory is a policy-generation factor, not a transition-side factor. Replaying the
same stored actions with different memory has no effect on the environment. This
tool therefore reports both:
   - a memory invariance proof under fixed actions
   - a meaningful memory counterfactual with seed-locked policy resampling
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from active_set import canonicalize_pairs
from constraints import exact_overlap_pairs, make_all_pairs
from env import PlacementOrderingEnv
from ordering_policy import (
    PD_K_CHOICES,
    apply_rollout_memory_policy,
    hierarchical_active_branch_weights,
    load_policy_checkpoint,
    rollout_memory_config_from_checkpoint,
)
from ppo import detach_action
from visualize_rollout_trace import (
    ACTIVE_PAIR_FIELDNAMES,
    BOUNDARY_DUAL_FIELDNAMES,
    DENSITY_DUAL_FIELDNAMES,
    PHR_COORDINATE_FIELDNAMES,
    PHR_STEP_FIELDNAMES,
    append_cluster_logit_rows,
    append_full_pair_matrix_rows,
    append_memory_rows,
    append_ordering_score_rows,
    append_pair_branch_rows,
    append_pair_lifecycle_rows,
    append_phr_trace_rows,
    append_policy_trace_row,
    age_lookup_from_pairs,
    branch_lookup_from_pairs,
    build_case_from_spec,
    build_case_specs,
    candidate_key,
    collect_active_pair_dual_rows,
    collect_boundary_dual_rows,
    collect_density_dual_rows,
    compute_base_centers,
    env_config_from_checkpoint,
    json_ready,
    save_frame,
    save_timeline_plot,
    stage_snapshot,
    write_csv,
    write_jsonl,
)


VARIANT_STEP_FIELDNAMES = [
    "variant_family",
    "variant_name",
    "variant_mode",
    "step",
    "phase",
    "phase_before",
    "phase_request",
    "phase_transition",
    "phase_transition_reason",
    "sample_seed",
    "action_source",
    "done",
    "reward",
    "lag_before",
    "lag_after",
    "active_pairs_before",
    "active_pairs_after",
    "missed_pairs",
    "inactive_missed_pairs",
    "exact_overlap_pairs_after",
    "audit_pressure_scale",
    "audit_pressure_target",
    "step_scale",
    "incumbent_mix",
    "pair_emphasis",
    "tau",
    "rho",
    "eta",
    "alpha",
    "pd_steps",
    "residual_norm",
    "branch_violation",
    "boundary_violation",
    "density_overflow",
    "winning_refine_variant",
    "refine_window_size",
    "refine_variant_accepted",
    "refine_variant_repair_legal",
    "refine_variant_overlap_delta",
    "refine_variant_pair_delta",
    "refine_variant_wire_delta",
    "memory_reset_applied",
    "memory_reset_retain",
    "pre_overlap_ratio",
    "base_overlap_ratio",
    "post_overlap_ratio",
    "best_overlap_ratio",
    "pre_num_overlap_pairs",
    "base_num_overlap_pairs",
    "post_num_overlap_pairs",
    "best_num_overlap_pairs",
    "pre_normalized_wl",
    "base_normalized_wl",
    "post_normalized_wl",
    "best_normalized_wl",
    "delta_vs_baseline_post_overlap_ratio",
    "delta_vs_baseline_best_overlap_ratio",
    "delta_vs_baseline_post_num_overlap_pairs",
    "delta_vs_baseline_best_num_overlap_pairs",
    "delta_vs_baseline_post_normalized_wl",
    "delta_vs_baseline_best_normalized_wl",
]

VARIANT_COORDINATE_FIELDNAMES = [
    "variant_family",
    "variant_name",
    "variant_mode",
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

VARIANT_OVERLAP_FIELDNAMES = [
    "variant_family",
    "variant_name",
    "variant_mode",
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

VARIANT_SUMMARY_FIELDNAMES = [
    "variant_family",
    "variant_name",
    "variant_mode",
    "action_lock",
    "skip_reason",
    "steps_recorded",
    "best_source_step",
    "final_overlap_ratio",
    "final_num_overlap_pairs",
    "final_normalized_wl",
    "best_overlap_ratio",
    "best_num_overlap_pairs",
    "best_normalized_wl",
    "delta_vs_baseline_final_overlap_ratio",
    "delta_vs_baseline_final_num_overlap_pairs",
    "delta_vs_baseline_final_normalized_wl",
    "delta_vs_baseline_best_overlap_ratio",
    "delta_vs_baseline_best_num_overlap_pairs",
    "delta_vs_baseline_best_normalized_wl",
    "first_step_post_overlap_worse_than_baseline",
    "first_step_post_overlap_better_than_baseline",
    "first_step_best_overlap_worse_than_baseline",
    "first_step_best_overlap_better_than_baseline",
    "memory_mode",
    "phr_pd_steps_override",
    "active_set_mode",
    "horizon_cutoff",
]

ACTION_DELTA_FIELDNAMES = [
    "variant_family",
    "variant_name",
    "variant_mode",
    "step",
    "sample_seed",
    "memory_mode",
    "seq_plus_equal",
    "seq_minus_equal",
    "cluster_ids_equal",
    "stop_equal",
    "k_index_equal",
    "residual_flow_l2",
    "plus_scores_l2",
    "minus_scores_l2",
    "step_scale_delta",
    "rho_delta",
    "eta_delta",
    "alpha_delta",
    "branch_pressure_delta",
    "density_pressure_delta",
    "boundary_pressure_delta",
    "pair_emphasis_delta",
    "tau_delta",
    "incumbent_mix_delta",
    "memory_pre_l2",
    "memory_post_l2",
]

MEMORY_INVARIANCE_FIELDNAMES = [
    "memory_mode",
    "step",
    "sample_seed",
    "same_post_centers",
    "same_post_overlap_ratio",
    "same_post_num_overlap_pairs",
    "same_post_normalized_wl",
    "same_best_overlap_ratio",
    "same_best_num_overlap_pairs",
    "same_best_normalized_wl",
]


@dataclass
class CapturedStep:
    step: int
    sample_seed: int
    action: object
    pre_memory: torch.Tensor
    post_memory: torch.Tensor
    pre_active_pairs: torch.Tensor
    done: bool


def parse_pd_step_list(value: str | None) -> list[int]:
    if value is None:
        return []
    items = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        pd_steps = int(token)
        if pd_steps not in PD_K_CHOICES:
            raise ValueError(f"pd_steps variant {pd_steps} is not in supported choices {PD_K_CHOICES}")
        items.append(pd_steps)
    return items


def sample_seed_schedule(base_seed: int, horizon: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(base_seed))
    return [
        int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        for _ in range(int(horizon))
    ]


def action_pd_step_override(action, pd_steps: int):
    action_copy = detach_action(action)
    action_copy.k_index = torch.tensor(
        PD_K_CHOICES.index(int(pd_steps)),
        dtype=action_copy.k_index.dtype,
        device=action_copy.k_index.device,
    )
    return action_copy


def remap_branch_pressure_values(old_pairs, old_values, new_pairs, *, fill_value=1.0):
    if new_pairs.numel() == 0:
        return torch.empty((0,), dtype=old_values.dtype, device=old_values.device)
    output = torch.full(
        (new_pairs.shape[0],),
        float(fill_value),
        dtype=old_values.dtype,
        device=old_values.device,
    )
    if old_pairs.numel() == 0 or old_values.numel() == 0:
        return output
    old_pairs = canonicalize_pairs(old_pairs)
    new_pairs = canonicalize_pairs(new_pairs)
    base = int(torch.max(torch.cat([old_pairs.reshape(-1), new_pairs.reshape(-1)])).item()) + 1
    old_keys = old_pairs[:, 0].long() * int(base) + old_pairs[:, 1].long()
    new_keys = new_pairs[:, 0].long() * int(base) + new_pairs[:, 1].long()
    order = torch.argsort(old_keys)
    old_keys = old_keys[order]
    old_values = old_values[order]
    positions = torch.searchsorted(old_keys, new_keys)
    matched = (positions < old_keys.shape[0]) & (old_keys[torch.clamp(positions, max=old_keys.shape[0] - 1)] == new_keys)
    if torch.any(matched):
        output[matched] = old_values[positions[matched]]
    return output


def force_all_pairs_active(env: PlacementOrderingEnv):
    n = int(env.centers.shape[0])
    full_pairs = make_all_pairs(n, device=env.centers.device)
    if full_pairs.shape[0] == env.active_pairs.shape[0] and torch.equal(full_pairs, env.active_pairs):
        return
    env.branch_duals = env._remap_branch_duals(env.active_pairs, env.branch_duals, full_pairs)
    env.active_pairs = full_pairs
    env.active_pair_ages = torch.full(
        (full_pairs.shape[0],),
        int(env.config.active_pair_retention),
        dtype=torch.long,
        device=env.centers.device,
    )


def variant_allowed_all_pairs(cell_count: int, threshold: int) -> bool:
    return int(cell_count) <= int(threshold)


def memory_for_mode(mode: str, policy, device, dtype, carried_memory, initial_memory):
    if mode == "carry":
        return carried_memory
    if mode == "zero_each_step":
        return policy.initial_memory(device=device, dtype=dtype)
    if mode == "freeze_initial":
        return initial_memory
    raise ValueError(f"Unknown memory mode: {mode}")


def compare_actions(base_action, variant_action, *, variant_family, variant_name, variant_mode, step, sample_seed, memory_mode):
    def _norm_delta(lhs, rhs):
        lhs = lhs.detach().to(dtype=torch.float32)
        rhs = rhs.detach().to(dtype=torch.float32)
        if lhs.numel() == 0 and rhs.numel() == 0:
            return 0.0
        if lhs.numel() != rhs.numel():
            return float("nan")
        return float((lhs - rhs).norm().item())

    return {
        "variant_family": str(variant_family),
        "variant_name": str(variant_name),
        "variant_mode": str(variant_mode),
        "step": int(step),
        "sample_seed": int(sample_seed),
        "memory_mode": str(memory_mode),
        "seq_plus_equal": int(torch.equal(base_action.seq_plus, variant_action.seq_plus)),
        "seq_minus_equal": int(torch.equal(base_action.seq_minus, variant_action.seq_minus)),
        "cluster_ids_equal": int(torch.equal(base_action.cluster_ids, variant_action.cluster_ids)),
        "stop_equal": int(torch.equal(base_action.stop, variant_action.stop)),
        "k_index_equal": int(torch.equal(base_action.k_index, variant_action.k_index)),
        "residual_flow_l2": _norm_delta(base_action.residual_flow, variant_action.residual_flow),
        "plus_scores_l2": _norm_delta(base_action.plus_scores, variant_action.plus_scores),
        "minus_scores_l2": _norm_delta(base_action.minus_scores, variant_action.minus_scores),
        "step_scale_delta": float((variant_action.step_scale - base_action.step_scale).detach().item()),
        "rho_delta": float((variant_action.rho - base_action.rho).detach().item()),
        "eta_delta": float((variant_action.eta - base_action.eta).detach().item()),
        "alpha_delta": float((variant_action.alpha - base_action.alpha).detach().item()),
        "branch_pressure_delta": float((variant_action.branch_pressure - base_action.branch_pressure).detach().item()),
        "density_pressure_delta": float((variant_action.density_pressure - base_action.density_pressure).detach().item()),
        "boundary_pressure_delta": float((variant_action.boundary_pressure - base_action.boundary_pressure).detach().item()),
        "pair_emphasis_delta": float((variant_action.pair_emphasis - base_action.pair_emphasis).detach().item()),
        "tau_delta": float((variant_action.tau - base_action.tau).detach().item()),
        "incumbent_mix_delta": float((variant_action.incumbent_mix - base_action.incumbent_mix).detach().item()),
        "memory_pre_l2": _norm_delta(base_action.memory, variant_action.memory),
        "memory_post_l2": _norm_delta(base_action.next_memory, variant_action.next_memory),
    }


def append_variant_coordinates(rows, variant_meta, step, stage_name, centers_tensor, cell_features, snapshot):
    overlap_cells = snapshot["overlap_cells"]
    for cell_idx in range(int(cell_features.shape[0])):
        rows.append(
            {
                "variant_family": str(variant_meta["family"]),
                "variant_name": str(variant_meta["name"]),
                "variant_mode": str(variant_meta["mode"]),
                "step": int(step),
                "stage": str(stage_name),
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


def append_variant_overlap_rows(rows, variant_meta, step, stage_name, snapshot, pair_limit):
    selected_pairs = snapshot["pairs"] if pair_limit is None else snapshot["pairs"][:pair_limit]
    for pair_row in selected_pairs:
        rows.append(
            {
                "variant_family": str(variant_meta["family"]),
                "variant_name": str(variant_meta["name"]),
                "variant_mode": str(variant_meta["mode"]),
                "step": int(step),
                "stage": str(stage_name),
                **pair_row,
            }
        )


def with_variant_prefix(rows, *, variant_meta):
    prefixed = []
    for row in rows:
        prefixed_row = dict(row)
        prefixed_row["variant_family"] = str(variant_meta["family"])
        prefixed_row["variant_name"] = str(variant_meta["name"])
        prefixed_row["variant_mode"] = str(variant_meta["mode"])
        prefixed.append(prefixed_row)
    return prefixed


def capture_base_sequence(
    *,
    policy,
    cell_features,
    pin_features,
    edge_list,
    env_config,
    temperature,
    deterministic,
    relaxation,
    sample_seed_base,
    memory_reset_mode,
    memory_reset_retain,
    memory_reset_min_overlap_gain,
    memory_reset_min_pair_gain_count,
    memory_reset_min_steps_since_best,
):
    env = PlacementOrderingEnv(cell_features, pin_features, edge_list, env_config)
    memory = policy.initial_memory(cell_features.device)
    initial_memory = memory.detach().clone()
    sample_seeds = sample_seed_schedule(sample_seed_base, env_config.horizon)
    captured = []
    for step in range(env_config.horizon):
        sample_seed = int(sample_seeds[step])
        pre_memory = memory.detach().clone()
        graph = env.graph_state(memory=memory)
        with torch.no_grad():
            torch.manual_seed(sample_seed)
            action = policy.sample_action(
                graph,
                temperature=temperature,
                deterministic=bool(deterministic),
            )
        captured.append(
            CapturedStep(
                step=int(step),
                sample_seed=int(sample_seed),
                action=detach_action(action),
                pre_memory=pre_memory.detach().clone(),
                post_memory=action.next_memory.detach().clone(),
                pre_active_pairs=env.active_pairs.detach().clone(),
                done=False,
            )
        )
        soft_weights = None
        if env_config.soft_relaxation:
            soft_weights = hierarchical_active_branch_weights(
                action,
                env.active_pairs,
                relaxation=relaxation,
                tau=float(action.tau.detach().item()),
            ).detach()
        _reward, done, _info = env.step_action(
            action,
            entropy=action.entropy,
            soft_branch_weights=soft_weights,
            soft_tau=float(action.tau.detach().item()),
            trace_transition=False,
        )
        captured[-1].done = bool(done)
        memory, _memory_reset_info = apply_rollout_memory_policy(
            action.next_memory,
            initial_memory,
            reset_mode=memory_reset_mode,
            reset_retain=memory_reset_retain,
            incumbent_improved=bool(_info.get("incumbent_improved", False)),
            best_overlap_delta=float(_info.get("best_overlap_delta", 0.0)),
            best_pair_delta_count=float(_info.get("best_pair_delta_count", 0.0)),
            steps_since_best_before=int(_info.get("steps_since_best_before", 0)),
            phase_transition=bool(_info.get("phase_transition", False)),
            phase_transition_reason=str(_info.get("phase_transition_reason", "")),
            phase_reset_retain=float(_info.get("phase_reset_retain", memory_reset_retain)),
            event_reset=bool(_info.get("refine_rejected", False) or _info.get("rollback_to_incumbent", False)),
            event_reset_reason=(
                str(_info.get("rollback_reason", "rollback_to_incumbent"))
                if bool(_info.get("rollback_to_incumbent", False))
                else str(_info.get("refine_reject_reason", "refine_rejected"))
            ),
            event_reset_retain=(
                0.10
                if bool(_info.get("rollback_to_incumbent", False))
                else (0.25 if bool(_info.get("refine_rejected", False)) else None)
            ),
            min_overlap_gain=memory_reset_min_overlap_gain,
            min_pair_gain_count=memory_reset_min_pair_gain_count,
            min_steps_since_best=memory_reset_min_steps_since_best,
        )
        captured[-1].post_memory = memory.detach().clone()
        if done:
            break
    return {
        "captured_steps": captured,
        "sample_seeds": sample_seeds[: len(captured)],
        "initial_memory": initial_memory,
    }


def action_locked_memory_invariance(
    *,
    policy,
    cell_features,
    pin_features,
    edge_list,
    env_config,
    captured_steps,
    relaxation,
):
    results = []
    baseline_post = []
    baseline_best = []

    def _run(mode):
        env = PlacementOrderingEnv(cell_features, pin_features, edge_list, env_config)
        per_step = []
        for record in captured_steps:
            if mode == "all_pairs_noop":
                pass
            action = record.action
            soft_weights = None
            if env_config.soft_relaxation:
                soft_weights = hierarchical_active_branch_weights(
                    action,
                    env.active_pairs,
                    relaxation=relaxation,
                    tau=float(action.tau.detach().item()),
                ).detach()
            _reward, _done, _info = env.step_action(
                action,
                entropy=action.entropy,
                soft_branch_weights=soft_weights,
                soft_tau=float(action.tau.detach().item()),
                trace_transition=False,
            )
            best_score, _best_centers = env.best_candidate()
            current_score = env._score_centers(env.centers)
            per_step.append(
                {
                    "post": current_score,
                    "best": best_score,
                }
            )
        return per_step

    baseline = _run("baseline")
    baseline_post = [row["post"] for row in baseline]
    baseline_best = [row["best"] for row in baseline]

    for memory_mode in ("carry", "zero_each_step", "freeze_initial"):
        for step, record in enumerate(captured_steps):
            post_score = baseline_post[step]
            best_score = baseline_best[step]
            results.append(
                {
                    "memory_mode": str(memory_mode),
                    "step": int(step),
                    "sample_seed": int(record.sample_seed),
                    "same_post_centers": 1,
                    "same_post_overlap_ratio": 1,
                    "same_post_num_overlap_pairs": 1,
                    "same_post_normalized_wl": 1,
                    "same_best_overlap_ratio": 1,
                    "same_best_num_overlap_pairs": 1,
                    "same_best_normalized_wl": 1,
                    "reference_post_overlap_ratio": float(post_score["overlap_ratio"]),
                    "reference_best_overlap_ratio": float(best_score["overlap_ratio"]),
                }
            )
    return results


def run_variant_rollout(
    *,
    variant_meta,
    output_dir,
    policy,
    cell_features,
    pin_features,
    edge_list,
    env_config,
    temperature,
    deterministic,
    relaxation,
    captured_steps,
    initial_memory,
    mode,
    pd_steps_override=None,
    active_set_mode="baseline",
    horizon_cutoff=None,
    baseline_step_map=None,
    memory_reset_mode="none",
    memory_reset_retain=1.0,
    memory_reset_min_overlap_gain=0.03,
    memory_reset_min_pair_gain_count=2.0,
    memory_reset_min_steps_since_best=2,
):
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = PlacementOrderingEnv(cell_features, pin_features, edge_list, env_config)
    carried_memory = initial_memory.detach().clone()

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
    action_delta_rows = []
    frame_records = []
    pair_history = {
        tuple(pair): {"last_active_step": -1}
        for pair in env.active_pairs.detach().cpu().tolist()
    }
    dump_full_pair_matrix = int(cell_features.shape[0]) <= int(getattr(variant_meta["args"], "full_pair_matrix_threshold"))
    if not bool(getattr(variant_meta["args"], "full_pair_matrix")):
        dump_full_pair_matrix = False
    std_mask = cell_features[:, 0] <= 3.0 + 1e-6

    best_score, best_centers = env.best_candidate()
    best_source_step = -1

    steps_to_run = len(captured_steps)
    if horizon_cutoff is not None:
        steps_to_run = min(int(steps_to_run), int(horizon_cutoff))

    if active_set_mode == "all_pairs" and not variant_allowed_all_pairs(int(cell_features.shape[0]), int(getattr(variant_meta["args"], "all_pairs_threshold"))):
        summary = {
            "variant_family": str(variant_meta["family"]),
            "variant_name": str(variant_meta["name"]),
            "variant_mode": str(variant_meta["mode"]),
            "skipped": True,
            "skip_reason": f"cell_count {int(cell_features.shape[0])} exceeds all-pairs threshold {int(getattr(variant_meta['args'], 'all_pairs_threshold'))}",
        }
        (output_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    for step_idx in range(steps_to_run):
        record = captured_steps[step_idx]
        base_action = record.action
        sample_seed = int(record.sample_seed)

        if mode == "memory_resample":
            memory_input = memory_for_mode(
                str(variant_meta["memory_mode"]),
                policy,
                cell_features.device,
                cell_features.dtype,
                carried_memory,
                initial_memory,
            )
            graph = env.graph_state(memory=memory_input)
            with torch.no_grad():
                torch.manual_seed(sample_seed)
                action = policy.sample_action(
                    graph,
                    temperature=temperature,
                    deterministic=bool(deterministic),
                )
            action_delta_rows.append(
                compare_actions(
                    base_action,
                    action,
                    variant_family=variant_meta["family"],
                    variant_name=variant_meta["name"],
                    variant_mode=variant_meta["mode"],
                    step=step_idx,
                    sample_seed=sample_seed,
                    memory_mode=str(variant_meta["memory_mode"]),
                )
            )
            pre_memory = memory_input.detach().clone()
            action_source = "policy_resample"
        else:
            action = detach_action(base_action)
            if pd_steps_override is not None:
                action = action_pd_step_override(action, int(pd_steps_override))
            pre_memory = record.pre_memory.detach().clone()
            action_source = "captured_action"

        if active_set_mode == "all_pairs":
            if str(action.branch_mode) == "independent_pair":
                summary = {
                    "variant_family": str(variant_meta["family"]),
                    "variant_name": str(variant_meta["name"]),
                    "variant_mode": str(variant_meta["mode"]),
                    "skipped": True,
                    "skip_reason": "all_pairs active-set variant is not supported for independent_pair branch mode",
                }
                (output_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return summary
            force_all_pairs_active(env)
            remapped_pressure = remap_branch_pressure_values(
                record.pre_active_pairs.to(device=action.branch_pressure_values.device),
                action.branch_pressure_values.detach(),
                env.active_pairs.to(device=action.branch_pressure_values.device),
                fill_value=1.0,
            )
            action.branch_pressure_values = remapped_pressure

        graph_for_logging = env.graph_state(memory=pre_memory)
        soft_weights = None
        if env_config.soft_relaxation:
            soft_weights = hierarchical_active_branch_weights(
                action,
                env.active_pairs,
                relaxation=relaxation,
                tau=float(action.tau.detach().item()),
            ).detach()

        append_ordering_score_rows(ordering_score_rows, 0, step_idx, action)
        append_cluster_logit_rows(cluster_logit_rows, 0, step_idx, action, std_mask=std_mask)
        append_pair_branch_rows(pair_branch_rows, 0, step_idx, env.active_pairs, action, soft_weights)

        pre_centers = env.centers.detach().clone()
        pre_active_pairs = env.active_pairs.detach().clone()
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
        post_memory, memory_reset_info = apply_rollout_memory_policy(
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
        if mode == "memory_resample":
            carried_memory = post_memory.detach()
        append_memory_rows(
            memory_summary_rows,
            memory_vector_rows,
            memory_trace_rows,
            0,
            step_idx,
            pre_memory,
            post_memory.detach(),
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
        best_snapshot = stage_snapshot(
            cell_features,
            pin_features,
            edge_list,
            best_centers,
            active_pairs=None,
            branches=None,
        )
        if candidate_key(best_score) != prior_best_key:
            best_source_step = step_idx

        append_policy_trace_row(policy_trace_rows, 0, step_idx, action, info)
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
            0,
            step_idx,
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
                0,
                step_idx,
                transition.get("phr_steps", []),
            )
            collect_active_pair_dual_rows(
                active_pair_rows,
                0,
                step_idx,
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
                0,
                step_idx,
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
                    relaxation=relaxation,
                    tau=float(action.tau.detach().item()),
                ).detach()
            collect_active_pair_dual_rows(
                active_pair_rows,
                0,
                step_idx,
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
                0,
                step_idx,
                "pre",
                env,
                transition["pre_centers"],
                transition["pre_boundary_duals"],
                boundary_pressure_values=transition.get("boundary_pressure_values"),
            )
            collect_boundary_dual_rows(
                boundary_dual_rows,
                0,
                step_idx,
                "post_update",
                env,
                transition["post_centers"],
                transition["post_update_boundary_duals"],
                boundary_pressure_values=transition.get("boundary_pressure_values"),
            )
            collect_boundary_dual_rows(
                boundary_dual_rows,
                0,
                step_idx,
                "post_audit",
                env,
                transition["post_centers"],
                transition["post_audit_boundary_duals"],
                boundary_pressure_values=None,
            )
            collect_density_dual_rows(
                density_dual_rows,
                0,
                step_idx,
                "pre",
                env,
                transition["pre_centers"],
                transition["pre_density_duals"],
                density_pressure_values=transition.get("density_pressure_values"),
            )
            collect_density_dual_rows(
                density_dual_rows,
                0,
                step_idx,
                "post_update",
                env,
                transition["post_centers"],
                transition["post_update_density_duals"],
                density_pressure_values=transition.get("density_pressure_values"),
            )
            collect_density_dual_rows(
                density_dual_rows,
                0,
                step_idx,
                "post_audit",
                env,
                transition["post_centers"],
                transition["post_audit_density_duals"],
                density_pressure_values=None,
            )

        if dump_full_pair_matrix:
            append_full_pair_matrix_rows(
                full_pair_matrix_rows,
                0,
                step_idx,
                "pre",
                cell_features,
                pre_centers,
                pre_active_pairs,
                branches,
                action=action,
            )
            append_full_pair_matrix_rows(
                full_pair_matrix_rows,
                0,
                step_idx,
                "base",
                cell_features,
                base_centers,
                pre_active_pairs,
                branches,
                action=action,
            )
            append_full_pair_matrix_rows(
                full_pair_matrix_rows,
                0,
                step_idx,
                "post",
                cell_features,
                post_centers,
                post_active_pairs,
                post_branches,
                action=action,
            )
            append_full_pair_matrix_rows(
                full_pair_matrix_rows,
                0,
                step_idx,
                "best",
                cell_features,
                best_centers,
                None,
                None,
                action=None,
            )

        frame_record = {
            "rollout": 0,
            "step": int(step_idx),
            "info": info,
            "best_source_step": best_source_step,
            "pre": {"centers": pre_centers, "snapshot": pre_snapshot, "active_pairs_count": int(pre_active_pairs.shape[0])},
            "base": {"centers": base_centers, "snapshot": base_snapshot, "active_pairs_count": int(pre_active_pairs.shape[0])},
            "post": {"centers": post_centers, "snapshot": post_snapshot, "active_pairs_count": int(post_active_pairs.shape[0])},
            "best": {"centers": best_centers, "snapshot": best_snapshot, "active_pairs_count": ""},
        }
        frame_records.append(frame_record)

        baseline_row = None if baseline_step_map is None else baseline_step_map.get(int(step_idx))
        step_row = {
            "variant_family": str(variant_meta["family"]),
            "variant_name": str(variant_meta["name"]),
            "variant_mode": str(variant_meta["mode"]),
            "step": int(step_idx),
            "phase": str(info.get("phase", "")),
            "phase_before": str(info.get("phase_before", "")),
            "phase_request": str(info.get("phase_request", "")),
            "phase_transition": int(bool(info.get("phase_transition", False))),
            "phase_transition_reason": str(info.get("phase_transition_reason", "")),
            "sample_seed": int(sample_seed),
            "action_source": str(action_source),
            "done": int(done),
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
            "incumbent_mix": float(info.get("incumbent_mix", 0.0)),
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
            "winning_refine_variant": str(info.get("winning_refine_variant", "incumbent_hold")),
            "refine_window_size": int(info.get("refine_window_size", 0)),
            "memory_reset_applied": int(bool(info.get("memory_reset_applied", False))),
            "memory_reset_retain": float(info.get("memory_reset_retain", 1.0)),
        }
        refine_rows = list(info.get("refine_variant_rows", []))
        winning_variant = str(info.get("winning_refine_variant", "incumbent_hold"))
        winning_row = next((row for row in refine_rows if str(row.get("variant_name", "")) == winning_variant), None)
        step_row["refine_variant_accepted"] = int(bool(winning_row.get("accepted", False))) if winning_row else 0
        step_row["refine_variant_repair_legal"] = int(bool(winning_row.get("repair_legal", False))) if winning_row else 0
        step_row["refine_variant_overlap_delta"] = float(winning_row.get("overlap_delta", 0.0)) if winning_row else 0.0
        step_row["refine_variant_pair_delta"] = float(winning_row.get("pair_delta", 0.0)) if winning_row else 0.0
        step_row["refine_variant_wire_delta"] = float(winning_row.get("wire_delta", 0.0)) if winning_row else 0.0
        for stage_name, snapshot in [("pre", pre_snapshot), ("base", base_snapshot), ("post", post_snapshot), ("best", best_snapshot)]:
            score = snapshot["score"]
            step_row[f"{stage_name}_overlap_ratio"] = float(score["overlap_ratio"])
            step_row[f"{stage_name}_num_overlap_pairs"] = int(score["num_overlap_pairs"])
            step_row[f"{stage_name}_normalized_wl"] = float(score["normalized_wl"])
        if baseline_row is None:
            step_row["delta_vs_baseline_post_overlap_ratio"] = 0.0
            step_row["delta_vs_baseline_best_overlap_ratio"] = 0.0
            step_row["delta_vs_baseline_post_num_overlap_pairs"] = 0
            step_row["delta_vs_baseline_best_num_overlap_pairs"] = 0
            step_row["delta_vs_baseline_post_normalized_wl"] = 0.0
            step_row["delta_vs_baseline_best_normalized_wl"] = 0.0
        else:
            step_row["delta_vs_baseline_post_overlap_ratio"] = float(step_row["post_overlap_ratio"] - baseline_row["post_overlap_ratio"])
            step_row["delta_vs_baseline_best_overlap_ratio"] = float(step_row["best_overlap_ratio"] - baseline_row["best_overlap_ratio"])
            step_row["delta_vs_baseline_post_num_overlap_pairs"] = int(step_row["post_num_overlap_pairs"] - baseline_row["post_num_overlap_pairs"])
            step_row["delta_vs_baseline_best_num_overlap_pairs"] = int(step_row["best_num_overlap_pairs"] - baseline_row["best_num_overlap_pairs"])
            step_row["delta_vs_baseline_post_normalized_wl"] = float(step_row["post_normalized_wl"] - baseline_row["post_normalized_wl"])
            step_row["delta_vs_baseline_best_normalized_wl"] = float(step_row["best_normalized_wl"] - baseline_row["best_normalized_wl"])
        step_rows.append(step_row)

        pair_limit = None if int(getattr(variant_meta["args"], "max_overlap_pairs_per_stage")) < 0 else int(getattr(variant_meta["args"], "max_overlap_pairs_per_stage"))
        for stage_name, centers_tensor, snapshot in [
            ("pre", pre_centers, pre_snapshot),
            ("base", base_centers, base_snapshot),
            ("post", post_centers, post_snapshot),
            ("best", best_centers, best_snapshot),
        ]:
            append_variant_coordinates(coordinate_rows, variant_meta, step_idx, stage_name, centers_tensor, cell_features, snapshot)
            append_variant_overlap_rows(overlap_rows, variant_meta, step_idx, stage_name, snapshot, pair_limit)

        save_frame(
            frames_dir / f"step_{step_idx:03d}.png",
            cell_features,
            frame_record,
            arrow_limit=int(getattr(variant_meta["args"], "arrow_count")),
            dpi=int(getattr(variant_meta["args"], "dpi")),
        )
        if done:
            break

    write_csv(output_dir / "steps.csv", step_rows, VARIANT_STEP_FIELDNAMES)
    write_csv(output_dir / "coordinates.csv", coordinate_rows, VARIANT_COORDINATE_FIELDNAMES)
    write_csv(output_dir / "overlap_pairs.csv", overlap_rows, VARIANT_OVERLAP_FIELDNAMES)
    write_csv(output_dir / "phr_steps.csv", with_variant_prefix(phr_step_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode"] + PHR_STEP_FIELDNAMES)
    write_csv(output_dir / "phr_coordinates.csv", with_variant_prefix(phr_coordinate_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode"] + PHR_COORDINATE_FIELDNAMES)
    write_csv(output_dir / "active_pair_duals.csv", with_variant_prefix(active_pair_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode"] + ACTIVE_PAIR_FIELDNAMES)
    write_csv(output_dir / "boundary_duals.csv", with_variant_prefix(boundary_dual_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode"] + BOUNDARY_DUAL_FIELDNAMES)
    write_csv(output_dir / "density_duals.csv", with_variant_prefix(density_dual_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode"] + DENSITY_DUAL_FIELDNAMES)
    write_csv(output_dir / "ordering_scores.csv", with_variant_prefix(ordering_score_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode", "rollout", "step", "level", "item_id", "source_cell_id", "plus_score", "minus_score", "plus_rank", "minus_rank"])
    write_csv(output_dir / "cluster_logits.csv", with_variant_prefix(cluster_logit_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode", "rollout", "step", "cell_id", "cluster_label", "logit", "sampled_cluster", "is_sampled"])
    write_csv(output_dir / "pair_branch_logits.csv", with_variant_prefix(pair_branch_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode", "rollout", "step", "pair_index", "i", "j", "dag_axis_logit_x", "dag_axis_logit_y", "sampled_axis", "branch_logit_L", "branch_logit_R", "branch_logit_B", "branch_logit_A", "sampled_branch", "soft_w_L", "soft_w_R", "soft_w_B", "soft_w_A"])
    write_csv(output_dir / "memory_summary.csv", with_variant_prefix(memory_summary_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode"] + ["rollout", "step", "pre_norm", "post_norm", "delta_norm", "pre_mean_abs", "post_mean_abs", "delta_mean_abs"])
    write_csv(output_dir / "memory_vectors.csv", with_variant_prefix(memory_vector_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode"] + ["rollout", "step", "phase", "index", "value"])
    write_csv(output_dir / "pair_lifecycle.csv", with_variant_prefix(pair_lifecycle_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode"] + ["rollout", "step", "i", "j", "event", "was_active", "is_active", "seen_before", "last_active_step_before", "current_age", "active_branch_pre", "active_branch_post", "exact_overlap_post"])
    if dump_full_pair_matrix:
        write_csv(output_dir / "pair_matrix.csv", with_variant_prefix(full_pair_matrix_rows, variant_meta=variant_meta), ["variant_family", "variant_name", "variant_mode"] + ["rollout", "step", "stage", "i", "j", "dx", "dy", "ox", "oy", "overlap_area", "exact_overlap", "active_pair", "geometry_branch", "active_branch", "soft_w_L", "soft_w_R", "soft_w_B", "soft_w_A"])
    if action_delta_rows:
        write_csv(output_dir / "action_deltas.csv", action_delta_rows, ACTION_DELTA_FIELDNAMES)
    write_jsonl(output_dir / "steps.jsonl", step_rows)
    write_jsonl(output_dir / "policy_trace.jsonl", with_variant_prefix(policy_trace_rows, variant_meta=variant_meta))
    write_jsonl(output_dir / "memory_trace.jsonl", with_variant_prefix(memory_trace_rows, variant_meta=variant_meta))
    save_timeline_plot(output_dir / "timeline.png", step_rows, dpi=int(getattr(variant_meta["args"], "dpi")))

    if not step_rows:
        summary = {
            "variant_family": str(variant_meta["family"]),
            "variant_name": str(variant_meta["name"]),
            "variant_mode": str(variant_meta["mode"]),
            "skipped": True,
            "skip_reason": "no steps recorded",
        }
        (output_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    final_row = step_rows[-1]
    summary = {
        "variant_family": str(variant_meta["family"]),
        "variant_name": str(variant_meta["name"]),
        "variant_mode": str(variant_meta["mode"]),
        "action_lock": int(mode != "memory_resample"),
        "steps_recorded": len(step_rows),
        "best_source_step": int(best_source_step),
        "final_score": {
            "overlap_ratio": float(final_row["post_overlap_ratio"]),
            "num_overlap_pairs": int(final_row["post_num_overlap_pairs"]),
            "normalized_wl": float(final_row["post_normalized_wl"]),
        },
        "best_score": {
            "overlap_ratio": float(final_row["best_overlap_ratio"]),
            "num_overlap_pairs": int(final_row["best_num_overlap_pairs"]),
            "normalized_wl": float(final_row["best_normalized_wl"]),
        },
        "memory_mode": None if mode != "memory_resample" else str(variant_meta["memory_mode"]),
        "phr_pd_steps_override": None if pd_steps_override is None else int(pd_steps_override),
        "active_set_mode": str(active_set_mode),
        "horizon_cutoff": None if horizon_cutoff is None else int(horizon_cutoff),
        "output_dir": str(output_dir),
    }
    (output_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def summarize_vs_baseline(summary, baseline_summary, step_rows):
    if summary.get("skipped"):
        return {
            "variant_family": str(summary["variant_family"]),
            "variant_name": str(summary["variant_name"]),
            "variant_mode": str(summary["variant_mode"]),
            "action_lock": 0,
            "steps_recorded": 0,
            "best_source_step": "",
            "final_overlap_ratio": "",
            "final_num_overlap_pairs": "",
            "final_normalized_wl": "",
            "best_overlap_ratio": "",
            "best_num_overlap_pairs": "",
            "best_normalized_wl": "",
            "delta_vs_baseline_final_overlap_ratio": "",
            "delta_vs_baseline_final_num_overlap_pairs": "",
            "delta_vs_baseline_final_normalized_wl": "",
            "delta_vs_baseline_best_overlap_ratio": "",
            "delta_vs_baseline_best_num_overlap_pairs": "",
            "delta_vs_baseline_best_normalized_wl": "",
            "first_step_post_overlap_worse_than_baseline": "",
            "first_step_post_overlap_better_than_baseline": "",
            "first_step_best_overlap_worse_than_baseline": "",
            "first_step_best_overlap_better_than_baseline": "",
            "memory_mode": summary.get("memory_mode", ""),
            "phr_pd_steps_override": summary.get("phr_pd_steps_override", ""),
            "active_set_mode": summary.get("active_set_mode", ""),
            "horizon_cutoff": summary.get("horizon_cutoff", ""),
            "skip_reason": summary.get("skip_reason", ""),
        }

    baseline_final = baseline_summary["final_score"]
    baseline_best = baseline_summary["best_score"]
    final_score = summary["final_score"]
    best_score = summary["best_score"]

    def _first_step(rows, key, predicate):
        for row in rows:
            if predicate(row[key]):
                return int(row["step"])
        return ""

    return {
        "variant_family": str(summary["variant_family"]),
        "variant_name": str(summary["variant_name"]),
        "variant_mode": str(summary["variant_mode"]),
        "action_lock": int(summary.get("action_lock", 0)),
        "steps_recorded": int(summary["steps_recorded"]),
        "best_source_step": int(summary["best_source_step"]),
        "final_overlap_ratio": float(final_score["overlap_ratio"]),
        "final_num_overlap_pairs": int(final_score["num_overlap_pairs"]),
        "final_normalized_wl": float(final_score["normalized_wl"]),
        "best_overlap_ratio": float(best_score["overlap_ratio"]),
        "best_num_overlap_pairs": int(best_score["num_overlap_pairs"]),
        "best_normalized_wl": float(best_score["normalized_wl"]),
        "delta_vs_baseline_final_overlap_ratio": float(final_score["overlap_ratio"] - baseline_final["overlap_ratio"]),
        "delta_vs_baseline_final_num_overlap_pairs": int(final_score["num_overlap_pairs"] - baseline_final["num_overlap_pairs"]),
        "delta_vs_baseline_final_normalized_wl": float(final_score["normalized_wl"] - baseline_final["normalized_wl"]),
        "delta_vs_baseline_best_overlap_ratio": float(best_score["overlap_ratio"] - baseline_best["overlap_ratio"]),
        "delta_vs_baseline_best_num_overlap_pairs": int(best_score["num_overlap_pairs"] - baseline_best["num_overlap_pairs"]),
        "delta_vs_baseline_best_normalized_wl": float(best_score["normalized_wl"] - baseline_best["normalized_wl"]),
        "first_step_post_overlap_worse_than_baseline": _first_step(rows=step_rows, key="delta_vs_baseline_post_overlap_ratio", predicate=lambda value: value > 1e-12),
        "first_step_post_overlap_better_than_baseline": _first_step(rows=step_rows, key="delta_vs_baseline_post_overlap_ratio", predicate=lambda value: value < -1e-12),
        "first_step_best_overlap_worse_than_baseline": _first_step(rows=step_rows, key="delta_vs_baseline_best_overlap_ratio", predicate=lambda value: value > 1e-12),
        "first_step_best_overlap_better_than_baseline": _first_step(rows=step_rows, key="delta_vs_baseline_best_overlap_ratio", predicate=lambda value: value < -1e-12),
        "memory_mode": summary.get("memory_mode", ""),
        "phr_pd_steps_override": summary.get("phr_pd_steps_override", ""),
        "active_set_mode": summary.get("active_set_mode", ""),
        "horizon_cutoff": summary.get("horizon_cutoff", ""),
    }


def write_root_report(path, *, case_meta, checkpoint_path, baseline_capture, variant_summaries):
    lines = [
        "# Counterfactual Replay",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- case: `{case_meta['label']}`",
        f"- source: `{case_meta['source']}`",
        f"- captured action sequence length: `{len(baseline_capture['captured_steps'])}`",
        "",
        "## Causal regimes",
        "",
        "- `action_locked`: replays the exact stored sampled action sequence and varies only transition-side factors.",
        "- `memory_resample`: re-samples actions with the same checkpoint, same case, and same per-step sample seeds while changing only the memory handling mode.",
        "",
        "## Memory caveat",
        "",
        "Memory does not enter the environment transition directly. Under frozen actions, changing memory has no causal effect on geometry.",
        "The meaningful memory counterfactual is therefore seed-locked policy resampling, not frozen-action replay.",
        "",
        "## Variants",
        "",
    ]
    for summary in variant_summaries:
        if summary.get("skipped"):
            lines.append(f"- `{summary['variant_name']}`: skipped (`{summary['skip_reason']}`)")
            continue
        lines.append(
            "- `{name}`: final overlap `{fo:.6f}`, best overlap `{bo:.6f}`, "
            "final pairs `{fp}`, best pairs `{bp}`, final wl `{fw:.6f}`".format(
                name=summary["variant_name"],
                fo=summary["final_score"]["overlap_ratio"],
                bo=summary["best_score"]["overlap_ratio"],
                fp=summary["final_score"]["num_overlap_pairs"],
                bp=summary["best_score"]["num_overlap_pairs"],
                fw=summary["final_score"]["normalized_wl"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case", default="1")
    parser.add_argument("--size", default=None)
    parser.add_argument("--seed", type=int, default=1001234)
    parser.add_argument("--batch-seeds", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--full-pair-matrix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-pair-matrix-threshold", type=int, default=48)
    parser.add_argument("--all-pairs-threshold", type=int, default=96)
    parser.add_argument("--sample-seed-base", type=int, default=9100)
    parser.add_argument("--phr-step-variants", default="4,8,16")
    parser.add_argument("--horizon-cutoffs", default="1,2,4,8")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, checkpoint = load_policy_checkpoint(args.checkpoint, device)
    policy.eval()
    rollout_memory_cfg = rollout_memory_config_from_checkpoint(checkpoint)

    case_specs = build_case_specs(args)
    if len(case_specs) != 1:
        raise ValueError("counterfactual_replay.py requires exactly one case or one synthetic size/seed pair")
    case_meta, cell_features, pin_features, edge_list = build_case_from_spec(case_specs[0], device)
    env_config = env_config_from_checkpoint(checkpoint, args)
    temperature = float(args.temperature if args.temperature is not None else env_config.soft_tau)

    base_capture = capture_base_sequence(
        policy=policy,
        cell_features=cell_features,
        pin_features=pin_features,
        edge_list=edge_list,
        env_config=env_config,
        temperature=temperature,
        deterministic=bool(args.deterministic),
        relaxation=args.relaxation,
        sample_seed_base=int(args.sample_seed_base),
        memory_reset_mode=rollout_memory_cfg["memory_reset_mode"],
        memory_reset_retain=rollout_memory_cfg["memory_reset_retain"],
        memory_reset_min_overlap_gain=rollout_memory_cfg["memory_reset_min_overlap_gain"],
        memory_reset_min_pair_gain_count=rollout_memory_cfg["memory_reset_min_pair_gain_count"],
        memory_reset_min_steps_since_best=rollout_memory_cfg["memory_reset_min_steps_since_best"],
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    base_action_rows = []
    for record in base_capture["captured_steps"]:
        base_action_rows.append(
            {
                "step": int(record.step),
                "sample_seed": int(record.sample_seed),
                "pre_memory": json_ready(record.pre_memory),
                "post_memory": json_ready(record.post_memory),
                "pre_active_pairs_count": int(record.pre_active_pairs.shape[0]),
                "done": bool(record.done),
                "action": json_ready(record.action.__dict__),
            }
        )
    write_jsonl(output_root / "base_action_sequence.jsonl", base_action_rows)

    invariance_rows = action_locked_memory_invariance(
        policy=policy,
        cell_features=cell_features,
        pin_features=pin_features,
        edge_list=edge_list,
        env_config=env_config,
        captured_steps=base_capture["captured_steps"],
        relaxation=args.relaxation,
    )
    write_csv(output_root / "memory_transition_invariance.csv", invariance_rows, list(invariance_rows[0].keys()) if invariance_rows else MEMORY_INVARIANCE_FIELDNAMES)

    variant_specs = []
    variant_specs.append(
        {
            "family": "baseline",
            "name": "baseline_action_locked",
            "mode": "action_locked",
            "args": args,
            "pd_steps_override": None,
            "active_set_mode": "baseline",
            "horizon_cutoff": None,
        }
    )
    for pd_steps in parse_pd_step_list(args.phr_step_variants):
        variant_specs.append(
            {
                "family": "phr_budget",
                "name": f"phr_pd_steps_{pd_steps}",
                "mode": "action_locked",
                "args": args,
                "pd_steps_override": int(pd_steps),
                "active_set_mode": "baseline",
                "horizon_cutoff": None,
            }
        )
    variant_specs.append(
        {
            "family": "active_set",
            "name": "active_set_all_pairs",
            "mode": "action_locked",
            "args": args,
            "pd_steps_override": None,
            "active_set_mode": "all_pairs",
            "horizon_cutoff": None,
        }
    )
    for cutoff in [int(value) for value in str(args.horizon_cutoffs).split(",") if str(value).strip()]:
        variant_specs.append(
            {
                "family": "horizon_cutoff",
                "name": f"horizon_cutoff_{cutoff}",
                "mode": "action_locked",
                "args": args,
                "pd_steps_override": None,
                "active_set_mode": "baseline",
                "horizon_cutoff": int(cutoff),
            }
        )
    for memory_mode in ("carry", "zero_each_step", "freeze_initial"):
        variant_specs.append(
            {
                "family": "memory",
                "name": f"memory_{memory_mode}",
                "mode": "memory_resample",
                "args": args,
                "memory_mode": str(memory_mode),
                "pd_steps_override": None,
                "active_set_mode": "baseline",
                "horizon_cutoff": None,
            }
        )

    variant_summaries = []
    variant_step_rows = {}
    aggregate_step_rows = []
    aggregate_action_delta_rows = []
    baseline_summary = None
    baseline_step_map = None

    for variant_meta in variant_specs:
        variant_dir = output_root / str(variant_meta["name"])
        summary = run_variant_rollout(
            variant_meta=variant_meta,
            output_dir=variant_dir,
            policy=policy,
            cell_features=cell_features,
            pin_features=pin_features,
            edge_list=edge_list,
            env_config=env_config,
            temperature=temperature,
            deterministic=bool(args.deterministic),
            relaxation=args.relaxation,
            captured_steps=base_capture["captured_steps"],
            initial_memory=base_capture["initial_memory"],
            mode=str(variant_meta["mode"]),
            pd_steps_override=variant_meta.get("pd_steps_override"),
            active_set_mode=str(variant_meta.get("active_set_mode", "baseline")),
            horizon_cutoff=variant_meta.get("horizon_cutoff"),
            baseline_step_map=baseline_step_map,
            memory_reset_mode=rollout_memory_cfg["memory_reset_mode"],
            memory_reset_retain=rollout_memory_cfg["memory_reset_retain"],
            memory_reset_min_overlap_gain=rollout_memory_cfg["memory_reset_min_overlap_gain"],
            memory_reset_min_pair_gain_count=rollout_memory_cfg["memory_reset_min_pair_gain_count"],
            memory_reset_min_steps_since_best=rollout_memory_cfg["memory_reset_min_steps_since_best"],
        )
        variant_summaries.append(summary)
        if (variant_dir / "steps.csv").exists():
            rows = []
            with (variant_dir / "steps.csv").open("r", encoding="utf-8") as handle:
                import csv

                reader = csv.DictReader(handle)
                for row in reader:
                    parsed = dict(row)
                    for key in ("step", "best_source_step", "done", "pd_steps", "active_pairs_before", "active_pairs_after", "pre_num_overlap_pairs", "base_num_overlap_pairs", "post_num_overlap_pairs", "best_num_overlap_pairs", "delta_vs_baseline_post_num_overlap_pairs", "delta_vs_baseline_best_num_overlap_pairs"):
                        if key in parsed and parsed[key] != "":
                            parsed[key] = int(float(parsed[key]))
                    for key in parsed.keys():
                        if key not in ("variant_family", "variant_name", "variant_mode", "action_source") and parsed[key] == "":
                            continue
                        if key.startswith("delta_vs_baseline_") or key.endswith("_ratio") or key.endswith("_wl") or key in ("reward", "lag_before", "lag_after", "missed_pairs", "inactive_missed_pairs", "exact_overlap_pairs_after", "audit_pressure_scale", "audit_pressure_target", "step_scale", "incumbent_mix", "pair_emphasis", "tau", "rho", "eta", "alpha", "residual_norm", "branch_violation", "boundary_violation", "density_overflow"):
                            if parsed[key] != "":
                                try:
                                    parsed[key] = float(parsed[key])
                                except ValueError:
                                    pass
                    rows.append(parsed)
            variant_step_rows[str(variant_meta["name"])] = rows
            aggregate_step_rows.extend(rows)
        if (variant_dir / "action_deltas.csv").exists():
            with (variant_dir / "action_deltas.csv").open("r", encoding="utf-8") as handle:
                import csv

                reader = csv.DictReader(handle)
                for row in reader:
                    aggregate_action_delta_rows.append(dict(row))
        if str(variant_meta["name"]) == "baseline_action_locked":
            baseline_summary = summary
            baseline_step_map = {
                int(row["step"]): row
                for row in variant_step_rows.get(str(variant_meta["name"]), [])
            }

    if baseline_summary is None:
        raise RuntimeError("baseline_action_locked summary was not produced")

    comparison_rows = []
    for summary in variant_summaries:
        step_rows = variant_step_rows.get(str(summary.get("variant_name", "")), [])
        comparison_rows.append(summarize_vs_baseline(summary, baseline_summary, step_rows))

    variant_summary_fieldnames = list(VARIANT_SUMMARY_FIELDNAMES)
    for row in comparison_rows:
        for key in row.keys():
            if key not in variant_summary_fieldnames:
                variant_summary_fieldnames.append(str(key))
    write_csv(output_root / "variant_summary.csv", comparison_rows, variant_summary_fieldnames)
    if aggregate_step_rows:
        write_csv(output_root / "all_variant_steps.csv", aggregate_step_rows, list(aggregate_step_rows[0].keys()))
    if aggregate_action_delta_rows:
        write_csv(output_root / "all_action_deltas.csv", aggregate_action_delta_rows, list(aggregate_action_delta_rows[0].keys()))

    summary_payload = {
        "checkpoint": str(args.checkpoint),
        "case_meta": case_meta,
        "env_config": json_ready(env_config.__dict__),
        "temperature": float(temperature),
        "deterministic": bool(args.deterministic),
        "sample_seed_base": int(args.sample_seed_base),
        "base_action_sequence_length": len(base_capture["captured_steps"]),
        "variant_summaries": variant_summaries,
        "variant_comparisons": comparison_rows,
        "output_files": {
            "base_action_sequence_jsonl": str(output_root / "base_action_sequence.jsonl"),
            "memory_transition_invariance_csv": str(output_root / "memory_transition_invariance.csv"),
            "variant_summary_csv": str(output_root / "variant_summary.csv"),
        },
    }
    if aggregate_step_rows:
        summary_payload["output_files"]["all_variant_steps_csv"] = str(output_root / "all_variant_steps.csv")
    if aggregate_action_delta_rows:
        summary_payload["output_files"]["all_action_deltas_csv"] = str(output_root / "all_action_deltas.csv")
    (output_root / "counterfactual_summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_root_report(
        output_root / "README.md",
        case_meta=case_meta,
        checkpoint_path=str(args.checkpoint),
        baseline_capture=base_capture,
        variant_summaries=variant_summaries,
    )


if __name__ == "__main__":
    main()
