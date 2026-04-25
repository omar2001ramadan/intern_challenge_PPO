"""Train the sequence-pair ordering policy with PPO."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from env import EnvConfig, PlacementOrderingEnv
from ordering_policy import OrderingPolicy, hierarchical_active_branch_weights, load_policy_checkpoint, save_policy_checkpoint
from placement import generate_placement_input
from ppo import Transition, detach_action, detach_graph, ppo_update


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


def linear_schedule(update_idx, total_updates, start, end):
    if total_updates <= 1:
        return end
    mix = min(max(update_idx / (total_updates - 1), 0.0), 1.0)
    return start * (1.0 - mix) + end * mix


def temperature_at(update_idx, total_updates, start=1.4, end=0.55):
    return linear_schedule(update_idx, total_updates, start, end)


def soft_tau_at(update_idx, total_updates, start=2.0, end=0.10):
    return linear_schedule(update_idx, total_updates, start, end)


def make_problem(sizes, device, seed, forced_size=None):
    num_macros, num_std_cells = forced_size if forced_size is not None else random.choice(sizes)
    torch.manual_seed(seed)
    cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)
    cell_features = initialize_random_spread(cell_features, seed=seed + 17)
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
):
    cell_features, pin_features, edge_list, size = make_problem(sizes, device, seed, forced_size=forced_size)
    env = PlacementOrderingEnv(cell_features, pin_features, edge_list, env_config)
    transitions = []
    infos = []
    memory = policy.initial_memory(device)

    for step_idx in range(env_config.horizon):
        graph = env.graph_state(memory=memory)
        action = policy.sample_action(graph, temperature=temperature, deterministic=deterministic)
        memory = action.next_memory.detach()
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

    best_score, _ = env.best_candidate()
    return transitions, {
        "size": size,
        "steps": len(transitions),
        "best_overlap": best_score["overlap_ratio"],
        "best_overlap_cells": best_score["overlap_cells"],
        "best_wl": best_score["normalized_wl"],
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
    }


def validate_policy(policy, sizes, env_config, device, seed, temperature, soft_tau=None, relaxation="sigmoid", episodes=4):
    policy.eval()
    rows = []
    for episode_idx in range(max(int(episodes), 1)):
        forced_size = sizes[episode_idx % len(sizes)]
        _transitions, info = collect_episode(
            policy,
            sizes,
            env_config,
            device,
            seed + episode_idx,
            temperature,
            soft_tau=soft_tau,
            relaxation=relaxation,
            forced_size=forced_size,
            deterministic=True,
        )
        rows.append(info)
    policy.train()

    def mean(key):
        return sum(float(row.get(key, 0.0)) for row in rows) / max(len(rows), 1)

    return {
        "validation_episodes": len(rows),
        "validation_overlap": mean("best_overlap"),
        "validation_wirelength": mean("best_wl"),
        "validation_branch_violation": mean("branch_violation"),
        "validation_missed_pairs": mean("missed_pairs"),
        "validation_exact_overlap_pairs": mean("exact_overlap_pairs"),
        "validation_hard_pair_age_mean": mean("hard_pair_age_mean"),
        "validation_audit_pressure_scale": mean("audit_pressure_scale"),
        "validation_stop_probability": mean("stop_probability"),
        "validation_stop_gated_rate": mean("stop_gated"),
        "validation_stop_overlap": mean("stop_overlap"),
        "validation_false_stop_rate": mean("false_stop"),
        "validation_stop_rate": sum(1.0 if row.get("stop", False) else 0.0 for row in rows) / max(len(rows), 1),
    }


def update_metric_gated_tau(
    current_tau,
    overlap,
    branch_violation,
    missed_pairs,
    state,
    *,
    tau_min,
    tau_max,
    gamma_down,
    gamma_up,
    overlap_epsilon,
    branch_violation_max,
    missed_pairs_max,
    patience,
):
    improved = overlap < state["best_overlap"] - overlap_epsilon
    stable = branch_violation <= branch_violation_max and missed_pairs <= missed_pairs_max
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
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--reward-mode", choices=["aligned", "legacy"], default="aligned")
    parser.add_argument("--lag-reward-coef", type=float, default=1.0)
    parser.add_argument("--overlap-reward-coef", type=float, default=4.0)
    parser.add_argument("--overlap-regression-coef", type=float, default=16.0)
    parser.add_argument("--wirelength-reward-coef", type=float, default=0.20)
    parser.add_argument("--branch-violation-penalty", type=float, default=4.0)
    parser.add_argument("--missed-pair-penalty", type=float, default=0.02)
    parser.add_argument("--stop-gate-penalty", type=float, default=5.0)
    parser.add_argument("--stop-gate-overlap", type=float, default=0.02)
    parser.add_argument("--stop-no-progress-penalty", type=float, default=4.0)
    parser.add_argument("--soft-branch-epsilon", type=float, default=1e-4)
    parser.add_argument("--audit-missed-target", type=float, default=64.0)
    parser.add_argument("--audit-pressure-gamma", type=float, default=1.0)
    parser.add_argument("--audit-pressure-max", type=float, default=4.0)
    parser.add_argument("--wire-overlap-threshold", type=float, default=0.05)
    parser.add_argument("--no-soft-relax", action="store_true")
    parser.add_argument("--no-residual-flow", action="store_true")
    parser.add_argument("--no-phr-layer", action="store_true")
    parser.add_argument("--no-exact-audit", action="store_true")
    parser.add_argument("--no-density", action="store_true")
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
        fixed_pd_controls=args.fixed_pd_controls,
        ordering_representation=args.ordering_representation,
        branch_mode=args.branch_mode,
        al_mode=args.al_mode,
        reward_mode=args.reward_mode,
        lag_reward_coef=args.lag_reward_coef,
        exact_overlap_reward_coef=args.overlap_reward_coef,
        exact_overlap_regression_coef=args.overlap_regression_coef,
        exact_wirelength_reward_coef=args.wirelength_reward_coef,
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
    if args.resume_checkpoint:
        policy, _checkpoint = load_policy_checkpoint(args.resume_checkpoint, device)
    else:
        policy = OrderingPolicy(
            hidden_dim=args.hidden_dim,
            message_passes=args.message_passes,
            num_clusters=args.num_clusters,
            global_flow_rank=args.global_flow_rank,
        ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_exact_overlap = float("inf")
    best_lex_overlap = float("inf")
    best_lex_wl = float("inf")
    best_wire_under_threshold = float("inf")
    best_reward = -float("inf")
    hardening_state = {"best_overlap": float("inf"), "bad_windows": 0}
    soft_tau_state = float(args.soft_tau_start)
    started = time.time()

    for update_idx in range(args.updates):
        temperature = temperature_at(update_idx, args.updates)
        if args.metric_gated_hardening:
            use_soft = not args.no_soft_relax
            soft_tau = soft_tau_state if use_soft else None
        else:
            soft_cutoff = int(args.updates * args.soft_relax_frac)
            use_soft = (not args.no_soft_relax) and update_idx < soft_cutoff
            soft_tau = soft_tau_at(update_idx, max(soft_cutoff, 1), args.soft_tau_start, args.soft_tau_end) if use_soft else None
        transitions = []
        episode_infos = []
        for episode_idx in range(args.episodes_per_update):
            seed = args.seed + update_idx * 10_000 + episode_idx
            episode_transitions, info = collect_episode(
                policy,
                sizes,
                env_config,
                device,
                seed,
                temperature,
                soft_tau=soft_tau,
                relaxation=args.relaxation,
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
        avg_sampled_pairs = sum(info["sampled_pairs"] for info in episode_infos) / len(episode_infos)
        avg_cluster_pairs = sum(info["cluster_pairs"] for info in episode_infos) / len(episode_infos)
        avg_uncertain_pairs = sum(info["uncertain_pairs"] for info in episode_infos) / len(episode_infos)
        avg_new_active_pairs = sum(info["new_active_pairs"] for info in episode_infos) / len(episode_infos)
        avg_retained_pairs = sum(info["retained_pairs"] for info in episode_infos) / len(episode_infos)
        avg_hard_pair_age_mean = sum(info.get("hard_pair_age_mean", 0.0) for info in episode_infos) / len(episode_infos)
        avg_hard_pair_age_max = sum(info.get("hard_pair_age_max", 0.0) for info in episode_infos) / len(episode_infos)
        avg_hard_pair_age_min = sum(info.get("hard_pair_age_min", 0.0) for info in episode_infos) / len(episode_infos)
        avg_audit_pressure_scale = sum(info.get("audit_pressure_scale", 1.0) for info in episode_infos) / len(episode_infos)
        avg_retention_horizon = sum(info.get("retention_horizon", 0.0) for info in episode_infos) / len(episode_infos)
        stop_rate = sum(1.0 if info["stop"] else 0.0 for info in episode_infos) / len(episode_infos)
        avg_stop_probability = sum(info.get("stop_probability", 0.0) for info in episode_infos) / len(episode_infos)
        stop_gated_rate = sum(1.0 if info.get("stop_gated", False) else 0.0 for info in episode_infos) / len(episode_infos)
        false_stop_rate = sum(1.0 if info.get("false_stop", False) else 0.0 for info in episode_infos) / len(episode_infos)
        avg_stop_overlap = sum(info.get("stop_overlap", 0.0) for info in episode_infos) / len(episode_infos)
        avg_residual_norm = sum(info["residual_norm"] for info in episode_infos) / len(episode_infos)
        record = {
            "update": update_idx,
            "temperature": temperature,
            "soft_tau": soft_tau,
            "soft_relaxation": use_soft,
            "relaxation": args.relaxation,
            "ordering_representation": args.ordering_representation,
            "branch_mode": args.branch_mode,
            "al_mode": args.al_mode,
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
            "avg_sampled_pairs": avg_sampled_pairs,
            "avg_cluster_pairs": avg_cluster_pairs,
            "avg_uncertain_pairs": avg_uncertain_pairs,
            "avg_new_active_pairs": avg_new_active_pairs,
            "avg_retained_pairs": avg_retained_pairs,
            "avg_hard_pair_age_mean": avg_hard_pair_age_mean,
            "avg_hard_pair_age_max": avg_hard_pair_age_max,
            "avg_hard_pair_age_min": avg_hard_pair_age_min,
            "avg_audit_pressure_scale": avg_audit_pressure_scale,
            "avg_retention_horizon": avg_retention_horizon,
            "stop_rate": stop_rate,
            "avg_stop_probability": avg_stop_probability,
            "stop_gated_rate": stop_gated_rate,
            "false_stop_rate": false_stop_rate,
            "avg_stop_overlap": avg_stop_overlap,
            "avg_residual_norm": avg_residual_norm,
            "elapsed": time.time() - started,
            **metrics,
        }
        validation = None
        if args.validation_interval > 0 and update_idx % args.validation_interval == 0:
            validation = validate_policy(
                policy,
                sizes,
                env_config,
                device,
                args.seed + 1_000_000 + update_idx * 1000,
                temperature,
                soft_tau=soft_tau,
                relaxation=args.relaxation,
                episodes=args.validation_episodes,
            )
            record.update(validation)
        gate_overlap = record.get("validation_overlap", avg_overlap)
        gate_branch = record.get("validation_branch_violation", avg_branch_violation)
        gate_missed = record.get("validation_missed_pairs", avg_missed_pairs)
        if args.metric_gated_hardening:
            soft_tau_state = update_metric_gated_tau(
                soft_tau_state,
                gate_overlap,
                gate_branch,
                gate_missed,
                hardening_state,
                tau_min=args.soft_tau_end,
                tau_max=args.tau_max,
                gamma_down=args.tau_down,
                gamma_up=args.tau_up,
                overlap_epsilon=args.hardening_overlap_eps,
                branch_violation_max=args.hardening_branch_vmax,
                missed_pairs_max=args.hardening_missed_max,
                patience=args.hardening_patience,
            )
            record["next_soft_tau"] = soft_tau_state
            record["hardening_best_overlap"] = hardening_state["best_overlap"]
            record["hardening_bad_windows"] = hardening_state["bad_windows"]
        record["checkpoint_metric_source"] = "validation" if validation else "training"
        record["checkpoint_metric_overlap"] = record.get("validation_overlap", avg_overlap)
        record["checkpoint_metric_wirelength"] = record.get("validation_wirelength", avg_wl)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        metric_overlap = record["checkpoint_metric_overlap"]
        metric_wl = record["checkpoint_metric_wirelength"]

        latest_path = checkpoint_dir / "latest.pt"
        save_policy_checkpoint(policy, latest_path, config=vars(args), stats=record)
        save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_latest.pt", config=vars(args), stats=record)
        if avg_reward > best_reward:
            best_reward = avg_reward
            save_policy_checkpoint(policy, checkpoint_dir / "shaped_reward_debug.pt", config=vars(args), stats=record)
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_best_reward.pt", config=vars(args), stats=record)
        if metric_overlap < best_exact_overlap:
            best_exact_overlap = metric_overlap
            save_policy_checkpoint(policy, checkpoint_dir / "best_exact_overlap.pt", config=vars(args), stats=record)
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_best_overlap.pt", config=vars(args), stats=record)
        if (metric_overlap, metric_wl) < (best_lex_overlap, best_lex_wl):
            best_lex_overlap, best_lex_wl = metric_overlap, metric_wl
            save_policy_checkpoint(policy, checkpoint_dir / "best_lexicographic.pt", config=vars(args), stats=record)
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_best_validation.pt", config=vars(args), stats=record)
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy.pt", config=vars(args), stats=record)
        if metric_overlap <= args.wire_overlap_threshold and metric_wl < best_wire_under_threshold:
            best_wire_under_threshold = metric_wl
            save_policy_checkpoint(policy, checkpoint_dir / "best_wire_given_overlap_threshold.pt", config=vars(args), stats=record)


if __name__ == "__main__":
    main()
