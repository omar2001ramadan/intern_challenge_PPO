"""Train the sequence-pair ordering policy with PPO."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from env import EnvConfig, PlacementOrderingEnv
from ordering_policy import OrderingPolicy, hierarchical_active_branch_weights, save_policy_checkpoint
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


def make_problem(sizes, device, seed):
    num_macros, num_std_cells = random.choice(sizes)
    torch.manual_seed(seed)
    cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)
    cell_features = initialize_random_spread(cell_features, seed=seed + 17)
    return cell_features.to(device), pin_features.to(device), edge_list.to(device), (num_macros, num_std_cells)


def collect_episode(policy, sizes, env_config, device, seed, temperature, soft_tau=None, relaxation="sigmoid"):
    cell_features, pin_features, edge_list, size = make_problem(sizes, device, seed)
    env = PlacementOrderingEnv(cell_features, pin_features, edge_list, env_config)
    transitions = []
    infos = []
    memory = policy.initial_memory(device)

    for step_idx in range(env_config.horizon):
        graph = env.graph_state(memory=memory)
        action = policy.sample_action(graph, temperature=temperature)
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
        "sampled_pairs": infos[-1].get("sampled_pairs", 0) if infos else 0,
        "cluster_pairs": infos[-1].get("cluster_pairs", 0) if infos else 0,
        "uncertain_pairs": infos[-1].get("uncertain_pairs", 0) if infos else 0,
        "new_active_pairs": infos[-1].get("new_active_pairs", 0) if infos else 0,
        "retained_pairs": infos[-1].get("retained_pairs", 0) if infos else 0,
        "stop": infos[-1].get("stop", False) if infos else False,
        "residual_norm": infos[-1].get("residual_norm", 0.0) if infos else 0.0,
    }


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
    )
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

    best_overlap = float("inf")
    best_wl = float("inf")
    started = time.time()

    for update_idx in range(args.updates):
        temperature = temperature_at(update_idx, args.updates)
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
        avg_sampled_pairs = sum(info["sampled_pairs"] for info in episode_infos) / len(episode_infos)
        avg_cluster_pairs = sum(info["cluster_pairs"] for info in episode_infos) / len(episode_infos)
        avg_uncertain_pairs = sum(info["uncertain_pairs"] for info in episode_infos) / len(episode_infos)
        avg_new_active_pairs = sum(info["new_active_pairs"] for info in episode_infos) / len(episode_infos)
        avg_retained_pairs = sum(info["retained_pairs"] for info in episode_infos) / len(episode_infos)
        stop_rate = sum(1.0 if info["stop"] else 0.0 for info in episode_infos) / len(episode_infos)
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
            "avg_sampled_pairs": avg_sampled_pairs,
            "avg_cluster_pairs": avg_cluster_pairs,
            "avg_uncertain_pairs": avg_uncertain_pairs,
            "avg_new_active_pairs": avg_new_active_pairs,
            "avg_retained_pairs": avg_retained_pairs,
            "stop_rate": stop_rate,
            "avg_residual_norm": avg_residual_norm,
            "elapsed": time.time() - started,
            **metrics,
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        latest_path = checkpoint_dir / "ordering_policy_latest.pt"
        save_policy_checkpoint(policy, latest_path, config=vars(args), stats=record)
        if (avg_overlap, avg_wl) < (best_overlap, best_wl):
            best_overlap, best_wl = avg_overlap, avg_wl
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy.pt", config=vars(args), stats=record)


if __name__ == "__main__":
    main()
