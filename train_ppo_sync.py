"""Synchronized multi-GPU PPO for sequence-pair ordering policies."""

import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from env import EnvConfig
from ordering_policy import OrderingPolicy, load_policy_checkpoint, save_policy_checkpoint
from ppo import ppo_update
from train_ppo import collect_episode, parse_sizes, soft_tau_at, temperature_at, update_metric_gated_tau, validate_policy


def broadcast_parameters(module, src=0):
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=src)


def reduce_mean(value, device):
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor.item()


def worker(local_rank, world_size, args):
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    dist.init_process_group(
        backend=backend,
        rank=local_rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=args.dist_timeout_minutes),
    )

    random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed)
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
    broadcast_parameters(policy, src=0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    log_path = Path(args.log)
    if local_rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
        for episode_idx in range(args.episodes_per_rank):
            seed = args.seed + update_idx * 100_000 + local_rank * 10_000 + episode_idx
            forced_size = None
            if not args.random_rank_sizes:
                forced_size = sizes[(update_idx * args.episodes_per_rank + episode_idx) % len(sizes)]
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
            )
            transitions.extend(episode_transitions)
            episode_infos.append(info)

        metrics = ppo_update(
            policy,
            optimizer,
            transitions,
            update_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            sync_gradients=True,
            equivariance_coef=args.equivariance_coef,
        )

        local_overlap = sum(info["best_overlap"] for info in episode_infos) / len(episode_infos)
        local_wl = sum(info["best_wl"] for info in episode_infos) / len(episode_infos)
        local_reward = sum(info["last_reward"] for info in episode_infos) / len(episode_infos)
        local_density = sum(info["density_overflow"] for info in episode_infos) / len(episode_infos)
        local_branch_violation = sum(info["branch_violation"] for info in episode_infos) / len(episode_infos)
        local_boundary_violation = sum(info["boundary_violation"] for info in episode_infos) / len(episode_infos)
        local_lag_before = sum(info["lag_before"] for info in episode_infos) / len(episode_infos)
        local_lag_after = sum(info["lag_after"] for info in episode_infos) / len(episode_infos)
        local_pd_steps = sum(info["pd_steps"] for info in episode_infos) / len(episode_infos)
        local_rho = sum(info["rho"] for info in episode_infos) / len(episode_infos)
        local_eta = sum(info["eta"] for info in episode_infos) / len(episode_infos)
        local_alpha = sum(info["alpha"] for info in episode_infos) / len(episode_infos)
        local_step_scale = sum(info["step_scale"] for info in episode_infos) / len(episode_infos)
        local_pair_emphasis = sum(info["pair_emphasis"] for info in episode_infos) / len(episode_infos)
        local_tau = sum(info["tau"] for info in episode_infos) / len(episode_infos)
        local_branch_pressure = sum(info["branch_pressure"] for info in episode_infos) / len(episode_infos)
        local_density_pressure = sum(info["density_pressure"] for info in episode_infos) / len(episode_infos)
        local_boundary_pressure = sum(info["boundary_pressure"] for info in episode_infos) / len(episode_infos)
        local_missed_pairs = sum(info["missed_pairs"] for info in episode_infos) / len(episode_infos)
        local_exact_overlap_pairs = sum(info.get("exact_overlap_pairs", 0) for info in episode_infos) / len(episode_infos)
        local_sampled_pairs = sum(info["sampled_pairs"] for info in episode_infos) / len(episode_infos)
        local_cluster_pairs = sum(info["cluster_pairs"] for info in episode_infos) / len(episode_infos)
        local_uncertain_pairs = sum(info["uncertain_pairs"] for info in episode_infos) / len(episode_infos)
        local_new_active_pairs = sum(info["new_active_pairs"] for info in episode_infos) / len(episode_infos)
        local_retained_pairs = sum(info["retained_pairs"] for info in episode_infos) / len(episode_infos)
        local_hard_pair_age_mean = sum(info.get("hard_pair_age_mean", 0.0) for info in episode_infos) / len(episode_infos)
        local_hard_pair_age_max = sum(info.get("hard_pair_age_max", 0.0) for info in episode_infos) / len(episode_infos)
        local_hard_pair_age_min = sum(info.get("hard_pair_age_min", 0.0) for info in episode_infos) / len(episode_infos)
        local_audit_pressure_scale = sum(info.get("audit_pressure_scale", 1.0) for info in episode_infos) / len(episode_infos)
        local_retention_horizon = sum(info.get("retention_horizon", 0.0) for info in episode_infos) / len(episode_infos)
        local_stop_rate = sum(1.0 if info["stop"] else 0.0 for info in episode_infos) / len(episode_infos)
        local_stop_probability = sum(info.get("stop_probability", 0.0) for info in episode_infos) / len(episode_infos)
        local_stop_gated_rate = sum(1.0 if info.get("stop_gated", False) else 0.0 for info in episode_infos) / len(episode_infos)
        local_false_stop_rate = sum(1.0 if info.get("false_stop", False) else 0.0 for info in episode_infos) / len(episode_infos)
        local_stop_overlap = sum(info.get("stop_overlap", 0.0) for info in episode_infos) / len(episode_infos)
        local_residual_norm = sum(info["residual_norm"] for info in episode_infos) / len(episode_infos)
        local_dual_clamp = sum(info.get("dual_clamp_fraction", 0.0) for info in episode_infos) / len(episode_infos)
        local_overlap_delta = sum(info.get("overlap_delta", 0.0) for info in episode_infos) / len(episode_infos)
        local_branch_penalty = sum(info.get("branch_violation_penalty", 0.0) for info in episode_infos) / len(episode_infos)
        local_missed_penalty = sum(info.get("missed_pair_penalty", 0.0) for info in episode_infos) / len(episode_infos)
        avg_overlap = reduce_mean(local_overlap, device)
        avg_wl = reduce_mean(local_wl, device)
        avg_reward = reduce_mean(local_reward, device)
        avg_density = reduce_mean(local_density, device)
        avg_branch_violation = reduce_mean(local_branch_violation, device)
        avg_boundary_violation = reduce_mean(local_boundary_violation, device)
        avg_lag_before = reduce_mean(local_lag_before, device)
        avg_lag_after = reduce_mean(local_lag_after, device)
        avg_pd_steps = reduce_mean(local_pd_steps, device)
        avg_rho = reduce_mean(local_rho, device)
        avg_eta = reduce_mean(local_eta, device)
        avg_alpha = reduce_mean(local_alpha, device)
        avg_step_scale = reduce_mean(local_step_scale, device)
        avg_pair_emphasis = reduce_mean(local_pair_emphasis, device)
        avg_tau = reduce_mean(local_tau, device)
        avg_branch_pressure = reduce_mean(local_branch_pressure, device)
        avg_density_pressure = reduce_mean(local_density_pressure, device)
        avg_boundary_pressure = reduce_mean(local_boundary_pressure, device)
        avg_missed_pairs = reduce_mean(local_missed_pairs, device)
        avg_exact_overlap_pairs = reduce_mean(local_exact_overlap_pairs, device)
        avg_sampled_pairs = reduce_mean(local_sampled_pairs, device)
        avg_cluster_pairs = reduce_mean(local_cluster_pairs, device)
        avg_uncertain_pairs = reduce_mean(local_uncertain_pairs, device)
        avg_new_active_pairs = reduce_mean(local_new_active_pairs, device)
        avg_retained_pairs = reduce_mean(local_retained_pairs, device)
        avg_hard_pair_age_mean = reduce_mean(local_hard_pair_age_mean, device)
        avg_hard_pair_age_max = reduce_mean(local_hard_pair_age_max, device)
        avg_hard_pair_age_min = reduce_mean(local_hard_pair_age_min, device)
        avg_audit_pressure_scale = reduce_mean(local_audit_pressure_scale, device)
        avg_retention_horizon = reduce_mean(local_retention_horizon, device)
        stop_rate = reduce_mean(local_stop_rate, device)
        avg_stop_probability = reduce_mean(local_stop_probability, device)
        stop_gated_rate = reduce_mean(local_stop_gated_rate, device)
        false_stop_rate = reduce_mean(local_false_stop_rate, device)
        avg_stop_overlap = reduce_mean(local_stop_overlap, device)
        avg_residual_norm = reduce_mean(local_residual_norm, device)
        avg_dual_clamp = reduce_mean(local_dual_clamp, device)
        avg_overlap_delta = reduce_mean(local_overlap_delta, device)
        avg_branch_penalty = reduce_mean(local_branch_penalty, device)
        avg_missed_penalty = reduce_mean(local_missed_penalty, device)

        for key, value in list(metrics.items()):
            if key != "updates":
                metrics[key] = reduce_mean(value, device)

        if local_rank == 0:
            record = {
                "update": update_idx,
                "world_size": world_size,
                "episodes": args.episodes_per_rank * world_size,
                "transitions": len(transitions) * world_size,
                "temperature": temperature,
                "soft_tau": soft_tau,
                "soft_relaxation": use_soft,
                "random_rank_sizes": args.random_rank_sizes,
                "relaxation": args.relaxation,
                "ordering_representation": args.ordering_representation,
                "branch_mode": args.branch_mode,
                "al_mode": args.al_mode,
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
                "avg_dual_clamp_fraction": avg_dual_clamp,
                "avg_overlap_delta": avg_overlap_delta,
                "avg_branch_violation_penalty": avg_branch_penalty,
                "avg_missed_pair_penalty": avg_missed_penalty,
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
            if args.metric_gated_hardening:
                hardening_source = "hold"
                if validation is not None or args.validation_interval <= 0:
                    gate_overlap = record.get("validation_overlap", avg_overlap)
                    gate_branch = record.get("validation_branch_violation", avg_branch_violation)
                    gate_missed = record.get("validation_missed_pairs", avg_missed_pairs)
                    hardening_source = "validation" if validation is not None else "training_fallback"
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
                record["hardening_source"] = hardening_source
            authority_metrics_available = validation is not None or args.validation_interval <= 0
            record["checkpoint_metric_source"] = (
                "validation" if validation else ("training_fallback" if args.validation_interval <= 0 else "held")
            )
            record["checkpoint_metric_overlap"] = record.get("validation_overlap", avg_overlap)
            record["checkpoint_metric_wirelength"] = record.get("validation_wirelength", avg_wl)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

            metric_overlap = record["checkpoint_metric_overlap"]
            metric_wl = record["checkpoint_metric_wirelength"]

            save_policy_checkpoint(policy, checkpoint_dir / "latest.pt", config=vars(args), stats=record)
            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_latest.pt", config=vars(args), stats=record)
            if avg_reward > best_reward:
                best_reward = avg_reward
                save_policy_checkpoint(policy, checkpoint_dir / "shaped_reward_debug.pt", config=vars(args), stats=record)
                save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_best_reward.pt", config=vars(args), stats=record)
            if authority_metrics_available and metric_overlap < best_exact_overlap:
                best_exact_overlap = metric_overlap
                save_policy_checkpoint(policy, checkpoint_dir / "best_exact_overlap.pt", config=vars(args), stats=record)
                save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_best_overlap.pt", config=vars(args), stats=record)
            if authority_metrics_available and (metric_overlap, metric_wl) < (best_lex_overlap, best_lex_wl):
                best_lex_overlap, best_lex_wl = metric_overlap, metric_wl
                save_policy_checkpoint(policy, checkpoint_dir / "best_lexicographic.pt", config=vars(args), stats=record)
                save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_best_validation.pt", config=vars(args), stats=record)
                save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy.pt", config=vars(args), stats=record)
            if authority_metrics_available and metric_overlap <= args.wire_overlap_threshold and metric_wl < best_wire_under_threshold:
                best_wire_under_threshold = metric_wl
                save_policy_checkpoint(policy, checkpoint_dir / "best_wire_given_overlap_threshold.pt", config=vars(args), stats=record)

        if args.metric_gated_hardening:
            state_tensor = torch.tensor(
                [
                    float(soft_tau_state),
                    float(hardening_state["best_overlap"]),
                    float(hardening_state["bad_windows"]),
                ],
                dtype=torch.float32,
                device=device,
            )
            dist.broadcast(state_tensor, src=0)
            soft_tau_state = float(state_tensor[0].item())
            hardening_state["best_overlap"] = float(state_tensor[1].item())
            hardening_state["bad_windows"] = int(round(float(state_tensor[2].item())))

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-rank", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--coordinate-steps", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4)
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
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo_sync")
    parser.add_argument("--log", default="training_logs/ppo_sync.jsonl")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29571)
    parser.add_argument("--dist-timeout-minutes", type=int, default=60)
    parser.add_argument(
        "--random-rank-sizes",
        action="store_true",
        help="Let each rank sample problem sizes independently. Disabled by default to avoid sync rank skew.",
    )
    args = parser.parse_args()

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    world_size = len(gpu_ids)
    mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
