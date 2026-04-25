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
from ordering_policy import OrderingPolicy, save_policy_checkpoint
from ppo import ppo_update
from train_ppo import collect_episode, parse_sizes, soft_tau_at, temperature_at


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
    )
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
        local_sampled_pairs = sum(info["sampled_pairs"] for info in episode_infos) / len(episode_infos)
        local_cluster_pairs = sum(info["cluster_pairs"] for info in episode_infos) / len(episode_infos)
        local_uncertain_pairs = sum(info["uncertain_pairs"] for info in episode_infos) / len(episode_infos)
        local_new_active_pairs = sum(info["new_active_pairs"] for info in episode_infos) / len(episode_infos)
        local_retained_pairs = sum(info["retained_pairs"] for info in episode_infos) / len(episode_infos)
        local_stop_rate = sum(1.0 if info["stop"] else 0.0 for info in episode_infos) / len(episode_infos)
        local_residual_norm = sum(info["residual_norm"] for info in episode_infos) / len(episode_infos)
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
        avg_sampled_pairs = reduce_mean(local_sampled_pairs, device)
        avg_cluster_pairs = reduce_mean(local_cluster_pairs, device)
        avg_uncertain_pairs = reduce_mean(local_uncertain_pairs, device)
        avg_new_active_pairs = reduce_mean(local_new_active_pairs, device)
        avg_retained_pairs = reduce_mean(local_retained_pairs, device)
        stop_rate = reduce_mean(local_stop_rate, device)
        avg_residual_norm = reduce_mean(local_residual_norm, device)

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

            save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy_latest.pt", config=vars(args), stats=record)
            if (avg_overlap, avg_wl) < (best_overlap, best_wl):
                best_overlap, best_wl = avg_overlap, avg_wl
                save_policy_checkpoint(policy, checkpoint_dir / "ordering_policy.pt", config=vars(args), stats=record)

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
