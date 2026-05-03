"""Launch independent PPO trainers across multiple GPUs."""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys


def worker(gpu_id, args):
    cmd = [
        sys.executable,
        "train_ppo.py",
        "--device",
        "cuda:0",
        "--updates",
        str(args.updates),
        "--episodes-per-update",
        str(args.episodes_per_update),
        "--horizon",
        str(args.horizon),
        "--coordinate-steps",
        str(args.coordinate_steps),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--minibatch-size",
        str(args.minibatch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--message-passes",
        str(args.message_passes),
        "--num-clusters",
        str(args.num_clusters),
        "--global-flow-rank",
        str(args.global_flow_rank),
        "--lr",
        str(args.lr),
        "--equivariance-coef",
        str(args.equivariance_coef),
        "--distill-epochs",
        str(args.distill_epochs),
        "--distill-batch-size",
        str(args.distill_batch_size),
        "--distill-lr",
        str(args.distill_lr),
        "--distill-max-branch-pairs",
        str(args.distill_max_branch_pairs),
        "--teacher-lambda0",
        str(args.teacher_lambda0),
        "--teacher-anneal-updates",
        str(args.teacher_anneal_updates),
        "--teacher-aux-batch-size",
        str(args.teacher_aux_batch_size),
        "--teacher-aux-steps-per-update",
        str(args.teacher_aux_steps_per_update),
        "--ordering-representation",
        args.ordering_representation,
        "--branch-mode",
        args.branch_mode,
        "--al-mode",
        args.al_mode,
        "--relaxation",
        args.relaxation,
        "--seed",
        str(args.seed + 1000 * gpu_id),
        "--sizes",
        args.sizes,
        "--checkpoint-dir",
        f"{args.checkpoint_dir}/gpu{gpu_id}",
        "--log",
        f"{args.log_dir}/ppo_gpu{gpu_id}.jsonl",
    ]
    if args.teacher_dataset:
        cmd.extend(["--teacher-dataset", args.teacher_dataset])
    for flag, enabled in (
        ("--metric-gated-hardening", args.metric_gated_hardening),
        ("--no-residual-flow", args.no_residual_flow),
        ("--no-phr-layer", args.no_phr_layer),
        ("--no-exact-audit", args.no_exact_audit),
        ("--no-density", args.no_density),
        ("--fixed-pd-controls", args.fixed_pd_controls),
    ):
        if enabled:
            cmd.append(flag)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
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
    parser.add_argument("--teacher-dataset", default="")
    parser.add_argument("--distill-epochs", type=int, default=0)
    parser.add_argument("--distill-batch-size", type=int, default=1)
    parser.add_argument("--distill-lr", type=float, default=1e-4)
    parser.add_argument("--distill-max-branch-pairs", type=int, default=65_536)
    parser.add_argument("--teacher-lambda0", type=float, default=1.0)
    parser.add_argument("--teacher-anneal-updates", type=int, default=50)
    parser.add_argument("--teacher-aux-batch-size", type=int, default=1)
    parser.add_argument("--teacher-aux-steps-per-update", type=int, default=1)
    parser.add_argument("--metric-gated-hardening", action="store_true")
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
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo_multi")
    parser.add_argument("--log-dir", default="training_logs")
    args = parser.parse_args()

    gpu_ids = [int(item) for item in args.gpus.split(",") if item.strip()]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(gpu_id, args)) for gpu_id in gpu_ids]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(proc.exitcode)


if __name__ == "__main__":
    main()
