"""Run named PPO architecture ablations as executable train_ppo variants."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "baseline": [],
    "independent_pair_branches": ["--branch-mode", "independent_pair"],
    "cplus_only_al": ["--al-mode", "positive_only"],
    "dag_ordering": ["--ordering-representation", "dag"],
    "no_density": ["--no-density"],
    "no_phr_layer": ["--no-phr-layer"],
    "no_exact_audit": ["--no-exact-audit"],
    "fixed_pd_controls": ["--fixed-pd-controls"],
}


def read_last_jsonl(path):
    if not path.exists():
        return {}
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = json.loads(line)
    return last or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="baseline,independent_pair_branches,cplus_only_al")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--coordinate-steps", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-passes", type=int, default=2)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--global-flow-rank", type=int, default=2)
    parser.add_argument("--relaxation", choices=["sigmoid", "neuralsort", "gumbel_sinkhorn"], default="sigmoid")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sizes", default="2:20,3:25,2:30,3:50")
    parser.add_argument("--checkpoint-dir", default="checkpoints/ablations")
    parser.add_argument("--log-dir", default="training_logs/ablations")
    parser.add_argument("--summary", default="training_logs/ppo_ablation_train_summary.json")
    args = parser.parse_args()

    selected = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in selected if item not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {unknown}. Available: {sorted(VARIANTS)}")

    summary = {}
    for offset, variant in enumerate(selected):
        log_path = Path(args.log_dir) / f"{variant}.jsonl"
        checkpoint_dir = Path(args.checkpoint_dir) / variant
        command = [
            sys.executable,
            "train_ppo.py",
            "--device",
            args.device,
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
            "--relaxation",
            args.relaxation,
            "--seed",
            str(args.seed + 10_000 * offset),
            "--sizes",
            args.sizes,
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--log",
            str(log_path),
            *VARIANTS[variant],
        ]
        subprocess.run(command, check=True)
        summary[variant] = {
            "command": command,
            "checkpoint_dir": str(checkpoint_dir),
            "log": str(log_path),
            "last_record": read_last_jsonl(log_path),
        }

    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
