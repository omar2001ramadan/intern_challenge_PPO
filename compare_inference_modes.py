"""Report terminal-policy Mode A versus audited-ensemble Mode B."""

import argparse
import json
from pathlib import Path

import torch

from env import EnvConfig, PlacementOrderingEnv
from ordering_policy import hierarchical_active_branch_weights, load_policy_checkpoint
from validate_policy import make_case, parse_cases


def candidate_key(item):
    score = item[0]
    return (
        score["overlap_cells"],
        score["overlap_ratio"],
        score["normalized_wl"],
        score["num_overlap_pairs"],
    )


def run_rollout(policy, cell_features, pin_features, edge_list, env_config, args, mode, rollout_idx):
    torch.manual_seed(args.seed + rollout_idx)
    env = PlacementOrderingEnv(cell_features, pin_features, edge_list, env_config)
    memory = policy.initial_memory(cell_features.device)
    last_info = {}
    for _step in range(args.steps):
        graph = env.graph_state(memory=memory)
        with torch.no_grad():
            action = policy.sample_action(
                graph,
                temperature=args.temperature,
                deterministic=args.deterministic,
            )
            memory = action.next_memory.detach()
            soft_weights = None
            if env_config.soft_relaxation:
                soft_weights = hierarchical_active_branch_weights(
                    action,
                    graph["active_pairs"],
                    relaxation=args.relaxation,
                    tau=float(action.tau.detach().item()),
                ).detach()
        _reward, done, last_info = env.step_action(
            action,
            entropy=action.entropy,
            soft_branch_weights=soft_weights,
            soft_tau=float(action.tau.detach().item()),
        )
        if done:
            break

    if mode == "terminal_policy":
        return env._score_centers(env.centers), env.centers.detach().clone(), last_info
    score, centers = env.best_candidate()
    return score, centers, last_info


def evaluate_case(policy, case, device, args):
    test_id, cell_features, pin_features, edge_list = make_case(case, device)
    env_config = EnvConfig(
        horizon=args.steps,
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

    terminal = run_rollout(
        policy,
        cell_features,
        pin_features,
        edge_list,
        env_config,
        args,
        "terminal_policy",
        rollout_idx=10_000 + test_id,
    )
    audited_candidates = [
        run_rollout(
            policy,
            cell_features,
            pin_features,
            edge_list,
            env_config,
            args,
            "audited_policy_ensemble",
            rollout_idx=20_000 + 100 * test_id + rollout_idx,
        )
        for rollout_idx in range(max(args.rollouts, 1))
    ]
    audited = min(audited_candidates, key=candidate_key)
    terminal_score = terminal[0]
    audited_score = audited[0]
    return {
        "test_id": test_id,
        "terminal_overlap": terminal_score["overlap_ratio"],
        "terminal_overlap_cells": terminal_score["overlap_cells"],
        "terminal_overlap_pairs": terminal_score["num_overlap_pairs"],
        "terminal_wl": terminal_score["normalized_wl"],
        "audited_overlap": audited_score["overlap_ratio"],
        "audited_overlap_cells": audited_score["overlap_cells"],
        "audited_overlap_pairs": audited_score["num_overlap_pairs"],
        "audited_wl": audited_score["normalized_wl"],
        "gap_overlap": audited_score["overlap_ratio"] - terminal_score["overlap_ratio"],
        "gap_overlap_cells": audited_score["overlap_cells"] - terminal_score["overlap_cells"],
        "gap_wl": audited_score["normalized_wl"] - terminal_score["normalized_wl"],
        "terminal_last_info": terminal[2],
        "audited_last_info": audited[2],
    }


def summarize(rows):
    if not rows:
        return {}
    return {
        "cases": len(rows),
        "avg_terminal_overlap": sum(row["terminal_overlap"] for row in rows) / len(rows),
        "avg_terminal_wl": sum(row["terminal_wl"] for row in rows) / len(rows),
        "avg_audited_overlap": sum(row["audited_overlap"] for row in rows) / len(rows),
        "avg_audited_wl": sum(row["audited_wl"] for row in rows) / len(rows),
        "avg_gap_overlap": sum(row["gap_overlap"] for row in rows) / len(rows),
        "avg_gap_wl": sum(row["gap_wl"] for row in rows) / len(rows),
        "audited_better_or_equal": sum(
            1
            for row in rows
            if (
                row["audited_overlap_cells"],
                row["audited_overlap"],
                row["audited_wl"],
            )
            <= (
                row["terminal_overlap_cells"],
                row["terminal_overlap"],
                row["terminal_wl"],
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", default="first10")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--deterministic", action="store_true")
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
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, checkpoint = load_policy_checkpoint(args.checkpoint, device)
    rows = [evaluate_case(policy, case, device, args) for case in parse_cases(args.cases)]
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_stats": checkpoint.get("stats", {}),
        "settings": vars(args),
        "rows": rows,
        "summary": summarize(rows),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
