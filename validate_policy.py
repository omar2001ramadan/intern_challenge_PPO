"""Policy validation through declared terminal and audited inference modes."""

import argparse
import json
from pathlib import Path

import torch

from env import EnvConfig, PlacementOrderingEnv, write_positions
from ordering_policy import hierarchical_active_branch_weights, load_policy_checkpoint
from placement import calculate_normalized_metrics, generate_placement_input
from rollout import select_by_exact_overlap_then_wirelength
from test import TEST_CASES


def default_device_arg():
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def initialize_random_spread(cell_features):
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6
    angles = torch.rand(total_cells, device=cell_features.device) * 2 * 3.14159
    radii = torch.rand(total_cells, device=cell_features.device) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)
    return cell_features


def parse_cases(value):
    if value == "first10":
        return TEST_CASES[:10]
    selected = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        elif part:
            selected.add(int(part))
    return [case for case in TEST_CASES if case[0] in selected]


def make_case(case, device):
    test_id, num_macros, num_std_cells, seed = case
    torch.manual_seed(seed)
    cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)
    cell_features = initialize_random_spread(cell_features)
    return test_id, cell_features.to(device), pin_features.to(device), edge_list.to(device)


def score_current(cell_features, pin_features, edge_list):
    return calculate_normalized_metrics(
        cell_features.detach().cpu(),
        pin_features.detach().cpu(),
        edge_list.detach().cpu(),
    )


def geometry_result(cell_features, pin_features, edge_list):
    metrics = score_current(cell_features, pin_features, edge_list)
    metrics["source"] = "input_noop"
    return metrics


def build_env_config(args):
    return EnvConfig(
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


def run_rollout(policy, cell_features, pin_features, edge_list, env_config, args, rollout_idx, mode):
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
                deterministic=args.deterministic and rollout_idx == 0,
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
        centers = env.centers.detach().clone()
        score = env._score_centers(centers)
    else:
        score, centers = env.best_candidate()
    placed = write_positions(cell_features, centers)
    metrics = score_current(placed, pin_features, edge_list)
    metrics["env_overlap_ratio"] = score["overlap_ratio"]
    metrics["env_normalized_wl"] = score["normalized_wl"]
    metrics["num_overlap_pairs"] = int(score["num_overlap_pairs"])
    metrics["overlap_cells"] = int(metrics["num_cells_with_overlaps"])
    metrics["last_info"] = last_info
    metrics["centers"] = centers.detach().clone()
    return metrics


def policy_result(policy, cell_features, pin_features, edge_list, args, mode):
    env_config = build_env_config(args)
    if mode == "terminal_policy":
        metrics = run_rollout(
            policy,
            cell_features,
            pin_features,
            edge_list,
            env_config,
            args,
            rollout_idx=0,
            mode=mode,
        )
        metrics["source"] = "terminal_policy"
        return metrics

    candidates = []
    for rollout_idx in range(max(int(args.samples), 1)):
        metrics = run_rollout(
            policy,
            cell_features,
            pin_features,
            edge_list,
            env_config,
            args,
            rollout_idx=rollout_idx,
            mode="audited_policy_ensemble",
        )
        metrics["source"] = "audited_policy_ensemble"
        candidates.append(
            {
                "X": metrics["centers"],
                "overlap_cells": metrics["overlap_cells"],
                "overlap_ratio": metrics["overlap_ratio"],
                "normalized_wl": metrics["normalized_wl"],
                "num_overlap_pairs": metrics["num_overlap_pairs"],
                "metrics": metrics,
            }
        )
    selected = select_by_exact_overlap_then_wirelength(candidates)
    selected_metrics = dict(selected["metrics"])
    selected_metrics["candidate_count"] = len(candidates)
    return selected_metrics


def evaluate_case(policy, case, device, args):
    test_id, cell_features, pin_features, edge_list = make_case(case, device)
    baseline = geometry_result(cell_features, pin_features, edge_list)
    row = {
        "test_id": test_id,
        "geometry_overlap": baseline["overlap_ratio"],
        "geometry_wl": baseline["normalized_wl"],
    }

    if args.mode in {"terminal_policy", "both"}:
        terminal = policy_result(policy, cell_features, pin_features, edge_list, args, "terminal_policy")
        row.update(
            {
                "terminal_overlap": terminal["overlap_ratio"],
                "terminal_wl": terminal["normalized_wl"],
                "terminal_source": terminal["source"],
            }
        )

    if args.mode in {"audited_policy_ensemble", "both"}:
        audited = policy_result(policy, cell_features, pin_features, edge_list, args, "audited_policy_ensemble")
        row.update(
            {
                "audited_overlap": audited["overlap_ratio"],
                "audited_wl": audited["normalized_wl"],
                "audited_source": audited["source"],
                "audited_candidates": audited.get("candidate_count", max(int(args.samples), 1)),
            }
        )

    if args.mode == "terminal_policy":
        row["policy_overlap"] = row["terminal_overlap"]
        row["policy_wl"] = row["terminal_wl"]
        row["policy_source"] = row["terminal_source"]
    elif args.mode == "audited_policy_ensemble":
        row["policy_overlap"] = row["audited_overlap"]
        row["policy_wl"] = row["audited_wl"]
        row["policy_source"] = row["audited_source"]
    else:
        row["gap_overlap"] = row["audited_overlap"] - row["terminal_overlap"]
        row["gap_wl"] = row["audited_wl"] - row["terminal_wl"]
        row["policy_overlap"] = row["audited_overlap"]
        row["policy_wl"] = row["audited_wl"]
        row["policy_source"] = "both"

    row["delta_wl"] = row["policy_wl"] - baseline["normalized_wl"]
    row["delta_overlap"] = row["policy_overlap"] - baseline["overlap_ratio"]
    return row


def summarize(rows, mode):
    summary = {
        "cases": len(rows),
        "avg_geometry_overlap": sum(row["geometry_overlap"] for row in rows) / len(rows),
        "avg_geometry_wl": sum(row["geometry_wl"] for row in rows) / len(rows),
    }
    if mode in {"terminal_policy", "both"}:
        summary["avg_terminal_overlap"] = sum(row["terminal_overlap"] for row in rows) / len(rows)
        summary["avg_terminal_wl"] = sum(row["terminal_wl"] for row in rows) / len(rows)
    if mode in {"audited_policy_ensemble", "both"}:
        summary["avg_audited_overlap"] = sum(row["audited_overlap"] for row in rows) / len(rows)
        summary["avg_audited_wl"] = sum(row["audited_wl"] for row in rows) / len(rows)
    if mode == "both":
        summary["avg_gap_overlap"] = sum(row["gap_overlap"] for row in rows) / len(rows)
        summary["avg_gap_wl"] = sum(row["gap_wl"] for row in rows) / len(rows)
        summary["audited_better_or_equal"] = sum(
            1 for row in rows if (row["audited_overlap"], row["audited_wl"]) <= (row["terminal_overlap"], row["terminal_wl"])
        )
    summary["avg_policy_overlap"] = sum(row["policy_overlap"] for row in rows) / len(rows)
    summary["avg_policy_wl"] = sum(row["policy_wl"] for row in rows) / len(rows)
    summary["avg_delta_wl"] = sum(row["delta_wl"] for row in rows) / len(rows)
    summary["avg_delta_overlap"] = sum(row["delta_overlap"] for row in rows) / len(rows)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=default_device_arg())
    parser.add_argument("--cases", default="first10")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mode", choices=["terminal_policy", "audited_policy_ensemble", "both"], default="audited_policy_ensemble")
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
        "summary": summarize(rows, args.mode),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
