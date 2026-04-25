"""Policy-only validation against deterministic sequence-pair baselines."""

import argparse
import json
from pathlib import Path

import torch

from active_set import build_initial_active_pairs
from env import write_positions
from induce_branches import sequence_pair_from_centers
from ordering_policy import OrderingPolicy, build_graph_state, load_policy_checkpoint
from placement import calculate_normalized_metrics, generate_placement_input
from primal_dual import sequence_pair_legalize
from test import TEST_CASES


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


def score_sequence_pair(cell_features, pin_features, edge_list, seq_plus, seq_minus):
    centers = sequence_pair_legalize(
        cell_features,
        seq_plus,
        seq_minus,
        reference_centers=cell_features[:, 2:4],
    )
    placed = write_positions(cell_features, centers)
    metrics = calculate_normalized_metrics(placed.detach().cpu(), pin_features.detach().cpu(), edge_list.detach().cpu())
    return metrics


def geometry_result(cell_features, pin_features, edge_list):
    seq_plus, seq_minus = sequence_pair_from_centers(cell_features)
    return score_sequence_pair(cell_features, pin_features, edge_list, seq_plus, seq_minus)


def policy_result(policy, cell_features, pin_features, edge_list, samples=1, temperature=0.7):
    active_pairs = build_initial_active_pairs(cell_features)
    branch_duals = torch.zeros((active_pairs.shape[0], 4), dtype=cell_features.dtype, device=cell_features.device)
    boundary_duals = torch.zeros((cell_features.shape[0], 4), dtype=cell_features.dtype, device=cell_features.device)
    graph = build_graph_state(cell_features, pin_features, edge_list, active_pairs, branch_duals, boundary_duals)

    best = None
    with torch.no_grad():
        deterministic_action = policy.deterministic_sequence_pair(graph)
        candidates = [(deterministic_action.seq_plus, deterministic_action.seq_minus, "deterministic")]
        for idx in range(max(samples - 1, 0)):
            action = policy.sample_sequence_pair(graph, temperature=temperature)
            candidates.append((action.seq_plus, action.seq_minus, f"sample_{idx}"))

    for seq_plus, seq_minus, source in candidates:
        metrics = score_sequence_pair(cell_features, pin_features, edge_list, seq_plus, seq_minus)
        metrics["source"] = source
        key = (metrics["num_cells_with_overlaps"], metrics["overlap_ratio"], metrics["normalized_wl"])
        if best is None or key < best[0]:
            best = (key, metrics)
    return best[1]


def evaluate_cases(policy, cases, device, samples=1, temperature=0.7):
    rows = []
    for case in cases:
        test_id, cell_features, pin_features, edge_list = make_case(case, device)
        baseline = geometry_result(cell_features, pin_features, edge_list)
        policy_metrics = policy_result(policy, cell_features, pin_features, edge_list, samples=samples, temperature=temperature)
        row = {
            "test_id": test_id,
            "geometry_overlap": baseline["overlap_ratio"],
            "geometry_wl": baseline["normalized_wl"],
            "policy_overlap": policy_metrics["overlap_ratio"],
            "policy_wl": policy_metrics["normalized_wl"],
            "policy_source": policy_metrics["source"],
            "delta_wl": policy_metrics["normalized_wl"] - baseline["normalized_wl"],
            "delta_overlap": policy_metrics["overlap_ratio"] - baseline["overlap_ratio"],
        }
        rows.append(row)
    return rows


def summarize(rows):
    return {
        "cases": len(rows),
        "avg_geometry_overlap": sum(row["geometry_overlap"] for row in rows) / len(rows),
        "avg_geometry_wl": sum(row["geometry_wl"] for row in rows) / len(rows),
        "avg_policy_overlap": sum(row["policy_overlap"] for row in rows) / len(rows),
        "avg_policy_wl": sum(row["policy_wl"] for row in rows) / len(rows),
        "avg_delta_wl": sum(row["delta_wl"] for row in rows) / len(rows),
        "avg_delta_overlap": sum(row["delta_overlap"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", default="first10")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, _ = load_policy_checkpoint(args.checkpoint, device)
    rows = evaluate_cases(policy, parse_cases(args.cases), device, samples=args.samples, temperature=args.temperature)
    result = {"checkpoint": args.checkpoint, "rows": rows, "summary": summarize(rows)}
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
