"""Policy validation through declared environment transitions."""

import argparse
import json
from pathlib import Path

import torch

from env import EnvConfig, PlacementOrderingEnv, write_positions
from ordering_policy import load_policy_checkpoint
from placement import calculate_normalized_metrics, generate_placement_input
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


def score_current(cell_features, pin_features, edge_list):
    metrics = calculate_normalized_metrics(
        cell_features.detach().cpu(),
        pin_features.detach().cpu(),
        edge_list.detach().cpu(),
    )
    return metrics


def geometry_result(cell_features, pin_features, edge_list):
    metrics = score_current(cell_features, pin_features, edge_list)
    metrics["source"] = "input_noop"
    return metrics


def policy_result(policy, cell_features, pin_features, edge_list, samples=1, temperature=0.7, steps=4):
    best = None
    for idx in range(max(samples, 1)):
        config = EnvConfig(horizon=steps, soft_relaxation=False)
        env = PlacementOrderingEnv(cell_features, pin_features, edge_list, config)
        memory = policy.initial_memory(cell_features.device)
        for _ in range(steps):
            graph = env.graph_state(memory=memory)
            with torch.no_grad():
                action = policy.sample_action(
                    graph,
                    temperature=temperature,
                    deterministic=(idx == 0),
                )
                memory = action.next_memory.detach()
            _reward, done, _info = env.step_action(action)
            if done:
                break
        score, centers = env.best_candidate()
        placed = write_positions(cell_features, centers)
        metrics = score_current(placed, pin_features, edge_list)
        metrics["source"] = "deterministic_env" if idx == 0 else f"sample_env_{idx}"
        metrics["env_overlap_ratio"] = score["overlap_ratio"]
        metrics["env_normalized_wl"] = score["normalized_wl"]
        key = (metrics["num_cells_with_overlaps"], metrics["overlap_ratio"], metrics["normalized_wl"])
        if best is None or key < best[0]:
            best = (key, metrics)
    return best[1]


def evaluate_cases(policy, cases, device, samples=1, temperature=0.7, steps=4):
    rows = []
    for case in cases:
        test_id, cell_features, pin_features, edge_list = make_case(case, device)
        baseline = geometry_result(cell_features, pin_features, edge_list)
        policy_metrics = policy_result(
            policy,
            cell_features,
            pin_features,
            edge_list,
            samples=samples,
            temperature=temperature,
            steps=steps,
        )
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
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, _ = load_policy_checkpoint(args.checkpoint, device)
    rows = evaluate_cases(
        policy,
        parse_cases(args.cases),
        device,
        samples=args.samples,
        temperature=args.temperature,
        steps=args.steps,
    )
    result = {"checkpoint": args.checkpoint, "rows": rows, "summary": summarize(rows)}
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
