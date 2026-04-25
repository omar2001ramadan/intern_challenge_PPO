"""Ablations for PPO ordering quality."""

import argparse
import json
from pathlib import Path

import torch

from ordering_policy import OrderingPolicy, load_policy_checkpoint
from validate_policy import evaluate_cases, make_case, parse_cases, policy_result, summarize


def geometry_only_rows(cases, device):
    rows = []
    for case in cases:
        test_id, cell_features, pin_features, edge_list = make_case(case, device)
        from validate_policy import geometry_result

        baseline = geometry_result(cell_features, pin_features, edge_list)
        rows.append(
            {
                "test_id": test_id,
                "geometry_overlap": baseline["overlap_ratio"],
                "geometry_wl": baseline["normalized_wl"],
                "policy_overlap": baseline["overlap_ratio"],
                "policy_wl": baseline["normalized_wl"],
                "policy_source": "geometry",
                "delta_wl": 0.0,
                "delta_overlap": 0.0,
            }
        )
    return rows


def untrained_rows(cases, device, hidden_dim, message_passes, samples, temperature):
    policy = OrderingPolicy(hidden_dim=hidden_dim, message_passes=message_passes).to(device)
    return evaluate_cases(policy, cases, device, samples=samples, temperature=temperature)


def checkpoint_rows(path, cases, device, samples, temperature):
    policy, _ = load_policy_checkpoint(path, device)
    return evaluate_cases(policy, cases, device, samples=samples, temperature=temperature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", default="first10")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-passes", type=int, default=2)
    parser.add_argument("--output", default="training_logs/ppo_ablation.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    cases = parse_cases(args.cases)

    variants = {
        "geometry": geometry_only_rows(cases, device),
        "untrained": untrained_rows(
            cases,
            device,
            args.hidden_dim,
            args.message_passes,
            args.samples,
            args.temperature,
        ),
    }
    for checkpoint in args.checkpoint:
        variants[f"checkpoint:{checkpoint}"] = checkpoint_rows(
            checkpoint,
            cases,
            device,
            args.samples,
            args.temperature,
        )

    result = {name: {"rows": rows, "summary": summarize(rows)} for name, rows in variants.items()}
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
