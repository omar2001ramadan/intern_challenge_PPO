"""Offline teacher outcome dataset generation.

The teacher path is intentionally separate from placement inference.  This
module may call an older solver to produce demonstrations, but nothing here is
imported by ``placement.py`` or the evaluated ``train_placement`` path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable

import torch

from active_set import canonicalize_pairs, connected_cell_pairs
from constraints import (
    all_branch_signed_constraints,
    exact_overlap_pairs,
    make_all_pairs,
    overlap_ratio_from_pairs,
)
from env import normalized_wirelength, write_positions
from induce_branches import sequence_pair_from_centers
from placement import generate_placement_input


@dataclass
class TeacherQualityConfig:
    max_demo_overlap: float = 0.0
    alpha_o: float = 8.0
    alpha_w: float = 1.0
    lambda_o: float = 10.0
    lambda_w: float = 1.0
    max_flow_fraction: float = 0.40
    max_label_pairs: int = 200_000
    all_pair_label_limit: int = 2048
    near_window: int = 8
    branch_margin: float = 1e-5
    stop_overlap_threshold: float = 0.0


def parse_sizes(value: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        macros, std_cells = item.split(":")
        sizes.append((int(macros), int(std_cells)))
    if not sizes:
        raise ValueError("At least one size must be supplied.")
    return sizes


def initialize_random_spread(cell_features: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(int(seed))
    total_cells = int(cell_features.shape[0])
    total_area = float(cell_features[:, 0].sum().item())
    spread_radius = (total_area ** 0.5) * 0.6
    angles = torch.rand(total_cells, device=cell_features.device) * 2.0 * math.pi
    radii = torch.rand(total_cells, device=cell_features.device) * spread_radius
    cell_features = cell_features.clone()
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)
    return cell_features


def score_official_metrics(cell_features: torch.Tensor, pin_features: torch.Tensor, edge_list: torch.Tensor) -> dict:
    with torch.no_grad():
        pairs = exact_overlap_pairs(cell_features)
        overlap_ratio, overlap_cells = overlap_ratio_from_pairs(int(cell_features.shape[0]), pairs)
        norm_wl = normalized_wirelength(cell_features, pin_features, edge_list)
    return {
        "overlap_ratio": float(overlap_ratio),
        "overlap_cells": int(overlap_cells),
        "normalized_wl": float(norm_wl),
        "num_overlap_pairs": int(pairs.shape[0]),
    }


def _near_pairs_from_centers(centers: torch.Tensor, window: int) -> torch.Tensor:
    if centers.shape[0] <= 1 or window <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=centers.device)
    chunks = []
    for dim in (0, 1):
        order = torch.argsort(centers[:, dim])
        for offset in range(1, min(int(window), int(order.numel() - 1)) + 1):
            chunks.append(torch.stack([order[:-offset], order[offset:]], dim=1))
    if not chunks:
        return torch.empty((0, 2), dtype=torch.long, device=centers.device)
    return canonicalize_pairs(torch.cat(chunks, dim=0))


def _cap_pairs(pairs: torch.Tensor, max_pairs: int) -> torch.Tensor:
    pairs = canonicalize_pairs(pairs)
    if pairs.shape[0] <= int(max_pairs):
        return pairs
    return pairs[: int(max_pairs)]


def extract_teacher_active_pairs(
    x0: torch.Tensor,
    x_star: torch.Tensor,
    cell_features: torch.Tensor,
    pin_features: torch.Tensor | None = None,
    edge_list: torch.Tensor | None = None,
    *,
    near_window: int = 8,
    max_pairs: int = 200_000,
) -> torch.Tensor:
    initial = write_positions(cell_features, x0)
    final = write_positions(cell_features, x_star)
    chunks = [
        exact_overlap_pairs(initial),
        exact_overlap_pairs(final),
        _near_pairs_from_centers(x0, near_window),
        _near_pairs_from_centers(x_star, near_window),
    ]
    if pin_features is not None and edge_list is not None:
        chunks.append(connected_cell_pairs(pin_features, edge_list, device=cell_features.device, max_pairs=max_pairs))
    nonempty = [pairs for pairs in chunks if pairs.numel() > 0]
    if not nonempty:
        return torch.empty((0, 2), dtype=torch.long, device=cell_features.device)
    return _cap_pairs(torch.cat(nonempty, dim=0), max_pairs)


def extract_branch_labels(
    x_star: torch.Tensor,
    cell_features: torch.Tensor,
    pairs: torch.Tensor | None = None,
    *,
    max_pairs: int = 200_000,
    all_pair_limit: int = 2048,
    near_window: int = 8,
    margin: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(cell_features.shape[0])
    if pairs is None:
        if n <= int(all_pair_limit):
            pairs = make_all_pairs(n, device=cell_features.device)
        else:
            pairs = _near_pairs_from_centers(x_star, near_window)
    pairs = _cap_pairs(pairs, max_pairs)
    if pairs.numel() == 0:
        return pairs.reshape(0, 2), torch.empty(0, dtype=torch.long, device=cell_features.device)

    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    constraints = all_branch_signed_constraints(x_star, widths, heights, pairs)
    values, labels = torch.sort(constraints, dim=1)
    best = labels[:, 0].long()
    if constraints.shape[1] > 1 and float(margin) > 0.0:
        keep = (values[:, 1] - values[:, 0]) >= float(margin)
        if torch.any(keep):
            pairs = pairs[keep]
            best = best[keep]
    return pairs, best


def _flow_target(x0: torch.Tensor, x_star: torch.Tensor, cell_features: torch.Tensor, max_fraction: float) -> torch.Tensor:
    length_scale = torch.sqrt(torch.clamp(cell_features[:, 0].sum(), min=1.0))
    raw = (x_star - x0) / torch.clamp(length_scale, min=1.0)
    return torch.clamp(raw, min=-float(max_fraction), max=float(max_fraction))


def build_teacher_sample(
    cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
    teacher_result: dict,
    *,
    quality_cfg: TeacherQualityConfig,
    seed: int | None = None,
    size: tuple[int, int] | None = None,
) -> dict | None:
    if "final_cell_features" not in teacher_result:
        raise KeyError("teacher_result must contain 'final_cell_features'.")

    initial = cell_features.detach().clone()
    final = teacher_result["final_cell_features"].detach().clone().to(device=initial.device, dtype=initial.dtype)
    x0 = initial[:, 2:4].clone()
    x_star = final[:, 2:4].clone()

    metrics = score_official_metrics(final, pin_features, edge_list)
    if metrics["overlap_ratio"] > float(quality_cfg.max_demo_overlap):
        return None

    active_pairs = extract_teacher_active_pairs(
        x0,
        x_star,
        initial,
        pin_features,
        edge_list,
        near_window=quality_cfg.near_window,
        max_pairs=quality_cfg.max_label_pairs,
    )
    branch_pairs, branch_labels = extract_branch_labels(
        x_star,
        initial,
        active_pairs,
        max_pairs=quality_cfg.max_label_pairs,
        all_pair_limit=quality_cfg.all_pair_label_limit,
        near_window=quality_cfg.near_window,
        margin=quality_cfg.branch_margin,
    )
    seq_plus, seq_minus = sequence_pair_from_centers(final)
    weight = math.exp(
        -float(quality_cfg.alpha_o) * metrics["overlap_ratio"]
        -float(quality_cfg.alpha_w) * metrics["normalized_wl"]
    )
    stop_label = float(metrics["overlap_ratio"] <= float(quality_cfg.stop_overlap_threshold))
    value_target = -(
        float(quality_cfg.lambda_o) * metrics["overlap_ratio"]
        + float(quality_cfg.lambda_w) * metrics["normalized_wl"]
    )
    return {
        "cell_features": initial.cpu(),
        "pin_features": pin_features.detach().cpu(),
        "edge_list": edge_list.detach().cpu(),
        "final_cell_features": final.cpu(),
        "x0": x0.detach().cpu(),
        "x_star": x_star.detach().cpu(),
        "teacher_seq_plus": seq_plus.detach().cpu(),
        "teacher_seq_minus": seq_minus.detach().cpu(),
        "branch_pairs": branch_pairs.detach().cpu(),
        "branch_labels": branch_labels.detach().cpu(),
        "active_pair_labels": active_pairs.detach().cpu(),
        "flow_target": _flow_target(x0, x_star, initial, quality_cfg.max_flow_fraction).detach().cpu(),
        "stop_label": torch.tensor(stop_label, dtype=initial.dtype).cpu(),
        "initial_stop_label": torch.tensor(0.0, dtype=initial.dtype).cpu(),
        "value_target": torch.tensor(value_target, dtype=initial.dtype).cpu(),
        "weight": torch.tensor(weight, dtype=initial.dtype).cpu(),
        "metrics": metrics,
        "seed": None if seed is None else int(seed),
        "size": None if size is None else tuple(int(v) for v in size),
    }


@contextmanager
def offline_teacher_environment(disable_policy: bool = True):
    keys = ["PLACEMENT_USE_POLICY", "PLACEMENT_POLICY_TRANSITION"]
    previous = {key: os.environ.get(key) for key in keys}
    if disable_policy:
        os.environ["PLACEMENT_USE_POLICY"] = "0"
        os.environ["PLACEMENT_POLICY_TRANSITION"] = "0"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_teacher_dataset(
    generator: Callable[[int], tuple],
    teacher_solver: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict],
    num_cases: int,
    quality_cfg: TeacherQualityConfig | None = None,
    *,
    disable_policy: bool = True,
) -> list[dict]:
    quality_cfg = quality_cfg or TeacherQualityConfig()
    dataset: list[dict] = []
    with offline_teacher_environment(disable_policy=disable_policy):
        for seed in range(int(num_cases)):
            generated = generator(seed)
            if len(generated) == 4:
                _graph, cell_features, pin_features, edge_list = generated
            elif len(generated) == 3:
                cell_features, pin_features, edge_list = generated
            else:
                raise ValueError("generator must return (cell_features, pin_features, edge_list) or (G, cell_features, pin_features, edge_list)")
            teacher_result = teacher_solver(cell_features.clone(), pin_features.clone(), edge_list.clone())
            sample = build_teacher_sample(
                cell_features,
                pin_features,
                edge_list,
                teacher_result,
                quality_cfg=quality_cfg,
                seed=seed,
            )
            if sample is not None:
                dataset.append(sample)
    return dataset


def build_dagger_correction_dataset(
    policy,
    generator: Callable[[int], tuple],
    teacher_solver: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict],
    num_cases: int,
    quality_cfg: TeacherQualityConfig | None = None,
    *,
    rollout_steps: int = 2,
    temperature: float = 0.70,
    device: torch.device | None = None,
    disable_policy_teacher: bool = True,
) -> list[dict]:
    """Label states visited by the current policy with offline teacher outcomes.

    This is DAgger-style correction data.  The teacher is called only inside this
    offline dataset builder; the resulting samples are ordinary outcome labels.
    """
    from env import EnvConfig, PlacementOrderingEnv
    from ordering_policy import hierarchical_active_branch_weights

    quality_cfg = quality_cfg or TeacherQualityConfig()
    if device is None:
        device = next(policy.parameters()).device
    dataset: list[dict] = []
    policy.eval()
    with offline_teacher_environment(disable_policy=disable_policy_teacher):
        for seed in range(int(num_cases)):
            generated = generator(seed)
            if len(generated) == 4:
                _graph, cell_features, pin_features, edge_list = generated
            elif len(generated) == 3:
                cell_features, pin_features, edge_list = generated
            else:
                raise ValueError("generator returned unsupported tuple shape")
            cell_features = cell_features.to(device)
            pin_features = pin_features.to(device)
            edge_list = edge_list.to(device)

            env = PlacementOrderingEnv(
                cell_features,
                pin_features,
                edge_list,
                EnvConfig(horizon=max(int(rollout_steps), 1)),
            )
            memory = policy.initial_memory(device)
            with torch.no_grad():
                for _step in range(max(int(rollout_steps), 1)):
                    graph = env.graph_state(memory=memory)
                    action = policy.sample_action(graph, temperature=temperature, deterministic=False)
                    memory = action.next_memory.detach()
                    soft_weights = hierarchical_active_branch_weights(
                        action,
                        graph["active_pairs"],
                        relaxation="sigmoid",
                        tau=float(action.tau.detach().item()),
                    ).detach()
                    _reward, done, _info = env.step_action(
                        action,
                        entropy=action.entropy,
                        soft_branch_weights=soft_weights,
                        soft_tau=float(action.tau.detach().item()),
                    )
                    if done:
                        break

            visited = write_positions(cell_features, env.centers.detach())
            teacher_result = teacher_solver(visited.clone(), pin_features.clone(), edge_list.clone())
            sample = build_teacher_sample(
                visited,
                pin_features,
                edge_list,
                teacher_result,
                quality_cfg=quality_cfg,
                seed=seed,
            )
            if sample is not None:
                sample["dagger_source"] = True
                dataset.append(sample)
    policy.train()
    return dataset


def save_teacher_dataset(dataset: Iterable[dict], path: str | os.PathLike, metadata: dict | None = None) -> None:
    payload = {
        "version": 1,
        "metadata": metadata or {},
        "samples": list(dataset),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_teacher_dataset(path: str | os.PathLike) -> list[dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "samples" in payload:
        return list(payload["samples"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported teacher dataset format in {path}")


def _load_callable(spec: str):
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _make_synthetic_generator(sizes: list[tuple[int, int]], device: torch.device):
    def generator(seed: int):
        torch.manual_seed(int(seed))
        size = sizes[int(seed) % len(sizes)]
        cell_features, pin_features, edge_list = generate_placement_input(*size)
        cell_features = initialize_random_spread(cell_features, seed=seed + 17)
        return (
            cell_features.to(device),
            pin_features.to(device),
            edge_list.to(device),
        )

    return generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline teacher outcome dataset.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-solver", default="placement:train_placement")
    parser.add_argument("--num-cases", type=int, default=32)
    parser.add_argument("--sizes", default="2:20,3:25,2:30,3:50")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-demo-overlap", type=float, default=0.0)
    parser.add_argument("--max-label-pairs", type=int, default=200_000)
    parser.add_argument("--allow-policy-teacher", action="store_true")
    parser.add_argument("--dagger-policy-checkpoint", default="")
    parser.add_argument("--dagger-cases", type=int, default=0)
    parser.add_argument("--dagger-steps", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    quality_cfg = TeacherQualityConfig(
        max_demo_overlap=args.max_demo_overlap,
        max_label_pairs=args.max_label_pairs,
    )
    dataset = build_teacher_dataset(
        _make_synthetic_generator(parse_sizes(args.sizes), device),
        _load_callable(args.teacher_solver),
        args.num_cases,
        quality_cfg,
        disable_policy=not args.allow_policy_teacher,
    )
    if args.dagger_policy_checkpoint and args.dagger_cases > 0:
        from ordering_policy import load_policy_checkpoint

        policy, _checkpoint = load_policy_checkpoint(args.dagger_policy_checkpoint, device)
        corrections = build_dagger_correction_dataset(
            policy,
            _make_synthetic_generator(parse_sizes(args.sizes), device),
            _load_callable(args.teacher_solver),
            args.dagger_cases,
            quality_cfg,
            rollout_steps=args.dagger_steps,
            device=device,
            disable_policy_teacher=not args.allow_policy_teacher,
        )
        dataset.extend(corrections)
    metadata = {
        "quality_cfg": asdict(quality_cfg),
        "teacher_solver": args.teacher_solver,
        "sizes": args.sizes,
        "num_cases_requested": args.num_cases,
        "num_cases_accepted": len(dataset),
        "dagger_cases_requested": args.dagger_cases,
        "dagger_cases_accepted": sum(1 for sample in dataset if bool(sample.get("dagger_source", False))),
    }
    save_teacher_dataset(dataset, args.output, metadata=metadata)
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
