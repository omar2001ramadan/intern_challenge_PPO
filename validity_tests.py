"""Run-validity checks for the policy-conditioned PPO implementation."""

from __future__ import annotations

import inspect
import os

import torch

import placement
from active_set import build_initial_active_pairs
from constraints import _exact_overlap_pairs_all, _exact_overlap_pairs_spatial_hash, overlap_ratio_from_pairs
from distill import DistillConfig, outcome_distill, teacher_lambda_at
from env import EnvConfig, PlacementOrderingEnv
from induce_branches import branch_antisymmetry_error
from ordering_policy import OrderingPolicy
from rollout import select_by_exact_overlap_then_wirelength
from teacher_data import TeacherQualityConfig, build_teacher_sample, score_official_metrics


def _tiny_problem():
    cell_features = torch.tensor(
        [
            [4.0, 2.0, 0.0, 0.0, 2.0, 2.0],
            [4.0, 2.0, 0.5, 0.0, 2.0, 2.0],
            [1.0, 1.0, 4.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 5.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    pin_features = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    edge_list = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    return cell_features, pin_features, edge_list


def _pair_set(pairs):
    return {tuple(pair) for pair in pairs.detach().cpu().tolist()}


def test_exact_overlap_parity():
    cell_features, _pin_features, _edge_list = _tiny_problem()
    all_pairs = _exact_overlap_pairs_all(cell_features, chunk_size=64)
    hash_pairs = _exact_overlap_pairs_spatial_hash(cell_features)
    assert _pair_set(all_pairs) == _pair_set(hash_pairs)


def test_branch_antisymmetry():
    seq_plus = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    seq_minus = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    assert branch_antisymmetry_error(seq_plus, seq_minus) == 0.0


def test_relabeling_metric_invariance():
    cell_features, pin_features, edge_list = _tiny_problem()
    score = score_official_metrics(cell_features, pin_features, edge_list)
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    relabeled_cells = cell_features[perm].clone()
    relabeled_pins = pin_features.clone()
    relabeled_pins[:, 0] = inv[pin_features[:, 0].long()].to(relabeled_pins.dtype)
    relabeled_score = score_official_metrics(relabeled_cells, relabeled_pins, edge_list)
    assert score["overlap_cells"] == relabeled_score["overlap_cells"]
    assert abs(score["overlap_ratio"] - relabeled_score["overlap_ratio"]) < 1e-8
    assert abs(score["normalized_wl"] - relabeled_score["normalized_wl"]) < 1e-6


def test_no_teacher_in_inference_path():
    source = inspect.getsource(placement)
    forbidden = ["teacher_solver", "outcome_distill", "build_teacher_dataset", "lambda_teacher"]
    for token in forbidden:
        assert token not in source


def test_mode_b_no_repair_selection():
    x_a = torch.tensor([[0.0, 0.0], [0.5, 0.0]])
    x_b = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
    candidates = [
        {"X": x_a.clone(), "overlap_cells": 2, "overlap_ratio": 1.0, "normalized_wl": 0.1, "num_overlap_pairs": 1},
        {"X": x_b.clone(), "overlap_cells": 0, "overlap_ratio": 0.0, "normalized_wl": 0.5, "num_overlap_pairs": 0},
    ]
    before = [candidate["X"].clone() for candidate in candidates]
    selected = select_by_exact_overlap_then_wirelength(candidates)
    assert torch.equal(selected["X"], x_b)
    for candidate, saved in zip(candidates, before):
        assert torch.equal(candidate["X"], saved)


def test_stop_safety_reward():
    cell_features, pin_features, edge_list = _tiny_problem()
    env = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    bad = {"overlap_cells": 1, "overlap_ratio": 0.5, "normalized_wl": 0.1}
    before = {"overlap_cells": 1, "overlap_ratio": 0.4, "normalized_wl": 0.1}
    assert env._stop_reward(bad, before) < 0.0


def test_teacher_annealing_reaches_zero():
    assert teacher_lambda_at(0, lambda0=1.0, anneal_updates=10) > 0.0
    assert teacher_lambda_at(10, lambda0=1.0, anneal_updates=10) == 0.0


def test_large_active_set_smoke():
    n = 2000
    cell_features = torch.zeros((n, 6), dtype=torch.float32)
    cell_features[:, 0] = 1.0
    cell_features[:, 1] = 1.0
    cell_features[:, 2] = torch.linspace(0.0, 200.0, n)
    cell_features[:, 3] = torch.sin(torch.linspace(0.0, 20.0, n))
    cell_features[:, 4] = 1.0
    cell_features[:, 5] = 1.0
    pin_features = torch.zeros((n, 7), dtype=torch.float32)
    pin_features[:, 0] = torch.arange(n, dtype=torch.float32)
    edge_list = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=1)
    active = build_initial_active_pairs(
        cell_features,
        pin_features,
        edge_list,
        all_pair_limit=128,
        near_window=4,
        max_pairs=25_000,
    )
    assert 0 < active.shape[0] <= 25_000


def test_distill_smoke():
    cell_features, pin_features, edge_list = _tiny_problem()
    final = cell_features.clone()
    final[:, 2] = torch.tensor([0.0, 3.0, 5.0, 7.0])
    final[:, 3] = 0.0
    sample = build_teacher_sample(
        cell_features,
        pin_features,
        edge_list,
        {"final_cell_features": final},
        quality_cfg=TeacherQualityConfig(max_demo_overlap=0.0, max_label_pairs=64),
        seed=0,
        size=(2, 2),
    )
    assert sample is not None
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    stats = outcome_distill(
        policy,
        [sample],
        DistillConfig(epochs=1, batch_size=1, max_branch_pairs_per_sample=64, equivariance_coef=0.0),
        device=torch.device("cpu"),
    )
    assert stats["teacher_samples"] == 1
    assert stats["teacher_lambda_final"] == 0.0


def run_all():
    tests = [
        test_exact_overlap_parity,
        test_branch_antisymmetry,
        test_relabeling_metric_invariance,
        test_no_teacher_in_inference_path,
        test_mode_b_no_repair_selection,
        test_stop_safety_reward,
        test_teacher_annealing_reaches_zero,
        test_large_active_set_smoke,
        test_distill_smoke,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    os.environ.setdefault("PLACEMENT_USE_POLICY", "0")
    os.environ.setdefault("PLACEMENT_POLICY_TRANSITION", "0")
    run_all()
