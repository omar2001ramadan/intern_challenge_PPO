"""Run-validity checks for the policy-conditioned PPO implementation."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import torch

import active_set
import constraints
import env
import placement
import run_decision_tests
import run_structural_falsification
import structural_audit
import train_ppo
import wisdom_audit
from active_set import build_initial_active_pairs, update_active_pair_cache
from constraints import (
    _exact_overlap_pairs_all,
    _exact_overlap_pairs_spatial_hash,
    branch_signed_constraints,
    soft_signed_disjunction,
)
from distill import DistillConfig, outcome_distill, teacher_lambda_at
from env import EnvConfig, PlacementOrderingEnv, gated_wirelength_delta
from induce_branches import Branch, branch_antisymmetry_error
from ordering_policy import OrderingPolicy, PHASE_REQUEST_TO_INDEX, apply_rollout_memory_policy
from primal_dual import phr_dual_update
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
    forbidden = ["teacher_solver", "prior_solver", "outcome_distill", "build_teacher_dataset", "lambda_teacher"]
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


def test_public_wrapper_declares_inference_mode():
    signature = inspect.signature(placement.train_placement)
    for name in ("num_rollouts", "num_steps", "temperature", "mode"):
        assert name in signature.parameters


def test_training_loop_uses_teacher_annealing():
    source = inspect.getsource(train_ppo)
    assert "teacher_lambda_at(" in source
    assert "teacher_auxiliary_update(" in source


def test_soft_branch_support_is_finite():
    centers = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    widths = torch.tensor([2.0, 2.0], dtype=torch.float32)
    heights = torch.tensor([2.0, 2.0], dtype=torch.float32)
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    branch_weights = torch.zeros((1, 4), dtype=torch.float32)
    soft_g = soft_signed_disjunction(centers, widths, heights, pairs, branch_weights, tau=0.5, epsilon=1e-4)
    assert torch.isfinite(soft_g).all()


def test_branch_dual_identity_update():
    cell_features, pin_features, edge_list = _tiny_problem()
    env = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    env.active_pairs = torch.tensor([[0, 1]], dtype=torch.long)
    env.active_pair_ages = torch.tensor([4], dtype=torch.long)
    env.branch_duals = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    old_branch_duals = env.branch_duals.clone()
    old_boundary_duals = env.boundary_duals.clone()
    old_density_duals = env.density_duals.clone()
    branches = torch.tensor([int(Branch.L)], dtype=torch.long)

    widths = env.cell_features[:, 4]
    heights = env.cell_features[:, 5]
    branch_g = branch_signed_constraints(env.centers, widths, heights, env.active_pairs, branches) / env.length_scale
    branch_rho = env._effective_rho(env.config.rho, 1.0, None, branch_g)
    expected_active = phr_dual_update(
        old_branch_duals[torch.arange(env.active_pairs.shape[0]), branches],
        branch_g,
        branch_rho,
        max_value=env.config.dual_max,
    )

    env._update_duals(
        branches,
        old_branch_duals,
        old_boundary_duals,
        old_density_duals,
        rho=env.config.rho,
    )
    assert torch.allclose(env.branch_duals[:, 0], expected_active, atol=1e-6)
    assert torch.allclose(env.branch_duals[:, 1:], old_branch_duals[:, 1:] * 0.92, atol=1e-6)


def test_active_pair_retention_cache():
    active_pairs = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    pair_ages = torch.tensor([1, 3], dtype=torch.long)
    missed_pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    updated_pairs, updated_ages = update_active_pair_cache(
        active_pairs,
        pair_ages,
        missed_pairs,
        retention_horizon=4,
        max_pairs=8,
    )
    key_to_age = {tuple(pair): int(age) for pair, age in zip(updated_pairs.tolist(), updated_ages.tolist())}
    assert key_to_age[(0, 1)] == 4
    assert key_to_age[(1, 2)] == 4
    assert key_to_age[(0, 2)] == 2


def test_metric_authority_checkpoint_contract_present():
    source = inspect.getsource(train_ppo)
    required = [
        "best_exact_overlap.pt",
        "best_lexicographic.pt",
        "best_wire_given_overlap_threshold.pt",
        "latest.pt",
        "shaped_reward_debug.pt",
    ]
    for token in required:
        assert token in source


def test_ppo_logs_group_ratios():
    ppo_source = inspect.getsource(__import__("ppo"))
    assert 'aggregate_log_ratio' in ppo_source
    assert 'result[f"ratio_{group}"]' in ppo_source
    assert 'result[f"log_ratio_{group}"]' in ppo_source


def test_mac_path_has_no_explicit_cpu_fallbacks():
    targets = [
        inspect.getsource(constraints._exact_overlap_pairs_spatial_hash),
        inspect.getsource(active_set.update_active_pair_cache),
        inspect.getsource(env.PlacementOrderingEnv._inactive_exact_pairs),
        inspect.getsource(env.PlacementOrderingEnv._remap_branch_duals),
    ]
    for source in targets:
        assert ".cpu()" not in source
        assert ".tolist()" not in source


def test_ordering_path_has_stability_guards():
    policy_source = inspect.getsource(__import__("ordering_policy"))
    assert "_stabilize_ordering_scores" in policy_source
    assert "_sanitize_tensor" in policy_source
    assert "teacher_aux_skipped" in inspect.getsource(__import__("distill"))


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


def test_default_sequence_pair_run_excludes_inactive_heads():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(ordering_representation="sequence_pair", branch_mode="ordering"),
    )
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=8, global_flow_rank=1)
    action = policy.sample_action(env_instance.graph_state(), temperature=1.0, deterministic=False)
    assert "dag_ordering" not in action.group_logprobs
    assert "pair_branches" not in action.group_logprobs


def test_disable_clusters_and_stop_remove_policy_groups():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(enable_clusters=False, enable_stop=False),
    )
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=8, global_flow_rank=1)
    action = policy.sample_action(env_instance.graph_state(), temperature=1.0, deterministic=False)
    assert "cluster_ordering" not in action.group_logprobs
    assert "clusters" not in action.group_logprobs
    assert "stop" not in action.group_logprobs
    assert action.enable_clusters is False
    assert action.enable_stop is False


def test_graph_state_exposes_incumbent_fields():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig())
    graph = env_instance.graph_state()
    assert "incumbent_centers" in graph
    assert "incumbent_overlap_ratio" in graph
    assert "incumbent_normalized_wl" in graph
    assert "steps_since_best" in graph
    assert graph["incumbent_centers"].shape == env_instance.centers.shape
    assert float(graph["steps_since_best"].item()) == 0.0


def test_disable_incumbent_action_removes_policy_group():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(enable_incumbent_action=False),
    )
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=8, global_flow_rank=1)
    action = policy.sample_action(env_instance.graph_state(), temperature=1.0, deterministic=False)
    assert "incumbent" not in action.group_logprobs
    assert abs(float(action.incumbent_mix.detach().item())) < 1e-8


def test_legacy_feature_dims_still_run_with_incumbent_graph():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig())
    policy = OrderingPolicy(
        node_feature_dim=16,
        pair_feature_dim=10,
        hidden_dim=32,
        message_passes=1,
        num_clusters=8,
        global_flow_rank=1,
        enable_incumbent_controls=False,
    )
    action = policy.sample_action(env_instance.graph_state(), temperature=1.0, deterministic=False)
    assert "incumbent" not in action.group_logprobs
    assert action.residual_flow.shape == env_instance.centers.shape


def test_validation_suite_is_fixed():
    sizes = [(2, 20), (3, 25)]
    suite_a = train_ppo.build_validation_suite(sizes, 4, 123)
    suite_b = train_ppo.build_validation_suite(sizes, 4, 123)
    assert suite_a == suite_b
    assert suite_a[0]["size"] == sizes[0]
    assert suite_a[1]["size"] == sizes[1]
    assert suite_a[2]["size"] == sizes[0]


def test_metric_gated_tau_uses_baseline_before_hardening():
    state = {"best_overlap": None, "bad_windows": 0, "has_baseline": False}
    tau = train_ppo.update_metric_gated_tau(
        1.5,
        overlap=0.7,
        branch_violation=0.0,
        missed_pairs=0.0,
        exact_overlap_pairs=0.0,
        state=state,
        tau_min=0.15,
        tau_max=2.5,
        gamma_down=0.96,
        gamma_up=1.08,
        overlap_epsilon=0.005,
        branch_violation_max=0.01,
        missed_pairs_max=64.0,
        exact_overlap_pairs_max=8.0,
        patience=5,
    )
    assert abs(tau - 1.5) < 1e-8
    assert state["has_baseline"] is True
    assert abs(state["best_overlap"] - 0.7) < 1e-8


def test_metric_gated_tau_blocks_hardening_on_exact_overlap_pairs():
    state = {"best_overlap": 0.8, "bad_windows": 0, "has_baseline": True}
    tau = train_ppo.update_metric_gated_tau(
        1.5,
        overlap=0.6,
        branch_violation=0.0,
        missed_pairs=0.0,
        exact_overlap_pairs=16.0,
        state=state,
        tau_min=0.15,
        tau_max=2.5,
        gamma_down=0.96,
        gamma_up=1.08,
        overlap_epsilon=0.005,
        branch_violation_max=0.01,
        missed_pairs_max=64.0,
        exact_overlap_pairs_max=8.0,
        patience=5,
    )
    assert abs(tau - 1.5) < 1e-8
    assert abs(state["best_overlap"] - 0.6) < 1e-8


def test_metric_gated_temperature_tracks_soft_tau():
    assert abs(train_ppo.metric_gated_temperature(2.0) - 1.4) < 1e-8
    assert abs(train_ppo.metric_gated_temperature(0.20) - 0.55) < 1e-8
    assert abs(train_ppo.metric_gated_temperature(0.80) - 0.80) < 1e-8


def test_audit_pressure_responds_to_exact_overlap_burden():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(audit_missed_target=64.0, audit_pressure_gamma=1.0, audit_pressure_max=4.0),
    )
    assert env_instance._audit_pressure_scale(0, 12) > 1.0


def test_validation_uses_best_candidate_overlap_pairs():
    source = inspect.getsource(train_ppo.validate_policy)
    assert 'mean("best_exact_overlap_pairs")' in source


def test_ppo_uses_scaled_smooth_l1_value_loss():
    source = inspect.getsource(__import__("ppo"))
    assert "return_scale = torch.clamp(returns.std(unbiased=False), min=1.0)" in source
    assert "F.smooth_l1_loss(scaled_value, scaled_return)" in source


def test_hard_replay_suite_is_deterministic():
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    suite_a = train_ppo.build_hard_replay_suite(
        policy,
        sizes=[(1, 2)],
        env_config=EnvConfig(horizon=1, enable_clusters=False, enable_stop=False),
        device=torch.device("cpu"),
        seed_base=777,
        temperature=1.0,
        soft_tau=1.5,
        relaxation="sigmoid",
        pool_size=4,
        suite_size=2,
        memory_reset_mode="incumbent_improve",
        memory_reset_retain=0.25,
        memory_reset_min_overlap_gain=0.03,
        memory_reset_min_pair_gain_count=2.0,
        memory_reset_min_steps_since_best=2,
    )
    suite_b = train_ppo.build_hard_replay_suite(
        policy,
        sizes=[(1, 2)],
        env_config=EnvConfig(horizon=1, enable_clusters=False, enable_stop=False),
        device=torch.device("cpu"),
        seed_base=777,
        temperature=1.0,
        soft_tau=1.5,
        relaxation="sigmoid",
        pool_size=4,
        suite_size=2,
        memory_reset_mode="incumbent_improve",
        memory_reset_retain=0.25,
        memory_reset_min_overlap_gain=0.03,
        memory_reset_min_pair_gain_count=2.0,
        memory_reset_min_steps_since_best=2,
    )
    assert suite_a == suite_b
    assert len(suite_a) == 2


def test_rollout_memory_policy_resets_on_incumbent_improve():
    next_memory = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    initial_memory = torch.zeros_like(next_memory)
    updated, info = apply_rollout_memory_policy(
        next_memory,
        initial_memory,
        reset_mode="incumbent_improve",
        reset_retain=0.25,
        incumbent_improved=True,
    )
    assert torch.allclose(updated, next_memory * 0.25, atol=1e-6)
    assert info["memory_reset_applied"] is True
    assert info["memory_reset_reason"] == "incumbent_improved"


def test_rollout_memory_policy_resets_on_rollback_event():
    next_memory = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    initial_memory = torch.zeros_like(next_memory)
    updated, info = apply_rollout_memory_policy(
        next_memory,
        initial_memory,
        reset_mode="none",
        event_reset=True,
        event_reset_reason="rollback_to_incumbent",
        event_reset_retain=0.10,
    )
    assert torch.allclose(updated, next_memory * 0.10, atol=1e-6)
    assert info["memory_reset_applied"] is True
    assert info["memory_reset_reason"] == "rollback_to_incumbent"
    assert info["memory_reset_event"] is True


def test_rollout_memory_policy_material_or_stale_gate():
    next_memory = torch.tensor([1.0, 2.0], dtype=torch.float32)
    initial_memory = torch.zeros_like(next_memory)
    updated_small, info_small = apply_rollout_memory_policy(
        next_memory,
        initial_memory,
        reset_mode="incumbent_improve_material_or_stale",
        reset_retain=0.75,
        incumbent_improved=True,
        best_overlap_delta=0.01,
        best_pair_delta_count=0.0,
        steps_since_best_before=0,
        min_overlap_gain=0.03,
        min_pair_gain_count=2.0,
        min_steps_since_best=2,
    )
    assert torch.allclose(updated_small, next_memory, atol=1e-6)
    assert info_small["memory_reset_applied"] is False

    updated_stale, info_stale = apply_rollout_memory_policy(
        next_memory,
        initial_memory,
        reset_mode="incumbent_improve_material_or_stale",
        reset_retain=0.75,
        incumbent_improved=True,
        best_overlap_delta=0.01,
        best_pair_delta_count=0.0,
        steps_since_best_before=3,
        min_overlap_gain=0.03,
        min_pair_gain_count=2.0,
        min_steps_since_best=2,
    )
    assert torch.allclose(updated_stale, next_memory * 0.75, atol=1e-6)
    assert info_stale["memory_reset_applied"] is True
    assert info_stale["memory_reset_reason"] == "incumbent_improved_stale"


def test_training_loop_applies_rollout_memory_policy():
    source = inspect.getsource(train_ppo.collect_episode)
    assert "apply_rollout_memory_policy(" in source
    assert "incumbent_improved" in source
    assert "rollback_to_incumbent" in source
    assert "phase_transition=" in source
    assert "event_reset=" in source


def test_rollout_memory_policy_resets_on_phase_transition():
    next_memory = torch.tensor([2.0, -4.0], dtype=torch.float32)
    initial_memory = torch.zeros_like(next_memory)
    updated, info = apply_rollout_memory_policy(
        next_memory,
        initial_memory,
        reset_mode="none",
        phase_transition=True,
        phase_transition_reason="discover_exit",
        phase_reset_retain=0.75,
    )
    assert torch.allclose(updated, next_memory * 0.75, atol=1e-6)
    assert info["memory_reset_applied"] is True
    assert info["memory_reset_reason"] == "discover_exit"
    assert info["memory_reset_phase_transition"] is True


def test_rollout_memory_policy_resets_on_refine_reject_event():
    next_memory = torch.tensor([3.0, -1.0], dtype=torch.float32)
    initial_memory = torch.zeros_like(next_memory)
    updated, info = apply_rollout_memory_policy(
        next_memory,
        initial_memory,
        reset_mode="none",
        event_reset=True,
        event_reset_reason="refine_rejected",
        event_reset_retain=0.25,
    )
    assert torch.allclose(updated, next_memory * 0.25, atol=1e-6)
    assert info["memory_reset_applied"] is True
    assert info["memory_reset_reason"] == "refine_rejected"
    assert info["memory_reset_event"] is True


def test_phase_transition_guards():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1, enable_stop=False))
    env_instance.phase = env.PlacementPhase.DISCOVER
    env_instance.stagnation_steps = env_instance.config.discover_patience
    best_score = dict(env_instance.best_score)
    best_score["overlap_ratio"] = env_instance.config.discover_exit_overlap - 0.05
    transitioned, reason = env_instance._advance_phase(
        action=None,
        phase_before=env.PlacementPhase.DISCOVER,
        phase_request_name="advance",
        current_score=best_score,
        best_before=best_score,
        best_after=best_score,
    )
    assert transitioned is True
    assert env_instance.phase == env.PlacementPhase.LEGALIZE
    assert reason == "discover_exit"


def test_refine_entry_guard_stays_strict():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(
            horizon=1,
            refine_overlap_threshold=0.25,
            refine_entry_overlap_threshold=0.30,
            refine_pairs_threshold=8,
            legal_streak_required=2,
        ),
    )
    env_instance.phase = env.PlacementPhase.LEGALIZE
    env_instance.legal_streak = 2
    best_after = {
        "overlap_ratio": 0.2727,
        "overlap_cells": 2,
        "num_overlap_pairs": 5,
        "normalized_wl": 0.8,
    }
    transitioned, reason = env_instance._advance_phase(
        action=None,
        phase_before=env.PlacementPhase.LEGALIZE,
        phase_request_name="stay",
        current_score=best_after,
        best_before=best_after,
        best_after=best_after,
    )
    assert transitioned is False
    assert env_instance.phase == env.PlacementPhase.LEGALIZE
    assert reason == "phase_stay"


def test_refine_regression_falls_back_to_legalize():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1, enable_stop=False))
    env_instance.phase = env.PlacementPhase.REFINE
    best_score = {
        "overlap_ratio": 0.10,
        "overlap_cells": 0,
        "num_overlap_pairs": 1,
        "normalized_wl": 0.50,
    }
    current_score = {
        "overlap_ratio": 0.20,
        "overlap_cells": 1,
        "num_overlap_pairs": 4,
        "normalized_wl": 0.45,
    }
    transitioned, reason = env_instance._advance_phase(
        action=None,
        phase_before=env.PlacementPhase.REFINE,
        phase_request_name="stay",
        current_score=current_score,
        best_before=best_score,
        best_after=best_score,
    )
    assert transitioned is True
    assert env_instance.phase == env.PlacementPhase.LEGALIZE
    assert reason == "refine_regression"


def test_refine_acceptance_rejects_wirelength_regression():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1, enable_stop=False))
    incumbent_score = {
        "overlap_ratio": 0.10,
        "overlap_cells": 0,
        "num_overlap_pairs": 1,
        "normalized_wl": 0.50,
    }
    candidate_score = {
        "overlap_ratio": 0.10,
        "overlap_cells": 0,
        "num_overlap_pairs": 1,
        "normalized_wl": 0.515,
    }
    accepted, reason = env_instance._accept_refine_candidate(candidate_score, incumbent_score)
    assert accepted is False
    assert reason == "wirelength_regression"


def test_unlock_exits_after_fixed_horizon():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1, enable_stop=False))
    env_instance.phase = env.PlacementPhase.UNLOCK
    env_instance.unlock_remaining_steps = 1
    best_score = dict(env_instance.best_score)
    transitioned, reason = env_instance._advance_phase(
        action=None,
        phase_before=env.PlacementPhase.UNLOCK,
        phase_request_name="stay",
        current_score=best_score,
        best_before=best_score,
        best_after=best_score,
    )
    assert transitioned is True
    assert env_instance.phase == env.PlacementPhase.LEGALIZE
    assert reason == "unlock_complete"


def test_refine_stop_bias_prefers_preserving_stale_incumbent():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(
            horizon=1,
            enable_stop=True,
            stop_gate_overlap_threshold=1.0,
            refine_stop_bias=3.0,
            refine_stop_min_phase_steps=1,
        ),
    )
    env_instance.phase = env.PlacementPhase.REFINE
    env_instance.phase_step = 1
    env_instance.steps_since_best = 1
    env_instance.stagnation_steps = 1
    env_instance.best_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    env_instance.best_centers = env_instance.centers.detach().clone()
    env_instance.phase_entry_best_score = dict(env_instance.best_score)
    graph = env_instance.graph_state(memory=torch.zeros(0))
    assert float(graph["stop_logit_bias"].detach().item()) > 0.0


def test_refine_auto_stop_preserves_better_incumbent():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_stop=True, refine_auto_stop_stale_steps=1),
    )
    env_instance.phase = env.PlacementPhase.REFINE
    incumbent_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    current_score = {
        "overlap_ratio": 0.02,
        "overlap_cells": 1,
        "num_overlap_pairs": 2,
        "normalized_wl": 0.45,
    }
    should_stop = env_instance._should_auto_stop_refine(
        current_score=current_score,
        incumbent_score=incumbent_score,
        incumbent_improved=False,
        steps_since_best_before=1,
        refine_rejected=False,
    )
    assert should_stop is True


def test_refine_auto_stop_ignores_small_nonmaterial_regression():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_stop=True, refine_auto_stop_stale_steps=1),
    )
    env_instance.phase = env.PlacementPhase.REFINE
    incumbent_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    current_score = {
        "overlap_ratio": 0.005,
        "overlap_cells": 0,
        "num_overlap_pairs": 1,
        "normalized_wl": 0.43,
    }
    should_stop = env_instance._should_auto_stop_refine(
        current_score=current_score,
        incumbent_score=incumbent_score,
        incumbent_improved=False,
        steps_since_best_before=3,
        refine_rejected=False,
        continuation_risk=0.5,
    )
    assert should_stop is False


def test_refine_auto_stop_skips_when_current_matches_incumbent():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_stop=True, refine_auto_stop_stale_steps=1),
    )
    env_instance.phase = env.PlacementPhase.REFINE
    incumbent_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    current_score = dict(incumbent_score)
    should_stop = env_instance._should_auto_stop_refine(
        current_score=current_score,
        incumbent_score=incumbent_score,
        incumbent_improved=False,
        steps_since_best_before=2,
        refine_rejected=False,
    )
    assert should_stop is False


def test_refine_rollback_preserves_better_incumbent():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_stop=True),
    )
    incumbent_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    current_score = {
        "overlap_ratio": 0.02,
        "overlap_cells": 1,
        "num_overlap_pairs": 2,
        "normalized_wl": 0.45,
    }
    should_rollback = env_instance._should_rollback_refine_to_incumbent(
        current_score=current_score,
        incumbent_score=incumbent_score,
        incumbent_improved=False,
    )
    assert should_rollback is True


def test_refine_rollback_ignores_small_nonmaterial_regression():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_stop=True),
    )
    incumbent_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    current_score = {
        "overlap_ratio": 0.005,
        "overlap_cells": 0,
        "num_overlap_pairs": 1,
        "normalized_wl": 0.43,
    }
    should_rollback = env_instance._should_rollback_refine_to_incumbent(
        current_score=current_score,
        incumbent_score=incumbent_score,
        incumbent_improved=False,
        continuation_risk=1.0,
    )
    assert should_rollback is False


def test_refine_rollback_skips_when_incumbent_improves():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_stop=True),
    )
    incumbent_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    current_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.35,
    }
    should_rollback = env_instance._should_rollback_refine_to_incumbent(
        current_score=current_score,
        incumbent_score=incumbent_score,
        incumbent_improved=True,
    )
    assert should_rollback is False


def test_policy_phase_masks_disable_ordering_and_stop():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1, enable_clusters=True, enable_stop=True))
    graph_legalize = env_instance.graph_state(memory=policy.initial_memory(torch.device("cpu")))
    graph_legalize["phase_name"] = "LEGALIZE"
    action_legalize = policy.sample_action(graph_legalize, deterministic=True)
    assert action_legalize.enable_stop is False
    assert action_legalize.enable_unlock is False
    assert float(action_legalize.group_logprobs["ordering"].detach().item()) == 0.0

    graph_refine = dict(graph_legalize)
    graph_refine["phase_name"] = "REFINE"
    action_refine = policy.sample_action(graph_refine, deterministic=True)
    assert action_refine.enable_stop is True
    assert action_refine.enable_unlock is False


def test_refine_control_heads_ignore_recurrent_memory():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1, enable_clusters=True, enable_stop=True))
    graph_refine_a = env_instance.graph_state(memory=torch.zeros(policy.hidden_dim))
    graph_refine_a["phase_name"] = "REFINE"
    graph_refine_b = dict(graph_refine_a)
    graph_refine_b["memory"] = torch.full((policy.hidden_dim,), 3.0, dtype=graph_refine_a["cell_features"].dtype)
    action_a = policy.sample_action(graph_refine_a, deterministic=True)
    action_b = policy.sample_action(graph_refine_b, deterministic=True)
    assert torch.allclose(action_a.control_raw, action_b.control_raw, atol=1e-6)
    assert torch.allclose(action_a.residual_flow, action_b.residual_flow, atol=1e-6)
    assert torch.allclose(action_a.incumbent_mix_raw, action_b.incumbent_mix_raw, atol=1e-6)
    assert int(action_a.k_index.detach().item()) == int(action_b.k_index.detach().item())


def test_legalize_post_legal_control_heads_ignore_recurrent_memory():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1, enable_clusters=True, enable_stop=True))
    env_instance.phase = env.PlacementPhase.LEGALIZE
    env_instance.best_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    env_instance.best_centers = env_instance.centers.detach().clone()
    graph_a = env_instance.graph_state(memory=torch.zeros(policy.hidden_dim))
    graph_b = dict(graph_a)
    graph_b["memory"] = torch.full((policy.hidden_dim,), 5.0, dtype=graph_a["cell_features"].dtype)
    action_a = policy.sample_action(graph_a, deterministic=True)
    action_b = policy.sample_action(graph_b, deterministic=True)
    assert torch.allclose(action_a.control_raw, action_b.control_raw, atol=1e-6)
    assert torch.allclose(action_a.residual_flow, action_b.residual_flow, atol=1e-6)
    assert torch.allclose(action_a.incumbent_mix_raw, action_b.incumbent_mix_raw, atol=1e-6)
    assert int(action_a.k_index.detach().item()) == int(action_b.k_index.detach().item())


def test_discover_discrete_heads_ignore_recurrent_memory_after_step_one():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_clusters=True, enable_stop=False),
        discover_mode="macro_clearance",
    )
    env_instance.phase = env.PlacementPhase.DISCOVER
    env_instance.phase_step = 2
    graph_a = env_instance.graph_state(memory=torch.zeros(policy.hidden_dim))
    graph_b = dict(graph_a)
    graph_b["memory"] = torch.full((policy.hidden_dim,), 5.0, dtype=graph_a["cell_features"].dtype)
    action_a = policy.sample_action(graph_a, deterministic=True)
    action_b = policy.sample_action(graph_b, deterministic=True)
    assert torch.equal(action_a.seq_plus, action_b.seq_plus)
    assert torch.equal(action_a.seq_minus, action_b.seq_minus)
    assert torch.equal(action_a.cluster_ids, action_b.cluster_ids)


def test_discover_sequence_pair_freezes_to_incumbent_after_step_one():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_clusters=True, enable_stop=False),
        discover_mode="spread_first",
    )
    env_instance.phase = env.PlacementPhase.DISCOVER
    env_instance.phase_step = 1
    incumbent = env_instance.centers.detach().clone()
    incumbent[:, 0] = torch.linspace(2.0, -1.0, steps=incumbent.shape[0], dtype=incumbent.dtype)
    incumbent[:, 1] = torch.linspace(0.5, -0.5, steps=incumbent.shape[0], dtype=incumbent.dtype)
    env_instance.best_centers = incumbent
    env_instance.best_score = {
        "overlap_ratio": 0.25,
        "overlap_cells": 0,
        "num_overlap_pairs": 4,
        "normalized_wl": 0.6,
    }
    graph = env_instance.graph_state(memory=torch.zeros(policy.hidden_dim))
    action = policy.sample_action(graph, deterministic=True)
    expected_plus, expected_minus = env.sequence_pair_from_centers(incumbent)
    assert torch.equal(action.seq_plus, expected_plus)
    assert torch.equal(action.seq_minus, expected_minus)


def test_spread_first_discover_control_heads_ignore_recurrent_memory_after_step_one():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_clusters=True, enable_stop=False),
        discover_mode="spread_first",
    )
    env_instance.phase = env.PlacementPhase.DISCOVER
    env_instance.phase_step = 2
    graph_a = env_instance.graph_state(memory=torch.zeros(policy.hidden_dim))
    graph_b = dict(graph_a)
    graph_b["memory"] = torch.full((policy.hidden_dim,), 5.0, dtype=graph_a["cell_features"].dtype)
    action_a = policy.sample_action(graph_a, deterministic=True)
    action_b = policy.sample_action(graph_b, deterministic=True)
    assert torch.allclose(action_a.control_raw, action_b.control_raw, atol=1e-6)
    assert torch.allclose(action_a.residual_flow, action_b.residual_flow, atol=1e-6)
    assert torch.allclose(action_a.incumbent_mix_raw, action_b.incumbent_mix_raw, atol=1e-6)
    assert int(action_a.k_index.detach().item()) == int(action_b.k_index.detach().item())
    assert torch.allclose(action_a.branch_pressure_raw, action_b.branch_pressure_raw, atol=1e-6)
    assert torch.allclose(action_a.boundary_pressure_raw, action_b.boundary_pressure_raw, atol=1e-6)
    assert torch.allclose(action_a.next_memory, action_b.next_memory, atol=1e-6)


def test_discover_mode_caps_incumbent_mix():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_clusters=True, enable_stop=False),
        discover_mode="spread_first",
    )
    graph = env_instance.graph_state(memory=torch.zeros(policy.hidden_dim))
    action = policy.sample_action(graph, deterministic=True)
    assert float(action.incumbent_mix.detach().item()) <= 0.0500001


def test_graph_state_exposes_discover_mode():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1),
        discover_mode="macro_clearance",
    )
    graph = env_instance.graph_state(memory=torch.zeros(32))
    assert str(graph["discover_mode_name"]) == "macro_clearance"
    assert int(graph["discover_mode_index"].detach().item()) >= 0


def test_late_legalize_flag_no_longer_changes_policy_strictness():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    env_instance.phase = env.PlacementPhase.LEGALIZE
    env_instance.best_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    env_instance.best_centers = env_instance.centers.detach().clone()
    graph_a = env_instance.graph_state(memory=torch.zeros(policy.hidden_dim))
    graph_b = dict(graph_a)
    graph_b["late_legalize_mode"] = torch.ones((), dtype=graph_a["cell_features"].dtype)
    assert policy._strict_post_legal_mode(graph_a) == policy._strict_post_legal_mode(graph_b)


def test_validate_policy_portfolio_selects_lexicographically():
    saved_collect_episode = train_ppo.collect_episode

    def fake_collect_episode(
        policy,
        sizes,
        env_config,
        device,
        seed,
        temperature,
        soft_tau=None,
        relaxation="sigmoid",
        forced_size=None,
        deterministic=False,
        discover_mode="balanced",
        **kwargs,
    ):
        del policy, sizes, env_config, device, temperature, soft_tau, relaxation, forced_size, deterministic, kwargs
        scores = {
            "balanced": (0.30, 6, 0.60),
            "spread_first": (0.20, 7, 0.90),
            "wire_first": (0.20, 5, 0.70),
            "macro_clearance": (0.25, 5, 0.50),
        }
        refine_winners = {
            "balanced": "incumbent_hold",
            "spread_first": "wire_grad_local",
            "wire_first": "projection_local",
            "macro_clearance": "swap_or_reassign_local",
        }
        overlap, pairs, wl = scores[str(discover_mode)]
        return [], {
            "size": (2, 2),
            "seed": int(seed),
            "discover_mode": str(discover_mode),
            "best_overlap": float(overlap),
            "best_exact_overlap_pairs": int(pairs),
            "best_wl": float(wl),
            "winning_refine_variant": refine_winners[str(discover_mode)],
            "refine_variant_rows": [],
            "refine_window_size": 2,
            "phase_fraction_DISCOVER": 1.0,
            "phase_fraction_LEGALIZE": 0.0,
            "phase_fraction_REFINE": 0.0,
            "phase_fraction_UNLOCK": 0.0,
            "phase_improvements_DISCOVER": 0,
            "phase_improvements_LEGALIZE": 0,
            "phase_improvements_REFINE": 0,
            "phase_improvements_UNLOCK": 0,
            "phase_failure_score": 0.0,
            "branch_violation": 0.0,
            "missed_pairs": 0.0,
            "hard_pair_age_mean": 0.0,
            "audit_pressure_scale": 1.0,
            "audit_pressure_target": 0.0,
            "stop_probability": 0.0,
            "stop_gated": False,
            "stop_overlap": 0.0,
            "false_stop": False,
            "stop": False,
            "memory_reset_count": 0,
            "memory_reset_applied": False,
        }

    train_ppo.collect_episode = fake_collect_episode
    try:
        policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
        summary, rows = train_ppo.validate_policy(
            policy,
            [(2, 2)],
            EnvConfig(horizon=1),
            torch.device("cpu"),
            seed=7,
            temperature=0.1,
            episodes=1,
            return_rows=True,
        )
    finally:
        train_ppo.collect_episode = saved_collect_episode
    assert rows[0]["winning_discover_mode"] == "wire_first"
    assert rows[0]["winning_refine_variant"] == "projection_local"
    assert summary["validation_mode_wins_wire_first"] == 1
    assert summary["validation_refine_variant_wins_projection_local"] == 1


def test_unlock_can_only_start_from_legalize():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    env_instance.phase = env.PlacementPhase.DISCOVER
    env_instance.stagnation_steps = 5
    env_instance.legalize_stall_steps = 5
    env_instance.legalize_pair_stall_steps = 5
    best_score = dict(env_instance.best_score)
    transitioned, _reason = env_instance._advance_phase(
        action=None,
        phase_before=env.PlacementPhase.DISCOVER,
        phase_request_name="unlock",
        current_score=best_score,
        best_before=best_score,
        best_after=best_score,
    )
    assert env_instance.phase != env.PlacementPhase.UNLOCK


def test_post_legal_legalize_phase_request_disables_unlock():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1, enable_clusters=True, enable_stop=True))
    env_instance.phase = env.PlacementPhase.LEGALIZE
    env_instance.best_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    env_instance.best_centers = env_instance.centers.detach().clone()
    graph = env_instance.graph_state(memory=torch.zeros(policy.hidden_dim))
    h = policy.encode(graph)
    _pooled, base_context = policy._global_features(h)
    _memory, recurrent_context = policy._memory_pair(h, graph)
    context = policy._phase_context(recurrent_context, graph)
    logits = policy._phase_request_logits(context, "LEGALIZE", strict_post_legal_mode=True)
    assert float(logits[PHASE_REQUEST_TO_INDEX["unlock"]].detach().item()) < -1.0e8


def test_refine_compaction_operator_is_local():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_clusters=True, enable_stop=True, enable_phr_layer=False),
    )
    env_instance.phase = env.PlacementPhase.REFINE
    env_instance.best_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    env_instance.best_centers = env_instance.centers.detach().clone()
    graph = env_instance.graph_state(memory=torch.zeros(policy.hidden_dim))
    graph["phase_name"] = "REFINE"
    action = policy.sample_action(graph, deterministic=True)
    step_scale = float(action.step_scale.detach().item())
    residual = torch.tanh(action.residual_flow.detach()) * (step_scale * env_instance.length_scale)
    base_centers, window, _wire_grad = env_instance._build_refine_compaction_base_centers(
        action=action,
        incumbent_centers=env_instance.best_centers.detach().clone(),
        residual=residual,
        step_scale=step_scale,
    )
    assert int(window.numel()) > 0
    outside = torch.ones((env_instance.centers.shape[0],), dtype=torch.bool)
    outside[window.long()] = False
    if torch.any(outside):
        assert torch.allclose(
            base_centers[outside],
            env_instance.best_centers[outside],
            atol=1e-6,
        )
    assert torch.any(torch.norm(base_centers[window] - env_instance.best_centers[window], dim=1) > 0.0)


def test_refine_window_is_bounded_and_deterministic():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, refine_window_min_cells=2, refine_window_max_cells=3),
    )
    incumbent = env_instance.centers.detach().clone()
    current = incumbent.clone()
    current[2, 0] += 0.75
    residual = current - incumbent
    wire_grad = env_instance._wirelength_gradient_at(incumbent)
    window_a = env_instance._select_refine_window(incumbent, current_centers=current, residual=residual, wire_grad=wire_grad)
    window_b = env_instance._select_refine_window(incumbent, current_centers=current, residual=residual, wire_grad=wire_grad)
    assert torch.equal(window_a, window_b)
    assert 2 <= int(window_a.numel()) <= 3


def test_swap_or_reassign_variant_is_local():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    incumbent = env_instance.centers.detach().clone()
    window = torch.tensor([2, 3], dtype=torch.long)
    base_centers = env_instance._swap_or_reassign_local_base_centers(
        incumbent_centers=incumbent,
        window=window,
    )
    outside = torch.ones((incumbent.shape[0],), dtype=torch.bool)
    outside[window] = False
    assert torch.allclose(base_centers[outside], incumbent[outside], atol=1e-6)


def test_swap_or_reassign_metadata_exposes_window_and_groups():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    incumbent = env_instance.centers.detach().clone()
    window = torch.tensor([2, 3], dtype=torch.long)
    _base_centers, metadata = env_instance._swap_or_reassign_local_base_centers(
        incumbent_centers=incumbent,
        window=window,
        return_metadata=True,
    )
    assert metadata["window_indices"] == [2, 3]
    assert metadata["attempted_swaps"] >= 1
    assert metadata["same_size_groups"]
    assert metadata["same_size_groups"][0]["members"] == [2, 3]


def test_refine_portfolio_candidates_preserve_incumbent_contract():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, coordinate_steps=2, enable_phr_layer=True),
    )
    incumbent = env_instance.centers.detach().clone()
    incumbent_score = env_instance._score_centers(incumbent)
    refine_result = env_instance.run_post_legal_refine_portfolio(incumbent)
    variant_names = {row["variant_name"] for row in refine_result["variant_rows"]}
    assert variant_names == {"incumbent_hold", "wire_grad_local", "projection_local", "swap_or_reassign_local"}
    assert env_instance._refine_variant_rank_key(refine_result["score"]) <= env_instance._refine_variant_rank_key(incumbent_score)
    if refine_result["winning_variant"] != "incumbent_hold":
        assert env_instance._refine_variant_rank_key(refine_result["score"]) < env_instance._refine_variant_rank_key(incumbent_score)


def test_unlock_actions_are_localized_to_window():
    policy = OrderingPolicy(hidden_dim=32, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(
        cell_features,
        pin_features,
        edge_list,
        EnvConfig(horizon=1, enable_clusters=False, enable_stop=False, enable_phr_layer=False),
    )
    env_instance.phase = env.PlacementPhase.UNLOCK
    env_instance.unlock_remaining_steps = 2
    graph = env_instance.graph_state(memory=policy.initial_memory(torch.device("cpu")))
    action = policy.sample_action(graph, deterministic=True)
    reward, done, info = env_instance.step_action(action, entropy=action.entropy, trace_transition=True)
    transition = env_instance.last_transition_trace
    assert transition is not None
    unlock_cells = transition["unlock_window_cells"]
    base_centers = transition["base_centers"]
    incumbent_anchor = transition["incumbent_anchor"]
    outside = torch.ones((base_centers.shape[0],), dtype=torch.bool)
    if unlock_cells.numel() > 0:
        outside[unlock_cells.long()] = False
    if torch.any(outside):
        assert torch.allclose(base_centers[outside], incumbent_anchor[outside], atol=1e-6)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "phase" in info


def test_validation_rewind_trigger_logic():
    assert train_ppo.should_apply_validation_rewind(
        0.55,
        20.0,
        0.40,
        12.0,
        overlap_epsilon=0.02,
        pair_epsilon=2.0,
    ) is True
    assert train_ppo.should_apply_validation_rewind(
        0.41,
        15.5,
        0.40,
        12.0,
        overlap_epsilon=0.02,
        pair_epsilon=2.0,
    ) is True
    assert train_ppo.should_apply_validation_rewind(
        0.41,
        13.0,
        0.40,
        12.0,
        overlap_epsilon=0.02,
        pair_epsilon=2.0,
    ) is False


def test_validation_rewind_lr_decay_helper():
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    before, after = train_ppo.decay_optimizer_lr(optimizer, decay=0.5, min_lr=1e-5)
    assert abs(before - 1e-3) < 1e-12
    assert abs(after - 5e-4) < 1e-12
    before_2, after_2 = train_ppo.decay_optimizer_lr(optimizer, decay=0.01, min_lr=1e-5)
    assert abs(before_2 - 5e-4) < 1e-12
    assert abs(after_2 - 1e-5) < 1e-12


def test_training_loop_has_validation_rewind_controller():
    source = inspect.getsource(train_ppo)
    assert "should_apply_validation_rewind(" in source
    assert "latest_pre_rewind.pt" in source
    assert "validation_rewind_applied" in source
    assert "snapshot_training_state(" in source
    assert "restore_training_state(" in source


def test_validation_replay_suite_ranks_worst_cases():
    rows = [
        {"size": (2, 20), "seed": 1, "best_overlap": 0.2, "best_exact_overlap_pairs": 5.0, "best_wl": 0.4},
        {"size": (3, 25), "seed": 2, "best_overlap": 0.6, "best_exact_overlap_pairs": 12.0, "best_wl": 0.5},
        {"size": (2, 30), "seed": 3, "best_overlap": 0.4, "best_exact_overlap_pairs": 8.0, "best_wl": 0.3},
    ]
    ranked = train_ppo.build_validation_replay_suite(rows, suite_size=2)
    assert len(ranked) == 2
    assert ranked[0]["seed"] == 2
    assert ranked[1]["seed"] == 3


def test_wirelength_gate_requires_low_overlap_basin():
    delta, active = gated_wirelength_delta(
        best_overlap_delta=0.05,
        best_wirelength_delta=-0.8,
        best_after_overlap=0.19,
        best_after_pairs=6.0,
        gate_epsilon=0.002,
        overlap_threshold=0.25,
        pairs_threshold=8.0,
    )
    assert active is True
    assert abs(delta + 0.8) < 1e-8

    delta_high_overlap, active_high_overlap = gated_wirelength_delta(
        best_overlap_delta=0.05,
        best_wirelength_delta=-0.8,
        best_after_overlap=0.31,
        best_after_pairs=6.0,
        gate_epsilon=0.002,
        overlap_threshold=0.25,
        pairs_threshold=8.0,
    )
    assert active_high_overlap is False
    assert abs(delta_high_overlap) < 1e-8

    delta_high_pairs, active_high_pairs = gated_wirelength_delta(
        best_overlap_delta=0.05,
        best_wirelength_delta=0.3,
        best_after_overlap=0.19,
        best_after_pairs=12.0,
        gate_epsilon=0.002,
        overlap_threshold=0.25,
        pairs_threshold=8.0,
    )
    assert active_high_pairs is False
    assert abs(delta_high_pairs) < 1e-8

    delta_regressing, active_regressing = gated_wirelength_delta(
        best_overlap_delta=-0.01,
        best_wirelength_delta=0.3,
        best_after_overlap=0.19,
        best_after_pairs=6.0,
        gate_epsilon=0.002,
        overlap_threshold=0.25,
        pairs_threshold=8.0,
    )
    assert active_regressing is False
    assert abs(delta_regressing) < 1e-8


def test_validate_policy_can_return_rows():
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    summary, rows = train_ppo.validate_policy(
        policy,
        sizes=[(1, 2)],
        env_config=EnvConfig(horizon=1, enable_clusters=False, enable_stop=False),
        device=torch.device("cpu"),
        seed=123,
        temperature=1.0,
        soft_tau=1.5,
        relaxation="sigmoid",
        episodes=1,
        validation_suite=[{"size": (1, 2), "seed": 123}],
        memory_reset_mode="incumbent_improve_material_or_stale",
        memory_reset_retain=0.75,
        memory_reset_min_overlap_gain=0.03,
        memory_reset_min_pair_gain_count=2.0,
        memory_reset_min_steps_since_best=2,
        return_rows=True,
    )
    assert summary["validation_episodes"] == 1
    assert len(rows) == 1
    assert rows[0]["seed"] == 123


def test_training_loop_uses_validation_replay_suite():
    source = inspect.getsource(train_ppo)
    assert "validation_replay_suite" in source
    assert "build_validation_replay_suite(" in source
    assert "validation_replay_episodes" in source


def test_wisdom_catalog_covers_major_patterns():
    classes = {row["wisdom_class"] for row in wisdom_audit.WISDOM_CLASS_SPECS}
    required = {
        "stage_basin_formation",
        "stage_legalization",
        "stage_post_legal_cleanup",
        "stage_stuck_layout_escape",
        "selection_legality_first_ranking",
        "selection_candidate_competition",
        "locality_same_size_structure",
        "locality_translation_clean_scoring",
        "case_shape_routing",
        "case_shape_macro_context",
        "case_shape_large_scale_pruning",
        "search_preserve_good_basins",
        "search_relegalize_before_scoring",
    }
    assert required.issubset(classes)


def test_wisdom_transfer_mechanisms_are_valid():
    for row in wisdom_audit.WISDOM_CLASS_SPECS:
        assert row["transfer_mechanism"] in wisdom_audit.VALID_TRANSFER_MECHANISMS
        assert row["transfer_bucket"] in wisdom_audit.VALID_TRANSFER_BUCKETS
        assert row["wisdom_status"] in wisdom_audit.VALID_WISDOM_STATUSES


def test_wisdom_case_gap_uses_counterfactual_factor_mapping():
    validation_rows = [
        {
            "suite_index": 0,
            "size": "2:20",
            "size_macros": 2,
            "size_std_cells": 20,
            "seed": 1001234,
            "winning_discover_mode": "spread_first",
            "winning_refine_variant": "swap_or_reassign_local",
            "best_overlap": 0.0909,
            "best_exact_overlap_pairs": 1,
            "mode_sensitivity": 0.02,
            "variant_sensitivity": 0.12,
        }
    ]
    analyzed_rows = [
        {
            "suite_index": 0,
            "size": "2:20",
            "seed": 1001234,
            "dominant_factor": "memory_sensitivity",
        }
    ]
    gap_rows = wisdom_audit.build_case_gap_rows(validation_rows, analyzed_rows)
    assert gap_rows[0]["missing_wisdom_primary"] == "credit_assignment_too_diffuse"
    assert gap_rows[0]["gap_stage"] == "basin_formation"


def test_wisdom_case_gap_labels_post_legal_cleanup_from_ranking_only():
    validation_rows = [
        {
            "suite_index": 1,
            "size": "5:150",
            "size_macros": 5,
            "size_std_cells": 150,
            "seed": 1002000,
            "winning_discover_mode": "spread_first",
            "winning_refine_variant": "incumbent_hold",
            "best_overlap": 0.0909,
            "best_exact_overlap_pairs": 1,
            "mode_sensitivity": 0.01,
            "variant_sensitivity": 0.03,
        }
    ]
    gap_rows = wisdom_audit.build_case_gap_rows(validation_rows, [])
    assert gap_rows[0]["missing_wisdom_primary"] == "post_legal_cleanup_bias_missing"
    assert gap_rows[0]["missing_wisdom_secondary"] == "candidate_competition_missing"


def test_wisdom_audit_upgrades_internalization_signals():
    rows = {
        row["wisdom_class"]: row
        for row in wisdom_audit.build_wisdom_class_audit(Path(__file__).resolve().parent)
    }
    post_legal = rows["stage_post_legal_cleanup"]
    assert post_legal["wisdom_status"] == "partially_internalized"
    legality = rows["selection_legality_first_ranking"]
    assert legality["wisdom_status"] == "partially_internalized"
    assert legality["selected"] == "full"
    assert legality["optimized"] == "partial"
    routing = rows["case_shape_routing"]
    assert routing["wisdom_status"] == "partially_internalized"
    translation = rows["locality_translation_clean_scoring"]
    assert translation["wisdom_status"] == "partially_internalized"


def test_structural_node_table_covers_required_decisions():
    rows = structural_audit.build_structural_node_rows(Path(__file__).resolve().parent)
    by_name = {row["decision_node"]: row for row in rows}
    required = {
        "case_family_routing",
        "basin_generator_selection",
        "local_cleanup_operator_selection",
        "repaired_candidate_selection",
        "final_lexicographic_winner_selection",
        "candidate_competition",
    }
    assert required.issubset(by_name)
    assert by_name["candidate_competition"]["rl_type"] == "external_wrapper"
    assert by_name["case_family_routing"]["structural_status"] == "supervised_but_not_decisive"


def test_structural_audit_does_not_treat_aux_head_as_internalized():
    rows = {
        row["decision_node"]: row
        for row in structural_audit.build_structural_node_rows(Path(__file__).resolve().parent)
    }
    assert rows["case_family_routing"]["rl_type"] == "auxiliary_only"
    assert rows["case_family_routing"]["structural_status"] == "supervised_but_not_decisive"
    assert rows["repaired_candidate_selection"]["structural_status"] == "externally_imposed"


def test_structural_case_rows_map_full_suite_factors_to_structural_labels():
    validation_rows = [
        {
            "suite_index": 0,
            "size": "2:20",
            "seed": 1001234,
            "winning_discover_mode": "spread_first",
            "winning_refine_variant": "incumbent_hold",
            "mode_sensitivity": 0.90,
            "variant_sensitivity": 0.53,
            "best_overlap": 0.0909,
            "best_exact_overlap_pairs": 1,
            "best_wl": 0.72,
        },
        {
            "suite_index": 1,
            "size": "3:25",
            "seed": 1001235,
            "winning_discover_mode": "macro_clearance",
            "winning_refine_variant": "swap_or_reassign_local",
            "mode_sensitivity": 0.95,
            "variant_sensitivity": 0.45,
            "best_overlap": 0.10,
            "best_exact_overlap_pairs": 2,
            "best_wl": 0.71,
        },
    ]
    analyzed_rows = [
        {
            "suite_index": 0,
            "seed": 1001234,
            "dominant_factor": "wire_recovery_missing",
        },
        {
            "suite_index": 1,
            "seed": 1001235,
            "dominant_factor": "continuation_after_good_basin",
        },
    ]
    rows = structural_audit.build_structural_case_rows(validation_rows, analyzed_rows)
    by_index = {row["suite_index"]: row for row in rows}
    assert by_index[0]["primary_structural_gap"] == "missing_internal_cleanup_selector"
    assert by_index[0]["secondary_structural_gap"] == "missing_repair_value_model"
    assert by_index[1]["primary_structural_gap"] == "missing_internal_ranker"
    assert by_index[1]["secondary_structural_gap"] == "missing_internal_case_router"


def test_structural_hypotheses_cover_h1_to_h5():
    node_rows = structural_audit.build_structural_node_rows(Path(__file__).resolve().parent)
    case_rows = [
        {
            "primary_structural_gap": "missing_internal_ranker",
            "secondary_structural_gap": "single_trajectory_bias",
        },
        {
            "primary_structural_gap": "missing_internal_cleanup_selector",
            "secondary_structural_gap": "missing_repair_value_model",
        },
    ]
    validation_rows = [{"mode_sensitivity": 0.9}]
    rows = structural_audit.build_structural_hypothesis_rows(case_rows, validation_rows, node_rows)
    ids = {row["hypothesis_id"] for row in rows}
    assert ids == {"H1", "H2", "H3", "H4", "H5"}


def test_structural_gap_report_splits_required_categories():
    node_rows = structural_audit.build_structural_node_rows(Path(__file__).resolve().parent)
    case_rows = [
        {
            "primary_structural_gap": "missing_internal_ranker",
            "secondary_structural_gap": "single_trajectory_bias",
        },
        {
            "primary_structural_gap": "missing_internal_cleanup_selector",
            "secondary_structural_gap": "missing_repair_value_model",
        },
    ]
    validation_rows = [{"mode_sensitivity": 0.9}]
    hypothesis_rows = structural_audit.build_structural_hypothesis_rows(case_rows, validation_rows, node_rows)
    report = structural_audit.build_structural_gap_report(node_rows, case_rows, hypothesis_rows)
    assert "structurally_missing_high_impact" in report
    assert "present_only_through_scaffolding" in report
    assert "not_actually_missing" in report
    assert report["next_architecture_experiment"]["priority_gap"] == "missing_internal_ranker"


def test_ranker_falsification_case_rows_use_masked_per_mode_candidates():
    validation_rows = [
        {
            "suite_index": 0,
            "size": "2:20",
            "seed": 123,
            "external_winning_mode": "spread_first",
            "external_selected_source": "refine_variant:swap_or_reassign_local",
            "external_selected_overlap": 0.1,
            "external_selected_pairs": 1,
            "external_selected_wirelength": 0.7,
            "per_mode_info_rows": [
                {
                    "discover_mode": "spread_first",
                    "candidate_records": [
                        train_ppo.candidate_record(source="final_selected", overlap=0.1, pairs=1, wirelength=0.7),
                        train_ppo.candidate_record(source="rollout_best", overlap=0.2, pairs=2, wirelength=0.8, live_input=True),
                    ],
                },
                {
                    "discover_mode": "macro_clearance",
                    "candidate_records": [
                        train_ppo.candidate_record(source="final_selected", overlap=0.3, pairs=2, wirelength=0.9),
                    ],
                },
            ],
        }
    ]
    mode_rows = [
        {
            "suite_index": 0,
            "seed": 123,
            "discover_mode": "spread_first",
            "chooser_selected_source": "rollout_best",
            "chooser_selected_overlap": 0.2,
            "chooser_selected_pairs": 2,
            "chooser_selected_wirelength": 0.8,
        },
        {
            "suite_index": 0,
            "seed": 123,
            "discover_mode": "macro_clearance",
            "chooser_selected_source": "final_selected",
            "chooser_selected_overlap": 0.15,
            "chooser_selected_pairs": 2,
            "chooser_selected_wirelength": 0.9,
        },
    ]
    rows = run_structural_falsification.build_case_falsification_rows(validation_rows, mode_rows)
    assert rows[0]["external_winning_mode"] == "spread_first"
    assert rows[0]["chooser_selected_winning_mode"] == "macro_clearance"
    assert rows[0]["overlap_regret"] > 0.0


def test_ranker_falsification_summary_counts_regret_and_matches():
    mode_rows = [
        {"metric_match": True, "overlap_regret": 0.0, "pair_regret": 0, "wire_regret": 0.0},
        {"metric_match": False, "overlap_regret": 0.1, "pair_regret": 1, "wire_regret": 0.2},
    ]
    case_rows = [
        {"metric_match": False, "overlap_regret": 0.1, "pair_regret": 0, "wire_regret": 0.3},
    ]
    summary = run_structural_falsification.build_summary(mode_rows, case_rows)
    assert summary["mode_match_rate"] == 0.5
    assert summary["case_match_rate"] == 0.0
    assert summary["mode_overlap_worse_count"] == 1
    assert summary["case_wire_worse_count"] == 1


def test_refine_supervision_rows_are_lexicographically_ranked():
    rows = train_ppo.enrich_refine_supervision_rows(
        [
            {
                "variant_name": "incumbent_hold",
                "accepted": False,
                "repair_legal": True,
                "overlap_ratio": 0.2,
                "num_overlap_pairs": 2,
                "normalized_wl": 0.7,
                "overlap_delta": 0.0,
                "pair_delta": 0,
                "wire_delta": 0.0,
            },
            {
                "variant_name": "swap_or_reassign_local",
                "accepted": True,
                "repair_legal": True,
                "overlap_ratio": 0.1,
                "num_overlap_pairs": 1,
                "normalized_wl": 0.8,
                "overlap_delta": 0.1,
                "pair_delta": 1,
                "wire_delta": -0.1,
            },
        ]
    )
    by_name = {row["variant_name"]: row for row in rows}
    assert by_name["swap_or_reassign_local"]["candidate_lexi_rank"] == 0
    assert by_name["incumbent_hold"]["candidate_lexi_rank"] == 1


def test_auxiliary_predictions_expose_cleanup_and_mode_heads():
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    graph = env.build_graph_state(
        cell_features,
        pin_features,
        edge_list,
        cleanup_feature_vector=torch.ones(5),
        case_descriptor=torch.ones(6),
        phase_name="REFINE",
    )
    aux = policy.auxiliary_predictions(graph)
    assert aux["cleanup_variant_logits"].shape[0] == len(train_ppo.REFINE_VARIANT_NAMES)
    assert aux["cleanup_accept_logit"].shape[0] == len(train_ppo.REFINE_VARIANT_NAMES)
    assert aux["cleanup_rank_value"].shape[0] == len(train_ppo.REFINE_VARIANT_NAMES)
    assert aux["mode_selector_logits"].shape[0] == len(train_ppo.DISCOVER_MODE_NAMES)
    assert aux["continuation_preserve_logit"].numel() == 1


def test_translation_clean_features_only_activate_post_legal():
    cell_features, pin_features, edge_list = _tiny_problem()
    placement_env = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    discover_features = placement_env._cleanup_feature_vector(
        placement_env.centers.detach().clone(),
        phase_name=placement_env.phase,
        incumbent_centers=placement_env.best_centers.detach().clone(),
    )
    assert torch.allclose(discover_features, torch.zeros_like(discover_features))
    refine_features = placement_env._cleanup_feature_vector(
        placement_env.best_centers.detach().clone(),
        phase_name=env.PlacementPhase.REFINE,
        incumbent_centers=placement_env.best_centers.detach().clone(),
    )
    assert refine_features.shape[0] == 5


def test_validate_policy_rows_expose_cleanup_supervision():
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    _summary, rows = train_ppo.validate_policy(
        policy,
        sizes=[(1, 2)],
        env_config=EnvConfig(horizon=1, enable_clusters=False, enable_stop=False),
        device=torch.device("cpu"),
        seed=123,
        temperature=1.0,
        soft_tau=1.5,
        relaxation="sigmoid",
        episodes=1,
        validation_suite=[{"size": (1, 2), "seed": 123}],
        memory_reset_mode="incumbent_improve_material_or_stale",
        memory_reset_retain=0.75,
        memory_reset_min_overlap_gain=0.03,
        memory_reset_min_pair_gain_count=2.0,
        memory_reset_min_steps_since_best=2,
        return_rows=True,
    )
    assert "per_mode_info_rows" in rows[0]
    assert "cleanup_supervision_available" in rows[0]
    assert "chooser_selected_source" in rows[0]["per_mode_info_rows"][0]
    assert "external_selected_source" in rows[0]["per_mode_info_rows"][0]
    assert "chooser_case_match" in rows[0]


def test_continuation_margin_regressed_requires_material_gap():
    best = train_ppo.candidate_record(source="rollout_best", overlap=0.0, pairs=0, wirelength=0.40, repair_legal=True)
    small = train_ppo.candidate_record(source="final_selected", overlap=0.005, pairs=1, wirelength=0.43, repair_legal=True)
    large = train_ppo.candidate_record(source="final_selected", overlap=0.05, pairs=3, wirelength=0.55, repair_legal=True)
    assert train_ppo.continuation_margin_regressed(best, small) is False
    assert train_ppo.continuation_margin_regressed(best, large) is True


def test_continuation_supervision_only_uses_good_refine_rows():
    eligible = {
        "best_candidate_phase": "REFINE",
        "best_overlap": 0.08,
        "best_exact_overlap_pairs": 2,
    }
    poor_legalize = {
        "best_candidate_phase": "LEGALIZE",
        "best_overlap": 0.08,
        "best_exact_overlap_pairs": 2,
    }
    poor_overlap = {
        "best_candidate_phase": "REFINE",
        "best_overlap": 0.25,
        "best_exact_overlap_pairs": 2,
    }
    poor_pairs = {
        "best_candidate_phase": "REFINE",
        "best_overlap": 0.08,
        "best_exact_overlap_pairs": 8,
    }
    assert train_ppo.continuation_supervision_eligible(eligible) is True
    assert train_ppo.continuation_supervision_eligible(poor_legalize) is False
    assert train_ppo.continuation_supervision_eligible(poor_overlap) is False
    assert train_ppo.continuation_supervision_eligible(poor_pairs) is False


def test_auxiliary_supervision_update_emits_continuation_and_gate_metrics():
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    cell_features, pin_features, edge_list = _tiny_problem()
    graph = env.build_graph_state(
        cell_features,
        pin_features,
        edge_list,
        cleanup_feature_vector=torch.tensor([0.2, 0.6, 0.1, 0.3, 0.0]),
        case_descriptor=torch.ones(6),
        phase_name="REFINE",
        continuation_risk=torch.tensor(1.0),
    )
    candidate_records = train_ppo.finalize_candidate_records(
        [
            train_ppo.candidate_record(
                source="rollout_best",
                overlap=0.0,
                pairs=0,
                wirelength=0.4,
                repair_legal=True,
                accepted=True,
                live_input=True,
                generation_index=0,
            ),
            train_ppo.candidate_record(
                source="refine_variant:swap_or_reassign_local",
                overlap=0.1,
                pairs=1,
                wirelength=0.5,
                repair_legal=False,
                accepted=False,
                live_input=True,
                generation_index=1,
                variant_name="swap_or_reassign_local",
            ),
            train_ppo.candidate_record(
                source="final_selected",
                overlap=0.1,
                pairs=1,
                wirelength=0.5,
                repair_legal=False,
                live_input=False,
                diagnostic_only=True,
                external_selected=True,
                generation_index=2,
                origin_source="refine_variant:swap_or_reassign_local",
            ),
        ]
    )
    refine_rows = train_ppo.enrich_refine_supervision_rows(
        [
            {
                "variant_name": "incumbent_hold",
                "accepted": False,
                "repair_legal": True,
                "overlap_ratio": 0.0,
                "num_overlap_pairs": 0,
                "normalized_wl": 0.45,
                "overlap_delta": 0.0,
                "pair_delta": 0,
                "wire_delta": -0.05,
            },
            {
                "variant_name": "swap_or_reassign_local",
                "accepted": True,
                "repair_legal": True,
                "overlap_ratio": 0.0,
                "num_overlap_pairs": 0,
                "normalized_wl": 0.35,
                "overlap_delta": 0.0,
                "pair_delta": 0,
                "wire_delta": 0.05,
            },
        ]
    )
    rows = [
        {
            "best_so_far_returned": False,
            "winning_discover_mode": "spread_first",
            "aux_graph": graph,
            "per_mode_info_rows": [
                {
                    "discover_mode": "spread_first",
                    "aux_graph": graph,
                    "refine_supervision_rows": refine_rows,
                    "candidate_records": candidate_records,
                    "best_overlap": 0.0,
                    "best_exact_overlap_pairs": 0,
                    "best_wl": 0.35,
                },
                {
                    "discover_mode": "balanced",
                    "aux_graph": graph,
                    "refine_supervision_rows": refine_rows,
                    "candidate_records": candidate_records,
                    "best_overlap": 0.1,
                    "best_exact_overlap_pairs": 1,
                    "best_wl": 0.5,
                },
            ],
        }
    ]
    stats = train_ppo.auxiliary_supervision_update(policy, optimizer, rows)
    assert "continuation_aux_loss" in stats
    assert "continuation_preserve_accuracy" in stats
    assert "refine_gate_top1" in stats
    assert "refine_gate_top2" in stats
    assert "chooser_aux_loss" in stats
    assert "chooser_top1_match_rate_mode" in stats


def test_live_chooser_input_excludes_final_selected():
    records = train_ppo.finalize_candidate_records(
        [
            train_ppo.candidate_record(
                source="rollout_best",
                overlap=0.0,
                pairs=0,
                wirelength=0.4,
                repair_legal=True,
                accepted=True,
                live_input=True,
                generation_index=0,
            ),
            train_ppo.candidate_record(
                source="refine_variant:projection_local",
                overlap=0.0,
                pairs=0,
                wirelength=0.35,
                repair_legal=True,
                accepted=True,
                live_input=True,
                generation_index=1,
                variant_name="projection_local",
            ),
            train_ppo.candidate_record(
                source="final_selected",
                overlap=0.0,
                pairs=0,
                wirelength=0.35,
                repair_legal=True,
                live_input=False,
                diagnostic_only=True,
                generation_index=2,
                origin_source="refine_variant:projection_local",
            ),
        ]
    )
    chooser_records = train_ppo.chooser_input_records(records)
    assert len(chooser_records) == 2
    assert all(row["candidate_source"] != "final_selected" for row in chooser_records)


def test_external_teacher_winner_uses_stable_generation_order():
    records = train_ppo.finalize_candidate_records(
        [
            train_ppo.candidate_record(
                source="rollout_best",
                overlap=0.0,
                pairs=0,
                wirelength=0.4,
                repair_legal=True,
                live_input=True,
                generation_index=0,
            ),
            train_ppo.candidate_record(
                source="refine_variant:wire_grad_local",
                overlap=0.0,
                pairs=0,
                wirelength=0.4,
                repair_legal=True,
                live_input=True,
                generation_index=1,
                variant_name="wire_grad_local",
            ),
        ]
    )
    winner = train_ppo.external_candidate_teacher_winner(train_ppo.chooser_input_records(records))
    assert winner["candidate_source"] == "rollout_best"


def test_choose_repaired_candidate_bypasses_single_candidate():
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    graph = env.build_graph_state(cell_features, pin_features, edge_list, phase_name="REFINE")
    chooser_features = torch.zeros((1, 16), dtype=torch.float32)
    logits, chosen_index = policy.choose_repaired_candidate(graph, chooser_features)
    assert logits.shape == (1,)
    assert chosen_index == 0


def test_choose_repaired_candidate_ignores_legacy_rank_residual_by_default():
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    graph = env.build_graph_state(cell_features, pin_features, edge_list, phase_name="REFINE")
    chooser_features = torch.zeros((2, 16), dtype=torch.float32)
    legacy_features = torch.tensor(
        [
            [0.0, 0.0, 0.4, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.2, 1.0, 1.0, -0.2],
        ],
        dtype=torch.float32,
    )
    logits, chosen_index = policy.choose_repaired_candidate(
        graph,
        chooser_features,
        legacy_candidate_features=legacy_features,
    )
    assert torch.allclose(logits, torch.zeros_like(logits))
    assert chosen_index == 0


def test_chooser_pairwise_weight_penalizes_legality_errors_more_than_wire_only():
    better_legality = train_ppo.candidate_record(
        source="rollout_best",
        overlap=0.047619047619047616,
        pairs=3,
        wirelength=0.8215680122375488,
        repair_legal=True,
        accepted=True,
        generation_index=0,
    )
    worse_legality = train_ppo.candidate_record(
        source="refine_variant:swap_or_reassign_local",
        overlap=0.05714285714285714,
        pairs=4,
        wirelength=0.8171555399894714,
        repair_legal=True,
        accepted=True,
        generation_index=1,
        variant_name="swap_or_reassign_local",
    )
    wire_only_worse = train_ppo.candidate_record(
        source="refine_variant:projection_local",
        overlap=0.047619047619047616,
        pairs=3,
        wirelength=0.8315680122375488,
        repair_legal=True,
        accepted=True,
        generation_index=2,
        variant_name="projection_local",
    )
    legality_weight = train_ppo.chooser_pairwise_weight(better_legality, worse_legality)
    wire_weight = train_ppo.chooser_pairwise_weight(better_legality, wire_only_worse)
    assert legality_weight > wire_weight
    assert legality_weight >= 4.0


def test_chooser_pairwise_weight_upweights_rollout_vs_swap_legality_regressions():
    rollout_teacher = train_ppo.candidate_record(
        source="rollout_best",
        overlap=0.047619047619047616,
        pairs=3,
        wirelength=0.8128370642662048,
        repair_legal=True,
        accepted=True,
        generation_index=0,
    )
    swap_legality_regression = train_ppo.candidate_record(
        source="refine_variant:swap_or_reassign_local",
        overlap=0.05714285714285714,
        pairs=4,
        wirelength=0.8017693758010864,
        repair_legal=True,
        accepted=True,
        generation_index=1,
        variant_name="swap_or_reassign_local",
    )
    projection_legality_regression = train_ppo.candidate_record(
        source="refine_variant:projection_local",
        overlap=0.05714285714285714,
        pairs=4,
        wirelength=0.8017693758010864,
        repair_legal=True,
        accepted=True,
        generation_index=2,
        variant_name="projection_local",
    )
    swap_weight = train_ppo.chooser_pairwise_weight(rollout_teacher, swap_legality_regression)
    projection_weight = train_ppo.chooser_pairwise_weight(rollout_teacher, projection_legality_regression)
    assert swap_weight > projection_weight
    assert swap_weight >= projection_weight * 1.5


def test_chooser_candidate_features_emit_large_case_swap_conflict_flag():
    graph = {
        "cell_features": torch.zeros((105, 6), dtype=torch.float32),
    }
    records = train_ppo.finalize_candidate_records(
        [
            train_ppo.candidate_record(
                source="rollout_best",
                overlap=0.047619047619047616,
                pairs=3,
                wirelength=0.8128370642662048,
                repair_legal=True,
                accepted=True,
                live_input=True,
                generation_index=0,
            ),
            train_ppo.candidate_record(
                source="refine_variant:swap_or_reassign_local",
                overlap=0.05714285714285714,
                pairs=4,
                wirelength=0.8017693758010864,
                repair_legal=True,
                accepted=True,
                live_input=True,
                generation_index=1,
                variant_name="swap_or_reassign_local",
            ),
        ]
    )
    bundle = train_ppo.chooser_candidate_features(records, device=torch.device("cpu"), dtype=torch.float32, graph=graph)
    assert bundle is not None
    chooser_features = bundle["chooser_features"]
    assert chooser_features.shape[1] == 16
    assert float(chooser_features[0, 13].item()) == 0.0
    assert float(chooser_features[1, 13].item()) == 1.0
    assert float(chooser_features[1, 14].item()) == 1.0


def test_choose_repaired_candidate_has_no_explicit_large_case_bias():
    policy = OrderingPolicy(hidden_dim=16, message_passes=1, num_clusters=2, global_flow_rank=1)
    cell_features, pin_features, edge_list = _tiny_problem()
    graph = env.build_graph_state(cell_features, pin_features, edge_list, phase_name="REFINE")
    chooser_features = torch.tensor(
        [
            [0.04761905, 3.0, 0.81283706, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.05714286, 4.0, 0.80176938, 1.0, 1.0, 0.00952381, 1.0, -0.01106769, 0.5, 1.0, -0.00952381, -1.0, 0.01106769, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logits, chosen_index = policy.choose_repaired_candidate(graph, chooser_features)
    assert chosen_index == 0
    assert torch.allclose(logits, torch.zeros_like(logits))


def test_large_case_swap_legality_conflict_filter_marks_candidate_non_live():
    n = 105
    cell_features = torch.zeros((n, 6), dtype=torch.float32)
    cell_features[:, 0] = 1.0
    cell_features[:, 1] = 1.0
    cell_features[:, 4] = 1.0
    cell_features[:, 5] = 1.0
    pin_features = torch.zeros((n, 7), dtype=torch.float32)
    pin_features[:, 0] = torch.arange(n, dtype=torch.float32)
    edge_list = torch.zeros((0, 2), dtype=torch.long)
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    incumbent_score = {
        "overlap_ratio": 0.047619047619047616,
        "num_overlap_pairs": 3,
        "normalized_wl": 0.8128370642662048,
    }
    candidate_score = {
        "overlap_ratio": 0.05714285714285714,
        "num_overlap_pairs": 4,
        "normalized_wl": 0.8017693758010864,
    }
    assert env_instance._large_case_swap_legality_conflict(
        incumbent_score=incumbent_score,
        candidate_score=candidate_score,
    )


def test_refine_variant_selection_always_keeps_incumbent_hold():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    incumbent = env_instance.centers.detach().clone()
    current = incumbent.clone()
    current[2, 0] += 0.5
    residual = current - incumbent
    wire_grad = env_instance._wirelength_gradient_at(incumbent)
    window = env_instance._select_refine_window(incumbent, current_centers=current, residual=residual, wire_grad=wire_grad)
    selected = env_instance._select_refine_variants(incumbent_centers=incumbent, window=window, policy=None)
    assert "incumbent_hold" in selected


def test_adaptive_pd_scope_excludes_discover():
    cell_features = torch.zeros((80, 6), dtype=torch.float32)
    cell_features[:, 0] = 1.0
    cell_features[:, 1] = 1.0
    cell_features[:, 2] = torch.linspace(0.0, 8.0, 80)
    cell_features[:, 3] = 0.0
    cell_features[:, 4] = 1.0
    cell_features[:, 5] = 1.0
    pin_features = torch.zeros((80, 7), dtype=torch.float32)
    pin_features[:, 0] = torch.arange(80, dtype=torch.float32)
    edge_list = torch.stack([torch.arange(79), torch.arange(1, 80)], dim=1)
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    score = {
        "overlap_ratio": 0.25,
        "overlap_cells": 8,
        "num_overlap_pairs": 10,
        "normalized_wl": 0.8,
    }
    assert env_instance._case_descriptor_bucket(env_instance.centers) == "mid_dense"
    extra_discover, _bucket, applied_discover = env_instance._adaptive_pd_step_adjustment(
        phase_name=env.PlacementPhase.DISCOVER,
        current_score=score,
        best_score=score,
    )
    assert extra_discover == 0
    assert applied_discover is False
    extra_legalize, _bucket, applied_legalize = env_instance._adaptive_pd_step_adjustment(
        phase_name=env.PlacementPhase.LEGALIZE,
        current_score=score,
        best_score=score,
    )
    assert extra_legalize >= 0
    assert applied_legalize in {True, False}


def test_continuation_risk_only_appears_in_late_refine():
    cell_features, pin_features, edge_list = _tiny_problem()
    env_instance = PlacementOrderingEnv(cell_features, pin_features, edge_list, EnvConfig(horizon=1))
    best_score = {
        "overlap_ratio": 0.0,
        "overlap_cells": 0,
        "num_overlap_pairs": 0,
        "normalized_wl": 0.40,
    }
    current_score = {
        "overlap_ratio": 0.05,
        "overlap_cells": 1,
        "num_overlap_pairs": 3,
        "normalized_wl": 0.55,
    }
    env_instance.phase = env.PlacementPhase.LEGALIZE
    env_instance.phase_step = 3
    assert env_instance._continuation_risk(
        current_score=current_score,
        best_score=best_score,
        phase_name=env_instance.phase,
        steps_since_best=3,
    ) == 0.0
    env_instance.phase = env.PlacementPhase.REFINE
    env_instance.phase_step = 1
    assert env_instance._continuation_risk(
        current_score=current_score,
        best_score=best_score,
        phase_name=env_instance.phase,
        steps_since_best=3,
    ) == 0.0
    env_instance.phase_step = 3
    assert env_instance._continuation_risk(
        current_score=current_score,
        best_score=best_score,
        phase_name=env_instance.phase,
        steps_since_best=3,
    ) > 0.0


def test_decision_tests_tolerate_missing_counterfactual_files(tmp_dir_name="tmp_decision_suite"):
    root = Path("/tmp") / tmp_dir_name
    if root.exists():
        for child in sorted(root.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            else:
                child.rmdir()
        root.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    try:
        diagnosis_dir = root / "diag"
        diagnosis_dir.mkdir()
        case_csv = diagnosis_dir / "case_diagnosis.csv"
        case_csv.write_text(
            "suite_index,size,seed,dominant_factor,counterfactual_dir\n"
            f"0,2:20,1001234,continuation_after_good_basin,{diagnosis_dir / 'missing_suite'}\n",
            encoding="utf-8",
        )
        rows = run_decision_tests.read_csv(case_csv)
        summary = run_decision_tests.summarize_case(rows[0])
        assert summary["resolved"] is False
        results = run_decision_tests.build_results([summary])
        assert results["thresholds"]["unresolved_cases"] == 1
    finally:
        if root.exists():
            for child in sorted(root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()
            root.rmdir()


def run_all():
    tests = [
        test_exact_overlap_parity,
        test_branch_antisymmetry,
        test_relabeling_metric_invariance,
        test_no_teacher_in_inference_path,
        test_mode_b_no_repair_selection,
        test_stop_safety_reward,
        test_teacher_annealing_reaches_zero,
        test_public_wrapper_declares_inference_mode,
        test_training_loop_uses_teacher_annealing,
        test_soft_branch_support_is_finite,
        test_branch_dual_identity_update,
        test_active_pair_retention_cache,
        test_metric_authority_checkpoint_contract_present,
        test_ppo_logs_group_ratios,
        test_mac_path_has_no_explicit_cpu_fallbacks,
        test_ordering_path_has_stability_guards,
        test_large_active_set_smoke,
        test_distill_smoke,
        test_default_sequence_pair_run_excludes_inactive_heads,
        test_disable_clusters_and_stop_remove_policy_groups,
        test_graph_state_exposes_incumbent_fields,
        test_disable_incumbent_action_removes_policy_group,
        test_legacy_feature_dims_still_run_with_incumbent_graph,
        test_validation_suite_is_fixed,
        test_metric_gated_tau_uses_baseline_before_hardening,
        test_metric_gated_tau_blocks_hardening_on_exact_overlap_pairs,
        test_metric_gated_temperature_tracks_soft_tau,
        test_audit_pressure_responds_to_exact_overlap_burden,
        test_validation_uses_best_candidate_overlap_pairs,
        test_ppo_uses_scaled_smooth_l1_value_loss,
        test_hard_replay_suite_is_deterministic,
        test_rollout_memory_policy_resets_on_incumbent_improve,
        test_rollout_memory_policy_resets_on_rollback_event,
        test_rollout_memory_policy_material_or_stale_gate,
        test_training_loop_applies_rollout_memory_policy,
        test_rollout_memory_policy_resets_on_phase_transition,
        test_rollout_memory_policy_resets_on_refine_reject_event,
        test_phase_transition_guards,
        test_refine_entry_guard_stays_strict,
        test_refine_regression_falls_back_to_legalize,
        test_refine_acceptance_rejects_wirelength_regression,
        test_unlock_exits_after_fixed_horizon,
        test_refine_stop_bias_prefers_preserving_stale_incumbent,
        test_refine_auto_stop_preserves_better_incumbent,
        test_refine_auto_stop_ignores_small_nonmaterial_regression,
        test_refine_auto_stop_skips_when_current_matches_incumbent,
        test_refine_rollback_preserves_better_incumbent,
        test_refine_rollback_ignores_small_nonmaterial_regression,
        test_refine_rollback_skips_when_incumbent_improves,
        test_policy_phase_masks_disable_ordering_and_stop,
        test_refine_control_heads_ignore_recurrent_memory,
        test_legalize_post_legal_control_heads_ignore_recurrent_memory,
        test_discover_discrete_heads_ignore_recurrent_memory_after_step_one,
        test_discover_sequence_pair_freezes_to_incumbent_after_step_one,
        test_spread_first_discover_control_heads_ignore_recurrent_memory_after_step_one,
        test_discover_mode_caps_incumbent_mix,
        test_graph_state_exposes_discover_mode,
        test_late_legalize_flag_no_longer_changes_policy_strictness,
        test_validate_policy_portfolio_selects_lexicographically,
        test_unlock_can_only_start_from_legalize,
        test_post_legal_legalize_phase_request_disables_unlock,
        test_refine_compaction_operator_is_local,
        test_refine_window_is_bounded_and_deterministic,
        test_swap_or_reassign_variant_is_local,
        test_swap_or_reassign_metadata_exposes_window_and_groups,
        test_refine_portfolio_candidates_preserve_incumbent_contract,
        test_unlock_actions_are_localized_to_window,
        test_validation_rewind_trigger_logic,
        test_validation_rewind_lr_decay_helper,
        test_training_loop_has_validation_rewind_controller,
        test_validation_replay_suite_ranks_worst_cases,
        test_wirelength_gate_requires_low_overlap_basin,
        test_validate_policy_can_return_rows,
        test_training_loop_uses_validation_replay_suite,
        test_wisdom_catalog_covers_major_patterns,
        test_wisdom_transfer_mechanisms_are_valid,
        test_wisdom_case_gap_uses_counterfactual_factor_mapping,
        test_wisdom_case_gap_labels_post_legal_cleanup_from_ranking_only,
        test_wisdom_audit_upgrades_internalization_signals,
        test_structural_node_table_covers_required_decisions,
        test_structural_audit_does_not_treat_aux_head_as_internalized,
        test_structural_case_rows_map_full_suite_factors_to_structural_labels,
        test_structural_hypotheses_cover_h1_to_h5,
        test_structural_gap_report_splits_required_categories,
        test_ranker_falsification_case_rows_use_masked_per_mode_candidates,
        test_ranker_falsification_summary_counts_regret_and_matches,
        test_refine_supervision_rows_are_lexicographically_ranked,
        test_auxiliary_predictions_expose_cleanup_and_mode_heads,
        test_translation_clean_features_only_activate_post_legal,
        test_validate_policy_rows_expose_cleanup_supervision,
        test_continuation_margin_regressed_requires_material_gap,
        test_continuation_supervision_only_uses_good_refine_rows,
        test_auxiliary_supervision_update_emits_continuation_and_gate_metrics,
        test_live_chooser_input_excludes_final_selected,
        test_external_teacher_winner_uses_stable_generation_order,
        test_choose_repaired_candidate_bypasses_single_candidate,
        test_choose_repaired_candidate_ignores_legacy_rank_residual_by_default,
        test_chooser_pairwise_weight_penalizes_legality_errors_more_than_wire_only,
        test_chooser_pairwise_weight_upweights_rollout_vs_swap_legality_regressions,
        test_chooser_candidate_features_emit_large_case_swap_conflict_flag,
        test_choose_repaired_candidate_has_no_explicit_large_case_bias,
        test_large_case_swap_legality_conflict_filter_marks_candidate_non_live,
        test_refine_variant_selection_always_keeps_incumbent_hold,
        test_adaptive_pd_scope_excludes_discover,
        test_continuation_risk_only_appears_in_late_refine,
        test_decision_tests_tolerate_missing_counterfactual_files,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    os.environ.setdefault("PLACEMENT_USE_POLICY", "0")
    os.environ.setdefault("PLACEMENT_POLICY_TRANSITION", "0")
    run_all()
