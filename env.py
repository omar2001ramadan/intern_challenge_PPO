"""CMDP environment for ordering PPO placement."""

import math
from dataclasses import dataclass

import torch
import torch.optim as optim

from active_set import (
    build_initial_active_pairs,
    canonicalize_pairs,
    pair_sort_keys,
    sort_pairs_and_payload,
    update_active_pair_cache,
)
from constraints import (
    boundary_signed_constraints,
    branch_signed_constraints,
    density_bin_constraints,
    density_pressure_per_cell,
    density_spread_violation,
    exact_overlap_pairs,
    outline_from_cells,
    overlap_ratio_from_pairs,
    overlap_repulsion_for_pairs,
    soft_signed_disjunction,
)
from induce_branches import Branch, induce_branches_from_sequence_pair
from ordering_policy import (
    DISCOVER_MODE_NAMES,
    DISCOVER_MODE_TO_INDEX,
    build_graph_state,
    hierarchical_active_branch_weights,
)
from primal_dual import phr_dual_update, phr_inequality_penalty


class PlacementPhase:
    DISCOVER = "DISCOVER"
    LEGALIZE = "LEGALIZE"
    REFINE = "REFINE"
    UNLOCK = "UNLOCK"


REFINE_VARIANT_NAMES = (
    "incumbent_hold",
    "wire_grad_local",
    "projection_local",
    "swap_or_reassign_local",
)


@dataclass
class EnvConfig:
    horizon: int = 4
    coordinate_steps: int = 8
    coordinate_lr: float = 0.01
    rho: float = 8.0
    lambda_wirelength: float = 1.0
    lambda_overlap: float = 16.0
    lambda_density: float = 4.0
    entropy_reward_coef: float = 0.005
    terminal_feasible_bonus: float = 2.0
    terminal_wirelength_coef: float = 0.25
    active_pair_limit: int = 500_000
    active_pair_retention: int = 4
    density_bins: int = 8
    density_rho_max: float = 0.85
    soft_relaxation: bool = True
    soft_tau: float = 1.0
    exact_overlap_reward_coef: float = 8.0
    exact_wirelength_reward_coef: float = 0.10
    exact_overlap_pairs_reward_coef: float = 1.0
    current_overlap_penalty_coef: float = 1.0
    current_overlap_pairs_penalty_coef: float = 0.25
    lag_reward_coef: float = 0.10
    lag_reward_tanh_scale: float = 25.0
    exact_overlap_regression_coef: float = 16.0
    exact_wirelength_gate_epsilon: float = 0.002
    exact_wirelength_reward_overlap_threshold: float = 0.25
    exact_wirelength_reward_pairs_threshold: float = 8.0
    branch_violation_penalty_coef: float = 2.0
    missed_pair_penalty_coef: float = 0.01
    movement_penalty_coef: float = 0.001
    stop_overlap_penalty: float = 2.0
    stop_no_progress_penalty: float = 2.0
    stop_wirelength_coef: float = 0.25
    enable_stop_gate: bool = True
    stop_gate_overlap_threshold: float = 0.02
    stop_gate_penalty: float = 4.0
    soft_branch_epsilon: float = 1e-4
    rho_min: float = 0.05
    rho_max: float = 128.0
    dual_max: float = 10_000.0
    audit_missed_target: float = 64.0
    audit_pressure_gamma: float = 1.0
    audit_pressure_max: float = 4.0
    incumbent_overlap_gap_penalty_coef: float = 4.0
    incumbent_pair_gap_penalty_coef: float = 1.0
    incumbent_position_gap_penalty_coef: float = 0.25
    incumbent_blend_max: float = 1.0
    enable_residual_flow: bool = True
    enable_phr_layer: bool = True
    enable_exact_audit: bool = True
    enable_density: bool = True
    enable_clusters: bool = True
    enable_stop: bool = True
    enable_incumbent_state: bool = True
    enable_incumbent_action: bool = True
    fixed_pd_controls: bool = False
    ordering_representation: str = "sequence_pair"
    branch_mode: str = "ordering"
    al_mode: str = "signed_phr"
    discover_exit_overlap: float = 0.40
    discover_patience: int = 2
    refine_overlap_threshold: float = 0.25
    refine_pairs_threshold: int = 8
    refine_entry_overlap_threshold: float = 0.30
    discover_mode_legalize_carry_steps: int = 2
    late_legalize_overlap_threshold: float = 0.35
    late_legalize_pairs_threshold: int = 8
    legal_streak_required: int = 2
    late_legalize_min_pd_steps: int = 8
    late_legalize_step_scale_multiplier: float = 0.5
    late_legalize_min_incumbent_mix: float = 0.75
    refine_regression_overlap: float = 0.01
    refine_regression_pairs: int = 1
    refine_wirelength_regression_epsilon: float = 0.002
    refine_compaction_fraction: float = 0.25
    refine_compaction_radius_scale: float = 2.0
    refine_compaction_max_cells: int = 16
    refine_compaction_gradient_weight: float = 1.0
    refine_compaction_residual_weight: float = 0.5
    refine_compaction_step_multiplier: float = 0.75
    refine_use_compaction_operator: bool = False
    refine_window_min_cells: int = 8
    refine_window_max_cells: int = 16
    refine_window_displacement_weight: float = 0.75
    refine_window_active_pair_weight: float = 0.50
    refine_window_same_size_bonus: float = 0.25
    swap_reassign_window_max_cells: int = 20
    swap_reassign_max_passes: int = 6
    large_case_swap_filter_min_std_cells: int = 100
    refine_stop_bias: float = 3.0
    refine_stop_min_phase_steps: int = 1
    refine_stop_wirelength_margin: float = 0.01
    refine_auto_stop_stale_steps: int = 1
    continuation_risk_stale_steps: int = 2
    continuation_risk_overlap_gap: float = 0.02
    continuation_risk_pair_gap: int = 2
    continuation_stop_bias: float = 0.75
    continuation_refine_reject_scale: float = 0.5
    continuation_return_best_bias: float = 0.95
    continuation_aux_min_overlap: float = 0.10
    continuation_margin_overlap: float = 0.03
    continuation_margin_pairs: int = 2
    continuation_margin_wire: float = 0.08
    continuation_refine_phase_bonus: float = 0.05
    continuation_refine_min_phase_steps: int = 2
    adaptive_pd_mid_cells_min: int = 45
    adaptive_pd_mid_cells_max: int = 110
    adaptive_pd_density_threshold: float = 0.08
    adaptive_pd_extra_steps_small: int = 4
    adaptive_pd_extra_steps_large: int = 8
    unlock_patience: int = 2
    unlock_horizon: int = 2
    unlock_window_max_cells: int = 12


def write_positions(cell_features, centers):
    updated = cell_features.clone()
    updated[:, 2:4] = centers
    return updated


def wirelength_loss(cell_features, pin_features, edge_list):
    """Differentiable normalized pin-pair wirelength surrogate."""
    if edge_list.shape[0] == 0:
        return cell_features[:, 2:4].sum() * 0.0

    cell_positions = cell_features[:, 2:4]
    cell_indices = pin_features[:, 0].long()
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]

    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()
    dx = torch.abs(pin_absolute_x[src_pins] - pin_absolute_x[tgt_pins])
    dy = torch.abs(pin_absolute_y[src_pins] - pin_absolute_y[tgt_pins])
    alpha = 0.1
    smooth_manhattan = alpha * torch.logsumexp(torch.stack([dx / alpha, dy / alpha], dim=0), dim=0)
    return smooth_manhattan.mean()


def normalized_wirelength(cell_features, pin_features, edge_list):
    if edge_list.shape[0] == 0:
        return 0.0
    with torch.no_grad():
        total_area = torch.clamp(cell_features[:, 0].sum(), min=1.0)
        return (wirelength_loss(cell_features, pin_features, edge_list) / torch.sqrt(total_area)).item()


def translation_clean_normalized_wirelength(cell_features, pin_features, edge_list):
    if edge_list.shape[0] == 0:
        return 0.0
    with torch.no_grad():
        pin_to_cell = pin_features[:, 0].long()
        src = edge_list[:, 0].long()
        dst = edge_list[:, 1].long()
        keep = pin_to_cell[src] != pin_to_cell[dst]
        if not torch.any(keep):
            return 0.0
        total_area = torch.clamp(cell_features[:, 0].sum(), min=1.0)
        return (wirelength_loss(cell_features, pin_features, edge_list[keep]) / torch.sqrt(total_area)).item()


def gated_wirelength_delta(
    *,
    best_overlap_delta,
    best_wirelength_delta,
    best_after_overlap,
    best_after_pairs,
    gate_epsilon,
    overlap_threshold,
    pairs_threshold,
):
    non_regressing = float(best_overlap_delta) >= -float(gate_epsilon)
    low_overlap = float(best_after_overlap) <= float(overlap_threshold)
    low_pairs = float(best_after_pairs) <= float(pairs_threshold)
    gate_active = bool(non_regressing and low_overlap and low_pairs)
    return (float(best_wirelength_delta) if gate_active else 0.0), gate_active


def sequence_pair_from_centers(centers):
    x = centers[:, 0]
    y = centers[:, 1]
    seq_plus = torch.argsort(x + 1.0e-3 * y, descending=False)
    seq_minus = torch.argsort(x - 1.0e-3 * y, descending=False)
    return seq_plus, seq_minus


class PlacementOrderingEnv:
    """One placement episode driven by global ordering actions."""

    def __init__(self, cell_features, pin_features, edge_list, config=None, discover_mode="balanced"):
        self.config = config or EnvConfig()
        self.cell_features = cell_features.clone()
        self.pin_features = pin_features.clone()
        self.edge_list = edge_list.clone()
        self.centers = self.cell_features[:, 2:4].clone()
        self.initial_centers = self.centers.clone()
        self.length_scale = torch.sqrt(torch.clamp(self.cell_features[:, 0].sum(), min=1.0))
        self.area_scale = torch.clamp(self.cell_features[:, 0].mean(), min=1.0)
        self.bounds = outline_from_cells(self.cell_features)
        self.active_pairs = build_initial_active_pairs(
            write_positions(self.cell_features, self.centers),
            self.pin_features,
            self.edge_list,
            max_pairs=self.config.active_pair_limit,
        )
        self.active_pair_ages = torch.full(
            (self.active_pairs.shape[0],),
            self.config.active_pair_retention,
            dtype=torch.long,
            device=self.cell_features.device,
        )
        self.branch_duals = torch.zeros((self.active_pairs.shape[0], 4), dtype=self.cell_features.dtype, device=self.cell_features.device)
        self.boundary_duals = torch.zeros((self.cell_features.shape[0], 4), dtype=self.cell_features.dtype, device=self.cell_features.device)
        density_g, assignment = self._density_constraints(self.centers)
        self.density_duals = torch.zeros_like(density_g)
        self.saved_candidates = []
        self.best_score = None
        self.best_centers = self.centers.detach().clone()
        self.best_candidate_step = 0
        self.steps_since_best = 0
        self.step_index = 0
        self.last_transition_trace = None
        self._save_candidate(self.centers, candidate_step=0)
        self.phase = PlacementPhase.DISCOVER
        self.phase_step = 0
        self.legal_streak = 0
        self.stagnation_steps = 0
        self.legalize_stall_steps = 0
        self.legalize_pair_stall_steps = 0
        self.unlock_remaining_steps = 0
        self.unlock_window_cells = torch.empty((0,), dtype=torch.long, device=self.cell_features.device)
        self.unlock_anchor_centers = self.centers.detach().clone()
        self.unlock_anchor_score = dict(self.best_score)
        discover_mode = str(discover_mode).lower()
        if discover_mode not in DISCOVER_MODE_TO_INDEX:
            discover_mode = DISCOVER_MODE_NAMES[0]
        self.discover_mode = discover_mode
        self.discover_mode_index = DISCOVER_MODE_TO_INDEX[discover_mode]
        self.discover_mode_carry_steps = 0
        self.phase_entry_best_score = dict(self.best_score)
        self.phase_entry_best_centers = self.best_centers.detach().clone()
        self.best_candidate_phase = str(self.phase)
        self.best_candidate_discover_mode = str(self.discover_mode)
        self.best_candidate_refine_variant = "incumbent_hold"

    def graph_state(self, memory=None):
        density_g, assignment = self._density_constraints(self.centers)
        current_score = self._score_centers(self.centers)
        best_score, best_centers = self.best_candidate()
        continuation_risk = self._continuation_risk(
            current_score=current_score,
            best_score=best_score,
            phase_name=self.phase,
            steps_since_best=self.steps_since_best,
        )
        overlap_ratio = torch.tensor(
            current_score["overlap_ratio"],
            dtype=self.centers.dtype,
            device=self.centers.device,
        )
        stop_logit_bias = 0.0
        if self.config.enable_stop and self.config.enable_stop_gate and current_score["overlap_ratio"] > self.config.stop_gate_overlap_threshold:
            stop_logit_bias = -float(self.config.stop_gate_penalty)
        if self.phase == PlacementPhase.REFINE and self._is_refine_ready(best_score):
            if int(self.phase_step) >= int(self.config.refine_stop_min_phase_steps):
                current_not_beating_incumbent = (
                    float(current_score["overlap_ratio"]) > float(best_score["overlap_ratio"]) + float(self.config.refine_regression_overlap)
                    or int(current_score["num_overlap_pairs"]) > int(best_score["num_overlap_pairs"]) + int(self.config.refine_regression_pairs)
                    or float(current_score["normalized_wl"]) >= float(best_score["normalized_wl"]) - float(self.config.refine_stop_wirelength_margin)
                )
                if current_not_beating_incumbent or int(self.steps_since_best) > 0 or int(self.stagnation_steps) > 0:
                    stop_logit_bias += float(self.config.refine_stop_bias)
        if self.phase == PlacementPhase.REFINE and float(continuation_risk) > 0.0:
            stop_logit_bias += float(self.config.continuation_stop_bias) * float(continuation_risk)
        return build_graph_state(
            write_positions(self.cell_features, self.centers),
            self.pin_features,
            self.edge_list,
            self.active_pairs,
            self.branch_duals,
            self.boundary_duals,
            self.density_duals,
            self._wirelength_gradient(),
            density_pressure_per_cell(density_g, assignment, self.density_duals),
            memory,
            exact_overlap_ratio=overlap_ratio,
            current_normalized_wl=torch.tensor(
                current_score["normalized_wl"],
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            current_num_overlap_pairs=torch.tensor(
                current_score["num_overlap_pairs"],
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            incumbent_centers=best_centers,
            incumbent_overlap_ratio=torch.tensor(
                best_score["overlap_ratio"],
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            incumbent_normalized_wl=torch.tensor(
                best_score["normalized_wl"],
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            incumbent_num_overlap_pairs=torch.tensor(
                best_score["num_overlap_pairs"],
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            steps_since_best=torch.tensor(
                float(self.steps_since_best),
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            stop_logit_bias=stop_logit_bias,
            phase_name=self.phase,
            phase_step=torch.tensor(
                float(self.phase_step),
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            legal_streak=torch.tensor(
                float(self.legal_streak),
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            stagnation_steps=torch.tensor(
                float(self.stagnation_steps),
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            discover_mode_name=self.discover_mode,
            discover_mode_index=self.discover_mode_index,
            discover_mode_carry_steps=torch.tensor(
                float(self.discover_mode_carry_steps),
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            unlock_remaining_steps=torch.tensor(
                float(self.unlock_remaining_steps),
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            late_legalize_mode=torch.zeros((), dtype=self.centers.dtype, device=self.centers.device),
            phase_entry_overlap=torch.tensor(
                float(self.phase_entry_best_score["overlap_ratio"]),
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            phase_entry_wirelength=torch.tensor(
                float(self.phase_entry_best_score["normalized_wl"]),
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
            ordering_representation=self.config.ordering_representation,
            branch_mode=self.config.branch_mode,
            enable_clusters=self.config.enable_clusters,
            enable_stop=self.config.enable_stop,
            enable_incumbent_state=self.config.enable_incumbent_state,
            enable_incumbent_action=self.config.enable_incumbent_action,
            cleanup_feature_vector=self._cleanup_feature_vector(
                self.centers.detach().clone(),
                phase_name=self.phase,
                incumbent_centers=best_centers.detach().clone(),
            ),
            case_descriptor=self._case_descriptor_vector(self.centers.detach().clone()),
            continuation_risk=torch.tensor(
                float(continuation_risk),
                dtype=self.centers.dtype,
                device=self.centers.device,
            ),
        )

    def auxiliary_graph_for_candidate(self, centers, *, phase_name="REFINE"):
        centers = centers.detach().clone().to(self.centers.device)
        positioned = write_positions(self.cell_features, centers)
        score = self._score_centers(centers)
        continuation_risk = self._continuation_risk(
            current_score=score,
            best_score=score,
            phase_name=phase_name,
            steps_since_best=0,
        )
        active_pairs = build_initial_active_pairs(
            positioned,
            self.pin_features,
            self.edge_list,
            max_pairs=self.config.active_pair_limit,
        )
        branch_duals = torch.zeros((active_pairs.shape[0], 4), dtype=self.cell_features.dtype, device=self.cell_features.device)
        boundary_duals = torch.zeros_like(self.boundary_duals)
        density_g, assignment = self._density_constraints(centers)
        density_duals = torch.zeros_like(density_g)
        return build_graph_state(
            positioned,
            self.pin_features,
            self.edge_list,
            active_pairs,
            branch_duals,
            boundary_duals,
            density_duals,
            self._wirelength_gradient_at(centers),
            density_pressure_per_cell(density_g, assignment, density_duals),
            torch.zeros(0, dtype=self.centers.dtype, device=self.centers.device),
            exact_overlap_ratio=torch.tensor(float(score["overlap_ratio"]), dtype=self.centers.dtype, device=self.centers.device),
            current_normalized_wl=torch.tensor(float(score["normalized_wl"]), dtype=self.centers.dtype, device=self.centers.device),
            current_num_overlap_pairs=torch.tensor(float(score["num_overlap_pairs"]), dtype=self.centers.dtype, device=self.centers.device),
            incumbent_centers=centers,
            incumbent_overlap_ratio=torch.tensor(float(score["overlap_ratio"]), dtype=self.centers.dtype, device=self.centers.device),
            incumbent_normalized_wl=torch.tensor(float(score["normalized_wl"]), dtype=self.centers.dtype, device=self.centers.device),
            incumbent_num_overlap_pairs=torch.tensor(float(score["num_overlap_pairs"]), dtype=self.centers.dtype, device=self.centers.device),
            steps_since_best=torch.tensor(0.0, dtype=self.centers.dtype, device=self.centers.device),
            stop_logit_bias=torch.tensor(0.0, dtype=self.centers.dtype, device=self.centers.device),
            phase_name=str(phase_name),
            phase_step=torch.tensor(1.0, dtype=self.centers.dtype, device=self.centers.device),
            legal_streak=torch.tensor(float(self.legal_streak), dtype=self.centers.dtype, device=self.centers.device),
            stagnation_steps=torch.tensor(float(self.stagnation_steps), dtype=self.centers.dtype, device=self.centers.device),
            discover_mode_name=self.discover_mode,
            discover_mode_index=self.discover_mode_index,
            discover_mode_carry_steps=torch.tensor(0.0, dtype=self.centers.dtype, device=self.centers.device),
            unlock_remaining_steps=torch.tensor(0.0, dtype=self.centers.dtype, device=self.centers.device),
            late_legalize_mode=torch.zeros((), dtype=self.centers.dtype, device=self.centers.device),
            phase_entry_overlap=torch.tensor(float(score["overlap_ratio"]), dtype=self.centers.dtype, device=self.centers.device),
            phase_entry_wirelength=torch.tensor(float(score["normalized_wl"]), dtype=self.centers.dtype, device=self.centers.device),
            ordering_representation=self.config.ordering_representation,
            branch_mode=self.config.branch_mode,
            enable_clusters=self.config.enable_clusters,
            enable_stop=self.config.enable_stop,
            enable_incumbent_state=self.config.enable_incumbent_state,
            enable_incumbent_action=self.config.enable_incumbent_action,
            cleanup_feature_vector=self._cleanup_feature_vector(
                centers,
                phase_name=str(phase_name),
                incumbent_centers=centers,
            ),
            case_descriptor=self._case_descriptor_vector(centers),
            continuation_risk=torch.tensor(float(continuation_risk), dtype=self.centers.dtype, device=self.centers.device),
        )

    def _density_constraints(self, centers):
        if not self.config.enable_density:
            return (
                centers.new_zeros(0),
                centers.new_zeros((centers.shape[0], 0)),
            )
        return density_bin_constraints(
            centers,
            self.cell_features,
            self.bounds,
            bins=self.config.density_bins,
            rho_max=self.config.density_rho_max,
        )

    def step_action(self, action, entropy=None, soft_branch_weights=None, soft_tau=None, trace_transition=False):
        """Apply the full policy-conditioned transition from the proposal."""
        self.last_transition_trace = None
        phase_before = str(self.phase)
        phase_request_name = self._phase_request_name(action)
        entropy = action.entropy if entropy is None else entropy
        use_soft = self.config.soft_relaxation and soft_branch_weights is not None
        active_soft_weights = soft_branch_weights if use_soft else None
        active_soft_tau = float(action.tau.detach().item()) if soft_tau is None else float(soft_tau)
        if self.config.fixed_pd_controls:
            rho = float(self.config.rho)
            eta = 0.02
            alpha = float(self.config.coordinate_lr)
            pd_steps = int(self.config.coordinate_steps)
        else:
            rho = float(action.rho.detach().item())
            eta = float(action.eta.detach().item())
            alpha = float(action.alpha.detach().item())
            pd_steps = int(action.pd_steps)
        pressure = {
            "branch": 1.0 if self.config.fixed_pd_controls else float(action.branch_pressure.detach().item()),
            "density": 1.0 if self.config.fixed_pd_controls else float(action.density_pressure.detach().item()),
            "boundary": 1.0 if self.config.fixed_pd_controls else float(action.boundary_pressure.detach().item()),
        }
        per_constraint_pressure = self._per_constraint_pressure(action)

        branches = self._induce_action_branches(action)
        old_branch_duals = self.branch_duals.detach().clone()
        old_boundary_duals = self.boundary_duals.detach().clone()
        old_density_duals = self.density_duals.detach().clone()

        before_score = self._score_centers(self.centers)
        best_before, best_before_centers = self.best_candidate()
        steps_since_best_before = int(self.steps_since_best)
        continuation_risk_before = self._continuation_risk(
            current_score=before_score,
            best_score=best_before,
            phase_name=phase_before,
            steps_since_best=steps_since_best_before,
        )
        adaptive_pd_extra_steps, adaptive_pd_case_descriptor_bucket, adaptive_pd_applied = self._adaptive_pd_step_adjustment(
            phase_name=phase_before,
            current_score=before_score,
            best_score=best_before,
        )
        trace_payload = None
        phr_trace_steps = None
        if trace_transition:
            trace_payload = {
                "phase_before": phase_before,
                "phase_request": phase_request_name,
                "pre_centers": self.centers.detach().clone(),
                "pre_active_pairs": self.active_pairs.detach().clone(),
                "pre_active_pair_ages": self.active_pair_ages.detach().clone(),
                "pre_branch_duals": self.branch_duals.detach().clone(),
                "pre_boundary_duals": self.boundary_duals.detach().clone(),
                "pre_density_duals": self.density_duals.detach().clone(),
                "rho": rho,
                "eta": eta,
                "alpha": alpha,
                "pd_steps": pd_steps,
                "continuation_risk_before": float(continuation_risk_before),
                "adaptive_pd_extra_steps": int(adaptive_pd_extra_steps),
                "pressure": dict(pressure),
                "branch_pressure_values": None
                if per_constraint_pressure["branch"] is None
                else per_constraint_pressure["branch"].detach().clone(),
                "boundary_pressure_values": None
                if per_constraint_pressure["boundary"] is None
                else per_constraint_pressure["boundary"].detach().clone(),
                "density_pressure_values": None
                if per_constraint_pressure["density"] is None
                else per_constraint_pressure["density"].detach().clone(),
            }
            phr_trace_steps = []
        if adaptive_pd_applied:
            pd_steps = int(max(pd_steps, int(self.config.coordinate_steps) + int(adaptive_pd_extra_steps)))
        step_scale = float(action.step_scale.detach().item())
        incumbent_mix = 0.0
        if self.config.enable_incumbent_action and hasattr(action, "incumbent_mix"):
            incumbent_mix = float(action.incumbent_mix.detach().item())
        incumbent_mix = max(0.0, min(incumbent_mix, float(self.config.incumbent_blend_max)))
        if phase_before == PlacementPhase.DISCOVER:
            if self.discover_mode == "spread_first":
                incumbent_mix = min(incumbent_mix, 0.05)
            elif self.discover_mode == "wire_first":
                incumbent_mix = min(incumbent_mix, 0.10)
            elif self.discover_mode == "macro_clearance":
                incumbent_mix = min(incumbent_mix, 0.20)
        incumbent_anchor = self.centers
        if self.config.enable_incumbent_action and incumbent_mix > 0.0:
            incumbent_anchor = torch.lerp(self.centers, best_before_centers.to(self.centers.device), incumbent_mix)
        residual = torch.tanh(action.residual_flow.detach()) * (step_scale * self.length_scale)
        if not self.config.enable_residual_flow:
            residual = torch.zeros_like(residual)
        unlock_window_cells = torch.empty((0,), dtype=torch.long, device=self.centers.device)
        refine_compaction_window = torch.empty((0,), dtype=torch.long, device=self.centers.device)
        refine_compaction_grad = None
        if phase_before == PlacementPhase.UNLOCK:
            unlock_window_cells = self._select_unlock_window(action)
            unlock_mask = torch.zeros((self.centers.shape[0],), dtype=torch.bool, device=self.centers.device)
            if unlock_window_cells.numel() > 0:
                unlock_mask[unlock_window_cells.long()] = True
            outside_anchor = self.unlock_anchor_centers.to(self.centers.device).detach().clone()
            residual = residual * unlock_mask.unsqueeze(1).to(residual.dtype)
            base_centers = outside_anchor.clone()
            base_centers[unlock_mask] = outside_anchor[unlock_mask] + residual[unlock_mask]
            self.unlock_window_cells = unlock_window_cells.detach().clone()
        else:
            self.unlock_window_cells = torch.empty((0,), dtype=torch.long, device=self.centers.device)
            base_centers = incumbent_anchor + residual
        if trace_payload is not None:
            trace_payload["branches_pre"] = branches.detach().clone()
            trace_payload["soft_branch_weights_pre"] = None if active_soft_weights is None else active_soft_weights.detach().clone()
            trace_payload["best_before_centers"] = best_before_centers.detach().clone()
            trace_payload["incumbent_anchor"] = incumbent_anchor.detach().clone()
            trace_payload["incumbent_mix"] = float(incumbent_mix)
            trace_payload["discover_mode"] = str(self.discover_mode)
            trace_payload["base_centers"] = base_centers.detach().clone()
            trace_payload["unlock_window_cells"] = unlock_window_cells.detach().clone()
            trace_payload["refine_compaction_window"] = refine_compaction_window.detach().clone()
            trace_payload["refine_compaction_grad"] = None if refine_compaction_grad is None else refine_compaction_grad.detach().clone()

        lag_before, before_logs = self._augmented_lagrangian(
            self.centers,
            branches,
            old_branch_duals,
            old_boundary_duals,
            old_density_duals,
            base_centers.detach(),
            rho=rho,
            eta=eta,
            pressure=pressure,
            branch_pressure_values=per_constraint_pressure["branch"],
            boundary_pressure_values=per_constraint_pressure["boundary"],
            density_pressure_values=per_constraint_pressure["density"],
            soft_branch_weights=active_soft_weights,
            soft_tau=active_soft_tau,
        )
        if self.config.enable_phr_layer:
            next_centers = self._coordinate_layer(
                branches,
                old_branch_duals,
                old_boundary_duals,
                old_density_duals,
                initial_centers=base_centers,
                anchor_centers=base_centers.detach(),
                coordinate_steps=pd_steps,
                coordinate_lr=alpha,
                rho=rho,
                eta=eta,
                pressure=pressure,
                branch_pressure_values=per_constraint_pressure["branch"],
                boundary_pressure_values=per_constraint_pressure["boundary"],
                density_pressure_values=per_constraint_pressure["density"],
                soft_branch_weights=active_soft_weights,
                soft_tau=active_soft_tau,
                trace_steps=phr_trace_steps,
            )
        else:
            next_centers = base_centers.detach()
        lag_after, after_logs = self._augmented_lagrangian(
            next_centers,
            branches,
            old_branch_duals,
            old_boundary_duals,
            old_density_duals,
            base_centers.detach(),
            rho=rho,
            eta=eta,
            pressure=pressure,
            branch_pressure_values=per_constraint_pressure["branch"],
            boundary_pressure_values=per_constraint_pressure["boundary"],
            density_pressure_values=per_constraint_pressure["density"],
            soft_branch_weights=active_soft_weights,
            soft_tau=active_soft_tau,
        )

        candidate_centers = next_centers.detach()
        candidate_lag_after = lag_after
        candidate_after_logs = dict(after_logs)
        candidate_score = self._score_centers(candidate_centers)
        winning_refine_variant = "incumbent_hold"
        refine_variant_rows = []
        refine_window = torch.empty((0,), dtype=torch.long, device=self.centers.device)
        refine_window_size = 0
        best_variant_before_refine = dict(best_before)
        if phase_before == PlacementPhase.REFINE:
            incumbent_anchor = best_before_centers.to(self.centers.device).detach().clone()
            refine_result = self._run_refine_variant_portfolio(
                policy=None,
                action=action,
                incumbent_centers=incumbent_anchor,
                current_centers=self.centers.detach().clone(),
                residual=residual,
                step_scale=step_scale,
                branches=branches,
                branch_duals=old_branch_duals,
                boundary_duals=old_boundary_duals,
                density_duals=old_density_duals,
                rho=rho,
                eta=eta,
                alpha=alpha,
                pd_steps=pd_steps,
                pressure=pressure,
                branch_pressure_values=per_constraint_pressure["branch"],
                boundary_pressure_values=per_constraint_pressure["boundary"],
                density_pressure_values=per_constraint_pressure["density"],
                soft_branch_weights=active_soft_weights,
                soft_tau=active_soft_tau,
            )
            candidate_centers = refine_result["centers"].detach().clone()
            candidate_score = dict(refine_result["score"])
            candidate_lag_after = refine_result["lag_after"]
            candidate_after_logs = dict(refine_result["after_logs"])
            winning_refine_variant = str(refine_result["winning_variant"])
            refine_variant_rows = list(refine_result["variant_rows"])
            refine_window = refine_result["window"].detach().clone()
            refine_window_size = int(refine_window.numel())
            next_centers = candidate_centers
            lag_after = candidate_lag_after
            after_logs = dict(candidate_after_logs)
        elif phase_before == PlacementPhase.REFINE and self.config.refine_use_compaction_operator:
            refine_window_size = int(refine_compaction_window.numel())
        refine_rejected = False
        refine_reject_reason = ""
        if phase_before == PlacementPhase.REFINE:
            accept_refine, refine_reject_reason = self._accept_refine_candidate(candidate_score, best_before)
            if not accept_refine:
                refine_rejected = True
                next_centers = best_before_centers.detach().clone()
                lag_after, after_logs = self._augmented_lagrangian(
                    next_centers,
                    branches,
                    old_branch_duals,
                    old_boundary_duals,
                    old_density_duals,
                    base_centers.detach(),
                    rho=rho,
                    eta=eta,
                    pressure=pressure,
                    branch_pressure_values=per_constraint_pressure["branch"],
                    boundary_pressure_values=per_constraint_pressure["boundary"],
                    density_pressure_values=per_constraint_pressure["density"],
                    soft_branch_weights=active_soft_weights,
                    soft_tau=active_soft_tau,
                )
                after_score = self._score_centers(next_centers)
            else:
                after_score = candidate_score
        else:
            after_score = candidate_score
        movement_penalty = self.config.movement_penalty_coef * (
            (next_centers - self.centers).square().mean() / torch.clamp(self.length_scale.square(), min=1.0)
        ).detach().item()
        entropy_term = 0.0 if entropy is None else float(entropy.detach().item())

        self.centers = next_centers.detach()
        if self.config.enable_phr_layer:
            self._update_duals(
                branches,
                old_branch_duals,
                old_boundary_duals,
                old_density_duals,
                rho=rho,
                pressure=pressure,
                branch_pressure_values=per_constraint_pressure["branch"],
                boundary_pressure_values=per_constraint_pressure["boundary"],
                density_pressure_values=per_constraint_pressure["density"],
                soft_branch_weights=active_soft_weights,
                soft_tau=active_soft_tau,
            )
        if trace_payload is not None:
            trace_payload["candidate_post_centers"] = candidate_centers.detach().clone()
            trace_payload["candidate_lag_after"] = float(candidate_lag_after.detach().item())
            trace_payload["candidate_after_logs"] = dict(candidate_after_logs)
            trace_payload["candidate_score"] = dict(candidate_score)
            trace_payload["refine_rejected"] = bool(refine_rejected)
            trace_payload["refine_reject_reason"] = refine_reject_reason
            trace_payload["winning_refine_variant"] = winning_refine_variant
            trace_payload["refine_variant_rows"] = [dict(row) for row in refine_variant_rows]
            trace_payload["refine_window"] = refine_window.detach().clone()
            trace_payload["post_centers"] = self.centers.detach().clone()
            trace_payload["phr_steps"] = phr_trace_steps or []
            trace_payload["post_update_branch_duals"] = self.branch_duals.detach().clone()
            trace_payload["post_update_boundary_duals"] = self.boundary_duals.detach().clone()
            trace_payload["post_update_density_duals"] = self.density_duals.detach().clone()
        audit_info = {
            "missed_pairs": 0,
            "inactive_missed_pairs": 0,
            "exact_overlap_pairs": 0,
            "sampled_pairs": 0,
            "cluster_pairs": 0,
            "uncertain_pairs": 0,
            "new_active_pairs": 0,
            "retained_pairs": int(self.active_pairs.shape[0]),
            "hard_pair_age_mean": self._age_mean(self.active_pair_ages),
            "hard_pair_age_max": self._age_max(self.active_pair_ages),
            "hard_pair_age_min": self._age_min(self.active_pair_ages),
            "audit_pressure_scale": 1.0,
            "audit_pressure_target": self._audit_pressure_target(),
            "retention_horizon": int(self.config.active_pair_retention),
        }
        if self.config.enable_exact_audit:
            audit_info = self._audit_active_set(
                action=action,
                cluster_ids=action.cluster_ids,
                pair_emphasis=float(action.pair_emphasis.detach().item()),
            )
        score = self._save_candidate(
            self.centers,
            candidate_step=self.step_index + 1,
            candidate_meta={
                "phase": phase_before,
                "discover_mode": self.discover_mode,
                "winning_refine_variant": winning_refine_variant,
            },
        )
        best_after, best_after_centers = self.best_candidate()
        incumbent_improved = self._candidate_rank_key(best_after) < self._candidate_rank_key(best_before)
        continuation_risk_after = self._continuation_risk(
            current_score=score,
            best_score=best_after,
            phase_name=phase_before,
            steps_since_best=self.steps_since_best,
        )
        rollback_to_incumbent = False
        rollback_reason = ""
        rollback_penalty = 0.0
        unlock_accepted = False
        unlock_reverted = False
        unlock_revert_reason = ""
        reward_score = dict(score)
        if phase_before == PlacementPhase.UNLOCK:
            unlock_accepted = bool(incumbent_improved)
            if not unlock_accepted:
                unlock_reverted = True
                unlock_revert_reason = "unlock_failed_to_improve_incumbent"
                if self._candidate_rank_key(best_after) <= self._candidate_rank_key(self.unlock_anchor_score):
                    self.centers = best_after_centers.detach().clone()
                    score = dict(best_after)
                else:
                    self.centers = self.unlock_anchor_centers.detach().clone()
                    score = dict(self.unlock_anchor_score)
                reward_score = dict(score)
        if phase_before == PlacementPhase.REFINE and self._should_rollback_refine_to_incumbent(
            current_score=score,
            incumbent_score=best_after,
            incumbent_improved=incumbent_improved,
            continuation_risk=continuation_risk_after,
        ):
            rollback_to_incumbent = True
            rollback_reason = "refine_non_improving_post_legal"
            self.centers = best_after_centers.detach().clone()
            score = dict(best_after)
            rollback_penalty = 0.5 * float(self.config.stop_no_progress_penalty)
        reward, reward_terms = self._aligned_reward(
            phase_before,
            lag_before,
            lag_after,
            before_score,
            reward_score,
            best_before,
            best_after,
            best_after_centers,
            after_logs,
            audit_info,
            entropy_term,
            movement_penalty,
        )
        refine_rejection_penalty = 0.0
        if refine_rejected:
            refine_rejection_penalty = max(
                float(self.config.stop_no_progress_penalty),
                0.25 * float(self.config.exact_overlap_regression_coef),
            )
            reward -= refine_rejection_penalty
        if rollback_to_incumbent:
            reward -= rollback_penalty

        legal_now = self._is_refine_ready(best_after)
        self.legal_streak = self.legal_streak + 1 if legal_now else 0
        self.stagnation_steps = 0 if incumbent_improved else self.stagnation_steps + 1
        if phase_before == PlacementPhase.LEGALIZE:
            self.legalize_stall_steps = 0 if incumbent_improved else self.legalize_stall_steps + 1
            pair_improved = int(best_after["num_overlap_pairs"]) < int(best_before["num_overlap_pairs"])
            self.legalize_pair_stall_steps = 0 if pair_improved else self.legalize_pair_stall_steps + 1
        else:
            self.legalize_stall_steps = 0
            self.legalize_pair_stall_steps = 0
        phase_transition, phase_transition_reason = self._advance_phase(
            action=action,
            phase_before=phase_before,
            phase_request_name=phase_request_name,
            current_score=score,
            best_before=best_before,
            best_after=best_after,
        )
        if phase_transition and self.phase == PlacementPhase.REFINE:
            self.centers = best_after_centers.detach().clone()
            score = dict(best_after)

        self.step_index += 1
        stop_requested = bool(getattr(action, "stop", torch.zeros((), device=self.centers.device)).detach().item() >= 0.5)
        stop_eligible = phase_before == PlacementPhase.REFINE and self._is_refine_ready(best_after)
        auto_stop = False
        stop_reason = ""
        if self.config.enable_stop and phase_before == PlacementPhase.REFINE:
            auto_stop = self._should_auto_stop_refine(
                current_score=score,
                incumbent_score=best_after,
                incumbent_improved=incumbent_improved,
                steps_since_best_before=steps_since_best_before,
                refine_rejected=refine_rejected,
                continuation_risk=continuation_risk_after,
            )
        if self.config.enable_stop:
            stopped = bool((stop_requested and stop_eligible) or auto_stop)
            stop_probability = float(getattr(action, "stop_probability", action.stop).detach().item())
            stop_logit_bias = float(getattr(action, "stop_logit_bias", action.stop.new_zeros(())).detach().item())
            if auto_stop:
                stop_reason = "refine_auto_stop_preserve_incumbent"
            elif stopped:
                stop_reason = "policy_stop"
        else:
            stopped = False
            stop_probability = 0.0
            stop_logit_bias = 0.0
        stop_overlap = (best_after["overlap_ratio"] if auto_stop else score["overlap_ratio"]) if stopped else 0.0
        false_stop = bool(stop_requested and not stop_eligible)
        done = stopped or self.step_index >= self.config.horizon
        if self.config.enable_stop and stopped:
            stop_score = best_after if auto_stop else score
            reward += self._stop_reward(stop_score, before_score)
        elif done:
            reward += self.terminal_reward()

        phase_after = str(self.phase)
        phase_reset_retain = self._phase_reset_retain(phase_before, phase_after)

        info = {
            "reward": reward,
            "phase": phase_after,
            "phase_before": phase_before,
            "phase_request": phase_request_name,
            "phase_transition": bool(phase_transition),
            "phase_transition_reason": phase_transition_reason,
            "phase_step": int(self.phase_step),
            "phase_reset_retain": float(phase_reset_retain),
            "discover_mode": str(self.discover_mode),
            "discover_mode_index": int(self.discover_mode_index),
            "discover_mode_carry_steps": int(self.discover_mode_carry_steps),
            "legal_streak": int(self.legal_streak),
            "stagnation_steps": int(self.stagnation_steps),
            "legalize_stall_steps": int(self.legalize_stall_steps),
            "unlock_remaining_steps": int(self.unlock_remaining_steps),
            "unlock_entered": bool(phase_transition and self.phase == PlacementPhase.UNLOCK),
            "unlock_accepted": bool(unlock_accepted),
            "unlock_reverted": bool(unlock_reverted),
            "unlock_revert_reason": unlock_revert_reason,
            "unlock_window_size": int(self.unlock_window_cells.numel()),
            "refine_compaction_window_size": int(refine_compaction_window.numel()),
            "refine_window_size": int(refine_window_size),
            "winning_refine_variant": str(winning_refine_variant),
            "refine_variant_rows": [dict(row) for row in refine_variant_rows],
            "best_candidate_phase": str(self.best_candidate_phase),
            "best_candidate_discover_mode": str(self.best_candidate_discover_mode),
            "best_candidate_refine_variant": str(self.best_candidate_refine_variant),
            "phase_entry_best_overlap": float(self.phase_entry_best_score["overlap_ratio"]),
            "phase_entry_best_wirelength": float(self.phase_entry_best_score["normalized_wl"]),
            "phase_entry_best_pairs": int(self.phase_entry_best_score["num_overlap_pairs"]),
            "lag_before": float(lag_before.detach().item()),
            "lag_after": float(lag_after.detach().item()),
            "wirelength": after_logs["wirelength"],
            "branch_violation": after_logs["branch_violation"],
            "boundary_violation": after_logs["boundary_violation"],
            "density_overflow": after_logs["density_overflow"],
            "overlap_ratio": score["overlap_ratio"],
            "normalized_wl": score["normalized_wl"],
            "active_pairs": int(self.active_pairs.shape[0]),
            "soft_relaxation": active_soft_weights is not None,
            "pd_steps": pd_steps,
            "rho": rho,
            "eta": eta,
            "alpha": alpha,
            "step_scale": step_scale,
            "incumbent_mix": incumbent_mix,
            "incumbent_improved": bool(incumbent_improved),
            "refine_rejected": bool(refine_rejected),
            "refine_reject_reason": refine_reject_reason,
            "rollback_to_incumbent": bool(rollback_to_incumbent),
            "rollback_reason": rollback_reason,
            "candidate_overlap_ratio": float(candidate_score["overlap_ratio"]),
            "candidate_num_overlap_pairs": int(candidate_score["num_overlap_pairs"]),
            "candidate_normalized_wl": float(candidate_score["normalized_wl"]),
            "refine_rejection_penalty": float(refine_rejection_penalty),
            "rollback_penalty": float(rollback_penalty),
            "steps_since_best_before": steps_since_best_before,
            "steps_since_best": int(self.steps_since_best),
            "best_candidate_step": int(self.best_candidate_step),
            "continuation_risk": float(continuation_risk_after),
            "best_so_far_returned": bool(self._candidate_rank_key(score) == self._candidate_rank_key(best_after)),
            "pair_emphasis": float(action.pair_emphasis.detach().item()),
            "tau": active_soft_tau,
            "stop": stopped,
            "stop_probability": stop_probability,
            "stop_logit_bias": stop_logit_bias,
            "stop_gated": stop_logit_bias < 0.0,
            "stop_overlap": stop_overlap,
            "false_stop": false_stop,
            "stop_requested": bool(stop_requested),
            "stop_eligible": bool(stop_eligible),
            "auto_stop": bool(auto_stop),
            "stop_reason": stop_reason,
            "residual_norm": float(residual.norm(dim=1).mean().detach().item()),
            **reward_terms,
            "movement_penalty": movement_penalty,
            "branch_pressure": pressure["branch"],
            "density_pressure": pressure["density"],
            "boundary_pressure": pressure["boundary"],
            "branch_pressure_mean": self._tensor_mean(per_constraint_pressure["branch"]),
            "boundary_pressure_mean": self._tensor_mean(per_constraint_pressure["boundary"]),
            "density_pressure_mean": self._tensor_mean(per_constraint_pressure["density"]),
            "dual_clamp_fraction": getattr(self, "last_dual_clamp_fraction", 0.0),
            "adaptive_pd_steps_applied": bool(adaptive_pd_applied),
            "adaptive_pd_steps_stage": str(phase_before) if adaptive_pd_applied else "",
            "adaptive_pd_steps_case_descriptor_bucket": str(adaptive_pd_case_descriptor_bucket),
            "adaptive_pd_extra_steps": int(adaptive_pd_extra_steps),
            **audit_info,
        }
        if trace_payload is not None:
            trace_payload["audit_info"] = dict(audit_info)
            trace_payload["post_audit_active_pairs"] = self.active_pairs.detach().clone()
            trace_payload["post_audit_active_pair_ages"] = self.active_pair_ages.detach().clone()
            trace_payload["post_audit_branch_duals"] = self.branch_duals.detach().clone()
            trace_payload["post_audit_boundary_duals"] = self.boundary_duals.detach().clone()
            trace_payload["post_audit_density_duals"] = self.density_duals.detach().clone()
            trace_payload["post_audit_branches"] = self._induce_action_branches(action).detach().clone()
            trace_payload["rollback_to_incumbent"] = bool(rollback_to_incumbent)
            trace_payload["rollback_reason"] = rollback_reason
            trace_payload["post_commit_centers"] = self.centers.detach().clone()
            trace_payload["reward"] = float(reward)
            trace_payload["done"] = bool(done)
            trace_payload["info"] = dict(info)
            trace_payload["phase_after"] = phase_after
            trace_payload["phase_transition_reason"] = phase_transition_reason
            trace_payload["auto_stop"] = bool(auto_stop)
            trace_payload["stop_reason"] = stop_reason
            self.last_transition_trace = trace_payload
        return reward, done, info

    def _aligned_reward(
        self,
        phase_name,
        lag_before,
        lag_after,
        before_score,
        current_score,
        best_before,
        best_after,
        best_after_centers,
        after_logs,
        audit_info,
        entropy_term,
        movement_penalty,
    ):
        phase_name = str(phase_name).upper()
        lag_delta = (lag_before.detach() - lag_after.detach()).item()
        lag_signal = math.tanh(lag_delta / max(float(self.config.lag_reward_tanh_scale), 1e-6))
        best_overlap_delta = best_before["overlap_ratio"] - best_after["overlap_ratio"]
        best_overlap_regression = max(-best_overlap_delta, 0.0)
        best_wirelength_delta = best_before["normalized_wl"] - best_after["normalized_wl"]
        gated_best_wire_delta, wirelength_gate_active = gated_wirelength_delta(
            best_overlap_delta=best_overlap_delta,
            best_wirelength_delta=best_wirelength_delta,
            best_after_overlap=best_after["overlap_ratio"],
            best_after_pairs=best_after["num_overlap_pairs"],
            gate_epsilon=self.config.exact_wirelength_gate_epsilon,
            overlap_threshold=self.config.exact_wirelength_reward_overlap_threshold,
            pairs_threshold=self.config.exact_wirelength_reward_pairs_threshold,
        )
        cell_count = max(int(self.centers.shape[0]), 1)
        best_pair_delta = (
            best_before["num_overlap_pairs"] - best_after["num_overlap_pairs"]
        ) / float(cell_count)
        best_pair_delta_count = (
            best_before["num_overlap_pairs"] - best_after["num_overlap_pairs"]
        )
        current_overlap_penalty = self.config.current_overlap_penalty_coef * current_score["overlap_ratio"]
        current_pair_penalty = self.config.current_overlap_pairs_penalty_coef * (
            current_score["num_overlap_pairs"] / float(cell_count)
        )
        incumbent_overlap_gap = max(current_score["overlap_ratio"] - best_after["overlap_ratio"], 0.0)
        incumbent_pair_gap = max(
            current_score["num_overlap_pairs"] - best_after["num_overlap_pairs"],
            0,
        ) / float(cell_count)
        incumbent_position_gap = (
            (self.centers.detach() - best_after_centers.detach()).square().mean()
            / torch.clamp(self.length_scale.square(), min=1.0)
        ).item()
        incumbent_gap_penalty = (
            self.config.incumbent_overlap_gap_penalty_coef * incumbent_overlap_gap
            + self.config.incumbent_pair_gap_penalty_coef * incumbent_pair_gap
            + self.config.incumbent_position_gap_penalty_coef * incumbent_position_gap
        )
        violation_penalty = self.config.branch_violation_penalty_coef * after_logs["branch_violation"]
        missed_pair_penalty = self.config.missed_pair_penalty_coef * float(audit_info.get("missed_pairs", 0))
        benchmark_reward = 0.0
        refine_regressed = (
            current_score["overlap_ratio"] > best_before["overlap_ratio"] + float(self.config.refine_regression_overlap)
            or current_score["num_overlap_pairs"] > best_before["num_overlap_pairs"] + int(self.config.refine_regression_pairs)
        )
        signed_current_wire_delta = before_score["normalized_wl"] - current_score["normalized_wl"]
        phase_wire_reward = 0.0
        if phase_name == PlacementPhase.DISCOVER:
            benchmark_reward = (
                self.config.exact_overlap_reward_coef * best_overlap_delta
                - self.config.exact_overlap_regression_coef * best_overlap_regression
                + self.config.exact_overlap_pairs_reward_coef * best_pair_delta
            )
        elif phase_name == PlacementPhase.LEGALIZE:
            benchmark_reward = (
                1.25 * self.config.exact_overlap_reward_coef * best_overlap_delta
                - 1.25 * self.config.exact_overlap_regression_coef * best_overlap_regression
                + 1.25 * self.config.exact_overlap_pairs_reward_coef * best_pair_delta
            )
            phase_wire_reward = 0.50 * self.config.exact_wirelength_reward_coef * gated_best_wire_delta
            benchmark_reward += phase_wire_reward
            if current_score["overlap_ratio"] > best_after["overlap_ratio"] + float(self.config.refine_regression_overlap):
                benchmark_reward = min(benchmark_reward, 0.0) - self.config.exact_overlap_regression_coef * (
                    current_score["overlap_ratio"] - best_after["overlap_ratio"]
                )
        elif phase_name == PlacementPhase.REFINE:
            if refine_regressed:
                benchmark_reward = -self.config.exact_overlap_regression_coef * (
                    max(current_score["overlap_ratio"] - best_before["overlap_ratio"], 0.0)
                    + max(
                        current_score["num_overlap_pairs"] - best_before["num_overlap_pairs"],
                        0,
                    )
                    / float(cell_count)
                )
            else:
                phase_wire_reward = self.config.exact_wirelength_reward_coef * signed_current_wire_delta
                benchmark_reward = (
                    phase_wire_reward
                    + 0.50 * self.config.exact_overlap_reward_coef * best_overlap_delta
                    + 0.50 * self.config.exact_overlap_pairs_reward_coef * best_pair_delta
                )
        elif phase_name == PlacementPhase.UNLOCK:
            phase_wire_reward = 0.50 * self.config.exact_wirelength_reward_coef * gated_best_wire_delta
            benchmark_reward = (
                self.config.exact_overlap_reward_coef * best_overlap_delta
                + self.config.exact_overlap_pairs_reward_coef * best_pair_delta
                + phase_wire_reward
            )
        reward = benchmark_reward
        if phase_name in {PlacementPhase.DISCOVER, PlacementPhase.LEGALIZE}:
            reward += self.config.lag_reward_coef * lag_signal
        elif phase_name == PlacementPhase.REFINE and not refine_regressed:
            reward += 0.25 * self.config.lag_reward_coef * lag_signal
        reward += self.config.entropy_reward_coef * entropy_term
        reward -= movement_penalty
        reward -= current_overlap_penalty
        reward -= current_pair_penalty
        reward -= incumbent_gap_penalty
        reward -= violation_penalty
        reward -= missed_pair_penalty
        return reward, {
            "phase_reward_mode": phase_name,
            "lag_reward": self.config.lag_reward_coef * lag_signal,
            "lag_signal": lag_signal,
            "benchmark_reward": benchmark_reward,
            "best_overlap_delta": best_overlap_delta,
            "best_wirelength_delta": best_wirelength_delta,
            "best_pair_delta": best_pair_delta,
            "best_pair_delta_count": best_pair_delta_count,
            "gated_best_wirelength_delta": gated_best_wire_delta,
            "phase_wire_reward": phase_wire_reward,
            "wirelength_gate_active": bool(wirelength_gate_active),
            "wirelength_gate_overlap_threshold": float(self.config.exact_wirelength_reward_overlap_threshold),
            "wirelength_gate_pairs_threshold": float(self.config.exact_wirelength_reward_pairs_threshold),
            "best_overlap_regression_penalty": self.config.exact_overlap_regression_coef * best_overlap_regression,
            "refine_regressed": bool(refine_regressed),
            "current_overlap_penalty": current_overlap_penalty,
            "current_pair_penalty": current_pair_penalty,
            "incumbent_overlap_gap_penalty": self.config.incumbent_overlap_gap_penalty_coef * incumbent_overlap_gap,
            "incumbent_pair_gap_penalty": self.config.incumbent_pair_gap_penalty_coef * incumbent_pair_gap,
            "incumbent_position_gap_penalty": self.config.incumbent_position_gap_penalty_coef * incumbent_position_gap,
            "branch_violation_penalty": violation_penalty,
            "missed_pair_penalty": missed_pair_penalty,
        }

    def _per_constraint_pressure(self, action):
        if self.config.fixed_pd_controls:
            return {"branch": None, "boundary": None, "density": None}

        branch_values = getattr(action, "branch_pressure_values", None)
        if branch_values is None or branch_values.numel() != self.active_pairs.shape[0]:
            branch_values = None
        else:
            branch_values = branch_values.detach().to(device=self.centers.device, dtype=self.centers.dtype).reshape(-1)

        boundary_values = getattr(action, "boundary_pressure_values", None)
        if boundary_values is None or boundary_values.numel() != self.boundary_duals.numel():
            boundary_values = None
        else:
            boundary_values = boundary_values.detach().to(device=self.centers.device, dtype=self.centers.dtype).reshape_as(self.boundary_duals)

        density_values = getattr(action, "density_pressure_values", None)
        if density_values is None or density_values.numel() != self.density_duals.numel():
            density_values = None
        else:
            density_values = density_values.detach().to(device=self.centers.device, dtype=self.centers.dtype).reshape_as(self.density_duals)

        return {"branch": branch_values, "boundary": boundary_values, "density": density_values}

    @staticmethod
    def _tensor_mean(value):
        if value is None or value.numel() == 0:
            return 0.0
        return float(value.detach().mean().item())

    def _effective_rho(self, rho, group_pressure=1.0, pressure_values=None, like=None):
        if like is None:
            return float(max(min(rho * group_pressure, self.config.rho_max), self.config.rho_min))
        rho_e = torch.full_like(like, float(rho) * float(group_pressure))
        if pressure_values is not None and pressure_values.numel() == like.numel():
            rho_e = rho_e * pressure_values.reshape_as(like).to(dtype=like.dtype, device=like.device)
        return torch.clamp(rho_e, min=float(self.config.rho_min), max=float(self.config.rho_max))

    def step(self, seq_plus, seq_minus, entropy=None, soft_branch_weights=None, soft_tau=None):
        branches = induce_branches_from_sequence_pair(seq_plus, seq_minus, self.active_pairs)
        old_branch_duals = self.branch_duals.detach().clone()
        old_boundary_duals = self.boundary_duals.detach().clone()
        old_density_duals = self.density_duals.detach().clone()
        active_soft_weights = soft_branch_weights if self.config.soft_relaxation else None
        active_soft_tau = self.config.soft_tau if soft_tau is None else soft_tau

        lag_before, before_logs = self._augmented_lagrangian(
            self.centers,
            branches,
            old_branch_duals,
            old_boundary_duals,
            old_density_duals,
            self.centers.detach(),
            soft_branch_weights=active_soft_weights,
            soft_tau=active_soft_tau,
        )
        next_centers = self._coordinate_layer(
            branches,
            old_branch_duals,
            old_boundary_duals,
            old_density_duals,
            soft_branch_weights=active_soft_weights,
            soft_tau=active_soft_tau,
        )
        lag_after, after_logs = self._augmented_lagrangian(
            next_centers,
            branches,
            old_branch_duals,
            old_boundary_duals,
            old_density_duals,
            self.centers.detach(),
            soft_branch_weights=active_soft_weights,
            soft_tau=active_soft_tau,
        )

        entropy_term = 0.0 if entropy is None else float(entropy.detach().item())
        reward = (lag_before.detach() - lag_after.detach()).item() + self.config.entropy_reward_coef * entropy_term

        self.centers = next_centers.detach()
        self._update_duals(
            branches,
            old_branch_duals,
            old_boundary_duals,
            old_density_duals,
            soft_branch_weights=active_soft_weights,
            soft_tau=active_soft_tau,
        )
        self._audit_active_set()
        score = self._save_candidate(self.centers)

        self.step_index += 1
        done = self.step_index >= self.config.horizon
        if done:
            reward += self.terminal_reward()

        info = {
            "reward": reward,
            "lag_before": float(lag_before.detach().item()),
            "lag_after": float(lag_after.detach().item()),
            "wirelength": after_logs["wirelength"],
            "branch_violation": after_logs["branch_violation"],
            "boundary_violation": after_logs["boundary_violation"],
            "density_overflow": after_logs["density_overflow"],
            "overlap_ratio": score["overlap_ratio"],
            "normalized_wl": score["normalized_wl"],
            "active_pairs": int(self.active_pairs.shape[0]),
            "soft_relaxation": active_soft_weights is not None,
        }
        return reward, done, info

    def _induce_action_branches(self, action):
        if self.active_pairs.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.centers.device)
        phase_name = str(self.phase).upper()
        if phase_name in {PlacementPhase.LEGALIZE, PlacementPhase.REFINE}:
            _best_score, best_centers = self.best_candidate()
            seq_plus, seq_minus = sequence_pair_from_centers(best_centers.to(device=self.centers.device))
            return induce_branches_from_sequence_pair(seq_plus, seq_minus, self.active_pairs)
        if self.config.branch_mode == "independent_pair" and hasattr(action, "pair_branch_choices"):
            return action.pair_branch_choices.to(dtype=torch.long, device=self.centers.device)
        if self.config.ordering_representation == "dag" and hasattr(action, "dag_axis"):
            branches = self._induce_dag_branches(action)
        else:
            branches = induce_branches_from_sequence_pair(action.seq_plus, action.seq_minus, self.active_pairs)

        branches = self._apply_macro_branches(action, branches)
        if not self.config.enable_clusters:
            return branches
        if (
            not hasattr(action, "cluster_ids")
            or not hasattr(action, "cluster_seq_plus")
            or not hasattr(action, "cluster_seq_minus")
        ):
            return branches

        cluster_ids = action.cluster_ids
        ci = cluster_ids[self.active_pairs[:, 0].long()]
        cj = cluster_ids[self.active_pairs[:, 1].long()]
        use_cluster = (ci >= 0) & (cj >= 0) & (ci != cj)
        if not torch.any(use_cluster):
            return branches

        k = int(action.cluster_seq_plus.numel())
        rank_plus = torch.empty(k, dtype=torch.long, device=branches.device)
        rank_minus = torch.empty(k, dtype=torch.long, device=branches.device)
        rank_plus[action.cluster_seq_plus.long()] = torch.arange(k, dtype=torch.long, device=branches.device)
        rank_minus[action.cluster_seq_minus.long()] = torch.arange(k, dtype=torch.long, device=branches.device)
        rpi = rank_plus[ci[use_cluster].long()]
        rpj = rank_plus[cj[use_cluster].long()]
        rmi = rank_minus[ci[use_cluster].long()]
        rmj = rank_minus[cj[use_cluster].long()]

        cluster_branches = torch.empty_like(branches[use_cluster])
        left = (rpi < rpj) & (rmi < rmj)
        right = (rpi > rpj) & (rmi > rmj)
        below = (rpi < rpj) & (rmi > rmj)
        cluster_branches[left] = int(Branch.L)
        cluster_branches[right] = int(Branch.R)
        cluster_branches[below] = int(Branch.B)
        cluster_branches[~(left | right | below)] = int(Branch.A)
        branches = branches.clone()
        branches[use_cluster] = cluster_branches
        return branches

    def _induce_dag_branches(self, action):
        i = self.active_pairs[:, 0].long()
        j = self.active_pairs[:, 1].long()
        axis = action.dag_axis.to(device=self.centers.device).long()
        use_x = axis == 0
        branches = torch.empty((self.active_pairs.shape[0],), dtype=torch.long, device=self.centers.device)
        i_before_x = action.plus_scores[i] >= action.plus_scores[j]
        i_before_y = action.minus_scores[i] >= action.minus_scores[j]
        branches[use_x & i_before_x] = int(Branch.L)
        branches[use_x & ~i_before_x] = int(Branch.R)
        branches[~use_x & i_before_y] = int(Branch.B)
        branches[~use_x & ~i_before_y] = int(Branch.A)
        return branches

    def _apply_macro_branches(self, action, branches):
        if (
            not hasattr(action, "macro_cell_indices")
            or action.macro_cell_indices.numel() == 0
            or action.macro_seq_plus.numel() == 0
            or action.macro_seq_minus.numel() == 0
        ):
            return branches
        local = torch.full((self.centers.shape[0],), -1, dtype=torch.long, device=self.centers.device)
        local[action.macro_cell_indices.to(device=self.centers.device).long()] = torch.arange(
            action.macro_cell_indices.numel(),
            dtype=torch.long,
            device=self.centers.device,
        )
        li = local[self.active_pairs[:, 0].long()]
        lj = local[self.active_pairs[:, 1].long()]
        use_macro = (li >= 0) & (lj >= 0)
        if not torch.any(use_macro):
            return branches
        local_pairs = torch.stack([li[use_macro], lj[use_macro]], dim=1)
        macro_branches = induce_branches_from_sequence_pair(
            action.macro_seq_plus.to(device=self.centers.device),
            action.macro_seq_minus.to(device=self.centers.device),
            local_pairs,
        )
        branches = branches.clone()
        branches[use_macro] = macro_branches
        return branches

    def terminal_reward(self):
        best_score, _ = self.best_candidate()
        feasible = 1.0 if best_score["overlap_cells"] == 0 else 0.0
        return self.config.terminal_feasible_bonus * feasible - self.config.terminal_wirelength_coef * best_score["normalized_wl"]

    def best_candidate(self):
        if self.best_score is not None:
            return self.best_score, self.best_centers
        return min(self.saved_candidates, key=lambda item: self._candidate_rank_key(item[0]))

    @staticmethod
    def _candidate_rank_key(score):
        return (
            score["overlap_cells"],
            score["overlap_ratio"],
            score["normalized_wl"],
            score["num_overlap_pairs"],
        )

    def _augmented_lagrangian(
        self,
        centers,
        branches,
        branch_duals,
        boundary_duals,
        density_duals,
        anchor_centers,
        rho=None,
        eta=None,
        pressure=None,
        branch_pressure_values=None,
        boundary_pressure_values=None,
        density_pressure_values=None,
        soft_branch_weights=None,
        soft_tau=None,
    ):
        rho_value = self.config.rho if rho is None else float(rho)
        eta_value = 0.02 if eta is None else float(eta)
        pressure = pressure or {"branch": 1.0, "density": 1.0, "boundary": 1.0}
        current = write_positions(self.cell_features, centers)
        widths = self.cell_features[:, 4]
        heights = self.cell_features[:, 5]
        wl = wirelength_loss(current, self.pin_features, self.edge_list)

        if self.active_pairs.numel() > 0:
            pair_idx = torch.arange(self.active_pairs.shape[0], device=centers.device)
            if soft_branch_weights is None:
                active_lam = branch_duals[pair_idx, branches]
                branch_g = branch_signed_constraints(centers, widths, heights, self.active_pairs, branches) / self.length_scale
            else:
                weights = torch.clamp(soft_branch_weights, min=0.0) + float(self.config.soft_branch_epsilon)
                weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
                active_lam = (branch_duals * weights).sum(dim=1)
                branch_g = (
                    soft_signed_disjunction(
                        centers,
                        widths,
                        heights,
                        self.active_pairs,
                        weights,
                        tau=self.config.soft_tau if soft_tau is None else soft_tau,
                        epsilon=self.config.soft_branch_epsilon,
                    )
                    / self.length_scale
                )
            branch_g_al = torch.relu(branch_g) if self.config.al_mode == "positive_only" else branch_g
            branch_rho = self._effective_rho(
                rho_value,
                pressure.get("branch", 1.0),
                branch_pressure_values,
                branch_g_al,
            )
            branch_al = phr_inequality_penalty(
                branch_g_al,
                active_lam,
                branch_rho,
            )
            overlap_loss = overlap_repulsion_for_pairs(centers, widths, heights, self.active_pairs, self.area_scale)
            branch_violation = torch.relu(branch_g).mean().detach().item()
        else:
            branch_al = centers.sum() * 0.0
            overlap_loss = centers.sum() * 0.0
            branch_violation = 0.0

        boundary_g = boundary_signed_constraints(centers, widths, heights, self.bounds) / self.length_scale
        boundary_g_al = torch.relu(boundary_g) if self.config.al_mode == "positive_only" else boundary_g
        boundary_rho = self._effective_rho(
            rho_value,
            pressure.get("boundary", 1.0),
            boundary_pressure_values,
            boundary_g_al.reshape(-1),
        )
        boundary_al = phr_inequality_penalty(
            boundary_g_al.reshape(-1),
            boundary_duals.reshape(-1),
            boundary_rho,
        )
        density_g, _assignment = self._density_constraints(centers)
        density_g_al = torch.relu(density_g) if self.config.al_mode == "positive_only" else density_g
        density_rho = self._effective_rho(
            rho_value,
            pressure.get("density", 1.0),
            density_pressure_values,
            density_g_al,
        )
        density_al = phr_inequality_penalty(
            density_g_al,
            density_duals,
            density_rho,
        )
        density_v = density_spread_violation(centers, self.cell_features) / self.length_scale
        if not self.config.enable_density:
            density_v = torch.zeros_like(density_v)
        prox = (centers - anchor_centers).square().mean() / torch.clamp(self.length_scale.square(), min=1.0)

        total = (
            self.config.lambda_wirelength * wl
            + (2.0 * self.config.lambda_overlap) * branch_al
            + (10.0 * self.config.lambda_overlap) * overlap_loss
            + 0.10 * self.config.lambda_overlap * boundary_al
            + self.config.lambda_density * density_al
            + 0.05 * self.config.lambda_density * density_v.square().sum()
            + eta_value * prox
        )
        logs = {
            "wirelength": float(wl.detach().item()),
            "branch_violation": branch_violation,
            "boundary_violation": float(torch.relu(boundary_g).mean().detach().item()),
            "density_overflow": float(torch.relu(density_g).mean().detach().item()),
        }
        return total, logs

    def _coordinate_layer(
        self,
        branches,
        branch_duals,
        boundary_duals,
        density_duals,
        initial_centers=None,
        anchor_centers=None,
        coordinate_steps=None,
        coordinate_lr=None,
        rho=None,
        eta=None,
        pressure=None,
        branch_pressure_values=None,
        boundary_pressure_values=None,
        density_pressure_values=None,
        soft_branch_weights=None,
        soft_tau=None,
        trace_steps=None,
    ):
        positions = (self.centers if initial_centers is None else initial_centers).clone().detach()
        positions.requires_grad_(True)
        optimizer = optim.Adam([positions], lr=self.config.coordinate_lr if coordinate_lr is None else float(coordinate_lr))
        anchor = self.centers.detach() if anchor_centers is None else anchor_centers.detach()
        steps = self.config.coordinate_steps if coordinate_steps is None else max(int(coordinate_steps), 1)
        previous_positions = positions.detach().clone()
        for inner_step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            lagrangian, _ = self._augmented_lagrangian(
                positions,
                branches,
                branch_duals,
                boundary_duals,
                density_duals,
                anchor,
                rho=rho,
                eta=eta,
                pressure=pressure,
                branch_pressure_values=branch_pressure_values,
                boundary_pressure_values=boundary_pressure_values,
                density_pressure_values=density_pressure_values,
                soft_branch_weights=soft_branch_weights,
                soft_tau=soft_tau,
            )
            lagrangian.backward()
            grad_norm = 0.0
            if positions.grad is not None:
                grad_norm = float(positions.grad.detach().norm().item())
            clipped_grad_norm = float(torch.nn.utils.clip_grad_norm_([positions], max_norm=10.0))
            optimizer.step()
            if trace_steps is not None:
                after_positions = positions.detach().clone()
                delta = after_positions - previous_positions
                _, step_logs = self._augmented_lagrangian(
                    after_positions,
                    branches,
                    branch_duals,
                    boundary_duals,
                    density_duals,
                    anchor,
                    rho=rho,
                    eta=eta,
                    pressure=pressure,
                    branch_pressure_values=branch_pressure_values,
                    boundary_pressure_values=boundary_pressure_values,
                    density_pressure_values=density_pressure_values,
                    soft_branch_weights=soft_branch_weights,
                    soft_tau=soft_tau,
                )
                trace_steps.append(
                    {
                        "inner_step": int(inner_step),
                        "lagrangian": float(lagrangian.detach().item()),
                        "wirelength": float(step_logs["wirelength"]),
                        "branch_violation": float(step_logs["branch_violation"]),
                        "boundary_violation": float(step_logs["boundary_violation"]),
                        "density_overflow": float(step_logs["density_overflow"]),
                        "grad_norm": float(grad_norm),
                        "grad_norm_clipped": float(clipped_grad_norm),
                        "delta_mean": float(delta.norm(dim=1).mean().item()) if delta.numel() > 0 else 0.0,
                        "delta_max": float(delta.norm(dim=1).max().item()) if delta.numel() > 0 else 0.0,
                        "positions": after_positions,
                    }
                )
                previous_positions = after_positions
        return positions.detach()

    def _update_duals(
        self,
        branches,
        old_branch_duals,
        old_boundary_duals,
        old_density_duals,
        rho=None,
        pressure=None,
        branch_pressure_values=None,
        boundary_pressure_values=None,
        density_pressure_values=None,
        soft_branch_weights=None,
        soft_tau=None,
    ):
        rho_value = self.config.rho if rho is None else float(rho)
        pressure = pressure or {"branch": 1.0, "density": 1.0, "boundary": 1.0}
        widths = self.cell_features[:, 4]
        heights = self.cell_features[:, 5]
        clamp_count = 0
        dual_count = 0
        if self.active_pairs.numel() > 0:
            pair_idx = torch.arange(self.active_pairs.shape[0], device=self.centers.device)
            if soft_branch_weights is None:
                active_lam = old_branch_duals[pair_idx, branches]
                branch_g = branch_signed_constraints(self.centers, widths, heights, self.active_pairs, branches) / self.length_scale
                branch_update_g = torch.relu(branch_g) if self.config.al_mode == "positive_only" else branch_g
                branch_rho = self._effective_rho(
                    rho_value,
                    pressure.get("branch", 1.0),
                    branch_pressure_values,
                    branch_update_g,
                )
                self.branch_duals = old_branch_duals * 0.92
                updated = phr_dual_update(active_lam, branch_update_g, branch_rho, max_value=self.config.dual_max)
                clamp_count += int((updated >= self.config.dual_max - 1e-6).sum().item())
                dual_count += int(updated.numel())
                self.branch_duals[pair_idx, branches] = updated
            else:
                weights = torch.clamp(soft_branch_weights, min=0.0) + float(self.config.soft_branch_epsilon)
                weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
                active_lam = (old_branch_duals * weights).sum(dim=1)
                branch_g = (
                    soft_signed_disjunction(
                        self.centers,
                        widths,
                        heights,
                        self.active_pairs,
                        weights,
                        tau=self.config.soft_tau if soft_tau is None else soft_tau,
                        epsilon=self.config.soft_branch_epsilon,
                    )
                    / self.length_scale
                )
                branch_update_g = torch.relu(branch_g) if self.config.al_mode == "positive_only" else branch_g
                branch_rho = self._effective_rho(
                    rho_value,
                    pressure.get("branch", 1.0),
                    branch_pressure_values,
                    branch_update_g,
                )
                updated_lam = phr_dual_update(active_lam, branch_update_g, branch_rho, max_value=self.config.dual_max)
                clamp_count += int((updated_lam >= self.config.dual_max - 1e-6).sum().item())
                dual_count += int(updated_lam.numel())
                self.branch_duals = old_branch_duals * 0.92 + weights * updated_lam.unsqueeze(1)
            self.branch_duals = torch.clamp(self.branch_duals, min=0.0, max=float(self.config.dual_max))

        boundary_g = boundary_signed_constraints(self.centers, widths, heights, self.bounds) / self.length_scale
        boundary_update_g = torch.relu(boundary_g) if self.config.al_mode == "positive_only" else boundary_g
        boundary_rho = self._effective_rho(
            rho_value,
            pressure.get("boundary", 1.0),
            boundary_pressure_values,
            boundary_update_g,
        )
        self.boundary_duals = phr_dual_update(old_boundary_duals, boundary_update_g, boundary_rho, max_value=self.config.dual_max)
        clamp_count += int((self.boundary_duals >= self.config.dual_max - 1e-6).sum().item())
        dual_count += int(self.boundary_duals.numel())
        density_g, _assignment = self._density_constraints(self.centers)
        density_update_g = torch.relu(density_g) if self.config.al_mode == "positive_only" else density_g
        density_rho = self._effective_rho(
            rho_value,
            pressure.get("density", 1.0),
            density_pressure_values,
            density_update_g,
        )
        self.density_duals = phr_dual_update(old_density_duals, density_update_g, density_rho, max_value=self.config.dual_max)
        clamp_count += int((self.density_duals >= self.config.dual_max - 1e-6).sum().item())
        dual_count += int(self.density_duals.numel())
        self.last_dual_clamp_fraction = float(clamp_count) / max(float(dual_count), 1.0)

    def _audit_active_set(self, action=None, cluster_ids=None, pair_emphasis=0.0):
        current = write_positions(self.cell_features, self.centers)
        exact_pairs = canonicalize_pairs(exact_overlap_pairs(current))
        exact_overlap_count = int(exact_pairs.shape[0])
        inactive_exact_pairs = self._inactive_exact_pairs(exact_pairs)
        inactive_missed_count = int(inactive_exact_pairs.shape[0])
        audit_pressure_scale = self._audit_pressure_scale(
            inactive_missed_count,
            exact_overlap_count,
        )
        scaled_pair_emphasis = max(0.0, min(float(pair_emphasis) * audit_pressure_scale, 1.0))
        retention_horizon = max(
            int(self.config.active_pair_retention),
            int(math.ceil(float(self.config.active_pair_retention) * audit_pressure_scale)),
        )
        cluster_pairs = self._sample_cluster_pairs(cluster_ids, scaled_pair_emphasis)
        uncertain_pairs = self._sample_uncertain_pairs(action, scaled_pair_emphasis)
        sampled_chunks = [pairs for pairs in (cluster_pairs, uncertain_pairs) if pairs.numel() > 0]
        sampled_pairs = (
            canonicalize_pairs(torch.cat(sampled_chunks, dim=0))
            if sampled_chunks
            else torch.empty((0, 2), dtype=torch.long, device=self.centers.device)
        )
        old_count = int(self.active_pairs.shape[0])
        candidate_pairs = exact_pairs
        if sampled_pairs.numel() > 0:
            candidate_pairs = torch.cat([candidate_pairs, sampled_pairs], dim=0) if candidate_pairs.numel() > 0 else sampled_pairs
        updated, updated_ages = update_active_pair_cache(
            self.active_pairs,
            self.active_pair_ages,
            candidate_pairs,
            retention_horizon=retention_horizon,
            max_pairs=self.config.active_pair_limit,
        )
        info = {
            "missed_pairs": inactive_missed_count,
            "inactive_missed_pairs": inactive_missed_count,
            "exact_overlap_pairs": exact_overlap_count,
            "sampled_pairs": int(sampled_pairs.shape[0]),
            "cluster_pairs": int(cluster_pairs.shape[0]),
            "uncertain_pairs": int(uncertain_pairs.shape[0]),
            "new_active_pairs": max(int(updated.shape[0]) - old_count, 0),
            "retained_pairs": int((updated_ages > 0).sum().item()) if updated_ages.numel() else 0,
            "hard_pair_age_mean": self._age_mean(updated_ages),
            "hard_pair_age_max": self._age_max(updated_ages),
            "hard_pair_age_min": self._age_min(updated_ages),
            "audit_pressure_scale": audit_pressure_scale,
            "audit_pressure_target": self._audit_pressure_target(),
            "retention_horizon": retention_horizon,
        }
        if updated.shape[0] == self.active_pairs.shape[0] and torch.equal(updated, self.active_pairs):
            self.active_pair_ages = updated_ages
            return info
        self.branch_duals = self._remap_branch_duals(self.active_pairs, self.branch_duals, updated)
        self.active_pairs = updated
        self.active_pair_ages = updated_ages
        return info

    def _inactive_exact_pairs(self, exact_pairs):
        exact_pairs = canonicalize_pairs(exact_pairs)
        if exact_pairs.numel() == 0 or self.active_pairs.numel() == 0:
            return exact_pairs
        active_pairs = canonicalize_pairs(self.active_pairs)
        if active_pairs.numel() == 0:
            return exact_pairs
        base = int(torch.max(torch.cat([exact_pairs.reshape(-1), active_pairs.reshape(-1)])).item()) + 1
        exact_pairs = sort_pairs_and_payload(exact_pairs, base=base)[0]
        active_pairs = sort_pairs_and_payload(active_pairs, base=base)[0]
        exact_keys = pair_sort_keys(exact_pairs, base=base)
        active_keys = pair_sort_keys(active_pairs, base=base)
        positions = torch.searchsorted(active_keys, exact_keys)
        clamped = torch.clamp(positions, max=active_keys.shape[0] - 1)
        matched = (positions < active_keys.shape[0]) & (active_keys[clamped] == exact_keys)
        return exact_pairs[~matched]

    def _audit_pressure_target(self):
        configured_target = float(self.config.audit_missed_target)
        size_scaled_target = max(4.0, 0.25 * float(self.centers.shape[0]))
        if configured_target <= 0.0:
            return size_scaled_target
        return min(configured_target, size_scaled_target)

    def _audit_pressure_scale(self, inactive_missed_count, exact_overlap_count=0):
        target = self._audit_pressure_target()
        if target <= 0.0:
            return 1.0
        unresolved_overlap_burden = max(float(inactive_missed_count), float(exact_overlap_count))
        excess = max((unresolved_overlap_burden - target) / (target + 1e-6), 0.0)
        scale = 1.0 + float(self.config.audit_pressure_gamma) * excess
        return max(1.0, min(scale, float(self.config.audit_pressure_max)))

    @staticmethod
    def _age_mean(ages):
        return float(ages.float().mean().item()) if ages is not None and ages.numel() else 0.0

    @staticmethod
    def _age_max(ages):
        return float(ages.max().item()) if ages is not None and ages.numel() else 0.0

    @staticmethod
    def _age_min(ages):
        return float(ages.min().item()) if ages is not None and ages.numel() else 0.0

    def _sample_uncertain_pairs(self, action=None, pair_emphasis=0.0):
        if action is None:
            return torch.empty((0, 2), dtype=torch.long, device=self.centers.device)
        emphasis = max(0.0, min(float(pair_emphasis), 1.0))
        if emphasis <= 1e-6 or self.centers.shape[0] < 2:
            return torch.empty((0, 2), dtype=torch.long, device=self.centers.device)

        window = 2 + int(10 * emphasis)
        chunks = []
        for dim in (0, 1):
            order = torch.argsort(self.centers[:, dim])
            for offset in range(1, min(window, int(order.numel() - 1)) + 1):
                chunks.append(torch.stack([order[:-offset], order[offset:]], dim=1))
        if not chunks:
            return torch.empty((0, 2), dtype=torch.long, device=self.centers.device)

        candidates = canonicalize_pairs(torch.cat(chunks, dim=0))
        if candidates.numel() == 0:
            return candidates
        max_candidates = min(candidates.shape[0], 4096 + int(8192 * emphasis))
        if candidates.shape[0] > max_candidates:
            candidates = candidates[:max_candidates]

        with torch.no_grad():
            weights = hierarchical_active_branch_weights(
                action,
                candidates,
                relaxation="sigmoid",
                tau=max(float(getattr(action, "tau", torch.tensor(1.0)).detach().item()), 1e-3),
            )
            entropy = -(weights * torch.log(torch.clamp(weights, min=1e-8))).sum(dim=1)
            keep = min(int(64 + 512 * emphasis), int(candidates.shape[0]))
            if keep <= 0:
                return torch.empty((0, 2), dtype=torch.long, device=self.centers.device)
            order = torch.topk(entropy, k=keep, largest=True).indices
        return canonicalize_pairs(candidates[order])

    def _sample_cluster_pairs(self, cluster_ids=None, pair_emphasis=0.0):
        if cluster_ids is None or cluster_ids.numel() != self.centers.shape[0]:
            return torch.empty((0, 2), dtype=torch.long, device=self.centers.device)
        emphasis = max(0.0, min(float(pair_emphasis), 1.0))
        if emphasis <= 1e-6:
            return torch.empty((0, 2), dtype=torch.long, device=self.centers.device)
        max_window = 1 + int(5 * emphasis)
        chunks = []
        valid_clusters = torch.unique(cluster_ids[cluster_ids >= 0])
        for cluster in valid_clusters.unbind(0):
            members = torch.where(cluster_ids == cluster)[0]
            if members.numel() < 2:
                continue
            for dim in (0, 1):
                order = members[torch.argsort(self.centers[members, dim])]
                for offset in range(1, min(max_window, int(order.numel() - 1)) + 1):
                    chunks.append(torch.stack([order[:-offset], order[offset:]], dim=1))
        if not chunks:
            return torch.empty((0, 2), dtype=torch.long, device=self.centers.device)
        pairs = canonicalize_pairs(torch.cat(chunks, dim=0))
        if pairs.shape[0] > self.config.active_pair_limit:
            pairs = pairs[: self.config.active_pair_limit]
        return pairs

    def _wirelength_gradient(self):
        positions = self.centers.clone().detach()
        positions.requires_grad_(True)
        current = write_positions(self.cell_features, positions)
        loss = wirelength_loss(current, self.pin_features, self.edge_list)
        grad = torch.autograd.grad(loss, positions, allow_unused=True)[0]
        if grad is None:
            return torch.zeros_like(positions)
        return grad.detach()

    def _wirelength_gradient_at(self, centers):
        positions = centers.clone().detach()
        positions.requires_grad_(True)
        current = write_positions(self.cell_features, positions)
        loss = wirelength_loss(current, self.pin_features, self.edge_list)
        grad = torch.autograd.grad(loss, positions, allow_unused=True)[0]
        if grad is None:
            return torch.zeros_like(positions)
        return grad.detach()

    def _macro_mask(self):
        areas = self.cell_features[:, 0]
        if areas.numel() <= 2:
            return torch.zeros((areas.shape[0],), dtype=torch.bool, device=areas.device)
        area_median = torch.median(areas)
        return areas > torch.clamp(area_median * 1.5, min=area_median + 1.0)

    def _same_size_mask(self, cell_index):
        widths = self.cell_features[:, 4]
        heights = self.cell_features[:, 5]
        target_w = widths[int(cell_index)]
        target_h = heights[int(cell_index)]
        return torch.isclose(widths, target_w, atol=1.0e-6) & torch.isclose(heights, target_h, atol=1.0e-6)

    def _translation_clean_wirelength_at(self, centers):
        return translation_clean_normalized_wirelength(
            write_positions(self.cell_features, centers),
            self.pin_features,
            self.edge_list,
        )

    def _same_size_group_sizes(self):
        widths = self.cell_features[:, 4]
        heights = self.cell_features[:, 5]
        groups = {}
        for idx in range(int(self.cell_features.shape[0])):
            key = (round(float(widths[idx].item()), 6), round(float(heights[idx].item()), 6))
            groups[key] = groups.get(key, 0) + 1
        counts = torch.zeros((self.cell_features.shape[0],), dtype=self.cell_features.dtype, device=self.cell_features.device)
        for idx in range(int(self.cell_features.shape[0])):
            key = (round(float(widths[idx].item()), 6), round(float(heights[idx].item()), 6))
            counts[idx] = float(groups.get(key, 1))
        return counts

    def _case_descriptor_vector(self, centers):
        dtype = self.cell_features.dtype
        device = self.cell_features.device
        area = self.cell_features[:, 0]
        macro_mask = self._macro_mask()
        std_mask = ~macro_mask
        total_area = torch.clamp(area.sum(), min=1.0)
        macro_area_share = area[macro_mask].sum() / total_area if torch.any(macro_mask) else centers.new_tensor(0.0)
        density_g, _assignment = self._density_constraints(centers)
        density_overflow = density_spread_violation(centers, self.cell_features).detach().max()
        pin_to_cell = self.pin_features[:, 0].long()
        if self.edge_list.numel() > 0:
            cell_edge_counts = torch.zeros((self.cell_features.shape[0],), dtype=dtype, device=device)
            src_cells = pin_to_cell[self.edge_list[:, 0].long()]
            dst_cells = pin_to_cell[self.edge_list[:, 1].long()]
            cell_edge_counts.scatter_add_(0, src_cells, torch.ones_like(src_cells, dtype=dtype))
            cell_edge_counts.scatter_add_(0, dst_cells, torch.ones_like(dst_cells, dtype=dtype))
            fanout_mean = cell_edge_counts.mean() / torch.clamp(cell_edge_counts.max(), min=1.0)
        else:
            fanout_mean = centers.new_tensor(0.0)
        same_size_counts = self._same_size_group_sizes()
        std_same_size = same_size_counts[std_mask] if torch.any(std_mask) else same_size_counts
        same_size_concentration = (
            std_same_size.max() / torch.clamp(std_mask.sum().to(dtype), min=1.0)
            if std_same_size.numel() > 0
            else centers.new_tensor(0.0)
        )
        return torch.tensor(
            [
                float(macro_mask.sum().item()),
                float(std_mask.sum().item()),
                float(macro_area_share.item()),
                float(density_overflow.item()),
                float(fanout_mean.item()),
                float(same_size_concentration.item()),
            ],
            dtype=dtype,
            device=device,
        )

    def _case_descriptor_bucket(self, centers):
        descriptor = self._case_descriptor_vector(centers)
        std_cells = int(round(float(descriptor[1].item())))
        density_overflow = float(descriptor[3].item())
        if (
            std_cells >= int(self.config.adaptive_pd_mid_cells_min)
            and std_cells <= int(self.config.adaptive_pd_mid_cells_max)
            and density_overflow >= float(self.config.adaptive_pd_density_threshold)
        ):
            return "mid_dense"
        return "default"

    def _large_case_swap_legality_conflict(self, *, incumbent_score, candidate_score):
        descriptor = self._case_descriptor_vector(self.centers)
        std_cells = int(round(float(descriptor[1].item())))
        if std_cells < int(self.config.large_case_swap_filter_min_std_cells):
            return False
        overlap_worse = float(candidate_score["overlap_ratio"]) > float(incumbent_score["overlap_ratio"]) + 1.0e-12
        pair_worse = int(candidate_score["num_overlap_pairs"]) > int(incumbent_score["num_overlap_pairs"])
        wire_better = float(candidate_score["normalized_wl"]) < float(incumbent_score["normalized_wl"]) - 1.0e-12
        return bool(wire_better and (overlap_worse or pair_worse))

    def _continuation_risk(self, *, current_score, best_score, phase_name, steps_since_best):
        phase_name = str(phase_name).upper()
        if phase_name != PlacementPhase.REFINE:
            return 0.0
        if int(self.phase_step) < int(self.config.continuation_refine_min_phase_steps):
            return 0.0
        if float(best_score["overlap_ratio"]) > float(self.config.continuation_aux_min_overlap):
            return 0.0
        if not self._continuation_margin_met(current_score=current_score, best_score=best_score):
            return 0.0
        risk = 0.0
        if int(steps_since_best) >= int(self.config.continuation_risk_stale_steps):
            risk += 0.30
        if float(current_score["overlap_ratio"]) > float(best_score["overlap_ratio"]) + float(self.config.continuation_risk_overlap_gap):
            risk += 0.30
        if int(current_score["num_overlap_pairs"]) > int(best_score["num_overlap_pairs"]) + int(self.config.continuation_risk_pair_gap):
            risk += 0.25
        risk += float(self.config.continuation_refine_phase_bonus)
        return max(0.0, min(risk, 1.0))

    def _continuation_margin_met(self, *, current_score, best_score):
        overlap_gap = float(current_score["overlap_ratio"]) - float(best_score["overlap_ratio"])
        if overlap_gap >= float(self.config.continuation_margin_overlap):
            return True
        if abs(overlap_gap) <= float(self.config.continuation_margin_overlap):
            pair_gap = int(current_score["num_overlap_pairs"]) - int(best_score["num_overlap_pairs"])
            if pair_gap >= int(self.config.continuation_margin_pairs):
                return True
            if abs(pair_gap) < int(self.config.continuation_margin_pairs):
                wire_gap = float(current_score["normalized_wl"]) - float(best_score["normalized_wl"])
                if wire_gap >= float(self.config.continuation_margin_wire):
                    return True
        return False

    def _adaptive_pd_step_adjustment(self, *, phase_name, current_score, best_score):
        phase_name = str(phase_name).upper()
        if phase_name not in {PlacementPhase.LEGALIZE, PlacementPhase.REFINE}:
            return 0, "", False
        bucket = self._case_descriptor_bucket(self.centers)
        if bucket != "mid_dense":
            return 0, bucket, False
        continuation_risk = self._continuation_risk(
            current_score=current_score,
            best_score=best_score,
            phase_name=phase_name,
            steps_since_best=self.steps_since_best,
        )
        if continuation_risk > 0.0:
            return 0, bucket, False
        if float(best_score["overlap_ratio"]) > float(self.config.refine_overlap_threshold):
            return int(self.config.adaptive_pd_extra_steps_large), bucket, True
        if int(best_score["num_overlap_pairs"]) > 1:
            return int(self.config.adaptive_pd_extra_steps_small), bucket, True
        return 0, bucket, False

    def _cleanup_feature_vector(self, centers, *, phase_name, incumbent_centers):
        dtype = self.cell_features.dtype
        device = self.cell_features.device
        if str(phase_name).upper() not in {PlacementPhase.LEGALIZE, PlacementPhase.REFINE}:
            return torch.zeros((5,), dtype=dtype, device=device)
        wire_grad = self._wirelength_gradient_at(incumbent_centers)
        window = self._select_refine_window(
            incumbent_centers,
            current_centers=centers,
            residual=(centers - incumbent_centers),
            wire_grad=wire_grad,
        )
        if window.numel() == 0:
            return torch.zeros((5,), dtype=dtype, device=device)
        translation_clean = self._translation_clean_wirelength_at(incumbent_centers)
        full_wire = self._normalized_wl_at_centers(incumbent_centers)
        same_size_counts = self._same_size_group_sizes()
        same_size_fraction = torch.clamp((same_size_counts[window] > 1).to(dtype).mean(), min=0.0, max=1.0)
        macro_mask = self._macro_mask()
        macro_centers = incumbent_centers[macro_mask]
        if macro_centers.numel() > 0:
            macro_dist = torch.cdist(incumbent_centers[window], macro_centers).min(dim=1).values
            macro_adjacent = (macro_dist <= max(float(self.length_scale) * 0.10, 1.0e-6)).to(dtype).mean()
        else:
            macro_adjacent = centers.new_tensor(0.0)
        displacement = torch.norm(centers[window] - incumbent_centers[window], dim=1).mean() / max(float(self.length_scale), 1.0e-6)
        score = self._score_centers(centers)
        repair_risk = float(score["overlap_ratio"]) + float(score["num_overlap_pairs"]) / max(float(self.cell_features.shape[0]), 1.0)
        return torch.tensor(
            [
                float(max(full_wire - translation_clean, 0.0)),
                float(same_size_fraction.item()),
                float(macro_adjacent.item()),
                float(displacement.item()),
                float(repair_risk),
            ],
            dtype=dtype,
            device=device,
        )

    def _select_refine_window(self, incumbent_centers, current_centers=None, residual=None, wire_grad=None):
        num_cells = int(self.centers.shape[0])
        if num_cells <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.centers.device)
        current_centers = incumbent_centers if current_centers is None else current_centers
        if wire_grad is None:
            wire_grad = self._wirelength_gradient_at(incumbent_centers)
        residual = torch.zeros_like(incumbent_centers) if residual is None else residual
        grad_norm = torch.norm(wire_grad, dim=1)
        displacement_norm = torch.norm(current_centers - incumbent_centers, dim=1)
        residual_norm = torch.norm(residual, dim=1)
        active_counts = torch.zeros((num_cells,), dtype=incumbent_centers.dtype, device=incumbent_centers.device)
        if self.active_pairs.numel() > 0:
            flat = self.active_pairs.reshape(-1).long()
            active_counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=active_counts.dtype))
        scores = (
            float(self.config.refine_compaction_gradient_weight) * grad_norm
            + float(self.config.refine_window_displacement_weight) * displacement_norm
            + float(self.config.refine_compaction_residual_weight) * residual_norm
            + float(self.config.refine_window_active_pair_weight) * active_counts
        )
        if not torch.isfinite(scores).all():
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        macro_mask = self._macro_mask()
        non_macro = torch.where(~macro_mask)[0]
        if non_macro.numel() > 0:
            seed_local = int(torch.argmax(scores[non_macro]).item())
            seed_idx = int(non_macro[seed_local].item())
        else:
            seed_idx = int(torch.argmax(scores).item())
        radius = max(float(self.config.refine_compaction_radius_scale) * float(self.length_scale), 1.0e-6)
        distances = torch.norm(incumbent_centers - incumbent_centers[seed_idx].unsqueeze(0), dim=1)
        radius_cells = torch.where(distances <= radius)[0]
        topk_count = max(int(self.config.refine_window_min_cells), int(math.ceil(float(self.config.refine_compaction_fraction) * float(num_cells))))
        topk_count = min(topk_count, num_cells)
        topk_cells = torch.topk(scores, k=topk_count, largest=True).indices
        chunks = [
            radius_cells.long(),
            topk_cells.long(),
            torch.tensor([seed_idx], dtype=torch.long, device=self.centers.device),
            torch.where(self._same_size_mask(seed_idx))[0].long(),
        ]
        if self.active_pairs.numel() > 0:
            chunks.append(self.active_pairs.reshape(-1).long())
        window = torch.unique(torch.cat(chunks, dim=0))
        max_cells = int(self.config.refine_window_max_cells)
        if max_cells > 0 and window.numel() > max_cells:
            window_scores = scores[window]
            keep = torch.topk(window_scores, k=max_cells, largest=True).indices
            window = window[keep]
        return torch.sort(window.long()).values

    def _build_refine_compaction_base_centers(self, *, action, incumbent_centers, residual, step_scale):
        wire_grad = self._wirelength_gradient_at(incumbent_centers)
        window = self._select_refine_window(
            incumbent_centers,
            current_centers=self.centers.detach().clone(),
            residual=residual,
            wire_grad=wire_grad,
        )
        base_centers = incumbent_centers.detach().clone()
        if window.numel() == 0:
            return base_centers, window, wire_grad
        grad_norm = torch.norm(wire_grad, dim=1, keepdim=True).clamp_min(1.0e-6)
        grad_direction = -wire_grad / grad_norm
        move_cap = (
            max(float(step_scale), 1.0e-4)
            * float(self.length_scale)
            * float(self.config.refine_compaction_step_multiplier)
        )
        compaction_delta = (
            float(self.config.refine_compaction_residual_weight) * residual
            + move_cap * grad_direction
        )
        delta_norm = torch.norm(compaction_delta, dim=1, keepdim=True).clamp_min(1.0e-6)
        compaction_delta = compaction_delta * torch.clamp(move_cap / delta_norm, max=1.0)
        compaction_mask = torch.zeros((self.centers.shape[0],), dtype=torch.bool, device=self.centers.device)
        compaction_mask[window] = True
        base_centers[compaction_mask] = incumbent_centers[compaction_mask] + compaction_delta[compaction_mask]
        return base_centers, window, wire_grad

    def _project_refine_local_base_centers(self, *, incumbent_centers, current_centers, residual, step_scale, window, wire_grad):
        base_centers = incumbent_centers.detach().clone()
        if window.numel() == 0:
            return base_centers
        grad_norm = torch.norm(wire_grad, dim=1, keepdim=True).clamp_min(1.0e-6)
        grad_direction = -wire_grad / grad_norm
        displacement = current_centers - incumbent_centers
        move_cap = max(float(step_scale), 1.0e-4) * float(self.length_scale) * 0.5
        projected_delta = displacement + move_cap * grad_direction
        delta_norm = torch.norm(projected_delta, dim=1, keepdim=True).clamp_min(1.0e-6)
        projected_delta = projected_delta * torch.clamp(move_cap / delta_norm, max=1.0)
        base_centers[window] = incumbent_centers[window] + projected_delta[window]
        return base_centers

    def _normalized_wl_at_centers(self, centers):
        return normalized_wirelength(write_positions(self.cell_features, centers), self.pin_features, self.edge_list)

    def _expand_swap_reassign_window(self, *, incumbent_centers, window):
        window = window.long()
        if window.numel() == 0:
            return window
        max_cells = max(int(self.config.swap_reassign_window_max_cells), int(window.numel()))
        if int(window.numel()) >= max_cells:
            return window
        widths = self.cell_features[:, 4]
        heights = self.cell_features[:, 5]
        centers = incumbent_centers
        selected = {int(idx) for idx in window.detach().cpu().tolist()}
        window_centroid = centers[window].mean(dim=0)
        candidate_rows = []
        for idx in range(int(self.centers.shape[0])):
            if idx in selected:
                continue
            same_mask = (
                torch.isclose(widths[window], widths[idx], atol=1.0e-6)
                & torch.isclose(heights[window], heights[idx], atol=1.0e-6)
            )
            if not bool(torch.any(same_mask)):
                continue
            distance = torch.norm(centers[idx] - window_centroid).item()
            candidate_rows.append((float(distance), int(idx)))
        candidate_rows.sort(key=lambda item: (item[0], item[1]))
        expanded = list(int(idx) for idx in window.detach().cpu().tolist())
        for _distance, idx in candidate_rows:
            if len(expanded) >= max_cells:
                break
            expanded.append(int(idx))
        return torch.tensor(expanded, dtype=torch.long, device=window.device)

    def _swap_or_reassign_local_base_centers(self, *, incumbent_centers, window, return_metadata=False):
        base_centers = incumbent_centers.detach().clone()
        window = self._expand_swap_reassign_window(
            incumbent_centers=incumbent_centers,
            window=window,
        )
        if window.numel() < 2:
            if return_metadata:
                return base_centers, {
                    "window_indices": [int(idx) for idx in window.detach().cpu().tolist()],
                    "same_size_groups": [],
                    "attempted_swaps": 0,
                    "accepted_swaps": [],
                    "passes": 0,
                    "best_wirelength": float(self._normalized_wl_at_centers(base_centers)),
                }
            return base_centers
        widths = self.cell_features[:, 4]
        heights = self.cell_features[:, 5]
        window = window.long()
        best_centers = base_centers.clone()
        best_wl = float(self._normalized_wl_at_centers(best_centers))
        metadata = {
            "window_indices": [int(idx) for idx in window.detach().cpu().tolist()],
            "same_size_groups": [],
            "attempted_swaps": 0,
            "accepted_swaps": [],
            "passes": 0,
            "best_wirelength": float(best_wl),
        }
        seen_groups = set()
        groups = []
        for seed in window.tolist():
            same_mask = (
                torch.isclose(widths[window], widths[int(seed)], atol=1.0e-6)
                & torch.isclose(heights[window], heights[int(seed)], atol=1.0e-6)
            )
            same_group = window[same_mask]
            if same_group.numel() < 2:
                continue
            group_key = tuple(sorted(int(idx) for idx in same_group.detach().cpu().tolist()))
            if group_key not in seen_groups:
                seen_groups.add(group_key)
                metadata["same_size_groups"].append(
                    {
                        "seed": int(seed),
                        "members": list(group_key),
                        "width": float(widths[int(seed)].item()),
                        "height": float(heights[int(seed)].item()),
                    }
                )
                groups.append(torch.tensor(group_key, dtype=torch.long, device=window.device))
        max_passes = max(int(self.config.swap_reassign_max_passes), 1)
        for _pass_idx in range(max_passes):
            metadata["passes"] += 1
            best_swap = None
            best_candidate = None
            best_candidate_wl = float(best_wl)
            for same_group in groups:
                for local_i in range(int(same_group.numel())):
                    for local_j in range(local_i + 1, int(same_group.numel())):
                        i = int(same_group[local_i].item())
                        j = int(same_group[local_j].item())
                        metadata["attempted_swaps"] += 1
                        candidate = best_centers.clone()
                        tmp = candidate[i].clone()
                        candidate[i] = candidate[j]
                        candidate[j] = tmp
                        candidate_wl = float(self._normalized_wl_at_centers(candidate))
                        if candidate_wl + 1.0e-9 < best_candidate_wl:
                            best_candidate_wl = candidate_wl
                            best_candidate = candidate
                            best_swap = (i, j)
            if best_swap is None:
                break
            previous_best_wl = float(best_wl)
            best_wl = float(best_candidate_wl)
            best_centers = best_candidate
            metadata["accepted_swaps"].append(
                {
                    "i": int(best_swap[0]),
                    "j": int(best_swap[1]),
                    "wire_before": previous_best_wl,
                    "wire_after": float(best_candidate_wl),
                }
            )
        metadata["best_wirelength"] = float(best_wl)
        if return_metadata:
            return best_centers, metadata
        return best_centers

    @staticmethod
    def _refine_variant_rank_key(score):
        return (
            float(score["overlap_ratio"]),
            int(score["num_overlap_pairs"]),
            float(score["normalized_wl"]),
        )

    def _repair_refine_variant(
        self,
        *,
        variant_name,
        base_centers,
        incumbent_centers,
        branches,
        branch_duals,
        boundary_duals,
        density_duals,
        rho,
        eta,
        alpha,
        pd_steps,
        pressure,
        branch_pressure_values,
        boundary_pressure_values,
        density_pressure_values,
        soft_branch_weights,
        soft_tau,
    ):
        repaired = self._coordinate_layer(
            branches,
            branch_duals,
            boundary_duals,
            density_duals,
            initial_centers=base_centers,
            anchor_centers=base_centers.detach(),
            coordinate_steps=pd_steps,
            coordinate_lr=alpha,
            rho=rho,
            eta=eta,
            pressure=pressure,
            branch_pressure_values=branch_pressure_values,
            boundary_pressure_values=boundary_pressure_values,
            density_pressure_values=density_pressure_values,
            soft_branch_weights=soft_branch_weights,
            soft_tau=soft_tau,
            trace_steps=None,
        )
        lag_after, after_logs = self._augmented_lagrangian(
            repaired,
            branches,
            branch_duals,
            boundary_duals,
            density_duals,
            incumbent_centers.detach(),
            rho=rho,
            eta=eta,
            pressure=pressure,
            branch_pressure_values=branch_pressure_values,
            boundary_pressure_values=boundary_pressure_values,
            density_pressure_values=density_pressure_values,
            soft_branch_weights=soft_branch_weights,
            soft_tau=soft_tau,
        )
        score = self._score_centers(repaired)
        return {
            "variant_name": str(variant_name),
            "centers": repaired.detach().clone(),
            "score": dict(score),
            "lag_after": lag_after,
            "after_logs": dict(after_logs),
        }

    def _select_refine_variants(self, *, incumbent_centers, window, policy=None):
        selected = {"incumbent_hold"}
        cleanup_features = self._cleanup_feature_vector(
            incumbent_centers,
            phase_name=PlacementPhase.REFINE,
            incumbent_centers=incumbent_centers,
        )
        translation_clean_gap = float(cleanup_features[0].item()) if cleanup_features.numel() > 0 else 0.0
        same_size_fraction = float(cleanup_features[1].item()) if cleanup_features.numel() > 1 else 0.0
        macro_adjacent = float(cleanup_features[2].item()) if cleanup_features.numel() > 2 else 0.0
        repair_risk = float(cleanup_features[4].item()) if cleanup_features.numel() > 4 else 0.0
        if same_size_fraction >= 0.35 and macro_adjacent <= 0.35:
            selected.add("swap_or_reassign_local")
        if same_size_fraction < 0.35 and translation_clean_gap >= 0.01:
            selected.add("projection_local")
        if repair_risk >= 0.30:
            return tuple(name for name in REFINE_VARIANT_NAMES if name in selected)
        if policy is not None:
            aux_graph = self.auxiliary_graph_for_candidate(incumbent_centers, phase_name=PlacementPhase.REFINE)
            with torch.no_grad():
                logits = policy.auxiliary_predictions(aux_graph)["cleanup_variant_logits"]
            topk = min(2, int(logits.numel()))
            for idx in torch.topk(logits, k=topk).indices.tolist():
                selected.add(REFINE_VARIANT_NAMES[int(idx)])
        if "wire_grad_local" in selected and same_size_fraction >= 0.5:
            selected.discard("wire_grad_local")
        return tuple(name for name in REFINE_VARIANT_NAMES if name in selected)

    def _run_refine_variant_portfolio(
        self,
        *,
        policy=None,
        action,
        incumbent_centers,
        current_centers,
        residual,
        step_scale,
        branches,
        branch_duals,
        boundary_duals,
        density_duals,
        rho,
        eta,
        alpha,
        pd_steps,
        pressure,
        branch_pressure_values,
        boundary_pressure_values,
        density_pressure_values,
        soft_branch_weights,
        soft_tau,
    ):
        wire_grad = self._wirelength_gradient_at(incumbent_centers)
        window = self._select_refine_window(
            incumbent_centers,
            current_centers=current_centers,
            residual=residual,
            wire_grad=wire_grad,
        )
        swap_base_centers, swap_metadata = self._swap_or_reassign_local_base_centers(
            incumbent_centers=incumbent_centers,
            window=window,
            return_metadata=True,
        )
        window_metadata = {
            "window_indices": [int(idx) for idx in window.detach().cpu().tolist()],
            "window_size": int(window.numel()),
            "selected_variants": [],
        }
        variant_bases = {
            "incumbent_hold": incumbent_centers.detach().clone(),
            "wire_grad_local": self._build_refine_compaction_base_centers(
                action=action,
                incumbent_centers=incumbent_centers,
                residual=residual,
                step_scale=step_scale,
            )[0],
            "projection_local": self._project_refine_local_base_centers(
                incumbent_centers=incumbent_centers,
                current_centers=current_centers,
                residual=residual,
                step_scale=step_scale,
                window=window,
                wire_grad=wire_grad,
            ),
            "swap_or_reassign_local": swap_base_centers,
        }
        incumbent_score = self._score_centers(incumbent_centers)
        best_result = {
            "winning_variant": "incumbent_hold",
            "centers": incumbent_centers.detach().clone(),
            "score": dict(incumbent_score),
            "lag_after": incumbent_centers.new_tensor(0.0),
            "after_logs": {
                "wirelength": float(incumbent_score["normalized_wl"]),
                "branch_violation": 0.0,
                "boundary_violation": 0.0,
                "density_overflow": 0.0,
            },
        }
        variant_rows = []
        candidate_options = []
        selected_variants = self._select_refine_variants(
            incumbent_centers=incumbent_centers,
            window=window,
            policy=policy,
        )
        window_metadata["selected_variants"] = [str(name) for name in selected_variants]
        for variant_name in REFINE_VARIANT_NAMES:
            if variant_name not in selected_variants:
                variant_rows.append(
                    {
                        "variant_name": str(variant_name),
                        "accepted": False,
                        "repair_legal": False,
                        "overlap_ratio": float(incumbent_score["overlap_ratio"]),
                        "num_overlap_pairs": int(incumbent_score["num_overlap_pairs"]),
                        "normalized_wl": float(incumbent_score["normalized_wl"]),
                        "overlap_delta": 0.0,
                        "pair_delta": 0,
                        "wire_delta": 0.0,
                        "evaluated": False,
                        "chooser_eligible": False,
                        "operator_metadata": {},
                    }
                )
                continue
            repaired = self._repair_refine_variant(
                variant_name=variant_name,
                base_centers=variant_bases[variant_name],
                incumbent_centers=incumbent_centers,
                branches=branches,
                branch_duals=branch_duals,
                boundary_duals=boundary_duals,
                density_duals=density_duals,
                rho=rho,
                eta=eta,
                alpha=alpha,
                pd_steps=pd_steps,
                pressure=pressure,
                branch_pressure_values=branch_pressure_values,
                boundary_pressure_values=boundary_pressure_values,
                density_pressure_values=density_pressure_values,
                soft_branch_weights=soft_branch_weights,
                soft_tau=soft_tau,
            )
            repair_legal = int(repaired["score"]["overlap_cells"]) == 0
            accepted = self._refine_variant_rank_key(repaired["score"]) < self._refine_variant_rank_key(incumbent_score)
            chooser_eligible = True
            if (
                variant_name == "swap_or_reassign_local"
                and self._large_case_swap_legality_conflict(
                    incumbent_score=incumbent_score,
                    candidate_score=repaired["score"],
                )
            ):
                chooser_eligible = False
            variant_rows.append(
                {
                    "variant_name": str(variant_name),
                    "accepted": bool(accepted),
                    "repair_legal": bool(repair_legal),
                    "evaluated": True,
                    "chooser_eligible": bool(chooser_eligible),
                    "overlap_ratio": float(repaired["score"]["overlap_ratio"]),
                    "num_overlap_pairs": int(repaired["score"]["num_overlap_pairs"]),
                    "normalized_wl": float(repaired["score"]["normalized_wl"]),
                    "overlap_delta": float(incumbent_score["overlap_ratio"]) - float(repaired["score"]["overlap_ratio"]),
                    "pair_delta": int(incumbent_score["num_overlap_pairs"]) - int(repaired["score"]["num_overlap_pairs"]),
                    "wire_delta": float(incumbent_score["normalized_wl"]) - float(repaired["score"]["normalized_wl"]),
                    "operator_metadata": dict(swap_metadata) if variant_name == "swap_or_reassign_local" else {},
                }
            )
            candidate_options.append(
                {
                    "source": f"refine_variant:{variant_name}",
                    "variant_name": str(variant_name),
                    "accepted": bool(accepted),
                    "repair_legal": bool(repair_legal),
                    "evaluated": True,
                    "live_input": bool(chooser_eligible),
                    "centers": repaired["centers"].detach().clone(),
                    "score": dict(repaired["score"]),
                    "generation_index": int(len(candidate_options) + 1),
                    "operator_metadata": dict(swap_metadata) if variant_name == "swap_or_reassign_local" else {},
                }
            )
            if accepted and self._refine_variant_rank_key(repaired["score"]) < self._refine_variant_rank_key(best_result["score"]):
                best_result = {
                    "winning_variant": str(variant_name),
                    "centers": repaired["centers"],
                    "score": dict(repaired["score"]),
                    "lag_after": repaired["lag_after"],
                    "after_logs": dict(repaired["after_logs"]),
                }
        best_result["variant_rows"] = variant_rows
        best_result["candidate_options"] = candidate_options
        best_result["window"] = window.detach().clone()
        best_result["window_metadata"] = window_metadata
        best_result["selected_variants"] = list(selected_variants)
        return best_result

    def run_post_legal_refine_portfolio(self, incumbent_centers, policy=None):
        incumbent_centers = incumbent_centers.detach().clone().to(self.centers.device)
        original_centers = self.centers.detach().clone()
        original_active_pairs = self.active_pairs.detach().clone()
        original_active_pair_ages = self.active_pair_ages.detach().clone()
        original_branch_duals = self.branch_duals.detach().clone()
        original_boundary_duals = self.boundary_duals.detach().clone()
        original_density_duals = self.density_duals.detach().clone()
        try:
            self.centers = incumbent_centers.detach().clone()
            self.active_pairs = build_initial_active_pairs(
                write_positions(self.cell_features, self.centers),
                self.pin_features,
                self.edge_list,
                max_pairs=self.config.active_pair_limit,
            )
            self.active_pair_ages = torch.full(
                (self.active_pairs.shape[0],),
                self.config.active_pair_retention,
                dtype=torch.long,
                device=self.centers.device,
            )
            self.branch_duals = torch.zeros((self.active_pairs.shape[0], 4), dtype=self.cell_features.dtype, device=self.cell_features.device)
            self.boundary_duals = torch.zeros_like(self.boundary_duals)
            density_g, _assignment = self._density_constraints(self.centers)
            self.density_duals = torch.zeros_like(density_g)
            seq_plus, seq_minus = sequence_pair_from_centers(self.centers)
            branches = induce_branches_from_sequence_pair(seq_plus, seq_minus, self.active_pairs)
            residual = torch.zeros_like(self.centers)
            return self._run_refine_variant_portfolio(
                policy=policy,
                action=None,
                incumbent_centers=incumbent_centers,
                current_centers=incumbent_centers,
                residual=residual,
                step_scale=0.25,
                branches=branches,
                branch_duals=self.branch_duals.detach().clone(),
                boundary_duals=self.boundary_duals.detach().clone(),
                density_duals=self.density_duals.detach().clone(),
                rho=float(self.config.rho),
                eta=0.02,
                alpha=float(self.config.coordinate_lr),
                pd_steps=int(self.config.coordinate_steps),
                pressure={"branch": 1.0, "density": 1.0, "boundary": 1.0},
                branch_pressure_values=None,
                boundary_pressure_values=None,
                density_pressure_values=None,
                soft_branch_weights=None,
                soft_tau=float(self.config.soft_tau),
            )
        finally:
            self.centers = original_centers
            self.active_pairs = original_active_pairs
            self.active_pair_ages = original_active_pair_ages
            self.branch_duals = original_branch_duals
            self.boundary_duals = original_boundary_duals
            self.density_duals = original_density_duals

    def _remap_branch_duals(self, old_pairs, old_duals, new_pairs):
        new_duals = torch.zeros((new_pairs.shape[0], 4), dtype=old_duals.dtype, device=old_duals.device)
        if old_pairs.numel() == 0 or new_pairs.numel() == 0:
            return new_duals
        old_pairs = canonicalize_pairs(old_pairs)
        new_pairs = canonicalize_pairs(new_pairs)
        base = int(torch.max(torch.cat([old_pairs.reshape(-1), new_pairs.reshape(-1)])).item()) + 1
        old_pairs, old_duals = sort_pairs_and_payload(old_pairs, old_duals, base=base)
        old_keys = pair_sort_keys(old_pairs, base=base)
        new_keys = pair_sort_keys(new_pairs, base=base)
        positions = torch.searchsorted(old_keys, new_keys)
        clamped = torch.clamp(positions, max=old_keys.shape[0] - 1)
        matched = (positions < old_keys.shape[0]) & (old_keys[clamped] == new_keys)
        if torch.any(matched):
            new_duals[matched] = old_duals[positions[matched]]
        return new_duals

    def _save_candidate(self, centers, candidate_step=None, candidate_meta=None):
        score = self._score_centers(centers)
        self.saved_candidates.append((score, centers.detach().clone()))
        if self.best_score is None or self._candidate_rank_key(score) < self._candidate_rank_key(self.best_score):
            self.best_score = dict(score)
            self.best_centers = centers.detach().clone()
            self.best_candidate_step = int(self.step_index if candidate_step is None else candidate_step)
            self.steps_since_best = 0
            meta = candidate_meta or {}
            self.best_candidate_phase = str(meta.get("phase", getattr(self, "phase", PlacementPhase.DISCOVER)))
            self.best_candidate_discover_mode = str(meta.get("discover_mode", getattr(self, "discover_mode", DISCOVER_MODE_NAMES[0])))
            self.best_candidate_refine_variant = str(meta.get("winning_refine_variant", "incumbent_hold"))
        else:
            current_step = int(self.step_index if candidate_step is None else candidate_step)
            self.steps_since_best = max(current_step - int(self.best_candidate_step), 0)
        return score

    def _score_centers(self, centers):
        current = write_positions(self.cell_features, centers)
        pairs = exact_overlap_pairs(current)
        overlap_ratio, overlap_cells = overlap_ratio_from_pairs(current.shape[0], pairs)
        return {
            "overlap_ratio": overlap_ratio,
            "overlap_cells": overlap_cells,
            "num_overlap_pairs": int(pairs.shape[0]),
            "normalized_wl": normalized_wirelength(current, self.pin_features, self.edge_list),
        }

    def _stop_reward(self, score, before_score=None):
        feasible = 1.0 if score["overlap_cells"] == 0 else 0.0
        if feasible:
            return self.config.terminal_feasible_bonus - self.config.stop_wirelength_coef * score["normalized_wl"]
        no_progress = 0.0
        if before_score is not None and score["overlap_ratio"] >= before_score["overlap_ratio"]:
            no_progress = float(self.config.stop_no_progress_penalty)
        return (
            -self.config.stop_overlap_penalty * (1.0 + score["overlap_ratio"])
            - no_progress
            - self.config.stop_wirelength_coef * score["normalized_wl"]
        )

    def _phase_request_name(self, action):
        phase_request = getattr(action, "phase_request", None)
        if phase_request is None or not torch.is_tensor(phase_request):
            return "stay"
        request_idx = int(phase_request.detach().item())
        mapping = {0: "stay", 1: "advance", 2: "unlock", 3: "stop"}
        return mapping.get(request_idx, "stay")

    def _is_refine_ready(self, score):
        return (
            float(score["overlap_ratio"]) <= float(self.config.refine_overlap_threshold)
            and int(score["num_overlap_pairs"]) <= int(self.config.refine_pairs_threshold)
        )

    def _is_refine_entry_ready(self, score):
        return self._is_refine_ready(score)

    def _is_late_legalize_ready(self, score):
        return (
            float(score["overlap_ratio"]) <= float(self.config.late_legalize_overlap_threshold)
            and int(score["num_overlap_pairs"]) <= int(self.config.late_legalize_pairs_threshold)
        )

    def _accept_refine_candidate(self, candidate_score, incumbent_score):
        if float(candidate_score["overlap_ratio"]) > float(incumbent_score["overlap_ratio"]) + float(self.config.refine_regression_overlap):
            return False, "overlap_regression"
        if int(candidate_score["num_overlap_pairs"]) > int(incumbent_score["num_overlap_pairs"]) + int(self.config.refine_regression_pairs):
            return False, "pair_regression"
        if float(candidate_score["normalized_wl"]) > float(incumbent_score["normalized_wl"]) + float(self.config.refine_wirelength_regression_epsilon):
            return False, "wirelength_regression"
        return True, ""

    def _refine_current_matches_or_beats_incumbent(self, current_score, incumbent_score):
        if float(current_score["overlap_ratio"]) > float(incumbent_score["overlap_ratio"]) + float(self.config.refine_regression_overlap):
            return False
        if int(current_score["num_overlap_pairs"]) > int(incumbent_score["num_overlap_pairs"]) + int(self.config.refine_regression_pairs):
            return False
        if float(current_score["normalized_wl"]) > float(incumbent_score["normalized_wl"]) + float(self.config.refine_wirelength_regression_epsilon):
            return False
        return True

    def _should_auto_stop_refine(self, *, current_score, incumbent_score, incumbent_improved, steps_since_best_before, refine_rejected, continuation_risk=0.0):
        if not self._is_refine_ready(incumbent_score):
            return False
        if bool(incumbent_improved):
            return False
        if self._refine_current_matches_or_beats_incumbent(current_score, incumbent_score):
            return False
        material_gap = self._continuation_margin_met(current_score=current_score, best_score=incumbent_score)
        if not material_gap:
            return False
        stale_enough = int(steps_since_best_before) >= int(self.config.refine_auto_stop_stale_steps)
        return bool(refine_rejected) or stale_enough or float(continuation_risk) >= float(self.config.continuation_return_best_bias)

    def _should_rollback_refine_to_incumbent(self, *, current_score, incumbent_score, incumbent_improved, continuation_risk=0.0):
        if not self._is_refine_ready(incumbent_score):
            return False
        if bool(incumbent_improved):
            return False
        material_gap = self._continuation_margin_met(current_score=current_score, best_score=incumbent_score)
        if not material_gap:
            return False
        if float(continuation_risk) >= float(self.config.continuation_return_best_bias):
            return True
        return not self._refine_current_matches_or_beats_incumbent(current_score, incumbent_score)

    def _phase_reset_retain(self, phase_before, phase_after):
        if phase_before == phase_after:
            return 1.0
        if phase_after == PlacementPhase.UNLOCK:
            return 0.25
        if phase_before == PlacementPhase.DISCOVER and phase_after == PlacementPhase.LEGALIZE:
            return 0.75
        if phase_before == PlacementPhase.LEGALIZE and phase_after == PlacementPhase.REFINE:
            return 0.50
        if phase_before == PlacementPhase.UNLOCK and phase_after == PlacementPhase.LEGALIZE:
            return 0.75
        if phase_before == PlacementPhase.REFINE and phase_after == PlacementPhase.LEGALIZE:
            return 0.75
        return 1.0

    def _set_phase(self, next_phase, best_score, best_centers):
        if next_phase == self.phase:
            self.phase_step += 1
            if next_phase == PlacementPhase.UNLOCK:
                self.unlock_remaining_steps = max(int(self.unlock_remaining_steps) - 1, 0)
            elif next_phase == PlacementPhase.LEGALIZE and int(self.discover_mode_carry_steps) > 0:
                self.discover_mode_carry_steps = max(int(self.discover_mode_carry_steps) - 1, 0)
            else:
                self.unlock_remaining_steps = 0
            return False
        previous_phase = self.phase
        self.phase = str(next_phase)
        self.phase_step = 0
        self.phase_entry_best_score = dict(best_score)
        self.phase_entry_best_centers = best_centers.detach().clone()
        if self.phase == PlacementPhase.UNLOCK:
            self.unlock_remaining_steps = int(self.config.unlock_horizon)
            self.unlock_anchor_centers = self.centers.detach().clone()
            self.unlock_anchor_score = self._score_centers(self.unlock_anchor_centers)
        else:
            self.unlock_remaining_steps = 0
        if self.phase != PlacementPhase.UNLOCK:
            self.unlock_window_cells = torch.empty((0,), dtype=torch.long, device=self.centers.device)
        if previous_phase == PlacementPhase.DISCOVER and self.phase == PlacementPhase.LEGALIZE:
            self.discover_mode_carry_steps = int(self.config.discover_mode_legalize_carry_steps)
        elif self.phase != PlacementPhase.LEGALIZE:
            self.discover_mode_carry_steps = 0
        return True

    def _advance_phase(self, *, action, phase_before, phase_request_name, current_score, best_before, best_after):
        next_phase = str(phase_before)
        reason = "phase_stay"
        refine_ready = self._is_refine_ready(best_after)
        refine_entry_ready = self._is_refine_entry_ready(best_after)
        discover_exit = (
            float(best_after["overlap_ratio"]) <= float(self.config.discover_exit_overlap)
            or int(self.stagnation_steps) >= int(self.config.discover_patience)
        )
        unlock_ready = (
            phase_before == PlacementPhase.LEGALIZE
            and int(self.legalize_stall_steps) >= int(self.config.unlock_patience)
            and int(self.legalize_pair_stall_steps) >= 1
            and float(best_after["overlap_ratio"]) > float(self.config.refine_overlap_threshold)
            and not refine_ready
        )
        refine_regressed = (
            phase_before == PlacementPhase.REFINE
            and (
                float(current_score["overlap_ratio"]) > float(best_before["overlap_ratio"]) + float(self.config.refine_regression_overlap)
                or int(current_score["num_overlap_pairs"]) > int(best_before["num_overlap_pairs"]) + int(self.config.refine_regression_pairs)
            )
        )
        if phase_before == PlacementPhase.UNLOCK:
            if int(self.unlock_remaining_steps) <= 1:
                next_phase = PlacementPhase.LEGALIZE
                reason = "unlock_complete"
        elif refine_regressed:
            next_phase = PlacementPhase.LEGALIZE
            reason = "refine_regression"
        elif unlock_ready and phase_request_name == "unlock":
            next_phase = PlacementPhase.UNLOCK
            reason = "unlock_request_after_stagnation"
        elif phase_before == PlacementPhase.DISCOVER and discover_exit:
            next_phase = PlacementPhase.LEGALIZE
            reason = "discover_exit"
        elif (
            phase_before == PlacementPhase.LEGALIZE
            and refine_entry_ready
            and int(self.legal_streak) >= int(self.config.legal_streak_required)
            and phase_request_name in {"advance", "stay"}
        ):
            next_phase = PlacementPhase.REFINE
            reason = "refine_entry"
        transitioned = self._set_phase(next_phase, best_after, self.best_centers)
        return transitioned, reason

    def _select_unlock_window(self, action):
        if self.active_pairs.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.centers.device)
        unlock_source_index = getattr(action, "unlock_source_index", None)
        unlock_radius_index = getattr(action, "unlock_radius_index", None)
        if unlock_source_index is None or unlock_radius_index is None:
            return torch.empty((0,), dtype=torch.long, device=self.centers.device)
        source_idx = int(torch.clamp(unlock_source_index.detach().to(device=self.centers.device), min=0, max=max(self.active_pairs.shape[0] - 1, 0)).item())
        radius_idx = int(torch.clamp(unlock_radius_index.detach().to(device=self.centers.device), min=0, max=2).item())
        pair = self.active_pairs[source_idx]
        seed_cells = pair.long().unique()
        center = self.centers[seed_cells].mean(dim=0)
        radius_mult = [1.5, 2.5, 4.0][radius_idx]
        radius = float(radius_mult) * float(self.length_scale)
        distances = torch.norm(self.centers - center.unsqueeze(0), dim=1)
        window = torch.where(distances <= radius)[0]
        if window.numel() == 0:
            window = seed_cells
        max_cells = max(int(self.config.unlock_window_max_cells), 1)
        if window.numel() > max_cells:
            nearest = torch.argsort(distances[window], descending=False)[:max_cells]
            window = window[nearest]
        return window.long()
