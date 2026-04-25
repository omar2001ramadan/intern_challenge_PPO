"""CMDP environment for ordering PPO placement."""

import math
from dataclasses import dataclass

import torch
import torch.optim as optim

from active_set import build_initial_active_pairs, canonicalize_pairs, update_active_pair_cache
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
from ordering_policy import build_graph_state, hierarchical_active_branch_weights
from primal_dual import phr_dual_update, phr_inequality_penalty


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
    exact_overlap_reward_coef: float = 1.0
    exact_wirelength_reward_coef: float = 0.20
    lag_reward_coef: float = 1.0
    exact_overlap_regression_coef: float = 8.0
    exact_wirelength_gate_epsilon: float = 0.002
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
    enable_residual_flow: bool = True
    enable_phr_layer: bool = True
    enable_exact_audit: bool = True
    enable_density: bool = True
    fixed_pd_controls: bool = False
    ordering_representation: str = "sequence_pair"
    branch_mode: str = "ordering"
    al_mode: str = "signed_phr"


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


class PlacementOrderingEnv:
    """One placement episode driven by global ordering actions."""

    def __init__(self, cell_features, pin_features, edge_list, config=None):
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
        self.step_index = 0
        self._save_candidate(self.centers)

    def graph_state(self, memory=None):
        density_g, assignment = self._density_constraints(self.centers)
        current_score = self._score_centers(self.centers)
        overlap_ratio = torch.tensor(
            current_score["overlap_ratio"],
            dtype=self.centers.dtype,
            device=self.centers.device,
        )
        if self.config.enable_stop_gate and current_score["overlap_ratio"] > self.config.stop_gate_overlap_threshold:
            stop_logit_bias = -float(self.config.stop_gate_penalty)
        else:
            stop_logit_bias = 0.0
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
            stop_logit_bias=stop_logit_bias,
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

    def step_action(self, action, entropy=None, soft_branch_weights=None, soft_tau=None):
        """Apply the full policy-conditioned transition from the proposal."""
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
        step_scale = float(action.step_scale.detach().item())
        residual = torch.tanh(action.residual_flow.detach()) * (step_scale * self.length_scale)
        if not self.config.enable_residual_flow:
            residual = torch.zeros_like(residual)
        base_centers = self.centers + residual

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

        after_score = self._score_centers(next_centers)
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
            "retention_horizon": int(self.config.active_pair_retention),
        }
        if self.config.enable_exact_audit:
            audit_info = self._audit_active_set(
                action=action,
                cluster_ids=action.cluster_ids,
                pair_emphasis=float(action.pair_emphasis.detach().item()),
            )
        score = self._save_candidate(self.centers)
        reward, reward_terms = self._aligned_reward(
            lag_before,
            lag_after,
            before_score,
            score,
            after_logs,
            audit_info,
            entropy_term,
            movement_penalty,
        )

        self.step_index += 1
        stopped = bool(action.stop.detach().item() >= 0.5)
        stop_probability = float(getattr(action, "stop_probability", action.stop).detach().item())
        stop_logit_bias = float(getattr(action, "stop_logit_bias", action.stop.new_zeros(())).detach().item())
        stop_overlap = score["overlap_ratio"] if stopped else 0.0
        false_stop = bool(stopped and score["overlap_ratio"] > 0.0)
        done = stopped or self.step_index >= self.config.horizon
        if stopped:
            reward += self._stop_reward(score, before_score)
        elif done:
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
            "pd_steps": pd_steps,
            "rho": rho,
            "eta": eta,
            "alpha": alpha,
            "step_scale": step_scale,
            "pair_emphasis": float(action.pair_emphasis.detach().item()),
            "tau": active_soft_tau,
            "stop": stopped,
            "stop_probability": stop_probability,
            "stop_logit_bias": stop_logit_bias,
            "stop_gated": stop_logit_bias < 0.0,
            "stop_overlap": stop_overlap,
            "false_stop": false_stop,
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
            **audit_info,
        }
        return reward, done, info

    def _aligned_reward(
        self,
        lag_before,
        lag_after,
        before_score,
        after_score,
        after_logs,
        audit_info,
        entropy_term,
        movement_penalty,
    ):
        lag_delta = (lag_before.detach() - lag_after.detach()).item()
        overlap_delta = before_score["overlap_ratio"] - after_score["overlap_ratio"]
        wirelength_delta = before_score["normalized_wl"] - after_score["normalized_wl"]
        overlap_regression = max(-overlap_delta, 0.0)
        gated_wire_delta = wirelength_delta if overlap_delta >= -self.config.exact_wirelength_gate_epsilon else 0.0
        exact_reward = (
            self.config.exact_overlap_reward_coef * overlap_delta
            - self.config.exact_overlap_regression_coef * overlap_regression
            + self.config.exact_wirelength_reward_coef * gated_wire_delta
        )
        violation_penalty = self.config.branch_violation_penalty_coef * after_logs["branch_violation"]
        missed_pair_penalty = self.config.missed_pair_penalty_coef * float(audit_info.get("missed_pairs", 0))
        reward = (
            self.config.lag_reward_coef * lag_delta
            + exact_reward
            + self.config.entropy_reward_coef * entropy_term
            - movement_penalty
            - violation_penalty
            - missed_pair_penalty
        )
        return reward, {
            "lag_reward": self.config.lag_reward_coef * lag_delta,
            "exact_reward": exact_reward,
            "overlap_delta": overlap_delta,
            "wirelength_delta": wirelength_delta,
            "gated_wirelength_delta": gated_wire_delta,
            "overlap_regression_penalty": self.config.exact_overlap_regression_coef * overlap_regression,
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
        if self.config.branch_mode == "independent_pair" and hasattr(action, "pair_branch_choices"):
            return action.pair_branch_choices.to(dtype=torch.long, device=self.centers.device)
        if self.config.ordering_representation == "dag" and hasattr(action, "dag_axis"):
            branches = self._induce_dag_branches(action)
        else:
            branches = induce_branches_from_sequence_pair(action.seq_plus, action.seq_minus, self.active_pairs)

        branches = self._apply_macro_branches(action, branches)
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
        return min(
            self.saved_candidates,
            key=lambda item: (
                item[0]["overlap_cells"],
                item[0]["overlap_ratio"],
                item[0]["normalized_wl"],
                item[0]["num_overlap_pairs"],
            ),
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
    ):
        positions = (self.centers if initial_centers is None else initial_centers).clone().detach()
        positions.requires_grad_(True)
        optimizer = optim.Adam([positions], lr=self.config.coordinate_lr if coordinate_lr is None else float(coordinate_lr))
        anchor = self.centers.detach() if anchor_centers is None else anchor_centers.detach()
        steps = self.config.coordinate_steps if coordinate_steps is None else max(int(coordinate_steps), 1)
        for _ in range(steps):
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
            torch.nn.utils.clip_grad_norm_([positions], max_norm=10.0)
            optimizer.step()
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
        audit_pressure_scale = self._audit_pressure_scale(inactive_missed_count)
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
        n = int(self.cell_features.shape[0])
        exact_keys = exact_pairs[:, 0] * n + exact_pairs[:, 1]
        active_keys = active_pairs[:, 0] * n + active_pairs[:, 1]
        if hasattr(torch, "isin"):
            inactive_mask = ~torch.isin(exact_keys, active_keys)
            return exact_pairs[inactive_mask]
        active_set = set(active_keys.detach().cpu().tolist())
        keep = [int(key) not in active_set for key in exact_keys.detach().cpu().tolist()]
        if not any(keep):
            return exact_pairs.new_empty((0, 2))
        return exact_pairs[torch.tensor(keep, dtype=torch.bool, device=exact_pairs.device)]

    def _audit_pressure_scale(self, inactive_missed_count):
        target = float(self.config.audit_missed_target)
        if target <= 0.0:
            return 1.0
        excess = max((float(inactive_missed_count) - target) / (target + 1e-6), 0.0)
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
        for cluster in valid_clusters.detach().cpu().tolist():
            members = torch.where(cluster_ids == int(cluster))[0]
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

    def _remap_branch_duals(self, old_pairs, old_duals, new_pairs):
        new_duals = torch.zeros((new_pairs.shape[0], 4), dtype=old_duals.dtype, device=old_duals.device)
        if old_pairs.numel() == 0 or new_pairs.numel() == 0:
            return new_duals
        old_cpu = canonicalize_pairs(old_pairs).detach().cpu().tolist()
        index = {tuple(pair): idx for idx, pair in enumerate(old_cpu)}
        for new_idx, pair in enumerate(canonicalize_pairs(new_pairs).detach().cpu().tolist()):
            old_idx = index.get(tuple(pair))
            if old_idx is not None:
                new_duals[new_idx] = old_duals[old_idx]
        return new_duals

    def _save_candidate(self, centers):
        score = self._score_centers(centers)
        self.saved_candidates.append((score, centers.detach().clone()))
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
