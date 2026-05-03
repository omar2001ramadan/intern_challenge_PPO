"""Graph-conditioned policy for policy-owned primal-dual placement PPO."""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LEGACY_NODE_FEATURE_DIM = 16
LEGACY_PAIR_FEATURE_DIM = 10
NODE_FEATURE_DIM = 24
PAIR_FEATURE_DIM = 16
CANDIDATE_RANK_FEATURE_DIM = 6
CHOOSER_CANDIDATE_FEATURE_DIM = 16
CHOOSER_OVERLAP_MARGIN_IDX = 10
CHOOSER_PAIR_MARGIN_IDX = 11
CHOOSER_WIRE_MARGIN_IDX = 12
CHOOSER_LARGE_SWAP_CONFLICT_IDX = 13
CHOOSER_LARGE_CASE_IDX = 14
CHOOSER_ROLLOUT_SOURCE_IDX = 15
PD_K_CHOICES = (1, 2, 4, 8, 12, 16)
PHASE_NAMES = ("DISCOVER", "LEGALIZE", "REFINE", "UNLOCK")
PHASE_TO_INDEX = {name: idx for idx, name in enumerate(PHASE_NAMES)}
PHASE_REQUEST_NAMES = ("stay", "advance", "unlock", "stop")
PHASE_REQUEST_TO_INDEX = {name: idx for idx, name in enumerate(PHASE_REQUEST_NAMES)}
UNLOCK_RADIUS_CHOICES = (1.5, 2.5, 4.0)
CONTROL_NAMES = (
    "step_scale",
    "rho",
    "eta",
    "alpha",
    "branch_pressure",
    "density_pressure",
    "boundary_pressure",
    "pair_emphasis",
    "tau",
)
DISCOVER_MODE_NAMES = ("balanced", "spread_first", "wire_first", "macro_clearance")
DISCOVER_MODE_TO_INDEX = {name: idx for idx, name in enumerate(DISCOVER_MODE_NAMES)}
REFINE_VARIANT_NAMES = (
    "incumbent_hold",
    "wire_grad_local",
    "projection_local",
    "swap_or_reassign_local",
)
REFINE_VARIANT_TO_INDEX = {name: idx for idx, name in enumerate(REFINE_VARIANT_NAMES)}
REFINE_READY_OVERLAP_THRESHOLD = 0.25
REFINE_READY_PAIR_THRESHOLD = 8.0

ORDERING_SCORE_CLAMP = 40.0
HIDDEN_STATE_CLAMP = 1.0e4
LOGPROB_CLAMP = 1.0e4


def _sanitize_tensor(tensor, *, pos=HIDDEN_STATE_CLAMP, neg=None):
    if neg is None:
        neg = -float(pos)
    return torch.nan_to_num(tensor, nan=0.0, posinf=float(pos), neginf=float(neg))


def _stabilize_ordering_scores(scores):
    scores = _sanitize_tensor(scores, pos=ORDERING_SCORE_CLAMP, neg=-ORDERING_SCORE_CLAMP)
    return torch.clamp(scores, min=-ORDERING_SCORE_CLAMP, max=ORDERING_SCORE_CLAMP)


@dataclass
class SequencePairAction:
    seq_plus: torch.Tensor
    seq_minus: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    plus_scores: torch.Tensor
    minus_scores: torch.Tensor


@dataclass
class PlacementPolicyAction:
    """Composite stochastic action from the unified PPO proposal."""

    seq_plus: torch.Tensor
    seq_minus: torch.Tensor
    macro_seq_plus: torch.Tensor
    macro_seq_minus: torch.Tensor
    macro_cell_indices: torch.Tensor
    cluster_seq_plus: torch.Tensor
    cluster_seq_minus: torch.Tensor
    residual_flow: torch.Tensor
    residual_local: torch.Tensor
    global_flow_latent: torch.Tensor
    control_raw: torch.Tensor
    incumbent_mix_raw: torch.Tensor
    k_index: torch.Tensor
    stop: torch.Tensor
    stop_probability: torch.Tensor
    stop_logit_bias: torch.Tensor
    cluster_ids: torch.Tensor
    dag_axis: torch.Tensor
    pair_branch_choices: torch.Tensor
    phase_request: torch.Tensor
    phase_request_logits: torch.Tensor
    unlock_source_index: torch.Tensor
    unlock_source_logits: torch.Tensor
    unlock_radius_index: torch.Tensor
    unlock_radius_logits: torch.Tensor
    ordering_representation: str
    branch_mode: str
    phase_name: str
    enable_clusters: bool
    enable_stop: bool
    enable_unlock: bool
    step_scale: torch.Tensor
    rho: torch.Tensor
    eta: torch.Tensor
    alpha: torch.Tensor
    branch_pressure: torch.Tensor
    density_pressure: torch.Tensor
    boundary_pressure: torch.Tensor
    pair_emphasis: torch.Tensor
    tau: torch.Tensor
    incumbent_mix: torch.Tensor
    memory: torch.Tensor
    next_memory: torch.Tensor
    branch_pressure_raw: torch.Tensor
    boundary_pressure_raw: torch.Tensor
    density_pressure_raw: torch.Tensor
    branch_pressure_values: torch.Tensor
    boundary_pressure_values: torch.Tensor
    density_pressure_values: torch.Tensor
    group_logprobs: dict
    group_entropies: dict
    group_token_counts: dict
    value: torch.Tensor
    plus_scores: torch.Tensor
    minus_scores: torch.Tensor
    macro_plus_scores: torch.Tensor
    macro_minus_scores: torch.Tensor
    cluster_plus_scores: torch.Tensor
    cluster_minus_scores: torch.Tensor
    dag_axis_logits: torch.Tensor
    pair_branch_logits: torch.Tensor
    cluster_logits: torch.Tensor

    @property
    def logprob(self):
        return sum(self.group_logprobs.values())

    @property
    def entropy(self):
        return sum(self.group_entropies.values())

    @property
    def pd_steps(self):
        return int(PD_K_CHOICES[int(self.k_index.detach().item())])


def build_graph_state(
    cell_features,
    pin_features,
    edge_list,
    active_pairs=None,
    branch_duals=None,
    boundary_duals=None,
    density_duals=None,
    wirelength_grad=None,
    density_pressure=None,
    memory=None,
    exact_overlap_ratio=None,
    current_normalized_wl=None,
    current_num_overlap_pairs=None,
    incumbent_centers=None,
    incumbent_overlap_ratio=None,
    incumbent_normalized_wl=None,
    incumbent_num_overlap_pairs=None,
    steps_since_best=None,
    stop_logit_bias=None,
    phase_name="DISCOVER",
    phase_index=None,
    phase_step=None,
    legal_streak=None,
    stagnation_steps=None,
    unlock_remaining_steps=None,
    late_legalize_mode=None,
    discover_mode_name="balanced",
    discover_mode_index=None,
    discover_mode_carry_steps=None,
    phase_entry_overlap=None,
    phase_entry_wirelength=None,
    ordering_representation="sequence_pair",
    branch_mode="ordering",
    enable_clusters=True,
    enable_stop=True,
    enable_incumbent_state=True,
    enable_incumbent_action=True,
    cleanup_feature_vector=None,
    case_descriptor=None,
    continuation_risk=None,
):
    """Build a policy graph dictionary from placement state tensors."""
    if active_pairs is None:
        active_pairs = torch.empty((0, 2), dtype=torch.long, device=cell_features.device)
    if branch_duals is None:
        branch_duals = torch.zeros((active_pairs.shape[0], 4), dtype=cell_features.dtype, device=cell_features.device)
    if boundary_duals is None:
        boundary_duals = torch.zeros((cell_features.shape[0], 4), dtype=cell_features.dtype, device=cell_features.device)
    if density_duals is None:
        density_duals = torch.zeros(0, dtype=cell_features.dtype, device=cell_features.device)
    if wirelength_grad is None:
        wirelength_grad = torch.zeros((cell_features.shape[0], 2), dtype=cell_features.dtype, device=cell_features.device)
    if density_pressure is None:
        density_pressure = torch.zeros(cell_features.shape[0], dtype=cell_features.dtype, device=cell_features.device)
    if memory is None:
        memory = torch.zeros(0, dtype=cell_features.dtype, device=cell_features.device)
    if exact_overlap_ratio is None:
        exact_overlap_ratio = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if current_normalized_wl is None:
        current_normalized_wl = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if current_num_overlap_pairs is None:
        current_num_overlap_pairs = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if incumbent_centers is None:
        incumbent_centers = cell_features[:, 2:4]
    if incumbent_overlap_ratio is None:
        incumbent_overlap_ratio = exact_overlap_ratio
    if incumbent_normalized_wl is None:
        incumbent_normalized_wl = current_normalized_wl
    if incumbent_num_overlap_pairs is None:
        incumbent_num_overlap_pairs = current_num_overlap_pairs
    if steps_since_best is None:
        steps_since_best = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if stop_logit_bias is None:
        stop_logit_bias = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if phase_index is None:
        phase_index = PHASE_TO_INDEX.get(str(phase_name).upper(), 0)
    if legal_streak is None:
        legal_streak = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if phase_step is None:
        phase_step = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if stagnation_steps is None:
        stagnation_steps = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if unlock_remaining_steps is None:
        unlock_remaining_steps = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if late_legalize_mode is None:
        late_legalize_mode = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if discover_mode_index is None:
        discover_mode_index = DISCOVER_MODE_TO_INDEX.get(str(discover_mode_name).lower(), 0)
    if discover_mode_carry_steps is None:
        discover_mode_carry_steps = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    if phase_entry_overlap is None:
        phase_entry_overlap = exact_overlap_ratio
    if phase_entry_wirelength is None:
        phase_entry_wirelength = current_normalized_wl
    if cleanup_feature_vector is None:
        cleanup_feature_vector = torch.zeros((5,), dtype=cell_features.dtype, device=cell_features.device)
    if case_descriptor is None:
        case_descriptor = torch.zeros((6,), dtype=cell_features.dtype, device=cell_features.device)
    if continuation_risk is None:
        continuation_risk = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
    return {
        "cell_features": cell_features.detach(),
        "pin_features": pin_features.detach(),
        "edge_list": edge_list.detach(),
        "active_pairs": active_pairs.detach(),
        "branch_duals": branch_duals.detach(),
        "boundary_duals": boundary_duals.detach(),
        "density_duals": density_duals.detach(),
        "wirelength_grad": wirelength_grad.detach(),
        "density_pressure": density_pressure.detach(),
        "memory": memory.detach(),
        "exact_overlap_ratio": exact_overlap_ratio.detach() if torch.is_tensor(exact_overlap_ratio) else torch.tensor(float(exact_overlap_ratio), dtype=cell_features.dtype, device=cell_features.device),
        "current_normalized_wl": current_normalized_wl.detach() if torch.is_tensor(current_normalized_wl) else torch.tensor(float(current_normalized_wl), dtype=cell_features.dtype, device=cell_features.device),
        "current_num_overlap_pairs": current_num_overlap_pairs.detach() if torch.is_tensor(current_num_overlap_pairs) else torch.tensor(float(current_num_overlap_pairs), dtype=cell_features.dtype, device=cell_features.device),
        "incumbent_centers": incumbent_centers.detach(),
        "incumbent_overlap_ratio": incumbent_overlap_ratio.detach() if torch.is_tensor(incumbent_overlap_ratio) else torch.tensor(float(incumbent_overlap_ratio), dtype=cell_features.dtype, device=cell_features.device),
        "incumbent_normalized_wl": incumbent_normalized_wl.detach() if torch.is_tensor(incumbent_normalized_wl) else torch.tensor(float(incumbent_normalized_wl), dtype=cell_features.dtype, device=cell_features.device),
        "incumbent_num_overlap_pairs": incumbent_num_overlap_pairs.detach() if torch.is_tensor(incumbent_num_overlap_pairs) else torch.tensor(float(incumbent_num_overlap_pairs), dtype=cell_features.dtype, device=cell_features.device),
        "steps_since_best": steps_since_best.detach() if torch.is_tensor(steps_since_best) else torch.tensor(float(steps_since_best), dtype=cell_features.dtype, device=cell_features.device),
        "stop_logit_bias": stop_logit_bias.detach() if torch.is_tensor(stop_logit_bias) else torch.tensor(float(stop_logit_bias), dtype=cell_features.dtype, device=cell_features.device),
        "phase_name": str(phase_name),
        "phase_index": torch.tensor(int(phase_index), dtype=torch.long, device=cell_features.device),
        "phase_step": phase_step.detach() if torch.is_tensor(phase_step) else torch.tensor(float(phase_step), dtype=cell_features.dtype, device=cell_features.device),
        "legal_streak": legal_streak.detach() if torch.is_tensor(legal_streak) else torch.tensor(float(legal_streak), dtype=cell_features.dtype, device=cell_features.device),
        "stagnation_steps": stagnation_steps.detach() if torch.is_tensor(stagnation_steps) else torch.tensor(float(stagnation_steps), dtype=cell_features.dtype, device=cell_features.device),
        "unlock_remaining_steps": unlock_remaining_steps.detach() if torch.is_tensor(unlock_remaining_steps) else torch.tensor(float(unlock_remaining_steps), dtype=cell_features.dtype, device=cell_features.device),
        "late_legalize_mode": late_legalize_mode.detach() if torch.is_tensor(late_legalize_mode) else torch.tensor(float(late_legalize_mode), dtype=cell_features.dtype, device=cell_features.device),
        "discover_mode_name": str(discover_mode_name),
        "discover_mode_index": torch.tensor(int(discover_mode_index), dtype=torch.long, device=cell_features.device),
        "discover_mode_carry_steps": discover_mode_carry_steps.detach() if torch.is_tensor(discover_mode_carry_steps) else torch.tensor(float(discover_mode_carry_steps), dtype=cell_features.dtype, device=cell_features.device),
        "phase_entry_overlap": phase_entry_overlap.detach() if torch.is_tensor(phase_entry_overlap) else torch.tensor(float(phase_entry_overlap), dtype=cell_features.dtype, device=cell_features.device),
        "phase_entry_wirelength": phase_entry_wirelength.detach() if torch.is_tensor(phase_entry_wirelength) else torch.tensor(float(phase_entry_wirelength), dtype=cell_features.dtype, device=cell_features.device),
        "ordering_representation": str(ordering_representation),
        "branch_mode": str(branch_mode),
        "enable_clusters": bool(enable_clusters),
        "enable_stop": bool(enable_stop),
        "enable_incumbent_state": bool(enable_incumbent_state),
        "enable_incumbent_action": bool(enable_incumbent_action),
        "cleanup_feature_vector": cleanup_feature_vector.detach(),
        "case_descriptor": case_descriptor.detach(),
        "continuation_risk": continuation_risk.detach() if torch.is_tensor(continuation_risk) else torch.tensor(float(continuation_risk), dtype=cell_features.dtype, device=cell_features.device),
    }


def graph_to_device(graph, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in graph.items()}


def _graph_bool(graph, key, default):
    value = graph.get(key, default)
    if torch.is_tensor(value):
        return bool(value.detach().item())
    return bool(value)


def _graph_str(graph, key, default):
    value = graph.get(key, default)
    return str(value)


def _graph_scalar(graph, key, default, *, dtype, device):
    value = graph.get(key, default)
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device).reshape(())
    return torch.tensor(float(value), dtype=dtype, device=device)


def rollout_memory_config_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return {
        "memory_reset_mode": str(config.get("memory_reset_mode", "none")),
        "memory_reset_retain": float(config.get("memory_reset_retain", 1.0)),
        "memory_reset_min_overlap_gain": float(config.get("memory_reset_min_overlap_gain", 0.03)),
        "memory_reset_min_pair_gain_count": float(config.get("memory_reset_min_pair_gain_count", 2.0)),
        "memory_reset_min_steps_since_best": int(config.get("memory_reset_min_steps_since_best", 2)),
    }


def apply_rollout_memory_policy(
    next_memory,
    initial_memory=None,
    *,
    reset_mode="none",
    reset_retain=1.0,
    incumbent_improved=False,
    best_overlap_delta=0.0,
    best_pair_delta_count=0.0,
    steps_since_best_before=0,
    phase_transition=False,
    phase_transition_reason="",
    phase_reset_retain=None,
    event_reset=False,
    event_reset_reason="",
    event_reset_retain=None,
    min_overlap_gain=0.03,
    min_pair_gain_count=2.0,
    min_steps_since_best=2,
):
    next_memory = next_memory.detach()
    retain = max(0.0, min(float(reset_retain), 1.0))
    reset_applied = False
    reset_reason = "none"
    policy_mode = str(reset_mode or "none")
    overlap_gain = float(best_overlap_delta)
    pair_gain_count = float(best_pair_delta_count)
    stale_steps = int(max(float(steps_since_best_before), 0.0))

    material_improvement = (
        overlap_gain >= float(min_overlap_gain)
        or pair_gain_count >= float(min_pair_gain_count)
    )
    stale_improvement = stale_steps >= int(max(int(min_steps_since_best), 0))

    should_reset = False
    if policy_mode == "incumbent_improve":
        should_reset = bool(incumbent_improved)
        if should_reset:
            reset_reason = "incumbent_improved"
    elif policy_mode == "incumbent_improve_material":
        should_reset = bool(incumbent_improved) and material_improvement
        if should_reset:
            reset_reason = "incumbent_improved_material"
    elif policy_mode == "incumbent_improve_stale":
        should_reset = bool(incumbent_improved) and stale_improvement
        if should_reset:
            reset_reason = "incumbent_improved_stale"
    elif policy_mode == "incumbent_improve_material_or_stale":
        should_reset = bool(incumbent_improved) and (material_improvement or stale_improvement)
        if should_reset:
            if material_improvement and stale_improvement:
                reset_reason = "incumbent_improved_material_and_stale"
            elif material_improvement:
                reset_reason = "incumbent_improved_material"
            else:
                reset_reason = "incumbent_improved_stale"

    if bool(phase_transition):
        should_reset = True
        retain = max(0.0, min(float(retain if phase_reset_retain is None else phase_reset_retain), 1.0))
        reset_reason = str(phase_transition_reason or "phase_transition")
    elif bool(event_reset):
        should_reset = True
        retain = max(0.0, min(float(retain if event_reset_retain is None else event_reset_retain), 1.0))
        reset_reason = str(event_reset_reason or "event_reset")

    if should_reset:
        if initial_memory is None or initial_memory.numel() != next_memory.numel():
            anchor = torch.zeros_like(next_memory)
        else:
            anchor = initial_memory.detach().to(dtype=next_memory.dtype, device=next_memory.device)
        next_memory = next_memory * retain + anchor * (1.0 - retain)
        next_memory = _sanitize_tensor(next_memory)
        reset_applied = True

    return next_memory, {
        "memory_reset_applied": reset_applied,
        "memory_reset_reason": reset_reason,
        "memory_reset_mode": policy_mode,
        "memory_reset_retain": retain,
        "memory_reset_material_improvement": bool(material_improvement),
        "memory_reset_stale_improvement": bool(stale_improvement),
        "memory_reset_best_overlap_delta": overlap_gain,
        "memory_reset_best_pair_delta_count": pair_gain_count,
        "memory_reset_steps_since_best_before": stale_steps,
        "memory_reset_phase_transition": bool(phase_transition),
        "memory_reset_event": bool(event_reset),
    }


def plackett_luce_logprob(scores, sequence):
    """Exact log probability for a permutation under Plackett-Luce scores."""
    if sequence.numel() <= 1:
        return scores.sum() * 0.0
    scores = _stabilize_ordering_scores(scores)
    ordered = scores[sequence.long()]
    if ordered.device.type == "mps":
        suffix_logsumexp = torch.empty_like(ordered)
        running = ordered[-1]
        suffix_logsumexp[-1] = running
        for idx in range(int(ordered.numel()) - 2, -1, -1):
            running = torch.logaddexp(ordered[idx], running)
            suffix_logsumexp[idx] = running
    else:
        suffix_logsumexp = torch.logcumsumexp(torch.flip(ordered, dims=[0]), dim=0).flip(0)
    logprob = (ordered - suffix_logsumexp).sum()
    return _sanitize_tensor(logprob, pos=LOGPROB_CLAMP, neg=-LOGPROB_CLAMP)


def categorical_entropy_from_scores(scores):
    if scores.numel() <= 1:
        return scores.sum() * 0.0
    scores = _stabilize_ordering_scores(scores)
    log_probs = F.log_softmax(scores, dim=0)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum()
    return _sanitize_tensor(entropy, pos=LOGPROB_CLAMP, neg=0.0)


def sample_plackett_luce(scores):
    """Sample a permutation via Gumbel top-k."""
    scores = _stabilize_ordering_scores(scores)
    eps = torch.finfo(scores.dtype).eps
    uniform = torch.rand_like(scores).clamp_(eps, 1.0 - eps)
    gumbel = -torch.log(-torch.log(uniform))
    return torch.argsort(scores + gumbel, descending=True)


def normal_logprob(value, mean, log_std):
    log_std = torch.clamp(log_std, min=-5.0, max=2.0)
    var = torch.exp(2.0 * log_std)
    logprob = -0.5 * (((value - mean).square() / var) + 2.0 * log_std + math.log(2.0 * math.pi))
    return _sanitize_tensor(logprob, pos=LOGPROB_CLAMP, neg=-LOGPROB_CLAMP)


def normal_entropy(log_std):
    log_std = torch.clamp(log_std, min=-5.0, max=2.0)
    entropy = log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))
    return _sanitize_tensor(entropy, pos=LOGPROB_CLAMP, neg=0.0)


def lognormal_from_raw(raw, scale=1.0, min_value=0.0, max_value=None):
    value = min_value + scale * torch.exp(torch.clamp(raw, min=-8.0, max=6.0))
    if max_value is not None:
        value = torch.clamp(value, max=float(max_value))
    return value


def bernoulli_logprob(value, logits):
    value = value.to(dtype=logits.dtype)
    return -F.binary_cross_entropy_with_logits(logits, value, reduction="none")


def bernoulli_entropy(logits):
    probs = torch.sigmoid(logits)
    log_p = F.logsigmoid(logits)
    log_not_p = F.logsigmoid(-logits)
    return -(probs * log_p + (1.0 - probs) * log_not_p)


def categorical_logprob(index, logits):
    logits = _sanitize_tensor(logits, pos=ORDERING_SCORE_CLAMP, neg=-ORDERING_SCORE_CLAMP)
    logprob = F.log_softmax(logits, dim=-1).gather(-1, index.long().reshape(1)).squeeze(0)
    return _sanitize_tensor(logprob, pos=LOGPROB_CLAMP, neg=-LOGPROB_CLAMP)


def categorical_entropy(logits):
    logits = _sanitize_tensor(logits, pos=ORDERING_SCORE_CLAMP, neg=-ORDERING_SCORE_CLAMP)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return _sanitize_tensor(entropy, pos=LOGPROB_CLAMP, neg=0.0)


def precedence_probabilities_from_scores(scores):
    """Pairwise soft precedence probabilities from scalar item scores."""
    diff = scores[:, None] - scores[None, :]
    return torch.sigmoid(diff)


def soft_ranks_from_scores(scores, tau=1.0, method="sigmoid"):
    """Differentiable expected ranks for soft ordering relaxations."""
    if scores.numel() <= 1:
        return torch.zeros_like(scores)
    tau = max(float(tau), 1e-4)
    method = str(method).lower()
    if method in {"neuralsort", "neural_sort", "sigmoid"}:
        pair_prob = torch.sigmoid((scores[None, :] - scores[:, None]) / tau)
        return pair_prob.sum(dim=1) - 0.5
    if method in {"gumbel_sinkhorn", "sinkhorn"}:
        n = int(scores.numel())
        positions = torch.linspace(1.0, 0.0, n, dtype=scores.dtype, device=scores.device)
        log_alpha = scores[:, None] * positions[None, :] / tau
        if method == "gumbel_sinkhorn":
            eps = torch.finfo(scores.dtype).eps
            uniform = torch.rand_like(log_alpha).clamp_(eps, 1.0 - eps)
            log_alpha = log_alpha - torch.log(-torch.log(uniform))
        matrix = torch.exp(log_alpha - log_alpha.max())
        for _ in range(12):
            matrix = matrix / torch.clamp(matrix.sum(dim=1, keepdim=True), min=1e-8)
            matrix = matrix / torch.clamp(matrix.sum(dim=0, keepdim=True), min=1e-8)
        pos = torch.arange(n, dtype=scores.dtype, device=scores.device)
        return matrix.matmul(pos)
    raise ValueError(f"Unknown relaxation method: {method}")


def active_branch_weights_from_scores(plus_scores, minus_scores, pairs, relaxation="sigmoid", tau=1.0):
    if pairs.numel() == 0:
        return torch.empty((0, 4), dtype=plus_scores.dtype, device=plus_scores.device)
    i = pairs[:, 0].long()
    j = pairs[:, 1].long()
    if str(relaxation).lower() in {"score", "scores", "sigmoid"}:
        p_plus = torch.sigmoid(plus_scores[i] - plus_scores[j])
        p_minus = torch.sigmoid(minus_scores[i] - minus_scores[j])
    else:
        plus_rank = soft_ranks_from_scores(plus_scores, tau=tau, method=relaxation)
        minus_rank = soft_ranks_from_scores(minus_scores, tau=tau, method=relaxation)
        p_plus = torch.sigmoid((plus_rank[j] - plus_rank[i]) / max(float(tau), 1e-4))
        p_minus = torch.sigmoid((minus_rank[j] - minus_rank[i]) / max(float(tau), 1e-4))
    return torch.stack(
        [
            p_plus * p_minus,
            (1.0 - p_plus) * (1.0 - p_minus),
            p_plus * (1.0 - p_minus),
            (1.0 - p_plus) * p_minus,
        ],
        dim=1,
    )


def soft_branch_weights(plus_scores, minus_scores):
    """Return q_ij(d) branch weights induced by soft ordering probabilities."""
    p_plus = precedence_probabilities_from_scores(plus_scores)
    p_minus = precedence_probabilities_from_scores(minus_scores)
    left = p_plus * p_minus
    right = (1.0 - p_plus) * (1.0 - p_minus)
    below = p_plus * (1.0 - p_minus)
    above = (1.0 - p_plus) * p_minus
    return torch.stack([left, right, below, above], dim=-1)


def active_branch_weights(plus_scores, minus_scores, pairs):
    """Return q_ij(d) only for active pairs to avoid dense N^2 storage."""
    if pairs.numel() == 0:
        return torch.empty((0, 4), dtype=plus_scores.dtype, device=plus_scores.device)
    return active_branch_weights_from_scores(plus_scores, minus_scores, pairs)


def hierarchical_active_branch_weights(action, pairs, relaxation="sigmoid", tau=1.0):
    """Soft branch weights from cell ordering, overridden by cluster ordering where applicable."""
    weights = active_branch_weights_from_scores(action.plus_scores, action.minus_scores, pairs, relaxation=relaxation, tau=tau)
    if (
        pairs.numel() == 0
        or not getattr(action, "enable_clusters", True)
        or not hasattr(action, "cluster_ids")
        or not hasattr(action, "cluster_plus_scores")
        or not hasattr(action, "cluster_minus_scores")
    ):
        return weights
    ci = action.cluster_ids[pairs[:, 0].long()]
    cj = action.cluster_ids[pairs[:, 1].long()]
    use_cluster = (ci >= 0) & (cj >= 0) & (ci != cj)
    if not torch.any(use_cluster):
        return weights
    cluster_pairs = torch.stack([ci[use_cluster], cj[use_cluster]], dim=1)
    weights = weights.clone()
    weights[use_cluster] = active_branch_weights_from_scores(
        action.cluster_plus_scores,
        action.cluster_minus_scores,
        cluster_pairs,
        relaxation=relaxation,
        tau=tau,
    )
    return weights


class OrderingPolicy(nn.Module):
    """Equivariant actor-critic with ordering, flow, PD-control, and stop heads."""

    def __init__(
        self,
        node_feature_dim=NODE_FEATURE_DIM,
        pair_feature_dim=PAIR_FEATURE_DIM,
        hidden_dim=128,
        message_passes=2,
        num_clusters=8,
        global_flow_rank=2,
        enable_incumbent_controls=True,
        chooser_legacy_residual_weight=0.0,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.pair_feature_dim = pair_feature_dim
        self.hidden_dim = hidden_dim
        self.message_passes = message_passes
        self.num_clusters = int(num_clusters)
        self.global_flow_rank = int(max(global_flow_rank, 1))
        self.enable_incumbent_controls = bool(enable_incumbent_controls)
        self.chooser_legacy_residual_weight = float(chooser_legacy_residual_weight)
        self.chooser_large_swap_conflict_bias = 0.0
        self.chooser_rollout_legality_bias = 0.0

        self.input_mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.plus_head = nn.Linear(hidden_dim, 1)
        self.minus_base_head = nn.Linear(hidden_dim, 1)
        self.minus_condition_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.memory_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.phase_embedding = nn.Embedding(len(PHASE_NAMES), hidden_dim)
        self.discover_mode_embedding = nn.Embedding(len(DISCOVER_MODE_NAMES), hidden_dim)
        self.macro_plus_head = nn.Linear(hidden_dim, 1)
        self.macro_minus_base_head = nn.Linear(hidden_dim, 1)
        self.macro_minus_condition_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cluster_head = nn.Linear(hidden_dim, self.num_clusters)
        self.cluster_plus_head = nn.Linear(hidden_dim, 1)
        self.cluster_minus_base_head = nn.Linear(hidden_dim, 1)
        self.cluster_minus_condition_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.residual_basis_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * self.global_flow_rank),
        )
        self.global_flow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * self.global_flow_rank),
        )
        self.k_head = nn.Linear(hidden_dim, len(PD_K_CHOICES))
        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * len(CONTROL_NAMES)),
        )
        if self.enable_incumbent_controls:
            self.incumbent_control_head = nn.Linear(hidden_dim, 2)
        else:
            self.incumbent_control_head = None
        self.dag_axis_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.pair_branch_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.branch_pressure_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.boundary_pressure_head = nn.Linear(hidden_dim, 8)
        self.density_pressure_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.phase_request_head = nn.Linear(hidden_dim, len(PHASE_REQUEST_NAMES))
        self.unlock_source_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.unlock_radius_head = nn.Linear(hidden_dim, len(UNLOCK_RADIUS_CHOICES))
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cleanup_variant_head = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(REFINE_VARIANT_NAMES)),
        )
        self.cleanup_accept_head = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(REFINE_VARIANT_NAMES)),
        )
        self.cleanup_rank_head = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(REFINE_VARIANT_NAMES)),
        )
        self.mode_selector_head = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(DISCOVER_MODE_NAMES)),
        )
        self.candidate_rank_head = nn.Sequential(
            nn.Linear(hidden_dim + CANDIDATE_RANK_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.repaired_candidate_embed = nn.Sequential(
            nn.Linear(hidden_dim + CHOOSER_CANDIDATE_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.repaired_candidate_choice_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.continuation_preserve_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.phase_embedding.weight)
        nn.init.zeros_(self.discover_mode_embedding.weight)
        nn.init.zeros_(self.phase_request_head.weight)
        self.phase_request_head.bias.data.copy_(
            torch.tensor([2.0, 0.5, -2.0, -2.0], dtype=self.phase_request_head.bias.dtype)
        )
        nn.init.zeros_(self.unlock_radius_head.weight)
        nn.init.zeros_(self.unlock_radius_head.bias)
        unlock_last = self.unlock_source_head[-1]
        nn.init.zeros_(unlock_last.weight)
        nn.init.zeros_(unlock_last.bias)
        chooser_last = self.repaired_candidate_choice_head[-1]
        nn.init.zeros_(chooser_last.weight)
        nn.init.zeros_(chooser_last.bias)

    def _cleanup_feature_vector(self, graph, *, dtype, device):
        value = graph.get("cleanup_feature_vector")
        if value is None:
            return torch.zeros((5,), dtype=dtype, device=device)
        if torch.is_tensor(value):
            return value.to(dtype=dtype, device=device).reshape(-1)
        return torch.tensor(value, dtype=dtype, device=device).reshape(-1)

    def _case_descriptor(self, graph, *, dtype, device):
        value = graph.get("case_descriptor")
        if value is None:
            return torch.zeros((6,), dtype=dtype, device=device)
        if torch.is_tensor(value):
            return value.to(dtype=dtype, device=device).reshape(-1)
        return torch.tensor(value, dtype=dtype, device=device).reshape(-1)

    def _continuation_risk(self, graph, *, dtype, device):
        value = graph.get("continuation_risk")
        if value is None:
            return torch.zeros((1,), dtype=dtype, device=device)
        if torch.is_tensor(value):
            return value.to(dtype=dtype, device=device).reshape(-1)[:1]
        return torch.tensor([float(value)], dtype=dtype, device=device)

    def auxiliary_predictions(self, graph):
        phase_context, base_context = self._phase_context_from_graph(graph)
        dtype = phase_context.dtype
        device = phase_context.device
        cleanup_features = self._cleanup_feature_vector(graph, dtype=dtype, device=device)
        case_descriptor = self._case_descriptor(graph, dtype=dtype, device=device)
        continuation_risk = self._continuation_risk(graph, dtype=dtype, device=device)
        cleanup_input = torch.cat([phase_context, cleanup_features], dim=0)
        mode_input = torch.cat([base_context, case_descriptor], dim=0)
        return {
            "cleanup_variant_logits": self.cleanup_variant_head(cleanup_input),
            "cleanup_accept_logit": self.cleanup_accept_head(cleanup_input),
            "cleanup_rank_value": self.cleanup_rank_head(cleanup_input),
            "mode_selector_logits": self.mode_selector_head(mode_input),
            "continuation_preserve_logit": self.continuation_preserve_head(
                torch.cat([phase_context, continuation_risk], dim=0)
            ).squeeze(-1),
        }

    def _phase_context_from_graph(self, graph):
        h = self.encode(graph)
        _pooled, base_context = self._global_features(h)
        phase_context = self._phase_context(base_context, graph)
        return phase_context, base_context

    def _candidate_rank_score_from_phase_context(self, phase_context, candidate_features):
        if candidate_features.dim() == 1:
            candidate_features = candidate_features.unsqueeze(0)
        candidate_features = candidate_features.to(dtype=phase_context.dtype, device=phase_context.device)
        context = phase_context.unsqueeze(0).expand(candidate_features.shape[0], -1)
        return self.candidate_rank_head(torch.cat([context, candidate_features], dim=1)).squeeze(-1)

    def candidate_rank_score(self, graph, candidate_features):
        phase_context, _base_context = self._phase_context_from_graph(graph)
        return self._candidate_rank_score_from_phase_context(phase_context, candidate_features)

    def choose_repaired_candidate(self, graph, candidate_features, legacy_candidate_features=None):
        phase_context, _base_context = self._phase_context_from_graph(graph)
        if candidate_features.dim() == 1:
            candidate_features = candidate_features.unsqueeze(0)
        candidate_features = candidate_features.to(dtype=phase_context.dtype, device=phase_context.device)
        num_candidates = int(candidate_features.shape[0])
        if num_candidates <= 1:
            logits = torch.zeros((num_candidates,), dtype=phase_context.dtype, device=phase_context.device)
            return logits, 0

        context = phase_context.unsqueeze(0).expand(num_candidates, -1)
        candidate_embed = self.repaired_candidate_embed(torch.cat([context, candidate_features], dim=1))
        pooled_mean = candidate_embed.mean(dim=0, keepdim=True).expand(num_candidates, -1)
        pooled_max = candidate_embed.max(dim=0, keepdim=True).values.expand(num_candidates, -1)
        chooser_logits = self.repaired_candidate_choice_head(
            torch.cat([candidate_embed, pooled_mean, pooled_max], dim=1)
        ).squeeze(-1)
        if candidate_features.shape[1] >= CHOOSER_CANDIDATE_FEATURE_DIM:
            chooser_logits = chooser_logits - (
                self.chooser_large_swap_conflict_bias
                * candidate_features[:, CHOOSER_LARGE_SWAP_CONFLICT_IDX]
            )
            rollout_bias = (
                candidate_features[:, CHOOSER_ROLLOUT_SOURCE_IDX]
                * torch.clamp(candidate_features[:, CHOOSER_OVERLAP_MARGIN_IDX], min=0.0)
            )
            chooser_logits = chooser_logits + (self.chooser_rollout_legality_bias * rollout_bias)
        if (
            self.chooser_legacy_residual_weight != 0.0
            and legacy_candidate_features is not None
            and legacy_candidate_features.numel() > 0
        ):
            chooser_logits = chooser_logits + (
                self.chooser_legacy_residual_weight
                * self._candidate_rank_score_from_phase_context(
                    phase_context,
                    legacy_candidate_features,
                )
            )
        chosen_index = int(torch.argmax(chooser_logits).item())
        return chooser_logits, chosen_index

    def encode(self, graph):
        cell_features = graph["cell_features"]
        pin_features = graph["pin_features"]
        edge_list = graph["edge_list"]
        x = self._node_features(graph)
        h = self.input_mlp(x)
        h = _sanitize_tensor(h)

        if edge_list.numel() > 0 and pin_features.numel() > 0:
            pin_to_cell = pin_features[:, 0].long()
            src = pin_to_cell[edge_list[:, 0].long()]
            dst = pin_to_cell[edge_list[:, 1].long()]
            mask = src != dst
            src = src[mask]
            dst = dst[mask]
            for _ in range(self.message_passes):
                messages = torch.zeros_like(h)
                counts = torch.zeros((cell_features.shape[0], 1), dtype=h.dtype, device=h.device)
                msg_src = self.message_mlp(h[src])
                msg_dst = self.message_mlp(h[dst])
                messages.index_add_(0, dst, msg_src)
                messages.index_add_(0, src, msg_dst)
                ones = torch.ones((src.numel(), 1), dtype=h.dtype, device=h.device)
                counts.index_add_(0, dst, ones)
                counts.index_add_(0, src, ones)
                messages = messages / torch.clamp(counts, min=1.0)
                h = self.update_mlp(torch.cat([h, messages], dim=-1))
                h = _sanitize_tensor(h)

        active_pairs = graph["active_pairs"]
        if active_pairs.numel() > 0:
            pair_messages = torch.zeros_like(h)
            pair_counts = torch.zeros((cell_features.shape[0], 1), dtype=h.dtype, device=h.device)
            pair_features = self._pair_features(graph)
            src = active_pairs[:, 0].long()
            dst = active_pairs[:, 1].long()
            msg_ij = self.pair_message_mlp(torch.cat([h[src], h[dst], pair_features], dim=1))
            msg_ji = self.pair_message_mlp(torch.cat([h[dst], h[src], pair_features], dim=1))
            pair_messages.index_add_(0, dst, msg_ij)
            pair_messages.index_add_(0, src, msg_ji)
            ones = torch.ones((active_pairs.shape[0], 1), dtype=h.dtype, device=h.device)
            pair_counts.index_add_(0, dst, ones)
            pair_counts.index_add_(0, src, ones)
            pair_messages = pair_messages / torch.clamp(pair_counts, min=1.0)
            h = self.update_mlp(torch.cat([h, pair_messages], dim=-1))
            h = _sanitize_tensor(h)

        return h

    def _global_features(self, h, graph=None):
        mean_pool = h.mean(dim=0)
        max_pool = h.max(dim=0).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        base_context = self.context_mlp(pooled)
        if graph is None:
            return pooled, base_context
        if self._discover_stateless_control_mode(graph):
            return pooled, _sanitize_tensor(base_context)
        memory = graph.get("memory")
        if memory is None or memory.numel() != self.hidden_dim:
            memory = torch.zeros(self.hidden_dim, dtype=h.dtype, device=h.device)
        else:
            memory = _sanitize_tensor(memory.to(dtype=h.dtype, device=h.device))
        next_memory = self.memory_cell(base_context.unsqueeze(0), memory.unsqueeze(0)).squeeze(0)
        next_memory = _sanitize_tensor(next_memory)
        return pooled, next_memory

    def _memory_pair(self, h, graph):
        if self._discover_stateless_control_mode(graph):
            _pooled, context = self._global_features(h, graph)
            memory = torch.zeros(self.hidden_dim, dtype=h.dtype, device=h.device)
            return memory, context
        memory = graph.get("memory")
        if memory is None or memory.numel() != self.hidden_dim:
            memory = torch.zeros(self.hidden_dim, dtype=h.dtype, device=h.device)
        else:
            memory = _sanitize_tensor(memory.to(dtype=h.dtype, device=h.device))
        _pooled, context = self._global_features(h, graph)
        return memory, context

    def _phase_name(self, graph):
        return str(graph.get("phase_name", "DISCOVER")).upper()

    def _incumbent_refine_ready(self, graph):
        incumbent_overlap_ratio = _graph_scalar(
            graph,
            "incumbent_overlap_ratio",
            0.0,
            dtype=next(self.parameters()).dtype,
            device=next(self.parameters()).device,
        )
        incumbent_num_overlap_pairs = _graph_scalar(
            graph,
            "incumbent_num_overlap_pairs",
            0.0,
            dtype=next(self.parameters()).dtype,
            device=next(self.parameters()).device,
        )
        return bool(
            float(incumbent_overlap_ratio.detach().item()) <= REFINE_READY_OVERLAP_THRESHOLD
            and float(incumbent_num_overlap_pairs.detach().item()) <= REFINE_READY_PAIR_THRESHOLD
        )

    def _strict_post_legal_mode(self, graph):
        phase_name = self._phase_name(graph)
        return phase_name in {"LEGALIZE", "REFINE"} and self._incumbent_refine_ready(graph)

    def _discover_mode_name(self, graph):
        return str(graph.get("discover_mode_name", "balanced")).lower()

    def _discover_mode_carry_steps(self, graph):
        value = graph.get("discover_mode_carry_steps", 0.0)
        if torch.is_tensor(value):
            return int(max(float(value.detach().item()), 0.0))
        return int(max(float(value), 0.0))

    def _phase_step(self, graph):
        value = graph.get("phase_step", 0.0)
        if torch.is_tensor(value):
            return int(max(float(value.detach().item()), 0.0))
        return int(max(float(value), 0.0))

    def _discover_discrete_stateless_mode(self, graph):
        return self._phase_name(graph) == "DISCOVER" and self._phase_step(graph) >= 1

    def _discover_frozen_sequence_mode(self, graph):
        return self._phase_name(graph) == "DISCOVER" and self._phase_step(graph) >= 1

    def _discover_stateless_control_mode(self, graph):
        return (
            self._phase_name(graph) == "DISCOVER"
            and self._discover_mode_name(graph) == "spread_first"
            and self._phase_step(graph) >= 1
        )

    def _discrete_basin_graph(self, graph):
        if not self._discover_discrete_stateless_mode(graph):
            return graph
        discrete_graph = dict(graph)
        centers = graph["cell_features"][:, 2:4]
        dtype = centers.dtype
        device = centers.device
        discrete_graph["incumbent_centers"] = centers.detach()
        discrete_graph["incumbent_overlap_ratio"] = _graph_scalar(
            graph, "exact_overlap_ratio", 0.0, dtype=dtype, device=device
        )
        discrete_graph["incumbent_normalized_wl"] = _graph_scalar(
            graph, "current_normalized_wl", 0.0, dtype=dtype, device=device
        )
        discrete_graph["incumbent_num_overlap_pairs"] = _graph_scalar(
            graph, "current_num_overlap_pairs", 0.0, dtype=dtype, device=device
        )
        discrete_graph["steps_since_best"] = torch.zeros((), dtype=dtype, device=device)
        discrete_graph["memory"] = torch.zeros_like(graph.get("memory", torch.zeros(self.hidden_dim, dtype=dtype, device=device)))
        return discrete_graph

    def _phase_context(self, context, graph):
        phase_name = self._phase_name(graph)
        phase_index = PHASE_TO_INDEX.get(phase_name, 0)
        phase_index = torch.tensor(phase_index, dtype=torch.long, device=context.device)
        phase_context = context + self.phase_embedding(phase_index)
        discover_mode_name = self._discover_mode_name(graph)
        discover_mode_index = DISCOVER_MODE_TO_INDEX.get(discover_mode_name, 0)
        discover_mode_index = torch.tensor(discover_mode_index, dtype=torch.long, device=context.device)
        carry_steps = self._discover_mode_carry_steps(graph)
        if phase_name == "DISCOVER":
            mode_scale = 1.0
        elif phase_name == "LEGALIZE" and carry_steps > 0:
            mode_scale = min(carry_steps / 2.0, 1.0)
        else:
            mode_scale = 0.0
        if mode_scale > 0.0:
            phase_context = phase_context + float(mode_scale) * self.discover_mode_embedding(discover_mode_index)
        return _sanitize_tensor(phase_context)

    def _control_context(self, recurrent_context, stateless_context, graph):
        phase_name = self._phase_name(graph)
        if (
            phase_name == "REFINE"
            or self._strict_post_legal_mode(graph)
            or self._discover_stateless_control_mode(graph)
        ):
            return stateless_context
        return recurrent_context

    def _control_node_embeddings(self, recurrent_h, stateless_h, graph):
        if self._discover_stateless_control_mode(graph):
            return stateless_h
        return recurrent_h

    def _discover_discrete_context(self, recurrent_context, stateless_context, graph):
        if self._discover_discrete_stateless_mode(graph):
            return stateless_context
        return recurrent_context

    def _phase_request_logits(self, context, phase_name, strict_post_legal_mode=False):
        logits = self.phase_request_head(context).clone()
        allowed = {
            "DISCOVER": {"stay", "advance"},
            "LEGALIZE": {"stay", "advance", "unlock"},
            "REFINE": {"stay", "advance", "stop"},
            "UNLOCK": {"stay", "advance"},
        }.get(str(phase_name).upper(), {"stay"})
        if strict_post_legal_mode and str(phase_name).upper() == "LEGALIZE":
            allowed = {"stay", "advance"}
        if strict_post_legal_mode and str(phase_name).upper() == "REFINE":
            allowed = {"stay", "stop"}
        for name, idx in PHASE_REQUEST_TO_INDEX.items():
            if name not in allowed:
                logits[idx] = -1.0e9
        return logits

    def _phase_adjust_controls(self, controls, phase_name, graph, strict_post_legal_mode=False):
        phase_name = str(phase_name).upper()
        adjusted = dict(controls)
        discover_mode_name = self._discover_mode_name(graph)
        carry_steps = self._discover_mode_carry_steps(graph)
        if phase_name == "DISCOVER":
            adjusted["step_scale"] = torch.clamp(adjusted["step_scale"], min=0.02, max=0.40)
            adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"], min=0.05, max=1.0)
            if discover_mode_name == "spread_first":
                adjusted["step_scale"] = torch.clamp(adjusted["step_scale"] * 1.30, min=0.03, max=0.50)
                adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"] * 1.25, min=0.15, max=1.0)
                adjusted["branch_pressure"] = torch.clamp(adjusted["branch_pressure"] * 1.20, min=0.25, max=8.0)
                adjusted["density_pressure"] = torch.clamp(adjusted["density_pressure"] * 1.10, min=0.05, max=8.0)
                adjusted["boundary_pressure"] = torch.clamp(adjusted["boundary_pressure"] * 1.10, min=0.05, max=8.0)
                adjusted["rho"] = torch.clamp(adjusted["rho"] * 1.10, min=0.05, max=128.0)
            elif discover_mode_name == "wire_first":
                adjusted["step_scale"] = torch.clamp(adjusted["step_scale"] * 0.80, min=0.015, max=0.28)
                adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"] * 0.60, min=0.02, max=0.70)
                adjusted["density_pressure"] = torch.clamp(adjusted["density_pressure"] * 0.70, min=0.02, max=6.0)
                adjusted["boundary_pressure"] = torch.clamp(adjusted["boundary_pressure"] * 0.80, min=0.02, max=6.0)
                adjusted["tau"] = torch.clamp(adjusted["tau"], min=0.10, max=1.50)
            elif discover_mode_name == "macro_clearance":
                adjusted["step_scale"] = torch.clamp(adjusted["step_scale"] * 1.10, min=0.02, max=0.42)
                adjusted["boundary_pressure"] = torch.clamp(adjusted["boundary_pressure"] * 1.40, min=0.10, max=10.0)
                adjusted["density_pressure"] = torch.clamp(adjusted["density_pressure"] * 1.35, min=0.10, max=10.0)
                adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"] * 1.05, min=0.05, max=1.0)
        elif phase_name == "LEGALIZE":
            adjusted["step_scale"] = torch.clamp(adjusted["step_scale"], min=0.01, max=0.22)
            adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"], min=0.50, max=1.0)
            adjusted["branch_pressure"] = torch.clamp(adjusted["branch_pressure"], min=0.25, max=8.0)
            adjusted["density_pressure"] = torch.clamp(adjusted["density_pressure"], min=0.10, max=8.0)
            if carry_steps > 0 and discover_mode_name == "spread_first":
                adjusted["step_scale"] = torch.clamp(adjusted["step_scale"] * 1.10, min=0.01, max=0.24)
                adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"] * 1.10, min=0.50, max=1.0)
            elif carry_steps > 0 and discover_mode_name == "wire_first":
                adjusted["step_scale"] = torch.clamp(adjusted["step_scale"] * 0.90, min=0.01, max=0.20)
                adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"] * 0.90, min=0.35, max=0.90)
            elif carry_steps > 0 and discover_mode_name == "macro_clearance":
                adjusted["density_pressure"] = torch.clamp(adjusted["density_pressure"] * 1.15, min=0.10, max=8.0)
                adjusted["boundary_pressure"] = torch.clamp(adjusted["boundary_pressure"] * 1.15, min=0.05, max=8.0)
            if strict_post_legal_mode:
                adjusted["step_scale"] = torch.clamp(adjusted["step_scale"], min=0.005, max=0.08)
                adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"], min=0.20, max=0.75)
                adjusted["branch_pressure"] = torch.clamp(adjusted["branch_pressure"], min=0.10, max=4.0)
                adjusted["density_pressure"] = torch.clamp(adjusted["density_pressure"], min=0.05, max=4.0)
                adjusted["boundary_pressure"] = torch.clamp(adjusted["boundary_pressure"], min=0.05, max=3.0)
                adjusted["tau"] = torch.clamp(adjusted["tau"], min=0.02, max=0.75)
        elif phase_name == "REFINE":
            adjusted["step_scale"] = torch.clamp(adjusted["step_scale"], min=0.005, max=0.10)
            adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"], min=0.05, max=0.55)
            adjusted["tau"] = torch.clamp(adjusted["tau"], min=0.02, max=1.5)
            if strict_post_legal_mode:
                adjusted["step_scale"] = torch.clamp(adjusted["step_scale"], min=0.002, max=0.05)
                adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"], min=0.02, max=0.20)
                adjusted["branch_pressure"] = torch.clamp(adjusted["branch_pressure"], min=0.05, max=2.0)
                adjusted["density_pressure"] = torch.clamp(adjusted["density_pressure"], min=0.02, max=2.0)
                adjusted["boundary_pressure"] = torch.clamp(adjusted["boundary_pressure"], min=0.02, max=2.0)
                adjusted["tau"] = torch.clamp(adjusted["tau"], min=0.02, max=0.60)
        elif phase_name == "UNLOCK":
            adjusted["step_scale"] = torch.clamp(adjusted["step_scale"], min=0.01, max=0.18)
            adjusted["pair_emphasis"] = torch.clamp(adjusted["pair_emphasis"], min=0.40, max=1.0)
        return adjusted

    def _masked_k_logits(self, context, phase_name, strict_post_legal_mode=False):
        logits = self.k_head(context).clone()
        allowed = {
            "DISCOVER": {1, 2, 4, 8, 12, 16},
            "LEGALIZE": {2, 4, 8, 12, 16},
            "REFINE": {1, 2, 4, 8},
            "UNLOCK": {2, 4, 8},
        }.get(str(phase_name).upper(), {1, 2, 4, 8})
        if strict_post_legal_mode and str(phase_name).upper() == "LEGALIZE":
            allowed = {1, 2, 4, 8}
        if strict_post_legal_mode and str(phase_name).upper() == "REFINE":
            allowed = {1, 2, 4}
        for idx, value in enumerate(PD_K_CHOICES):
            if value not in allowed:
                logits[idx] = -1.0e9
        return logits

    def _phase_adjust_global_distribution(self, global_mean, global_log_std, phase_name, graph, strict_post_legal_mode=False):
        phase_name = str(phase_name).upper()
        scale = 1.0
        discover_mode_name = self._discover_mode_name(graph)
        carry_steps = self._discover_mode_carry_steps(graph)
        if phase_name == "DISCOVER":
            if discover_mode_name == "spread_first":
                scale = 1.25
            elif discover_mode_name == "wire_first":
                scale = 0.75
            elif discover_mode_name == "macro_clearance":
                scale = 1.15
        elif phase_name == "LEGALIZE" and carry_steps > 0:
            if discover_mode_name == "spread_first":
                scale = 1.10
            elif discover_mode_name == "wire_first":
                scale = 0.90
        if phase_name == "REFINE":
            scale = 0.20
        if strict_post_legal_mode and phase_name == "LEGALIZE":
            scale = 0.25
        if strict_post_legal_mode and phase_name == "REFINE":
            scale = 0.05
        adjusted_mean = global_mean * scale
        adjusted_log_std = torch.clamp(global_log_std + math.log(max(scale, 1.0e-3)), min=-8.0, max=1.0)
        return adjusted_mean, adjusted_log_std

    def _frozen_sequence_pair(self, graph, centers):
        centers = centers.to(dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        x = centers[:, 0]
        y = centers[:, 1]
        seq_plus = torch.argsort(x + 1.0e-3 * y, descending=False)
        seq_minus = torch.argsort(x - 1.0e-3 * y, descending=False)
        return seq_plus, seq_minus

    def _discover_anchor_sequence_pair(self, graph):
        anchor_centers = graph.get("incumbent_centers", graph["cell_features"][:, 2:4])
        return self._frozen_sequence_pair(graph, anchor_centers)

    def forward(self, graph):
        h = self.encode(graph)
        plus_scores = self.plus_head(h).squeeze(-1)
        minus_scores = self.minus_base_head(h).squeeze(-1)
        pooled, _context = self._global_features(h)
        value = self.value_head(pooled).squeeze(-1)
        return plus_scores, minus_scores, value

    def _conditioned_minus_scores(self, h, base_minus_scores, seq_plus):
        n = int(seq_plus.numel())
        if n <= 1:
            rank_feature = torch.zeros((n, 1), dtype=h.dtype, device=h.device)
        else:
            ranks = torch.empty(n, dtype=h.dtype, device=h.device)
            ranks[seq_plus.long()] = torch.arange(n, dtype=h.dtype, device=h.device)
            rank_feature = (ranks / float(n - 1)).unsqueeze(1)
        centered_rank = rank_feature - rank_feature.mean()
        conditioned = self.minus_condition_head(torch.cat([h, rank_feature, centered_rank], dim=1)).squeeze(-1)
        return base_minus_scores + conditioned

    def _conditioned_cluster_minus_scores(self, cluster_h, base_minus_scores, seq_plus):
        n = int(seq_plus.numel())
        if n <= 1:
            rank_feature = torch.zeros((n, 1), dtype=cluster_h.dtype, device=cluster_h.device)
        else:
            ranks = torch.empty(n, dtype=cluster_h.dtype, device=cluster_h.device)
            ranks[seq_plus.long()] = torch.arange(n, dtype=cluster_h.dtype, device=cluster_h.device)
            rank_feature = (ranks / float(n - 1)).unsqueeze(1)
        centered_rank = rank_feature - rank_feature.mean()
        conditioned = self.cluster_minus_condition_head(
            torch.cat([cluster_h, rank_feature, centered_rank], dim=1)
        ).squeeze(-1)
        return base_minus_scores + conditioned

    def _conditioned_macro_minus_scores(self, macro_h, base_minus_scores, seq_plus):
        n = int(seq_plus.numel())
        if n <= 1:
            rank_feature = torch.zeros((n, 1), dtype=macro_h.dtype, device=macro_h.device)
        else:
            ranks = torch.empty(n, dtype=macro_h.dtype, device=macro_h.device)
            ranks[seq_plus.long()] = torch.arange(n, dtype=macro_h.dtype, device=macro_h.device)
            rank_feature = (ranks / float(n - 1)).unsqueeze(1)
        centered_rank = rank_feature - rank_feature.mean()
        conditioned = self.macro_minus_condition_head(
            torch.cat([macro_h, rank_feature, centered_rank], dim=1)
        ).squeeze(-1)
        return base_minus_scores + conditioned

    def _sample_macro_sequence_pair(self, h, graph, temperature, deterministic=False, seq_plus=None):
        macro_mask = self._macro_cell_mask(graph)
        macro_indices = torch.where(macro_mask)[0]
        if macro_indices.numel() == 0:
            empty_scores = h.new_zeros(0)
            empty_seq = torch.empty(0, dtype=torch.long, device=h.device)
            return macro_indices, empty_seq, empty_seq, empty_scores, empty_scores
        macro_h = h[macro_indices]
        macro_plus_scores = self.macro_plus_head(macro_h).squeeze(-1) / max(float(temperature), 1e-4)
        macro_plus_scores = _stabilize_ordering_scores(macro_plus_scores)
        if seq_plus is None:
            macro_seq_plus = torch.argsort(macro_plus_scores, descending=True) if deterministic else sample_plackett_luce(macro_plus_scores)
        else:
            macro_seq_plus = seq_plus
        macro_minus_scores = self._conditioned_macro_minus_scores(
            macro_h,
            self.macro_minus_base_head(macro_h).squeeze(-1),
            macro_seq_plus,
        )
        macro_minus_scores = macro_minus_scores / max(float(temperature), 1e-4)
        macro_minus_scores = _stabilize_ordering_scores(macro_minus_scores)
        macro_seq_minus = torch.argsort(macro_minus_scores, descending=True) if deterministic else sample_plackett_luce(macro_minus_scores)
        return macro_indices, macro_seq_plus, macro_seq_minus, macro_plus_scores, macro_minus_scores

    def _cluster_embeddings(self, h, cluster_ids, context):
        cluster_h = context.unsqueeze(0).expand(self.num_clusters, -1).clone()
        valid = cluster_ids >= 0
        if torch.any(valid):
            counts = torch.zeros((self.num_clusters, 1), dtype=h.dtype, device=h.device)
            sums = torch.zeros((self.num_clusters, h.shape[1]), dtype=h.dtype, device=h.device)
            ids = cluster_ids[valid].long()
            sums.index_add_(0, ids, h[valid])
            ones = torch.ones((ids.numel(), 1), dtype=h.dtype, device=h.device)
            counts.index_add_(0, ids, ones)
            nonempty = counts.squeeze(1) > 0
            cluster_h[nonempty] = sums[nonempty] / torch.clamp(counts[nonempty], min=1.0)
        return cluster_h

    def _sample_cluster_sequence_pair(self, cluster_h, temperature, deterministic=False, seq_plus=None):
        cluster_plus_scores = self.cluster_plus_head(cluster_h).squeeze(-1) / max(float(temperature), 1e-4)
        cluster_plus_scores = _stabilize_ordering_scores(cluster_plus_scores)
        if seq_plus is None:
            cluster_seq_plus = torch.argsort(cluster_plus_scores, descending=True) if deterministic else sample_plackett_luce(cluster_plus_scores)
        else:
            cluster_seq_plus = seq_plus
        cluster_minus_scores = self._conditioned_cluster_minus_scores(
            cluster_h,
            self.cluster_minus_base_head(cluster_h).squeeze(-1),
            cluster_seq_plus,
        )
        cluster_minus_scores = cluster_minus_scores / max(float(temperature), 1e-4)
        cluster_minus_scores = _stabilize_ordering_scores(cluster_minus_scores)
        cluster_seq_minus = torch.argsort(cluster_minus_scores, descending=True) if deterministic else sample_plackett_luce(cluster_minus_scores)
        return cluster_seq_plus, cluster_seq_minus, cluster_plus_scores, cluster_minus_scores

    def _pair_inputs(self, h, graph):
        active_pairs = graph["active_pairs"]
        if active_pairs.numel() == 0:
            return torch.empty((0, 2 * self.hidden_dim + self.pair_feature_dim), dtype=h.dtype, device=h.device)
        pair_features = self._pair_features(graph)
        src = active_pairs[:, 0].long()
        dst = active_pairs[:, 1].long()
        return torch.cat([h[src], h[dst], pair_features], dim=1)

    def _active_head_config(self, graph, std_mask=None):
        ordering_representation = _graph_str(graph, "ordering_representation", "sequence_pair")
        branch_mode = _graph_str(graph, "branch_mode", "ordering")
        enable_clusters = _graph_bool(graph, "enable_clusters", True)
        enable_stop = _graph_bool(graph, "enable_stop", True)
        enable_incumbent_action = _graph_bool(graph, "enable_incumbent_action", self.enable_incumbent_controls)
        phase_name = self._phase_name(graph)
        strict_post_legal_mode = self._strict_post_legal_mode(graph)
        has_std_cells = True if std_mask is None else bool(torch.any(std_mask).item())
        enable_ordering_actions = phase_name == "DISCOVER"
        enable_clusters = enable_clusters and has_std_cells and self.num_clusters > 1 and phase_name == "DISCOVER"
        enable_macro_ordering = phase_name == "DISCOVER"
        enable_stop = enable_stop and phase_name == "REFINE"
        enable_unlock = phase_name == "UNLOCK"
        return {
            "phase_name": phase_name,
            "ordering_representation": ordering_representation,
            "branch_mode": branch_mode,
            "enable_ordering_actions": enable_ordering_actions,
            "enable_macro_ordering": enable_macro_ordering,
            "enable_clusters": enable_clusters,
            "enable_stop": enable_stop,
            "enable_unlock": enable_unlock,
            "enable_incumbent_action": enable_incumbent_action and self.enable_incumbent_controls,
            "enable_dag_ordering": ordering_representation == "dag" and phase_name in {"DISCOVER", "UNLOCK"},
            "enable_pair_branches": branch_mode == "independent_pair" and phase_name in {"DISCOVER", "UNLOCK"},
            "strict_post_legal_mode": strict_post_legal_mode,
        }

    def _sample_pair_discrete_actions(self, h, graph, deterministic=False):
        pair_input = self._pair_inputs(h, graph)
        if pair_input.numel() == 0:
            device = h.device
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty((0, 2), dtype=h.dtype, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty((0, 4), dtype=h.dtype, device=device),
            )
        dag_axis_logits = self.dag_axis_mlp(pair_input)
        pair_branch_logits = self.pair_branch_mlp(pair_input)
        if deterministic:
            dag_axis = torch.argmax(dag_axis_logits, dim=1)
            pair_branch_choices = torch.argmax(pair_branch_logits, dim=1)
        else:
            dag_axis = torch.distributions.Categorical(logits=dag_axis_logits).sample()
            pair_branch_choices = torch.distributions.Categorical(logits=pair_branch_logits).sample()
        return dag_axis, dag_axis_logits, pair_branch_choices, pair_branch_logits

    def _sample_constraint_pressures(self, h, graph, context, deterministic=False):
        pair_input = self._pair_inputs(h, graph)
        if pair_input.numel() == 0:
            branch_raw = torch.empty(0, dtype=h.dtype, device=h.device)
            branch_mean = torch.empty(0, dtype=h.dtype, device=h.device)
            branch_log_std = torch.empty(0, dtype=h.dtype, device=h.device)
            branch_values = torch.empty(0, dtype=h.dtype, device=h.device)
        else:
            branch_params = self.branch_pressure_mlp(pair_input)
            branch_mean = branch_params[:, 0]
            branch_log_std = torch.clamp(branch_params[:, 1], min=-4.0, max=1.0)
            branch_raw = branch_mean if deterministic else branch_mean + torch.randn_like(branch_mean) * branch_log_std.exp()
            branch_values = lognormal_from_raw(branch_raw, scale=0.5, min_value=0.10, max_value=8.0)

        boundary_params = self.boundary_pressure_head(h).reshape(h.shape[0], 4, 2)
        boundary_mean = boundary_params[:, :, 0]
        boundary_log_std = torch.clamp(boundary_params[:, :, 1], min=-4.0, max=1.0)
        boundary_raw = boundary_mean if deterministic else boundary_mean + torch.randn_like(boundary_mean) * boundary_log_std.exp()
        boundary_values = lognormal_from_raw(boundary_raw, scale=0.35, min_value=0.05, max_value=6.0)

        density_duals = graph["density_duals"]
        if density_duals.numel() == 0:
            density_raw = torch.empty(0, dtype=h.dtype, device=h.device)
            density_mean = torch.empty(0, dtype=h.dtype, device=h.device)
            density_log_std = torch.empty(0, dtype=h.dtype, device=h.device)
            density_values = torch.empty(0, dtype=h.dtype, device=h.device)
        else:
            dual_norm = torch.clamp(density_duals.abs().max(), min=1.0)
            density_input = torch.cat(
                [
                    context.unsqueeze(0).expand(density_duals.numel(), -1),
                    (density_duals / dual_norm).unsqueeze(1),
                ],
                dim=1,
            )
            density_params = self.density_pressure_head(density_input)
            density_mean = density_params[:, 0]
            density_log_std = torch.clamp(density_params[:, 1], min=-4.0, max=1.0)
            density_raw = density_mean if deterministic else density_mean + torch.randn_like(density_mean) * density_log_std.exp()
            density_values = lognormal_from_raw(density_raw, scale=0.35, min_value=0.05, max_value=6.0)

        return {
            "branch_raw": branch_raw,
            "branch_mean": branch_mean,
            "branch_log_std": branch_log_std,
            "branch_values": branch_values,
            "boundary_raw": boundary_raw,
            "boundary_mean": boundary_mean,
            "boundary_log_std": boundary_log_std,
            "boundary_values": boundary_values,
            "density_raw": density_raw,
            "density_mean": density_mean,
            "density_log_std": density_log_std,
            "density_values": density_values,
        }

    def _scores_for_sequence_pair(self, graph, seq_plus=None, temperature=1.0):
        h = self.encode(graph)
        plus_scores = self.plus_head(h).squeeze(-1)
        base_minus_scores = self.minus_base_head(h).squeeze(-1)
        scaled_plus_scores = plus_scores / max(float(temperature), 1e-4)
        scaled_plus_scores = _stabilize_ordering_scores(scaled_plus_scores)
        if seq_plus is None:
            seq_plus = sample_plackett_luce(scaled_plus_scores)
        minus_scores = self._conditioned_minus_scores(h, base_minus_scores, seq_plus)
        scaled_minus_scores = minus_scores / max(float(temperature), 1e-4)
        scaled_minus_scores = _stabilize_ordering_scores(scaled_minus_scores)
        pooled, _context = self._global_features(h)
        value = self.value_head(pooled).squeeze(-1)
        return scaled_plus_scores, scaled_minus_scores, value, seq_plus

    def sample_sequence_pair(self, graph, temperature=1.0):
        plus_scores, minus_scores, value, seq_plus = self._scores_for_sequence_pair(graph, temperature=temperature)
        seq_minus = sample_plackett_luce(minus_scores)
        logprob = (
            plackett_luce_logprob(plus_scores, seq_plus)
            + plackett_luce_logprob(minus_scores, seq_minus)
        )
        entropy = categorical_entropy_from_scores(plus_scores) + categorical_entropy_from_scores(minus_scores)
        return SequencePairAction(seq_plus, seq_minus, logprob, entropy, value, plus_scores, minus_scores)

    def evaluate_sequence_pair(self, graph, seq_plus, seq_minus, temperature=1.0):
        plus_scores, minus_scores, value, _ = self._scores_for_sequence_pair(
            graph,
            seq_plus=seq_plus,
            temperature=temperature,
        )
        logprob = (
            plackett_luce_logprob(plus_scores, seq_plus)
            + plackett_luce_logprob(minus_scores, seq_minus)
        )
        entropy = categorical_entropy_from_scores(plus_scores) + categorical_entropy_from_scores(minus_scores)
        return logprob, entropy, value

    def deterministic_sequence_pair(self, graph):
        h = self.encode(graph)
        plus_scores = self.plus_head(h).squeeze(-1)
        base_minus_scores = self.minus_base_head(h).squeeze(-1)
        seq_plus = torch.argsort(plus_scores, descending=True)
        minus_scores = self._conditioned_minus_scores(h, base_minus_scores, seq_plus)
        seq_minus = torch.argsort(minus_scores, descending=True)
        pooled, _context = self._global_features(h)
        value = self.value_head(pooled).squeeze(-1)
        logprob = (
            plackett_luce_logprob(plus_scores, seq_plus)
            + plackett_luce_logprob(minus_scores, seq_minus)
        )
        entropy = categorical_entropy_from_scores(plus_scores) + categorical_entropy_from_scores(minus_scores)
        return SequencePairAction(seq_plus, seq_minus, logprob, entropy, value, plus_scores, minus_scores)

    def sample_action(self, graph, temperature=1.0, deterministic=False):
        h = self.encode(graph)
        discrete_graph = self._discrete_basin_graph(graph)
        h_discrete = h if discrete_graph is graph else self.encode(discrete_graph)
        pooled, base_context = self._global_features(h)
        memory, recurrent_context = self._memory_pair(h, graph)
        head_config = self._active_head_config(graph, std_mask=self._standard_cell_mask(graph))
        phase_name = head_config["phase_name"]
        strict_post_legal_mode = head_config["strict_post_legal_mode"]
        context = self._phase_context(recurrent_context, graph)
        stateless_phase_context = self._phase_context(base_context, graph)
        control_context = self._control_context(context, stateless_phase_context, graph)
        control_h = self._control_node_embeddings(h, h_discrete, graph)
        discrete_context = self._discover_discrete_context(context, stateless_phase_context, graph)
        value = self.value_head(pooled).squeeze(-1)
        std_mask = self._standard_cell_mask(graph)

        plus_scores = self.plus_head(h_discrete).squeeze(-1) / max(float(temperature), 1e-4)
        plus_scores = _stabilize_ordering_scores(plus_scores)
        if head_config["enable_ordering_actions"]:
            if self._discover_frozen_sequence_mode(graph):
                seq_plus, seq_minus = self._discover_anchor_sequence_pair(graph)
                minus_scores = self._conditioned_minus_scores(
                    h_discrete,
                    self.minus_base_head(h_discrete).squeeze(-1),
                    seq_plus,
                )
                minus_scores = minus_scores / max(float(temperature), 1e-4)
                minus_scores = _stabilize_ordering_scores(minus_scores)
            else:
                if deterministic:
                    seq_plus = torch.argsort(plus_scores, descending=True)
                else:
                    seq_plus = sample_plackett_luce(plus_scores)
                minus_scores = self._conditioned_minus_scores(h_discrete, self.minus_base_head(h_discrete).squeeze(-1), seq_plus)
                minus_scores = minus_scores / max(float(temperature), 1e-4)
                minus_scores = _stabilize_ordering_scores(minus_scores)
                if deterministic:
                    seq_minus = torch.argsort(minus_scores, descending=True)
                else:
                    seq_minus = sample_plackett_luce(minus_scores)
        else:
            frozen_centers = graph.get("incumbent_centers", graph["cell_features"][:, 2:4])
            seq_plus, seq_minus = self._frozen_sequence_pair(graph, frozen_centers)
            minus_scores = self.minus_base_head(h_discrete).squeeze(-1) / max(float(temperature), 1e-4)
            minus_scores = _stabilize_ordering_scores(minus_scores)

        if head_config["enable_macro_ordering"]:
            macro_cell_indices, macro_seq_plus, macro_seq_minus, macro_plus_scores, macro_minus_scores = (
                self._sample_macro_sequence_pair(h_discrete, discrete_graph, temperature, deterministic=deterministic)
            )
        else:
            macro_cell_indices = torch.empty(0, dtype=torch.long, device=h.device)
            macro_seq_plus = torch.empty(0, dtype=torch.long, device=h.device)
            macro_seq_minus = torch.empty(0, dtype=torch.long, device=h.device)
            macro_plus_scores = torch.empty(0, dtype=h.dtype, device=h.device)
            macro_minus_scores = torch.empty(0, dtype=h.dtype, device=h.device)

        node_context = control_context.unsqueeze(0).expand_as(control_h)
        residual_params = self.residual_head(torch.cat([control_h, node_context], dim=1))
        residual_mean = residual_params[:, :2]
        residual_log_std = torch.clamp(residual_params[:, 2:], min=-4.0, max=1.0)
        residual_basis = self.residual_basis_head(torch.cat([control_h, node_context], dim=1)).reshape(
            control_h.shape[0],
            2,
            self.global_flow_rank,
        )
        residual_basis = torch.tanh(residual_basis) / math.sqrt(float(self.global_flow_rank))
        global_params = self.global_flow_head(control_context)
        global_mean = global_params[: self.global_flow_rank]
        global_log_std = torch.clamp(global_params[self.global_flow_rank :], min=-4.0, max=1.0)
        global_mean, global_log_std = self._phase_adjust_global_distribution(
            global_mean,
            global_log_std,
            phase_name,
            graph,
            strict_post_legal_mode=strict_post_legal_mode,
        )
        if deterministic:
            residual_local = residual_mean
            global_flow_latent = global_mean
        else:
            residual_local = residual_mean + torch.randn_like(residual_mean) * residual_log_std.exp()
            global_flow_latent = global_mean + torch.randn_like(global_mean) * global_log_std.exp()
        residual_flow = residual_local + torch.einsum("ndr,r->nd", residual_basis, global_flow_latent)

        control_params = self.control_head(control_context)
        control_mean = control_params[: len(CONTROL_NAMES)]
        control_log_std = torch.clamp(control_params[len(CONTROL_NAMES) :], min=-4.0, max=1.0)
        if deterministic:
            control_raw = control_mean
        else:
            control_raw = control_mean + torch.randn_like(control_mean) * control_log_std.exp()
        controls = self._phase_adjust_controls(
            self._transform_controls(control_raw),
            phase_name,
            graph,
            strict_post_legal_mode=strict_post_legal_mode,
        )

        if head_config["enable_incumbent_action"] and self.incumbent_control_head is not None:
            incumbent_params = self.incumbent_control_head(control_context)
            incumbent_mean = incumbent_params[0]
            incumbent_log_std = torch.clamp(incumbent_params[1], min=-4.0, max=1.0)
            if deterministic:
                incumbent_mix_raw = incumbent_mean
            else:
                incumbent_mix_raw = incumbent_mean + torch.randn_like(incumbent_mean) * incumbent_log_std.exp()
            incumbent_mix = torch.sigmoid(incumbent_mix_raw)
            discover_mode_name = self._discover_mode_name(graph)
            if phase_name == "DISCOVER":
                if discover_mode_name == "spread_first":
                    incumbent_mix = torch.clamp(incumbent_mix, min=0.0, max=0.05)
                elif discover_mode_name == "wire_first":
                    incumbent_mix = torch.clamp(incumbent_mix, min=0.0, max=0.10)
                elif discover_mode_name == "macro_clearance":
                    incumbent_mix = torch.clamp(incumbent_mix, min=0.0, max=0.20)
            if phase_name == "REFINE":
                incumbent_mix = torch.clamp(incumbent_mix, min=0.50, max=1.0)
        else:
            incumbent_mean = torch.zeros((), dtype=h.dtype, device=h.device)
            incumbent_log_std = torch.zeros((), dtype=h.dtype, device=h.device)
            incumbent_mix_raw = torch.zeros((), dtype=h.dtype, device=h.device)
            incumbent_mix = torch.zeros((), dtype=h.dtype, device=h.device)

        k_logits = self._masked_k_logits(
            control_context,
            phase_name,
            strict_post_legal_mode=strict_post_legal_mode,
        )
        if deterministic:
            k_index = torch.argmax(k_logits, dim=-1)
        else:
            k_index = torch.distributions.Categorical(logits=k_logits).sample()

        phase_request_logits = self._phase_request_logits(
            context,
            phase_name,
            strict_post_legal_mode=strict_post_legal_mode,
        )
        if deterministic:
            phase_request = torch.argmax(phase_request_logits, dim=-1)
        else:
            phase_request = torch.distributions.Categorical(logits=phase_request_logits).sample()

        if head_config["enable_stop"]:
            stop_logit_bias = graph.get(
                "stop_logit_bias",
                torch.zeros((), dtype=h.dtype, device=h.device),
            ).to(dtype=h.dtype, device=h.device)
            stop_logits = self.stop_head(context).squeeze(-1) + stop_logit_bias
            stop_probability = torch.sigmoid(stop_logits)
            if deterministic:
                stop = (stop_probability > 0.5).to(dtype=h.dtype)
            else:
                stop = torch.bernoulli(stop_probability).to(dtype=h.dtype)
        else:
            stop_logit_bias = torch.zeros((), dtype=h.dtype, device=h.device)
            stop_logits = torch.zeros((), dtype=h.dtype, device=h.device)
            stop_probability = torch.zeros((), dtype=h.dtype, device=h.device)
            stop = torch.zeros((), dtype=h.dtype, device=h.device)

        if head_config["enable_unlock"]:
            pair_input = self._pair_inputs(h, graph)
            if pair_input.numel() > 0:
                unlock_source_logits = self.unlock_source_head(pair_input).squeeze(-1)
                if deterministic:
                    unlock_source_index = torch.argmax(unlock_source_logits, dim=-1)
                else:
                    unlock_source_index = torch.distributions.Categorical(logits=unlock_source_logits).sample()
            else:
                unlock_source_logits = torch.zeros((1,), dtype=h.dtype, device=h.device)
                unlock_source_index = torch.zeros((), dtype=torch.long, device=h.device)
            unlock_radius_logits = self.unlock_radius_head(context)
            if deterministic:
                unlock_radius_index = torch.argmax(unlock_radius_logits, dim=-1)
            else:
                unlock_radius_index = torch.distributions.Categorical(logits=unlock_radius_logits).sample()
        else:
            unlock_source_logits = torch.empty(0, dtype=h.dtype, device=h.device)
            unlock_source_index = torch.zeros((), dtype=torch.long, device=h.device)
            unlock_radius_logits = torch.empty(0, dtype=h.dtype, device=h.device)
            unlock_radius_index = torch.zeros((), dtype=torch.long, device=h.device)

        if head_config["enable_clusters"]:
            cluster_logits = self.cluster_head(h_discrete)
            if deterministic:
                sampled_clusters = torch.argmax(cluster_logits, dim=-1)
            else:
                sampled_clusters = torch.distributions.Categorical(logits=cluster_logits).sample()
            cluster_ids = torch.full((h.shape[0],), -1, dtype=torch.long, device=h.device)
            cluster_ids[std_mask] = sampled_clusters[std_mask]
            cluster_h = self._cluster_embeddings(h_discrete, cluster_ids, discrete_context)
            cluster_seq_plus, cluster_seq_minus, cluster_plus_scores, cluster_minus_scores = self._sample_cluster_sequence_pair(
                cluster_h,
                temperature,
                deterministic=deterministic,
            )
        else:
            cluster_logits = torch.empty((0, self.num_clusters), dtype=h.dtype, device=h.device)
            cluster_ids = torch.full((h.shape[0],), -1, dtype=torch.long, device=h.device)
            cluster_seq_plus = torch.empty(0, dtype=torch.long, device=h.device)
            cluster_seq_minus = torch.empty(0, dtype=torch.long, device=h.device)
            cluster_plus_scores = torch.empty(0, dtype=h.dtype, device=h.device)
            cluster_minus_scores = torch.empty(0, dtype=h.dtype, device=h.device)

        if head_config["enable_dag_ordering"] or head_config["enable_pair_branches"]:
            dag_axis, dag_axis_logits, pair_branch_choices, pair_branch_logits = self._sample_pair_discrete_actions(
                h,
                graph,
                deterministic=deterministic,
            )
        else:
            dag_axis = torch.empty(0, dtype=torch.long, device=h.device)
            dag_axis_logits = torch.empty((0, 2), dtype=h.dtype, device=h.device)
            pair_branch_choices = torch.empty(0, dtype=torch.long, device=h.device)
            pair_branch_logits = torch.empty((0, 4), dtype=h.dtype, device=h.device)
        pressure_action = self._sample_constraint_pressures(control_h, graph, control_context, deterministic=deterministic)

        group_logprobs, group_entropies, group_token_counts = self._group_stats(
            graph,
            seq_plus,
            seq_minus,
            macro_seq_plus,
            macro_seq_minus,
            cluster_seq_plus,
            cluster_seq_minus,
            plus_scores,
            minus_scores,
            macro_plus_scores,
            macro_minus_scores,
            cluster_plus_scores,
            cluster_minus_scores,
            dag_axis,
            dag_axis_logits,
            pair_branch_choices,
            pair_branch_logits,
            residual_flow,
            residual_local,
            global_flow_latent,
            residual_mean,
            residual_log_std,
            global_mean,
            global_log_std,
            control_raw,
            control_mean,
            control_log_std,
            incumbent_mix_raw,
            incumbent_mean,
            incumbent_log_std,
            k_index,
            k_logits,
            phase_request,
            phase_request_logits,
            stop,
            stop_logits,
            unlock_source_index,
            unlock_source_logits,
            unlock_radius_index,
            unlock_radius_logits,
            cluster_ids,
            cluster_logits,
            std_mask,
            pressure_action,
        )

        return PlacementPolicyAction(
            seq_plus=seq_plus,
            seq_minus=seq_minus,
            macro_seq_plus=macro_seq_plus,
            macro_seq_minus=macro_seq_minus,
            macro_cell_indices=macro_cell_indices,
            cluster_seq_plus=cluster_seq_plus,
            cluster_seq_minus=cluster_seq_minus,
            residual_flow=residual_flow,
            residual_local=residual_local,
            global_flow_latent=global_flow_latent,
            control_raw=control_raw,
            incumbent_mix_raw=incumbent_mix_raw,
            k_index=k_index,
            stop=stop,
            stop_probability=stop_probability,
            stop_logit_bias=stop_logit_bias,
            cluster_ids=cluster_ids,
            dag_axis=dag_axis,
            pair_branch_choices=pair_branch_choices,
            ordering_representation=head_config["ordering_representation"],
            branch_mode=head_config["branch_mode"],
            phase_name=phase_name,
            enable_clusters=head_config["enable_clusters"],
            enable_stop=head_config["enable_stop"],
            enable_unlock=head_config["enable_unlock"],
            memory=memory,
            next_memory=context,
            incumbent_mix=incumbent_mix,
            phase_request=phase_request,
            phase_request_logits=phase_request_logits,
            unlock_source_index=unlock_source_index,
            unlock_source_logits=unlock_source_logits,
            unlock_radius_index=unlock_radius_index,
            unlock_radius_logits=unlock_radius_logits,
            branch_pressure_raw=pressure_action["branch_raw"],
            boundary_pressure_raw=pressure_action["boundary_raw"],
            density_pressure_raw=pressure_action["density_raw"],
            branch_pressure_values=pressure_action["branch_values"],
            boundary_pressure_values=pressure_action["boundary_values"],
            density_pressure_values=pressure_action["density_values"],
            value=value,
            plus_scores=plus_scores,
            minus_scores=minus_scores,
            macro_plus_scores=macro_plus_scores,
            macro_minus_scores=macro_minus_scores,
            cluster_plus_scores=cluster_plus_scores,
            cluster_minus_scores=cluster_minus_scores,
            dag_axis_logits=dag_axis_logits,
            pair_branch_logits=pair_branch_logits,
            cluster_logits=cluster_logits,
            group_logprobs=group_logprobs,
            group_entropies=group_entropies,
            group_token_counts=group_token_counts,
            **controls,
        )

    def evaluate_action(self, graph, action, temperature=1.0):
        h = self.encode(graph)
        discrete_graph = self._discrete_basin_graph(graph)
        h_discrete = h if discrete_graph is graph else self.encode(discrete_graph)
        pooled, base_context = self._global_features(h)
        _memory, recurrent_context = self._memory_pair(h, graph)
        head_config = self._active_head_config(graph, std_mask=self._standard_cell_mask(graph))
        phase_name = head_config["phase_name"]
        strict_post_legal_mode = head_config["strict_post_legal_mode"]
        context = self._phase_context(recurrent_context, graph)
        stateless_phase_context = self._phase_context(base_context, graph)
        control_context = self._control_context(context, stateless_phase_context, graph)
        control_h = self._control_node_embeddings(h, h_discrete, graph)
        discrete_context = self._discover_discrete_context(context, stateless_phase_context, graph)
        value = self.value_head(pooled).squeeze(-1)
        std_mask = self._standard_cell_mask(graph)

        plus_scores = self.plus_head(h_discrete).squeeze(-1) / max(float(temperature), 1e-4)
        if head_config["enable_ordering_actions"]:
            minus_scores = self._conditioned_minus_scores(h_discrete, self.minus_base_head(h_discrete).squeeze(-1), action.seq_plus)
            minus_scores = minus_scores / max(float(temperature), 1e-4)
        else:
            minus_scores = self.minus_base_head(h_discrete).squeeze(-1) / max(float(temperature), 1e-4)
        if head_config["enable_macro_ordering"]:
            macro_cell_indices, macro_seq_plus, macro_seq_minus, macro_plus_scores, macro_minus_scores = (
                self._sample_macro_sequence_pair(
                    h_discrete,
                    discrete_graph,
                    temperature,
                    deterministic=False,
                    seq_plus=action.macro_seq_plus,
                )
            )
        else:
            macro_cell_indices = torch.empty(0, dtype=torch.long, device=h.device)
            macro_seq_plus = torch.empty(0, dtype=torch.long, device=h.device)
            macro_seq_minus = torch.empty(0, dtype=torch.long, device=h.device)
            macro_plus_scores = torch.empty(0, dtype=h.dtype, device=h.device)
            macro_minus_scores = torch.empty(0, dtype=h.dtype, device=h.device)

        node_context = control_context.unsqueeze(0).expand_as(control_h)
        residual_params = self.residual_head(torch.cat([control_h, node_context], dim=1))
        residual_mean = residual_params[:, :2]
        residual_log_std = torch.clamp(residual_params[:, 2:], min=-4.0, max=1.0)
        global_params = self.global_flow_head(control_context)
        global_mean = global_params[: self.global_flow_rank]
        global_log_std = torch.clamp(global_params[self.global_flow_rank :], min=-4.0, max=1.0)
        global_mean, global_log_std = self._phase_adjust_global_distribution(
            global_mean,
            global_log_std,
            phase_name,
            graph,
            strict_post_legal_mode=strict_post_legal_mode,
        )

        control_params = self.control_head(control_context)
        control_mean = control_params[: len(CONTROL_NAMES)]
        control_log_std = torch.clamp(control_params[len(CONTROL_NAMES) :], min=-4.0, max=1.0)
        if head_config["enable_incumbent_action"] and self.incumbent_control_head is not None:
            incumbent_params = self.incumbent_control_head(control_context)
            incumbent_mean = incumbent_params[0]
            incumbent_log_std = torch.clamp(incumbent_params[1], min=-4.0, max=1.0)
        else:
            incumbent_mean = torch.zeros((), dtype=h.dtype, device=h.device)
            incumbent_log_std = torch.zeros((), dtype=h.dtype, device=h.device)

        k_logits = self._masked_k_logits(
            control_context,
            phase_name,
            strict_post_legal_mode=strict_post_legal_mode,
        )
        phase_request_logits = self._phase_request_logits(
            context,
            phase_name,
            strict_post_legal_mode=strict_post_legal_mode,
        )
        if head_config["enable_stop"]:
            stop_logits = self.stop_head(context).squeeze(-1) + graph.get(
                "stop_logit_bias",
                torch.zeros((), dtype=h.dtype, device=h.device),
            ).to(dtype=h.dtype, device=h.device)
        else:
            stop_logits = torch.zeros((), dtype=h.dtype, device=h.device)
        if head_config["enable_unlock"]:
            pair_input = self._pair_inputs(h, graph)
            if pair_input.numel() > 0:
                unlock_source_logits = self.unlock_source_head(pair_input).squeeze(-1)
            else:
                unlock_source_logits = torch.zeros((1,), dtype=h.dtype, device=h.device)
            unlock_radius_logits = self.unlock_radius_head(context)
        else:
            unlock_source_logits = torch.empty(0, dtype=h.dtype, device=h.device)
            unlock_radius_logits = torch.empty(0, dtype=h.dtype, device=h.device)
        if head_config["enable_clusters"]:
            cluster_logits = self.cluster_head(h_discrete)
            cluster_h = self._cluster_embeddings(h_discrete, action.cluster_ids, discrete_context)
            cluster_seq_plus, cluster_seq_minus, cluster_plus_scores, cluster_minus_scores = self._sample_cluster_sequence_pair(
                cluster_h,
                temperature,
                deterministic=False,
                seq_plus=action.cluster_seq_plus,
            )
        else:
            cluster_logits = torch.empty((0, self.num_clusters), dtype=h.dtype, device=h.device)
            cluster_seq_plus = torch.empty(0, dtype=torch.long, device=h.device)
            cluster_seq_minus = torch.empty(0, dtype=torch.long, device=h.device)
            cluster_plus_scores = torch.empty(0, dtype=h.dtype, device=h.device)
            cluster_minus_scores = torch.empty(0, dtype=h.dtype, device=h.device)
        if head_config["enable_dag_ordering"] or head_config["enable_pair_branches"]:
            _dag_axis, dag_axis_logits, _pair_branch_choices, pair_branch_logits = self._sample_pair_discrete_actions(
                h,
                graph,
                deterministic=True,
            )
        else:
            dag_axis_logits = torch.empty((0, 2), dtype=h.dtype, device=h.device)
            pair_branch_logits = torch.empty((0, 4), dtype=h.dtype, device=h.device)
        pressure_eval = self._sample_constraint_pressures(control_h, graph, control_context, deterministic=True)
        pressure_eval["branch_raw"] = action.branch_pressure_raw
        pressure_eval["boundary_raw"] = action.boundary_pressure_raw
        pressure_eval["density_raw"] = action.density_pressure_raw

        group_logprobs, group_entropies, group_token_counts = self._group_stats(
            graph,
            action.seq_plus,
            action.seq_minus,
            action.macro_seq_plus,
            action.macro_seq_minus,
            action.cluster_seq_plus,
            action.cluster_seq_minus,
            plus_scores,
            minus_scores,
            macro_plus_scores,
            macro_minus_scores,
            cluster_plus_scores,
            cluster_minus_scores,
            action.dag_axis,
            dag_axis_logits,
            action.pair_branch_choices,
            pair_branch_logits,
            action.residual_flow,
            action.residual_local,
            action.global_flow_latent,
            residual_mean,
            residual_log_std,
            global_mean,
            global_log_std,
            action.control_raw,
            control_mean,
            control_log_std,
            action.incumbent_mix_raw,
            incumbent_mean,
            incumbent_log_std,
            action.k_index,
            k_logits,
            action.phase_request,
            phase_request_logits,
            action.stop,
            stop_logits,
            action.unlock_source_index,
            unlock_source_logits,
            action.unlock_radius_index,
            unlock_radius_logits,
            action.cluster_ids,
            cluster_logits,
            std_mask,
            pressure_eval,
        )
        return group_logprobs, group_entropies, value, group_token_counts

    def deterministic_action(self, graph):
        return self.sample_action(graph, deterministic=True)

    def initial_memory(self, device=None, dtype=None):
        parameter = next(self.parameters())
        return torch.zeros(
            self.hidden_dim,
            dtype=parameter.dtype if dtype is None else dtype,
            device=parameter.device if device is None else device,
        )

    def equivariance_loss(self, graph):
        """Permutation-equivariance sanity loss for node-index relabeling."""
        cell_features = graph["cell_features"]
        n = int(cell_features.shape[0])
        if n <= 1:
            return cell_features.sum() * 0.0
        perm = torch.randperm(n, device=cell_features.device)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(n, device=cell_features.device)

        relabeled = dict(graph)
        relabeled["cell_features"] = graph["cell_features"][perm]
        relabeled_pin_features = graph["pin_features"].clone()
        if relabeled_pin_features.numel() > 0:
            relabeled_pin_features[:, 0] = inv_perm[relabeled_pin_features[:, 0].long()].to(relabeled_pin_features.dtype)
        relabeled["pin_features"] = relabeled_pin_features
        if graph["active_pairs"].numel() > 0:
            mapped_pairs = inv_perm[graph["active_pairs"].long()]
            mapped_pairs = torch.stack(
                [torch.minimum(mapped_pairs[:, 0], mapped_pairs[:, 1]), torch.maximum(mapped_pairs[:, 0], mapped_pairs[:, 1])],
                dim=1,
            )
            relabeled["active_pairs"] = mapped_pairs
            relabeled["branch_duals"] = graph["branch_duals"]
        relabeled["boundary_duals"] = graph["boundary_duals"][perm]
        relabeled["wirelength_grad"] = graph["wirelength_grad"][perm]
        relabeled["density_pressure"] = graph["density_pressure"][perm]
        if "incumbent_centers" in graph:
            relabeled["incumbent_centers"] = graph["incumbent_centers"][perm]

        original = self._node_policy_outputs(graph)
        permuted = self._node_policy_outputs(relabeled)
        restored = {}
        for key, value in permuted.items():
            restored[key] = torch.empty_like(value)
            restored[key][perm] = value
        losses = []
        for key, value in original.items():
            losses.append(F.mse_loss(value, restored[key]))
        return torch.stack(losses).mean()

    def _node_policy_outputs(self, graph):
        h = self.encode(graph)
        pooled, context = self._global_features(h)
        node_context = context.unsqueeze(0).expand_as(h)
        residual_params = self.residual_head(torch.cat([h, node_context], dim=1))
        return {
            "plus": self.plus_head(h),
            "minus": self.minus_base_head(h),
            "cluster": self.cluster_head(h),
            "residual": residual_params,
        }

    def _transform_controls(self, raw):
        return {
            "step_scale": lognormal_from_raw(raw[0], scale=0.04, min_value=0.005, max_value=0.40),
            "rho": lognormal_from_raw(raw[1], scale=2.0, min_value=0.50, max_value=32.0),
            "eta": lognormal_from_raw(raw[2], scale=0.02, min_value=0.0005, max_value=1.0),
            "alpha": lognormal_from_raw(raw[3], scale=0.01, min_value=0.0001, max_value=0.25),
            "branch_pressure": lognormal_from_raw(raw[4], scale=0.50, min_value=0.05, max_value=8.0),
            "density_pressure": lognormal_from_raw(raw[5], scale=0.50, min_value=0.05, max_value=8.0),
            "boundary_pressure": lognormal_from_raw(raw[6], scale=0.35, min_value=0.05, max_value=6.0),
            "pair_emphasis": torch.sigmoid(raw[7]),
            "tau": lognormal_from_raw(raw[8], scale=0.25, min_value=0.02, max_value=4.0),
        }

    def _standard_cell_mask(self, graph):
        area = graph["cell_features"][:, 0]
        return area <= 3.0 + 1e-6

    def _macro_cell_mask(self, graph):
        area = graph["cell_features"][:, 0]
        return area > 3.0 + 1e-6

    def _group_stats(
        self,
        graph,
        seq_plus,
        seq_minus,
        macro_seq_plus,
        macro_seq_minus,
        cluster_seq_plus,
        cluster_seq_minus,
        plus_scores,
        minus_scores,
        macro_plus_scores,
        macro_minus_scores,
        cluster_plus_scores,
        cluster_minus_scores,
        dag_axis,
        dag_axis_logits,
        pair_branch_choices,
        pair_branch_logits,
        residual_flow,
        residual_local,
        global_flow_latent,
        residual_mean,
        residual_log_std,
        global_mean,
        global_log_std,
        control_raw,
        control_mean,
        control_log_std,
        incumbent_mix_raw,
        incumbent_mix_mean,
        incumbent_mix_log_std,
        k_index,
        k_logits,
        phase_request,
        phase_request_logits,
        stop,
        stop_logits,
        unlock_source_index,
        unlock_source_logits,
        unlock_radius_index,
        unlock_radius_logits,
        cluster_ids,
        cluster_logits,
        std_mask,
        pressure_action,
    ):
        zero = plus_scores.sum() * 0.0
        head_config = self._active_head_config(graph, std_mask=std_mask)
        dag_logprob = (
            F.log_softmax(dag_axis_logits, dim=-1).gather(1, dag_axis.long().reshape(-1, 1)).sum()
            if dag_axis_logits.numel() > 0
            else zero
        )
        pair_branch_logprob = (
            F.log_softmax(pair_branch_logits, dim=-1).gather(1, pair_branch_choices.long().reshape(-1, 1)).sum()
            if pair_branch_logits.numel() > 0
            else zero
        )
        pressure_logprob = (
            normal_logprob(
                pressure_action["branch_raw"],
                pressure_action["branch_mean"],
                pressure_action["branch_log_std"],
            ).sum()
            + normal_logprob(
                pressure_action["boundary_raw"],
                pressure_action["boundary_mean"],
                pressure_action["boundary_log_std"],
            ).sum()
            + normal_logprob(
                pressure_action["density_raw"],
                pressure_action["density_mean"],
                pressure_action["density_log_std"],
            ).sum()
        )
        group_logprobs = {
            "ordering": plackett_luce_logprob(plus_scores, seq_plus)
            + plackett_luce_logprob(minus_scores, seq_minus),
            "macro_ordering": plackett_luce_logprob(macro_plus_scores, macro_seq_plus)
            + plackett_luce_logprob(macro_minus_scores, macro_seq_minus),
            "constraint_pressure": pressure_logprob,
            "residual": normal_logprob(residual_local, residual_mean, residual_log_std).sum()
            + normal_logprob(global_flow_latent, global_mean, global_log_std).sum(),
            "pd_controls": normal_logprob(control_raw, control_mean, control_log_std).sum()
            + categorical_logprob(k_index, k_logits),
            "phase_control": categorical_logprob(phase_request, phase_request_logits),
        }
        group_entropies = {
            "ordering": categorical_entropy_from_scores(plus_scores) + categorical_entropy_from_scores(minus_scores),
            "macro_ordering": categorical_entropy_from_scores(macro_plus_scores)
            + categorical_entropy_from_scores(macro_minus_scores),
            "constraint_pressure": normal_entropy(pressure_action["branch_log_std"]).sum()
            + normal_entropy(pressure_action["boundary_log_std"]).sum()
            + normal_entropy(pressure_action["density_log_std"]).sum(),
            "residual": normal_entropy(residual_log_std).sum() + normal_entropy(global_log_std).sum(),
            "pd_controls": normal_entropy(control_log_std).sum() + categorical_entropy(k_logits),
            "phase_control": categorical_entropy(phase_request_logits),
        }
        group_token_counts = {
            "ordering": torch.tensor(max(int(seq_plus.numel() + seq_minus.numel()), 1), device=plus_scores.device),
            "macro_ordering": torch.tensor(
                max(int(macro_seq_plus.numel() + macro_seq_minus.numel()), 1),
                device=plus_scores.device,
            ),
            "constraint_pressure": torch.tensor(
                max(
                    int(
                        pressure_action["branch_raw"].numel()
                        + pressure_action["boundary_raw"].numel()
                        + pressure_action["density_raw"].numel()
                    ),
                    1,
                ),
                device=plus_scores.device,
            ),
            "residual": torch.tensor(
                max(int(residual_local.numel() + global_flow_latent.numel()), 1),
                device=plus_scores.device,
            ),
            "pd_controls": torch.tensor(len(CONTROL_NAMES) + 1, device=plus_scores.device),
            "phase_control": torch.tensor(1, device=plus_scores.device),
        }
        if not head_config["enable_ordering_actions"]:
            group_logprobs["ordering"] = zero
            group_entropies["ordering"] = zero
        if not head_config["enable_macro_ordering"]:
            group_logprobs["macro_ordering"] = zero
            group_entropies["macro_ordering"] = zero
            group_token_counts["macro_ordering"] = torch.tensor(1, device=plus_scores.device)
        if head_config["enable_incumbent_action"]:
            group_logprobs["incumbent"] = normal_logprob(
                incumbent_mix_raw,
                incumbent_mix_mean,
                incumbent_mix_log_std,
            ).sum()
            group_entropies["incumbent"] = normal_entropy(incumbent_mix_log_std).sum()
            group_token_counts["incumbent"] = torch.tensor(1, device=plus_scores.device)
        if head_config["enable_dag_ordering"]:
            group_logprobs["dag_ordering"] = dag_logprob
            group_entropies["dag_ordering"] = categorical_entropy(dag_axis_logits).sum() if dag_axis_logits.numel() > 0 else zero
            group_token_counts["dag_ordering"] = torch.tensor(max(int(dag_axis.numel()), 1), device=plus_scores.device)
        if head_config["enable_pair_branches"]:
            group_logprobs["pair_branches"] = pair_branch_logprob
            group_entropies["pair_branches"] = categorical_entropy(pair_branch_logits).sum() if pair_branch_logits.numel() > 0 else zero
            group_token_counts["pair_branches"] = torch.tensor(max(int(pair_branch_choices.numel()), 1), device=plus_scores.device)
        if head_config["enable_stop"]:
            group_logprobs["stop"] = bernoulli_logprob(stop, stop_logits).sum()
            group_entropies["stop"] = bernoulli_entropy(stop_logits).sum()
            group_token_counts["stop"] = torch.tensor(1, device=plus_scores.device)
        if head_config["enable_unlock"]:
            group_logprobs["unlock"] = categorical_logprob(unlock_source_index, unlock_source_logits) + categorical_logprob(
                unlock_radius_index,
                unlock_radius_logits,
            )
            group_entropies["unlock"] = categorical_entropy(unlock_source_logits).sum() + categorical_entropy(unlock_radius_logits).sum()
            group_token_counts["unlock"] = torch.tensor(2, device=plus_scores.device)
        if head_config["enable_clusters"]:
            group_logprobs["cluster_ordering"] = plackett_luce_logprob(cluster_plus_scores, cluster_seq_plus) + plackett_luce_logprob(cluster_minus_scores, cluster_seq_minus)
            group_entropies["cluster_ordering"] = categorical_entropy_from_scores(cluster_plus_scores) + categorical_entropy_from_scores(cluster_minus_scores)
            group_token_counts["cluster_ordering"] = torch.tensor(
                max(int(cluster_seq_plus.numel() + cluster_seq_minus.numel()), 1),
                device=plus_scores.device,
            )
        if head_config["enable_clusters"] and torch.any(std_mask):
            std_logits = cluster_logits[std_mask]
            std_clusters = torch.clamp(cluster_ids[std_mask], min=0)
            cluster_logprobs = F.log_softmax(std_logits, dim=-1).gather(1, std_clusters[:, None]).sum()
            cluster_entropies = categorical_entropy(std_logits).sum()
            cluster_tokens = int(std_clusters.numel())
            group_logprobs["clusters"] = cluster_logprobs
            group_entropies["clusters"] = cluster_entropies
            group_token_counts["clusters"] = torch.tensor(cluster_tokens, device=plus_scores.device)
        elif head_config["enable_clusters"]:
            cluster_logprobs = plus_scores.sum() * 0.0
            cluster_entropies = plus_scores.sum() * 0.0
            cluster_tokens = 1
            group_logprobs["clusters"] = cluster_logprobs
            group_entropies["clusters"] = cluster_entropies
            group_token_counts["clusters"] = torch.tensor(cluster_tokens, device=plus_scores.device)
        group_logprobs = {
            key: _sanitize_tensor(value, pos=LOGPROB_CLAMP, neg=-LOGPROB_CLAMP)
            for key, value in group_logprobs.items()
        }
        group_entropies = {
            key: _sanitize_tensor(value, pos=LOGPROB_CLAMP, neg=0.0)
            for key, value in group_entropies.items()
        }
        return group_logprobs, group_entropies, group_token_counts

    def _fit_feature_dim(self, features, target_dim):
        if features.shape[1] == target_dim:
            return features
        if features.shape[1] > target_dim:
            return features[:, :target_dim]
        pad = features.new_zeros((features.shape[0], target_dim - features.shape[1]))
        return torch.cat([features, pad], dim=1)

    def _node_features(self, graph):
        cell_features = graph["cell_features"]
        active_pairs = graph["active_pairs"]
        branch_duals = graph["branch_duals"]
        boundary_duals = graph["boundary_duals"]
        wirelength_grad = graph["wirelength_grad"]
        density_pressure_input = graph["density_pressure"]

        n = cell_features.shape[0]
        dtype = cell_features.dtype
        device = cell_features.device

        area = torch.clamp(cell_features[:, 0], min=1e-6)
        total_area = torch.clamp(area.sum(), min=1.0)
        length_scale = torch.sqrt(total_area)
        width = cell_features[:, 4]
        height = cell_features[:, 5]
        centers = cell_features[:, 2:4]
        pin_count = cell_features[:, 1]
        macro = (area > 3.0).to(dtype)

        active_count = torch.zeros(n, dtype=dtype, device=device)
        dual_pressure = torch.zeros(n, dtype=dtype, device=device)
        if active_pairs.numel() > 0:
            pair_pressure = branch_duals.sum(dim=1)
            ones = torch.ones(active_pairs.shape[0], dtype=dtype, device=device)
            i = active_pairs[:, 0].long()
            j = active_pairs[:, 1].long()
            active_count.index_add_(0, i, ones)
            active_count.index_add_(0, j, ones)
            dual_pressure.index_add_(0, i, pair_pressure)
            dual_pressure.index_add_(0, j, pair_pressure)

        boundary_pressure = boundary_duals.sum(dim=1)
        density_pressure = density_pressure_input.to(dtype)
        centered = centers - centers.mean(dim=0, keepdim=True)
        std = torch.clamp(centered.std(dim=0), min=1.0)
        use_incumbent_state = _graph_bool(graph, "enable_incumbent_state", True)
        incumbent_centers = graph.get("incumbent_centers")
        if (
            not use_incumbent_state
            or incumbent_centers is None
            or incumbent_centers.shape != centers.shape
        ):
            incumbent_centers = centers
            incumbent_active = 0.0
        else:
            incumbent_centers = incumbent_centers.to(dtype=dtype, device=device)
            incumbent_active = 1.0
        current_overlap_ratio = _graph_scalar(graph, "exact_overlap_ratio", 0.0, dtype=dtype, device=device)
        current_normalized_wl = _graph_scalar(graph, "current_normalized_wl", 0.0, dtype=dtype, device=device)
        current_num_overlap_pairs = _graph_scalar(graph, "current_num_overlap_pairs", 0.0, dtype=dtype, device=device)
        incumbent_overlap_ratio = _graph_scalar(graph, "incumbent_overlap_ratio", current_overlap_ratio, dtype=dtype, device=device)
        incumbent_normalized_wl = _graph_scalar(graph, "incumbent_normalized_wl", current_normalized_wl, dtype=dtype, device=device)
        incumbent_num_overlap_pairs = _graph_scalar(graph, "incumbent_num_overlap_pairs", current_num_overlap_pairs, dtype=dtype, device=device)
        steps_since_best = _graph_scalar(graph, "steps_since_best", 0.0, dtype=dtype, device=device)

        pin_norm = torch.clamp(pin_count.max(), min=1.0)
        active_norm = torch.clamp(active_count.max(), min=1.0)
        dual_norm = torch.clamp(dual_pressure.abs().max(), min=1.0)
        boundary_norm = torch.clamp(boundary_pressure.abs().max(), min=1.0)
        density_norm = torch.clamp(density_pressure.abs().max(), min=1.0)
        grad_norm = torch.clamp(wirelength_grad.norm(dim=1).max(), min=1.0)
        incumbent_delta = (centers - incumbent_centers) / length_scale
        incumbent_dist = incumbent_delta.norm(dim=1)
        overlap_gap = torch.clamp(current_overlap_ratio - incumbent_overlap_ratio, min=0.0).expand(n)
        wire_gap = (current_normalized_wl - incumbent_normalized_wl).expand(n)
        pair_gap = ((current_num_overlap_pairs - incumbent_num_overlap_pairs) / max(float(n), 1.0)).expand(n)
        steps_feature = (
            torch.log1p(torch.clamp(steps_since_best, min=0.0)) / math.log(10.0)
        ).expand(n)
        incumbent_active_feature = torch.full((n,), float(incumbent_active), dtype=dtype, device=device)

        features = torch.stack(
            [
                torch.log1p(area) / torch.log1p(total_area),
                width / length_scale,
                height / length_scale,
                pin_count / pin_norm,
                centered[:, 0] / std[0],
                centered[:, 1] / std[1],
                centers[:, 0] / length_scale,
                centers[:, 1] / length_scale,
                macro,
                active_count / active_norm,
                dual_pressure / dual_norm,
                boundary_pressure / boundary_norm,
                density_pressure / density_norm,
                wirelength_grad[:, 0] / grad_norm,
                wirelength_grad[:, 1] / grad_norm,
                torch.sqrt(area) / length_scale,
                incumbent_delta[:, 0],
                incumbent_delta[:, 1],
                incumbent_dist,
                overlap_gap,
                wire_gap,
                pair_gap,
                steps_feature,
                incumbent_active_feature,
            ],
            dim=1,
        )
        return self._fit_feature_dim(features, self.node_feature_dim)

    def _pair_features(self, graph):
        cell_features = graph["cell_features"]
        active_pairs = graph["active_pairs"]
        branch_duals = graph["branch_duals"]
        if active_pairs.numel() == 0:
            return torch.empty((0, self.pair_feature_dim), dtype=cell_features.dtype, device=cell_features.device)
        centers = cell_features[:, 2:4]
        widths = cell_features[:, 4]
        heights = cell_features[:, 5]
        total_area = torch.clamp(cell_features[:, 0].sum(), min=1.0)
        length_scale = torch.sqrt(total_area)
        use_incumbent_state = _graph_bool(graph, "enable_incumbent_state", True)
        incumbent_centers = graph.get("incumbent_centers")
        if (
            not use_incumbent_state
            or incumbent_centers is None
            or incumbent_centers.shape != centers.shape
        ):
            incumbent_centers = centers
        else:
            incumbent_centers = incumbent_centers.to(dtype=centers.dtype, device=centers.device)
        i = active_pairs[:, 0].long()
        j = active_pairs[:, 1].long()
        dx = (centers[j, 0] - centers[i, 0]) / length_scale
        dy = (centers[j, 1] - centers[i, 1]) / length_scale
        sep_x = torch.abs(dx)
        sep_y = torch.abs(dy)
        overlap_x = torch.relu(0.5 * (widths[i] + widths[j]) / length_scale - sep_x)
        overlap_y = torch.relu(0.5 * (heights[i] + heights[j]) / length_scale - sep_y)
        incumbent_dx = (incumbent_centers[j, 0] - incumbent_centers[i, 0]) / length_scale
        incumbent_dy = (incumbent_centers[j, 1] - incumbent_centers[i, 1]) / length_scale
        incumbent_overlap_x = torch.relu(
            0.5 * (widths[i] + widths[j]) / length_scale - torch.abs(incumbent_dx)
        )
        incumbent_overlap_y = torch.relu(
            0.5 * (heights[i] + heights[j]) / length_scale - torch.abs(incumbent_dy)
        )
        dual_norm = torch.clamp(branch_duals.abs().max(), min=1.0)
        features = torch.stack(
            [
                dx,
                dy,
                (widths[i] + widths[j]) / length_scale,
                (heights[i] + heights[j]) / length_scale,
                overlap_x,
                overlap_y,
                branch_duals[:, 0] / dual_norm,
                branch_duals[:, 1] / dual_norm,
                branch_duals[:, 2] / dual_norm,
                branch_duals[:, 3] / dual_norm,
                incumbent_dx,
                incumbent_dy,
                incumbent_overlap_x,
                incumbent_overlap_y,
                overlap_x - incumbent_overlap_x,
                overlap_y - incumbent_overlap_y,
            ],
            dim=1,
        )
        return self._fit_feature_dim(features, self.pair_feature_dim)


def policy_config_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return {
        "node_feature_dim": int(config.get("node_feature_dim", LEGACY_NODE_FEATURE_DIM)),
        "pair_feature_dim": int(config.get("pair_feature_dim", LEGACY_PAIR_FEATURE_DIM)),
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "message_passes": int(config.get("message_passes", 2)),
        "num_clusters": int(config.get("num_clusters", 8)),
        "global_flow_rank": int(config.get("global_flow_rank", 2)),
        "enable_incumbent_controls": bool(config.get("enable_incumbent_controls", False)),
        "chooser_legacy_residual_weight": float(config.get("chooser_legacy_residual_weight", 0.0)),
    }


def save_policy_checkpoint(policy, path, config=None, stats=None, extra=None):
    payload = {
        "model_state": policy.state_dict(),
        "config": {
            "node_feature_dim": policy.node_feature_dim,
            "pair_feature_dim": policy.pair_feature_dim,
            "hidden_dim": policy.hidden_dim,
            "message_passes": policy.message_passes,
            "num_clusters": policy.num_clusters,
            "global_flow_rank": policy.global_flow_rank,
            "enable_incumbent_controls": policy.enable_incumbent_controls,
            "chooser_legacy_residual_weight": policy.chooser_legacy_residual_weight,
            **(config or {}),
        },
        "stats": stats or {},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_policy_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = OrderingPolicy(**policy_config_from_checkpoint(checkpoint)).to(device)
    checkpoint_state = checkpoint["model_state"]
    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is None or target.shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state[key] = value
    load_result = model.load_state_dict(compatible_state, strict=False)
    checkpoint["_load_missing_keys"] = list(load_result.missing_keys)
    checkpoint["_load_unexpected_keys"] = list(load_result.unexpected_keys)
    checkpoint["_load_skipped_shape_keys"] = skipped_keys
    model.eval()
    return model, checkpoint
