"""Graph-conditioned policy for policy-owned primal-dual placement PPO."""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


NODE_FEATURE_DIM = 16
PD_K_CHOICES = (1, 2, 4, 8, 12, 16)
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
    k_index: torch.Tensor
    stop: torch.Tensor
    stop_probability: torch.Tensor
    stop_logit_bias: torch.Tensor
    cluster_ids: torch.Tensor
    dag_axis: torch.Tensor
    pair_branch_choices: torch.Tensor
    step_scale: torch.Tensor
    rho: torch.Tensor
    eta: torch.Tensor
    alpha: torch.Tensor
    branch_pressure: torch.Tensor
    density_pressure: torch.Tensor
    boundary_pressure: torch.Tensor
    pair_emphasis: torch.Tensor
    tau: torch.Tensor
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
    stop_logit_bias=None,
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
    if stop_logit_bias is None:
        stop_logit_bias = torch.zeros((), dtype=cell_features.dtype, device=cell_features.device)
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
        "stop_logit_bias": stop_logit_bias.detach() if torch.is_tensor(stop_logit_bias) else torch.tensor(float(stop_logit_bias), dtype=cell_features.dtype, device=cell_features.device),
    }


def graph_to_device(graph, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in graph.items()}


def plackett_luce_logprob(scores, sequence):
    """Exact log probability for a permutation under Plackett-Luce scores."""
    if sequence.numel() <= 1:
        return scores.sum() * 0.0
    ordered = scores[sequence.long()]
    suffix_logsumexp = torch.logcumsumexp(torch.flip(ordered, dims=[0]), dim=0).flip(0)
    return (ordered - suffix_logsumexp).sum()


def categorical_entropy_from_scores(scores):
    if scores.numel() <= 1:
        return scores.sum() * 0.0
    log_probs = F.log_softmax(scores, dim=0)
    probs = log_probs.exp()
    return -(probs * log_probs).sum()


def sample_plackett_luce(scores):
    """Sample a permutation via Gumbel top-k."""
    eps = torch.finfo(scores.dtype).eps
    uniform = torch.rand_like(scores).clamp_(eps, 1.0 - eps)
    gumbel = -torch.log(-torch.log(uniform))
    return torch.argsort(scores + gumbel, descending=True)


def normal_logprob(value, mean, log_std):
    log_std = torch.clamp(log_std, min=-5.0, max=2.0)
    var = torch.exp(2.0 * log_std)
    return -0.5 * (((value - mean).square() / var) + 2.0 * log_std + math.log(2.0 * math.pi))


def normal_entropy(log_std):
    log_std = torch.clamp(log_std, min=-5.0, max=2.0)
    return log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))


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
    return F.log_softmax(logits, dim=-1).gather(-1, index.long().reshape(1)).squeeze(0)


def categorical_entropy(logits):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


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
        hidden_dim=128,
        message_passes=2,
        num_clusters=8,
        global_flow_rank=2,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.message_passes = message_passes
        self.num_clusters = int(num_clusters)
        self.global_flow_rank = int(max(global_flow_rank, 1))

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
            nn.Linear(2 * hidden_dim + 10, hidden_dim),
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
        self.dag_axis_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 10, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.pair_branch_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 10, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.branch_pressure_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 10, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.boundary_pressure_head = nn.Linear(hidden_dim, 8)
        self.density_pressure_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, graph):
        cell_features = graph["cell_features"]
        pin_features = graph["pin_features"]
        edge_list = graph["edge_list"]
        x = self._node_features(graph)
        h = self.input_mlp(x)

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

        return h

    def _global_features(self, h, graph=None):
        mean_pool = h.mean(dim=0)
        max_pool = h.max(dim=0).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        base_context = self.context_mlp(pooled)
        if graph is None:
            return pooled, base_context
        memory = graph.get("memory")
        if memory is None or memory.numel() != self.hidden_dim:
            memory = torch.zeros(self.hidden_dim, dtype=h.dtype, device=h.device)
        else:
            memory = memory.to(dtype=h.dtype, device=h.device)
        next_memory = self.memory_cell(base_context.unsqueeze(0), memory.unsqueeze(0)).squeeze(0)
        return pooled, next_memory

    def _memory_pair(self, h, graph):
        memory = graph.get("memory")
        if memory is None or memory.numel() != self.hidden_dim:
            memory = torch.zeros(self.hidden_dim, dtype=h.dtype, device=h.device)
        else:
            memory = memory.to(dtype=h.dtype, device=h.device)
        _pooled, context = self._global_features(h, graph)
        return memory, context

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
        cluster_seq_minus = torch.argsort(cluster_minus_scores, descending=True) if deterministic else sample_plackett_luce(cluster_minus_scores)
        return cluster_seq_plus, cluster_seq_minus, cluster_plus_scores, cluster_minus_scores

    def _pair_inputs(self, h, graph):
        active_pairs = graph["active_pairs"]
        if active_pairs.numel() == 0:
            return torch.empty((0, 2 * self.hidden_dim + 10), dtype=h.dtype, device=h.device)
        pair_features = self._pair_features(graph)
        src = active_pairs[:, 0].long()
        dst = active_pairs[:, 1].long()
        return torch.cat([h[src], h[dst], pair_features], dim=1)

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
        if seq_plus is None:
            seq_plus = sample_plackett_luce(scaled_plus_scores)
        minus_scores = self._conditioned_minus_scores(h, base_minus_scores, seq_plus)
        scaled_minus_scores = minus_scores / max(float(temperature), 1e-4)
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
        pooled, _base_context = self._global_features(h)
        memory, context = self._memory_pair(h, graph)
        value = self.value_head(pooled).squeeze(-1)

        plus_scores = self.plus_head(h).squeeze(-1) / max(float(temperature), 1e-4)
        if deterministic:
            seq_plus = torch.argsort(plus_scores, descending=True)
        else:
            seq_plus = sample_plackett_luce(plus_scores)
        minus_scores = self._conditioned_minus_scores(h, self.minus_base_head(h).squeeze(-1), seq_plus)
        minus_scores = minus_scores / max(float(temperature), 1e-4)
        if deterministic:
            seq_minus = torch.argsort(minus_scores, descending=True)
        else:
            seq_minus = sample_plackett_luce(minus_scores)

        macro_cell_indices, macro_seq_plus, macro_seq_minus, macro_plus_scores, macro_minus_scores = (
            self._sample_macro_sequence_pair(h, graph, temperature, deterministic=deterministic)
        )

        node_context = context.unsqueeze(0).expand_as(h)
        residual_params = self.residual_head(torch.cat([h, node_context], dim=1))
        residual_mean = residual_params[:, :2]
        residual_log_std = torch.clamp(residual_params[:, 2:], min=-4.0, max=1.0)
        residual_basis = self.residual_basis_head(torch.cat([h, node_context], dim=1)).reshape(
            h.shape[0],
            2,
            self.global_flow_rank,
        )
        residual_basis = torch.tanh(residual_basis) / math.sqrt(float(self.global_flow_rank))
        global_params = self.global_flow_head(context)
        global_mean = global_params[: self.global_flow_rank]
        global_log_std = torch.clamp(global_params[self.global_flow_rank :], min=-4.0, max=1.0)
        if deterministic:
            residual_local = residual_mean
            global_flow_latent = global_mean
        else:
            residual_local = residual_mean + torch.randn_like(residual_mean) * residual_log_std.exp()
            global_flow_latent = global_mean + torch.randn_like(global_mean) * global_log_std.exp()
        residual_flow = residual_local + torch.einsum("ndr,r->nd", residual_basis, global_flow_latent)

        control_params = self.control_head(context)
        control_mean = control_params[: len(CONTROL_NAMES)]
        control_log_std = torch.clamp(control_params[len(CONTROL_NAMES) :], min=-4.0, max=1.0)
        if deterministic:
            control_raw = control_mean
        else:
            control_raw = control_mean + torch.randn_like(control_mean) * control_log_std.exp()
        controls = self._transform_controls(control_raw)

        k_logits = self.k_head(context)
        if deterministic:
            k_index = torch.argmax(k_logits, dim=-1)
        else:
            k_index = torch.distributions.Categorical(logits=k_logits).sample()

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

        cluster_logits = self.cluster_head(h)
        std_mask = self._standard_cell_mask(graph)
        if deterministic:
            sampled_clusters = torch.argmax(cluster_logits, dim=-1)
        else:
            sampled_clusters = torch.distributions.Categorical(logits=cluster_logits).sample()
        cluster_ids = torch.full((h.shape[0],), -1, dtype=torch.long, device=h.device)
        cluster_ids[std_mask] = sampled_clusters[std_mask]
        cluster_h = self._cluster_embeddings(h, cluster_ids, context)
        cluster_seq_plus, cluster_seq_minus, cluster_plus_scores, cluster_minus_scores = self._sample_cluster_sequence_pair(
            cluster_h,
            temperature,
            deterministic=deterministic,
        )
        dag_axis, dag_axis_logits, pair_branch_choices, pair_branch_logits = self._sample_pair_discrete_actions(
            h,
            graph,
            deterministic=deterministic,
        )
        pressure_action = self._sample_constraint_pressures(h, graph, context, deterministic=deterministic)

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
            k_index,
            k_logits,
            stop,
            stop_logits,
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
            k_index=k_index,
            stop=stop,
            stop_probability=stop_probability,
            stop_logit_bias=stop_logit_bias,
            cluster_ids=cluster_ids,
            dag_axis=dag_axis,
            pair_branch_choices=pair_branch_choices,
            memory=memory,
            next_memory=context,
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
            group_logprobs=group_logprobs,
            group_entropies=group_entropies,
            group_token_counts=group_token_counts,
            **controls,
        )

    def evaluate_action(self, graph, action, temperature=1.0):
        h = self.encode(graph)
        pooled, _base_context = self._global_features(h)
        _memory, context = self._memory_pair(h, graph)
        value = self.value_head(pooled).squeeze(-1)

        plus_scores = self.plus_head(h).squeeze(-1) / max(float(temperature), 1e-4)
        minus_scores = self._conditioned_minus_scores(h, self.minus_base_head(h).squeeze(-1), action.seq_plus)
        minus_scores = minus_scores / max(float(temperature), 1e-4)
        macro_cell_indices, macro_seq_plus, macro_seq_minus, macro_plus_scores, macro_minus_scores = (
            self._sample_macro_sequence_pair(
                h,
                graph,
                temperature,
                deterministic=False,
                seq_plus=action.macro_seq_plus,
            )
        )

        node_context = context.unsqueeze(0).expand_as(h)
        residual_params = self.residual_head(torch.cat([h, node_context], dim=1))
        residual_mean = residual_params[:, :2]
        residual_log_std = torch.clamp(residual_params[:, 2:], min=-4.0, max=1.0)
        global_params = self.global_flow_head(context)
        global_mean = global_params[: self.global_flow_rank]
        global_log_std = torch.clamp(global_params[self.global_flow_rank :], min=-4.0, max=1.0)

        control_params = self.control_head(context)
        control_mean = control_params[: len(CONTROL_NAMES)]
        control_log_std = torch.clamp(control_params[len(CONTROL_NAMES) :], min=-4.0, max=1.0)

        k_logits = self.k_head(context)
        stop_logits = self.stop_head(context).squeeze(-1) + graph.get(
            "stop_logit_bias",
            torch.zeros((), dtype=h.dtype, device=h.device),
        ).to(dtype=h.dtype, device=h.device)
        cluster_logits = self.cluster_head(h)
        std_mask = self._standard_cell_mask(graph)
        cluster_h = self._cluster_embeddings(h, action.cluster_ids, context)
        cluster_seq_plus, cluster_seq_minus, cluster_plus_scores, cluster_minus_scores = self._sample_cluster_sequence_pair(
            cluster_h,
            temperature,
            deterministic=False,
            seq_plus=action.cluster_seq_plus,
        )
        _dag_axis, dag_axis_logits, _pair_branch_choices, pair_branch_logits = self._sample_pair_discrete_actions(
            h,
            graph,
            deterministic=True,
        )
        pressure_eval = self._sample_constraint_pressures(h, graph, context, deterministic=True)
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
            action.k_index,
            k_logits,
            action.stop,
            stop_logits,
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
        k_index,
        k_logits,
        stop,
        stop_logits,
        cluster_ids,
        cluster_logits,
        std_mask,
        pressure_action,
    ):
        zero = plus_scores.sum() * 0.0
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
            "cluster_ordering": plackett_luce_logprob(cluster_plus_scores, cluster_seq_plus)
            + plackett_luce_logprob(cluster_minus_scores, cluster_seq_minus),
            "dag_ordering": dag_logprob,
            "pair_branches": pair_branch_logprob,
            "constraint_pressure": pressure_logprob,
            "residual": normal_logprob(residual_local, residual_mean, residual_log_std).sum()
            + normal_logprob(global_flow_latent, global_mean, global_log_std).sum(),
            "pd_controls": normal_logprob(control_raw, control_mean, control_log_std).sum()
            + categorical_logprob(k_index, k_logits),
            "stop": bernoulli_logprob(stop, stop_logits).sum(),
        }
        group_entropies = {
            "ordering": categorical_entropy_from_scores(plus_scores) + categorical_entropy_from_scores(minus_scores),
            "macro_ordering": categorical_entropy_from_scores(macro_plus_scores)
            + categorical_entropy_from_scores(macro_minus_scores),
            "cluster_ordering": categorical_entropy_from_scores(cluster_plus_scores)
            + categorical_entropy_from_scores(cluster_minus_scores),
            "dag_ordering": categorical_entropy(dag_axis_logits).sum() if dag_axis_logits.numel() > 0 else zero,
            "pair_branches": categorical_entropy(pair_branch_logits).sum() if pair_branch_logits.numel() > 0 else zero,
            "constraint_pressure": normal_entropy(pressure_action["branch_log_std"]).sum()
            + normal_entropy(pressure_action["boundary_log_std"]).sum()
            + normal_entropy(pressure_action["density_log_std"]).sum(),
            "residual": normal_entropy(residual_log_std).sum() + normal_entropy(global_log_std).sum(),
            "pd_controls": normal_entropy(control_log_std).sum() + categorical_entropy(k_logits),
            "stop": bernoulli_entropy(stop_logits).sum(),
        }
        group_token_counts = {
            "ordering": torch.tensor(max(int(seq_plus.numel() + seq_minus.numel()), 1), device=plus_scores.device),
            "macro_ordering": torch.tensor(
                max(int(macro_seq_plus.numel() + macro_seq_minus.numel()), 1),
                device=plus_scores.device,
            ),
            "cluster_ordering": torch.tensor(
                max(int(cluster_seq_plus.numel() + cluster_seq_minus.numel()), 1),
                device=plus_scores.device,
            ),
            "dag_ordering": torch.tensor(max(int(dag_axis.numel()), 1), device=plus_scores.device),
            "pair_branches": torch.tensor(max(int(pair_branch_choices.numel()), 1), device=plus_scores.device),
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
            "stop": torch.tensor(1, device=plus_scores.device),
        }
        if torch.any(std_mask):
            std_logits = cluster_logits[std_mask]
            std_clusters = torch.clamp(cluster_ids[std_mask], min=0)
            cluster_logprobs = F.log_softmax(std_logits, dim=-1).gather(1, std_clusters[:, None]).sum()
            cluster_entropies = categorical_entropy(std_logits).sum()
            cluster_tokens = int(std_clusters.numel())
        else:
            cluster_logprobs = plus_scores.sum() * 0.0
            cluster_entropies = plus_scores.sum() * 0.0
            cluster_tokens = 1
        group_logprobs["clusters"] = cluster_logprobs
        group_entropies["clusters"] = cluster_entropies
        group_token_counts["clusters"] = torch.tensor(cluster_tokens, device=plus_scores.device)
        return group_logprobs, group_entropies, group_token_counts

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

        pin_norm = torch.clamp(pin_count.max(), min=1.0)
        active_norm = torch.clamp(active_count.max(), min=1.0)
        dual_norm = torch.clamp(dual_pressure.abs().max(), min=1.0)
        boundary_norm = torch.clamp(boundary_pressure.abs().max(), min=1.0)
        density_norm = torch.clamp(density_pressure.abs().max(), min=1.0)
        grad_norm = torch.clamp(wirelength_grad.norm(dim=1).max(), min=1.0)

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
            ],
            dim=1,
        )
        return features

    def _pair_features(self, graph):
        cell_features = graph["cell_features"]
        active_pairs = graph["active_pairs"]
        branch_duals = graph["branch_duals"]
        if active_pairs.numel() == 0:
            return torch.empty((0, 10), dtype=cell_features.dtype, device=cell_features.device)
        centers = cell_features[:, 2:4]
        widths = cell_features[:, 4]
        heights = cell_features[:, 5]
        total_area = torch.clamp(cell_features[:, 0].sum(), min=1.0)
        length_scale = torch.sqrt(total_area)
        i = active_pairs[:, 0].long()
        j = active_pairs[:, 1].long()
        dx = (centers[j, 0] - centers[i, 0]) / length_scale
        dy = (centers[j, 1] - centers[i, 1]) / length_scale
        sep_x = torch.abs(dx)
        sep_y = torch.abs(dy)
        overlap_x = torch.relu(0.5 * (widths[i] + widths[j]) / length_scale - sep_x)
        overlap_y = torch.relu(0.5 * (heights[i] + heights[j]) / length_scale - sep_y)
        dual_norm = torch.clamp(branch_duals.abs().max(), min=1.0)
        return torch.stack(
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
            ],
            dim=1,
        )


def policy_config_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return {
        "node_feature_dim": int(config.get("node_feature_dim", NODE_FEATURE_DIM)),
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "message_passes": int(config.get("message_passes", 2)),
        "num_clusters": int(config.get("num_clusters", 8)),
        "global_flow_rank": int(config.get("global_flow_rank", 2)),
    }


def save_policy_checkpoint(policy, path, config=None, stats=None):
    payload = {
        "model_state": policy.state_dict(),
        "config": {
            "node_feature_dim": policy.node_feature_dim,
            "hidden_dim": policy.hidden_dim,
            "message_passes": policy.message_passes,
            "num_clusters": policy.num_clusters,
            "global_flow_rank": policy.global_flow_rank,
            **(config or {}),
        },
        "stats": stats or {},
    }
    torch.save(payload, path)


def load_policy_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = OrderingPolicy(**policy_config_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
