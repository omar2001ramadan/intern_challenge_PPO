"""Global ordering utilities for sequence-pair placement.

The challenge API only needs final coordinates, but keeping branch decisions
behind a global sequence pair prevents inconsistent pairwise non-overlap labels.
"""

from enum import IntEnum

import torch


class Branch(IntEnum):
    """Canonical branch labels for an unordered pair (i, j)."""

    L = 0  # i is left of j
    R = 1  # j is left of i
    B = 2  # i is below j
    A = 3  # j is below i


def ranks_from_sequence(sequence, n=None):
    """Return rank[cell_id] for a permutation tensor."""
    if n is None:
        n = int(sequence.numel())
    ranks = torch.empty(n, dtype=torch.long, device=sequence.device)
    ranks[sequence] = torch.arange(n, dtype=torch.long, device=sequence.device)
    return ranks


def sequence_pair_from_centers(cell_features, jitter_scale=0.0, generator=None):
    """Build a sequence pair from current geometry.

    Sorting by x+y and x-y maps left/right relationships to same-order pairs and
    above/below relationships to opposite-order pairs, matching the usual
    sequence-pair interpretation.
    """
    centers = cell_features[:, 2:4]
    plus_key = centers[:, 0] + centers[:, 1]
    minus_key = centers[:, 0] - centers[:, 1]

    if jitter_scale > 0.0:
        scale = torch.as_tensor(jitter_scale, device=centers.device, dtype=centers.dtype)
        plus_key = plus_key + scale * torch.rand(
            plus_key.shape, device=centers.device, dtype=centers.dtype, generator=generator
        )
        minus_key = minus_key + scale * torch.rand(
            minus_key.shape, device=centers.device, dtype=centers.dtype, generator=generator
        )

    return torch.argsort(plus_key), torch.argsort(minus_key)


def induce_branches_from_sequence_pair(seq_plus, seq_minus, pairs):
    """Derive one globally consistent branch for each canonical pair."""
    n = int(seq_plus.numel())
    rank_plus = ranks_from_sequence(seq_plus, n)
    rank_minus = ranks_from_sequence(seq_minus, n)

    i = pairs[:, 0].long()
    j = pairs[:, 1].long()
    plus_lt = rank_plus[i] < rank_plus[j]
    minus_lt = rank_minus[i] < rank_minus[j]

    branches = torch.empty(pairs.shape[0], dtype=torch.long, device=pairs.device)
    branches[plus_lt & minus_lt] = int(Branch.L)
    branches[~plus_lt & ~minus_lt] = int(Branch.R)
    branches[plus_lt & ~minus_lt] = int(Branch.B)
    branches[~plus_lt & minus_lt] = int(Branch.A)
    return branches


def branch_antisymmetry_error(seq_plus, seq_minus, sample_limit=4096):
    """Return zero when induced branches are antisymmetric under pair reversal."""
    n = int(seq_plus.numel())
    if n <= 1:
        return 0.0

    device = seq_plus.device
    total = min(sample_limit, n * (n - 1) // 2)
    if total == n * (n - 1) // 2 and n <= 128:
        pairs = torch.triu_indices(n, n, offset=1, device=device).t()
    else:
        i = torch.randint(0, n, (total,), device=device)
        j = torch.randint(0, n - 1, (total,), device=device)
        j = j + (j >= i).long()
        pairs = torch.stack([torch.minimum(i, j), torch.maximum(i, j)], dim=1)

    forward = induce_branches_from_sequence_pair(seq_plus, seq_minus, pairs)
    reverse_pairs = pairs.flip(1)
    reverse = induce_branches_from_sequence_pair(seq_plus, seq_minus, reverse_pairs)

    inverse = torch.empty_like(forward)
    inverse[forward == int(Branch.L)] = int(Branch.R)
    inverse[forward == int(Branch.R)] = int(Branch.L)
    inverse[forward == int(Branch.B)] = int(Branch.A)
    inverse[forward == int(Branch.A)] = int(Branch.B)
    return (inverse != reverse).float().mean().item()
