"""Primal-dual coordinate layer for sequence-pair placement."""

import torch

from constraints import exact_overlap_pairs


def phr_inequality_penalty(g, lam, rho, normalizer=None, weights=None):
    """Powell-Hestenes-Rockafellar inequality AL term for g(X) <= 0."""
    if g.numel() == 0:
        return g.sum() * 0.0
    rho_tensor = torch.as_tensor(rho, dtype=g.dtype, device=g.device)
    rho_tensor = torch.clamp(rho_tensor, min=1e-8)
    term = (torch.relu(lam + rho_tensor * g).square() - lam.square()) / (2.0 * rho_tensor)
    if weights is not None:
        term = term * weights.reshape_as(term)
    if normalizer is None:
        return term.mean()
    return term.sum() / max(float(normalizer), 1.0)


def phr_dual_update(lam, g, rho, max_value=None):
    """Projected multiplier update."""
    rho_tensor = torch.as_tensor(rho, dtype=g.dtype, device=g.device)
    rho_tensor = torch.clamp(rho_tensor, min=1e-8)
    updated = torch.relu(lam + rho_tensor * g.detach())
    if max_value is not None:
        updated = torch.clamp(updated, max=float(max_value))
    return updated


def sequence_pair_legalize(cell_features, seq_plus, seq_minus, reference_centers=None, limit=6000):
    """Produce a zero-overlap placement induced by a sequence pair.

    This dense O(N^2) dynamic program is intended for the benchmark-scale cases.
    It computes longest paths over the horizontal/vertical precedence graph using
    seq_plus as a shared topological order.
    """
    n = int(cell_features.shape[0])
    if n == 0:
        return cell_features[:, 2:4].clone()
    if n > limit:
        return shelf_legalize(cell_features, reference_centers)

    device = cell_features.device
    seq_plus_cpu = seq_plus.detach().cpu().tolist()
    seq_minus_cpu = seq_minus.detach().cpu().tolist()
    rank_minus = [0] * n
    for rank, cell in enumerate(seq_minus_cpu):
        rank_minus[cell] = rank

    widths = cell_features[:, 4].detach().cpu().tolist()
    heights = cell_features[:, 5].detach().cpu().tolist()
    total_area = max(float(cell_features[:, 0].detach().cpu().sum().item()), 1.0)
    gap = max(1e-3, (total_area ** 0.5) * 1e-6)
    x = [0.0] * n
    y = [0.0] * n

    for cur_pos, cur in enumerate(seq_plus_cpu):
        x_cur = widths[cur] * 0.5
        y_cur = heights[cur] * 0.5
        cur_minus = rank_minus[cur]
        for prev in seq_plus_cpu[:cur_pos]:
            if rank_minus[prev] < cur_minus:
                x_cur = max(x_cur, x[prev] + 0.5 * (widths[prev] + widths[cur]) + gap)
            else:
                y_cur = max(y_cur, y[prev] + 0.5 * (heights[prev] + heights[cur]) + gap)
        x[cur] = x_cur
        y[cur] = y_cur

    centers = torch.tensor(list(zip(x, y)), dtype=cell_features.dtype, device=device)
    if reference_centers is not None:
        centers = centers - centers.mean(dim=0, keepdim=True)
        centers = centers + reference_centers.detach().mean(dim=0, keepdim=True)
    return centers


def shelf_legalize(cell_features, reference_centers=None, utilization=0.68):
    """Linear-time legalizer for very large instances."""
    device = cell_features.device
    n = int(cell_features.shape[0])
    widths = cell_features[:, 4].detach().cpu()
    heights = cell_features[:, 5].detach().cpu()
    areas = cell_features[:, 0].detach().cpu()
    total_area = max(float(areas.sum().item()), 1.0)
    row_width = max(total_area / utilization, 1.0) ** 0.5
    row_width = max(row_width, float(widths.max().item()) * 1.2)
    gap = max(1e-3, (total_area ** 0.5) * 1e-6)

    if reference_centers is None:
        order = torch.argsort(-areas).cpu().tolist()
    else:
        ref = reference_centers.detach().cpu()
        order = torch.argsort(ref[:, 1] * row_width + ref[:, 0]).cpu().tolist()

    centers = torch.empty((n, 2), dtype=cell_features.dtype)
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0

    for cell in order:
        width = float(widths[cell].item())
        height = float(heights[cell].item())
        if x_cursor > 0.0 and x_cursor + width > row_width:
            y_cursor += row_height + gap
            x_cursor = 0.0
            row_height = 0.0
        centers[cell, 0] = x_cursor + 0.5 * width
        centers[cell, 1] = y_cursor + 0.5 * height
        x_cursor += width + gap
        row_height = max(row_height, height)

    centers = centers.to(device)
    centers = centers - centers.mean(dim=0, keepdim=True)
    if reference_centers is not None:
        centers = centers + reference_centers.detach().mean(dim=0, keepdim=True)
    return centers


def iterative_overlap_repair(cell_features, centers, max_iters=80):
    """Repair residual exact overlaps by separating along the cheaper axis."""
    repaired = centers.clone().detach()
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    total_area = max(float(cell_features[:, 0].detach().sum().item()), 1.0)
    gap = max(1e-3, (total_area ** 0.5) * 1e-6)

    for _ in range(max_iters):
        current = cell_features.clone()
        current[:, 2:4] = repaired
        pairs = exact_overlap_pairs(current)
        if pairs.numel() == 0:
            break

        i = pairs[:, 0].long()
        j = pairs[:, 1].long()
        dx = repaired[i, 0] - repaired[j, 0]
        dy = repaired[i, 1] - repaired[j, 1]
        overlap_x = torch.relu(0.5 * (widths[i] + widths[j]) - torch.abs(dx))
        overlap_y = torch.relu(0.5 * (heights[i] + heights[j]) - torch.abs(dy))
        use_x = overlap_x <= overlap_y

        direction_x = torch.where(dx >= 0, torch.ones_like(dx), -torch.ones_like(dx))
        direction_y = torch.where(dy >= 0, torch.ones_like(dy), -torch.ones_like(dy))
        shift_x = 0.5 * (overlap_x + gap) * direction_x
        shift_y = 0.5 * (overlap_y + gap) * direction_y

        delta = torch.zeros_like(repaired)
        counts = torch.zeros((repaired.shape[0], 1), dtype=repaired.dtype, device=repaired.device)

        move_i = torch.stack([torch.where(use_x, shift_x, torch.zeros_like(shift_x)),
                              torch.where(use_x, torch.zeros_like(shift_y), shift_y)], dim=1)
        move_j = -move_i

        delta.index_add_(0, i, move_i)
        delta.index_add_(0, j, move_j)
        ones = torch.ones((pairs.shape[0], 1), dtype=repaired.dtype, device=repaired.device)
        counts.index_add_(0, i, ones)
        counts.index_add_(0, j, ones)

        repaired = repaired + delta / torch.clamp(counts, min=1.0)

    return repaired
