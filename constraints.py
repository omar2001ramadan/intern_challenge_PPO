"""Constraint and audit helpers for placement optimization."""

import torch

from induce_branches import Branch


def make_all_pairs(n, device=None):
    """Return all unordered pairs i < j."""
    return torch.triu_indices(n, n, offset=1, device=device).t().contiguous()


def branch_signed_constraints(centers, widths, heights, pairs, branches):
    """Compute signed branch inequalities c_ij(X; d) <= 0."""
    i = pairs[:, 0].long()
    j = pairs[:, 1].long()

    xi = centers[i, 0]
    yi = centers[i, 1]
    xj = centers[j, 0]
    yj = centers[j, 1]
    wi = widths[i]
    wj = widths[j]
    hi = heights[i]
    hj = heights[j]

    left = xi + 0.5 * wi - xj + 0.5 * wj
    right = xj + 0.5 * wj - xi + 0.5 * wi
    below = yi + 0.5 * hi - yj + 0.5 * hj
    above = yj + 0.5 * hj - yi + 0.5 * hi

    constraints = torch.empty_like(left)
    constraints[branches == int(Branch.L)] = left[branches == int(Branch.L)]
    constraints[branches == int(Branch.R)] = right[branches == int(Branch.R)]
    constraints[branches == int(Branch.B)] = below[branches == int(Branch.B)]
    constraints[branches == int(Branch.A)] = above[branches == int(Branch.A)]
    return constraints


def all_branch_signed_constraints(centers, widths, heights, pairs):
    """Return signed constraints for L, R, B, A for every pair."""
    i = pairs[:, 0].long()
    j = pairs[:, 1].long()
    xi = centers[i, 0]
    yi = centers[i, 1]
    xj = centers[j, 0]
    yj = centers[j, 1]
    wi = widths[i]
    wj = widths[j]
    hi = heights[i]
    hj = heights[j]
    return torch.stack(
        [
            xi + 0.5 * wi - xj + 0.5 * wj,
            xj + 0.5 * wj - xi + 0.5 * wi,
            yi + 0.5 * hi - yj + 0.5 * hj,
            yj + 0.5 * hj - yi + 0.5 * hi,
        ],
        dim=1,
    )


def soft_signed_disjunction(centers, widths, heights, pairs, branch_weights, tau=1.0, epsilon=1e-4):
    """Soft minimum of signed branch constraints for continuation training."""
    constraints = all_branch_signed_constraints(centers, widths, heights, pairs)
    weights = torch.clamp(branch_weights, min=0.0) + float(epsilon)
    weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
    log_terms = torch.log(weights) - constraints / max(float(tau), 1e-6)
    return -float(tau) * torch.logsumexp(log_terms, dim=1)


def pair_overlap_areas(centers, widths, heights, pairs):
    """Return exact overlap area for the supplied pairs as a tensor."""
    i = pairs[:, 0].long()
    j = pairs[:, 1].long()
    dx = torch.abs(centers[i, 0] - centers[j, 0])
    dy = torch.abs(centers[i, 1] - centers[j, 1])
    overlap_x = torch.relu(0.5 * (widths[i] + widths[j]) - dx)
    overlap_y = torch.relu(0.5 * (heights[i] + heights[j]) - dy)
    return overlap_x * overlap_y


def overlap_repulsion_for_pairs(centers, widths, heights, pairs, area_scale=None):
    """Differentiable overlap loss for an active pair set."""
    if pairs.numel() == 0:
        return centers.sum() * 0.0
    overlap = pair_overlap_areas(centers, widths, heights, pairs)
    if area_scale is None:
        area_scale = torch.clamp((widths * heights).mean(), min=1.0)
    return torch.mean((overlap / area_scale) ** 2)


def outline_from_cells(cell_features, utilization=0.62, margin_scale=0.10):
    """Construct a soft outline box from total area.

    The benchmark has no fixed die outline, so this outline is deliberately soft
    and sized from total area. It counters center collapse without making the
    problem artificially infeasible.
    """
    areas = cell_features[:, 0]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    total_area = torch.clamp(areas.sum(), min=1.0)
    side = torch.sqrt(total_area / utilization)
    side = torch.maximum(side, widths.max() * 1.25)
    side = torch.maximum(side, heights.max() * 1.25)
    margin = margin_scale * side
    half = 0.5 * side + margin
    return (-half, half, -half, half)


def boundary_signed_constraints(centers, widths, heights, bounds):
    """Return boundary inequalities b(X) <= 0 with shape [N, 4]."""
    xmin, xmax, ymin, ymax = bounds
    left = xmin + 0.5 * widths - centers[:, 0]
    right = centers[:, 0] + 0.5 * widths - xmax
    bottom = ymin + 0.5 * heights - centers[:, 1]
    top = centers[:, 1] + 0.5 * heights - ymax
    return torch.stack([left, right, bottom, top], dim=1)


def density_spread_violation(centers, cell_features, utilization=0.65):
    """A cheap differentiable anti-collapse constraint.

    This is a moment-based density proxy rather than a hard bin model: it asks
    the placement variance to be large enough for the requested utilization.
    """
    total_area = torch.clamp(cell_features[:, 0].sum(), min=1.0)
    target_side = torch.sqrt(total_area / utilization)
    target_var = (target_side ** 2) / 24.0
    centered = centers - centers.mean(dim=0, keepdim=True)
    var = centered.square().mean(dim=0)
    return torch.relu(target_var - var)


def density_bin_constraints(
    centers,
    cell_features,
    bounds,
    bins=8,
    rho_max=0.85,
    sigma_scale=1.0,
):
    """Differentiable density-bin overflow constraints d_b(X) <= 0.

    Each cell area is softly assigned to bin centers by a Gaussian kernel. This
    is not a hard exact bin overlap model, but it is a true differentiable bin
    density constraint rather than a placement-moment proxy.
    """
    if centers.numel() == 0:
        empty = centers.new_zeros(0)
        return empty, centers.new_zeros((0, 0))

    xmin, xmax, ymin, ymax = bounds
    width = torch.clamp(xmax - xmin, min=1.0)
    height = torch.clamp(ymax - ymin, min=1.0)
    bins = int(max(bins, 1))
    bin_w = width / bins
    bin_h = height / bins

    x_centers = torch.linspace(
        xmin + 0.5 * bin_w,
        xmax - 0.5 * bin_w,
        bins,
        dtype=centers.dtype,
        device=centers.device,
    )
    y_centers = torch.linspace(
        ymin + 0.5 * bin_h,
        ymax - 0.5 * bin_h,
        bins,
        dtype=centers.dtype,
        device=centers.device,
    )
    grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
    bin_centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    sigma_x = torch.clamp(bin_w * sigma_scale, min=1e-3)
    sigma_y = torch.clamp(bin_h * sigma_scale, min=1e-3)
    dx = (centers[:, None, 0] - bin_centers[None, :, 0]) / sigma_x
    dy = (centers[:, None, 1] - bin_centers[None, :, 1]) / sigma_y
    logits = -0.5 * (dx.square() + dy.square())
    assignment = torch.softmax(logits, dim=1)

    area = cell_features[:, 0]
    bin_area = torch.clamp(bin_w * bin_h, min=1e-6)
    density = assignment.t().matmul(area) / bin_area
    return density - rho_max, assignment


def density_pressure_per_cell(density_g, assignment, density_duals=None):
    """Summarize bin overflow pressure incident on each cell."""
    if assignment.numel() == 0:
        return assignment.new_zeros(0)
    overflow = torch.relu(density_g)
    if density_duals is not None and density_duals.numel() == overflow.numel():
        overflow = overflow * (1.0 + density_duals.detach())
    return assignment.matmul(overflow)


def exact_overlap_pairs(cell_features, all_pair_limit=6000, chunk_size=2_000_000):
    """Return every overlapping pair found by an exact audit.

    For the challenge's first ten tests, all-pairs vectorization is exact and
    fast on GPU. Larger instances use an exact device-resident sweep-and-prune
    broad phase followed by exact overlap checks.
    """
    n = int(cell_features.shape[0])
    if n <= 1:
        return torch.empty((0, 2), dtype=torch.long, device=cell_features.device)

    if n <= all_pair_limit:
        return _exact_overlap_pairs_all(cell_features, chunk_size)
    return _exact_overlap_pairs_spatial_hash(cell_features)


def _exact_overlap_pairs_all(cell_features, chunk_size):
    device = cell_features.device
    n = int(cell_features.shape[0])
    centers = cell_features[:, 2:4]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    pairs = make_all_pairs(n, device=device)
    hits = []
    for start in range(0, pairs.shape[0], chunk_size):
        chunk = pairs[start : start + chunk_size]
        overlaps = pair_overlap_areas(centers, widths, heights, chunk)
        mask = overlaps > 0
        if torch.any(mask):
            hits.append(chunk[mask])
    if not hits:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.cat(hits, dim=0)


def _exact_overlap_pairs_spatial_hash(cell_features):
    centers = cell_features[:, 2:4]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    device = cell_features.device
    n = int(cell_features.shape[0])

    if n <= 1:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    left = centers[:, 0] - 0.5 * widths
    right = centers[:, 0] + 0.5 * widths
    bottom = centers[:, 1] - 0.5 * heights
    top = centers[:, 1] + 0.5 * heights

    order = torch.argsort(left)
    left_sorted = left[order]
    right_sorted = right[order]
    bottom_sorted = bottom[order]
    top_sorted = top[order]
    range_end = torch.searchsorted(left_sorted, right_sorted, right=False)

    hits = []
    chunk = 2048
    for start in range(0, n, chunk):
        row_idx = torch.arange(start, min(start + chunk, n), device=device)
        counts = torch.clamp(range_end[row_idx] - row_idx - 1, min=0)
        if not torch.any(counts > 0):
            continue

        max_count = int(counts.max().item())
        offsets = torch.arange(max_count, device=device)
        candidate_cols = row_idx[:, None] + 1 + offsets[None, :]
        mask = offsets[None, :] < counts[:, None]
        candidate_cols = candidate_cols[mask]
        candidate_rows = torch.repeat_interleave(row_idx, counts)

        overlap_y = torch.minimum(top_sorted[candidate_rows], top_sorted[candidate_cols]) - torch.maximum(
            bottom_sorted[candidate_rows],
            bottom_sorted[candidate_cols],
        )
        keep = overlap_y > 0
        if not torch.any(keep):
            continue

        src = order[candidate_rows[keep]]
        dst = order[candidate_cols[keep]]
        lo = torch.minimum(src, dst)
        hi = torch.maximum(src, dst)
        hits.append(torch.stack([lo, hi], dim=1))

    if not hits:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.unique(torch.cat(hits, dim=0), dim=0)


def overlap_ratio_from_pairs(n, pairs):
    """Official-style ratio: cells with at least one overlap divided by N."""
    if n == 0 or pairs.numel() == 0:
        return 0.0, 0
    cells = torch.unique(pairs.reshape(-1))
    return cells.numel() / n, int(cells.numel())
