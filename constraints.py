"""Constraint and audit helpers for placement optimization."""

import math

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
    fast on GPU. Larger instances fall back to a conservative spatial hash.
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
    cpu_features = cell_features.detach().cpu()
    centers = cpu_features[:, 2:4]
    widths = cpu_features[:, 4]
    heights = cpu_features[:, 5]
    n = int(cpu_features.shape[0])

    median_span = torch.median(torch.maximum(widths, heights)).item()
    cell_size = max(median_span * 4.0, 1.0)
    buckets = {}

    left = centers[:, 0] - 0.5 * widths
    right = centers[:, 0] + 0.5 * widths
    bottom = centers[:, 1] - 0.5 * heights
    top = centers[:, 1] + 0.5 * heights

    for idx in range(n):
        gx0 = math.floor(left[idx].item() / cell_size)
        gx1 = math.floor(right[idx].item() / cell_size)
        gy0 = math.floor(bottom[idx].item() / cell_size)
        gy1 = math.floor(top[idx].item() / cell_size)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                buckets.setdefault((gx, gy), []).append(idx)

    candidate_pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a_pos in range(len(members)):
            a = members[a_pos]
            for b in members[a_pos + 1 :]:
                i, j = (a, b) if a < b else (b, a)
                candidate_pairs.add((i, j))

    if not candidate_pairs:
        return torch.empty((0, 2), dtype=torch.long, device=cell_features.device)

    pairs = torch.tensor(sorted(candidate_pairs), dtype=torch.long)
    overlaps = pair_overlap_areas(centers, widths, heights, pairs)
    pairs = pairs[overlaps > 0]
    return pairs.to(cell_features.device)


def overlap_ratio_from_pairs(n, pairs):
    """Official-style ratio: cells with at least one overlap divided by N."""
    if n == 0 or pairs.numel() == 0:
        return 0.0, 0
    cells = torch.unique(pairs.reshape(-1))
    return cells.numel() / n, int(cells.numel())
