"""Cutting-plane active-set management for pair constraints."""

import torch

from constraints import exact_overlap_pairs, make_all_pairs


def canonicalize_pairs(pairs):
    """Sort endpoints and remove duplicate pairs."""
    if pairs.numel() == 0:
        return pairs.reshape(0, 2).long()
    lo = torch.minimum(pairs[:, 0], pairs[:, 1])
    hi = torch.maximum(pairs[:, 0], pairs[:, 1])
    pairs = torch.stack([lo, hi], dim=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.numel() == 0:
        return pairs.reshape(0, 2).long()
    pairs = pairs.long()
    keys = pair_sort_keys(pairs)
    order = torch.argsort(keys)
    pairs = pairs[order]
    keys = keys[order]
    unique_mask = torch.ones_like(keys, dtype=torch.bool)
    unique_mask[1:] = keys[1:] != keys[:-1]
    return pairs[unique_mask]


def pair_sort_keys(pairs, base=None):
    """Encode unordered pairs into sortable integer keys."""
    if pairs.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=pairs.device)
    if base is None:
        base = int(torch.max(pairs).item()) + 1
    return pairs[:, 0].long() * int(base) + pairs[:, 1].long()


def sort_pairs_and_payload(pairs, *payloads, base=None):
    """Sort unique pairs lexicographically and apply the same permutation to payloads."""
    if pairs.numel() == 0:
        outputs = [pairs.reshape(0, 2).long()]
        outputs.extend(payload[:0] for payload in payloads)
        return tuple(outputs)
    keys = pair_sort_keys(pairs, base=base)
    order = torch.argsort(keys)
    outputs = [pairs[order]]
    outputs.extend(payload[order] for payload in payloads)
    return tuple(outputs)


def connected_cell_pairs(pin_features, edge_list, device=None, max_pairs=500_000):
    """Return unordered cell pairs connected by at least one net edge."""
    if pin_features is None or edge_list is None or pin_features.numel() == 0 or edge_list.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    pin_to_cell = pin_features[:, 0].long()
    src = pin_to_cell[edge_list[:, 0].long()]
    dst = pin_to_cell[edge_list[:, 1].long()]
    mask = src != dst
    if not torch.any(mask):
        return torch.empty((0, 2), dtype=torch.long, device=pin_features.device if device is None else device)
    pairs = canonicalize_pairs(torch.stack([src[mask], dst[mask]], dim=1))
    if pairs.shape[0] > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs.to(pin_features.device if device is None else device)


def build_initial_active_pairs(
    cell_features,
    pin_features=None,
    edge_list=None,
    all_pair_limit=3500,
    near_window=8,
    max_pairs=2_500_000,
):
    """Build initial cutting-plane pairs.

    Small benchmark cases use all pairs. Larger cases start from exact overlaps
    plus local neighbors in x/y order, then the audit loop inserts missed pairs.
    """
    n = int(cell_features.shape[0])
    device = cell_features.device
    if n <= 1:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    if n <= all_pair_limit:
        return make_all_pairs(n, device=device)

    centers = cell_features[:, 2:4]
    candidates = [
        exact_overlap_pairs(cell_features),
        connected_cell_pairs(pin_features, edge_list, device=device, max_pairs=max_pairs),
    ]

    for dim in (0, 1):
        order = torch.argsort(centers[:, dim])
        for offset in range(1, near_window + 1):
            left = order[:-offset]
            right = order[offset:]
            candidates.append(torch.stack([left, right], dim=1))

    pairs = canonicalize_pairs(torch.cat(candidates, dim=0))
    if pairs.shape[0] > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def update_active_pairs(active_pairs, missed_pairs, max_pairs=2_500_000):
    """Union the current active set with exact-audit misses."""
    if missed_pairs.numel() == 0:
        return active_pairs
    if active_pairs.numel() == 0:
        merged = canonicalize_pairs(missed_pairs)
    else:
        merged = canonicalize_pairs(torch.cat([active_pairs, missed_pairs], dim=0))
    if merged.shape[0] > max_pairs:
        overlap_keep = canonicalize_pairs(missed_pairs)
        budget = max(max_pairs - overlap_keep.shape[0], 0)
        merged = canonicalize_pairs(torch.cat([overlap_keep, merged[:budget]], dim=0))
    return merged


def update_active_pair_cache(
    active_pairs,
    pair_ages,
    missed_pairs,
    *,
    retention_horizon=4,
    max_pairs=2_500_000,
):
    """Update a cutting-plane cache with fixed-horizon pair retention.

    Existing pairs age down each audit. Missed exact-overlap pairs are inserted
    or refreshed to `retention_horizon`, so no pair is declared permanently safe.
    """
    if active_pairs.numel() == 0:
        merged = canonicalize_pairs(missed_pairs)
        ages = torch.full(
            (merged.shape[0],),
            int(retention_horizon),
            dtype=torch.long,
            device=merged.device,
        )
        return merged, ages

    if pair_ages is None or pair_ages.numel() != active_pairs.shape[0]:
        pair_ages = torch.full(
            (active_pairs.shape[0],),
            int(retention_horizon),
            dtype=torch.long,
            device=active_pairs.device,
        )
    else:
        pair_ages = torch.clamp(pair_ages.to(active_pairs.device) - 1, min=0)

    keep = pair_ages > 0
    kept_pairs = active_pairs[keep].reshape(-1, 2).long()
    kept_ages = pair_ages[keep].reshape(-1).long()

    missed_pairs = canonicalize_pairs(missed_pairs)
    if missed_pairs.numel() == 0:
        if kept_pairs.shape[0] > max_pairs:
            kept_pairs = kept_pairs[:max_pairs]
            kept_ages = kept_ages[:max_pairs]
        return kept_pairs, kept_ages

    if kept_pairs.numel() == 0:
        ages = torch.full(
            (missed_pairs.shape[0],),
            int(retention_horizon),
            dtype=torch.long,
            device=missed_pairs.device,
        )
        pairs, ages = sort_pairs_and_payload(missed_pairs, ages)
    else:
        base = int(torch.max(torch.cat([kept_pairs.reshape(-1), missed_pairs.reshape(-1)])).item()) + 1
        kept_pairs, kept_ages = sort_pairs_and_payload(kept_pairs, kept_ages, base=base)
        missed_pairs = sort_pairs_and_payload(missed_pairs, base=base)[0]
        kept_keys = pair_sort_keys(kept_pairs, base=base)
        missed_keys = pair_sort_keys(missed_pairs, base=base)

        insert_positions = torch.searchsorted(kept_keys, missed_keys)
        if kept_keys.numel() > 0:
            clamped = torch.clamp(insert_positions, max=kept_keys.shape[0] - 1)
            matched = (insert_positions < kept_keys.shape[0]) & (kept_keys[clamped] == missed_keys)
        else:
            matched = torch.zeros_like(insert_positions, dtype=torch.bool)

        if torch.any(matched):
            kept_ages[insert_positions[matched]] = torch.maximum(
                kept_ages[insert_positions[matched]],
                torch.full_like(insert_positions[matched], int(retention_horizon), dtype=torch.long),
            )

        new_pairs = missed_pairs[~matched]
        if new_pairs.numel() == 0:
            pairs, ages = kept_pairs, kept_ages
        else:
            new_ages = torch.full(
                (new_pairs.shape[0],),
                int(retention_horizon),
                dtype=torch.long,
                device=new_pairs.device,
            )
            pairs = torch.cat([kept_pairs, new_pairs], dim=0)
            ages = torch.cat([kept_ages, new_ages], dim=0)
            pairs, ages = sort_pairs_and_payload(pairs, ages, base=base)

    if pairs.shape[0] > max_pairs:
        base = int(torch.max(pairs).item()) + 1
        keys = pair_sort_keys(pairs, base=base)
        priority = (int(retention_horizon) + 1 - ages.long()) * int(base * base + 1) + keys
        keep_order = torch.argsort(priority)[:max_pairs]
        pairs = pairs[keep_order]
        ages = ages[keep_order]
        pairs, ages = sort_pairs_and_payload(pairs, ages, base=base)
    return pairs, ages
