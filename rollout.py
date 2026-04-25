"""Proposal-compatible rollout/environment facade."""

from env import (  # noqa: F401
    EnvConfig,
    PlacementOrderingEnv,
    normalized_wirelength,
    wirelength_loss,
    write_positions,
)

from constraints import exact_overlap_pairs, overlap_ratio_from_pairs


def build_initial_state(cell_features, pin_features, edge_list, config=None):
    """Build the auditable environment state used by rollout transitions."""
    return PlacementOrderingEnv(cell_features, pin_features, edge_list, config or EnvConfig())


def exact_overlap_audit(X, cell_features, force_all_pairs=False):
    """Return exact overlapping pairs for centers X without moving geometry."""
    current = write_positions(cell_features, X)
    all_pair_limit = int(cell_features.shape[0]) + 1 if force_all_pairs else 6000
    return exact_overlap_pairs(current, all_pair_limit=all_pair_limit)


def score_official_metrics(X, cell_features, pin_features, edge_list):
    """Repository-equivalent exact overlap and normalized wirelength scoring."""
    current = write_positions(cell_features, X)
    pairs = exact_overlap_pairs(current)
    overlap_ratio, overlap_cells = overlap_ratio_from_pairs(int(current.shape[0]), pairs)
    return {
        "X": X.detach().clone(),
        "overlap_ratio": float(overlap_ratio),
        "overlap_cells": int(overlap_cells),
        "normalized_wl": float(normalized_wirelength(current, pin_features, edge_list)),
        "num_overlap_pairs": int(pairs.shape[0]),
    }


def select_by_exact_overlap_then_wirelength(candidates):
    """Select a saved candidate by official lexicographic metrics only."""
    return min(
        candidates,
        key=lambda item: (
            item["overlap_cells"],
            item["overlap_ratio"],
            item["normalized_wl"],
            item["num_overlap_pairs"],
        ),
    )
