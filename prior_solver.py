"""Offline prior solver entry point used for teacher demonstrations.

This module is intentionally thin: it exposes the deterministic offline teacher
under the paper-aligned "prior solver" name while keeping inference entirely
teacher-free.
"""

from teacher_solver import train_teacher_placement


def train_prior_solver_placement(cell_features, pin_features, edge_list, *, verbose=False):
    return train_teacher_placement(
        cell_features,
        pin_features,
        edge_list,
        verbose=verbose,
    )

