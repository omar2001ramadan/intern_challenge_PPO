"""Primal-dual coordinate layer for declared placement transitions."""

import torch


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
