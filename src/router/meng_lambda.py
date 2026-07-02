"""Meng-style normalized subgradient multiplier update (elfPlace TCAD 2021 Eq. 20-21)."""

import torch


def normalized_overflow_subgradient(
    overflows: torch.Tensor,
    overflow_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """g̃ = overflow / max(overflow_ref, eps) per constraint."""
    ref = torch.clamp(overflow_ref, min=eps)
    return overflows / ref


def update_multipliers_meng(
    lam: torch.Tensor,
    overflows: torch.Tensor,
    overflow_ref: torch.Tensor,
    step_size: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    λ ← λ + t · g̃ / ‖g̃‖₂, then clamp λ ≥ 0.

    overflow_ref is typically the initial overflow snapshot per physical edge.
    """
    g_tilde = normalized_overflow_subgradient(overflows, overflow_ref, eps=eps)
    norm = g_tilde.norm()
    if norm.item() > eps:
        lam = lam + step_size * (g_tilde / norm)
    else:
        lam = lam + step_size * g_tilde
    return torch.clamp(lam, min=0.0)
