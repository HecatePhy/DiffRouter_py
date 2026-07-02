"""Tests for Meng normalized subgradient λ update."""

import torch

from src.router.meng_lambda import normalized_overflow_subgradient, update_multipliers_meng


def test_normalized_subgradient():
    overflows = torch.tensor([2.0, 0.0, 1.0])
    overflow_ref = torch.tensor([4.0, 1.0, 1.0])
    g = normalized_overflow_subgradient(overflows, overflow_ref)
    expected = torch.tensor([0.5, 0.0, 1.0])
    assert torch.allclose(g, expected)


def test_meng_update_matches_formula():
    overflows = torch.tensor([2.0, 0.0, 1.0])
    overflow_ref = torch.tensor([4.0, 1.0, 1.0])
    lam = torch.zeros(3)
    t = 0.1
    g_tilde = normalized_overflow_subgradient(overflows, overflow_ref)
    expected = lam + t * (g_tilde / g_tilde.norm())
    result = update_multipliers_meng(lam, overflows, overflow_ref, t)
    assert torch.allclose(result, expected)


def test_lambda_stays_nonnegative():
    lam = torch.tensor([0.0, 0.5])
    overflows = torch.tensor([-1.0, 0.5])
    overflow_ref = torch.tensor([1.0, 1.0])
    updated = update_multipliers_meng(lam, overflows, overflow_ref, step_size=1.0)
    assert (updated >= 0).all()


def test_zero_overflow_no_nan():
    lam = torch.tensor([1.0, 2.0])
    overflows = torch.zeros(2)
    overflow_ref = torch.tensor([1.0, 1.0])
    updated = update_multipliers_meng(lam, overflows, overflow_ref, step_size=0.1)
    assert torch.isfinite(updated).all()
