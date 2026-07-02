"""Gradient smoke tests for the full differentiable objective."""

import torch

from tests.toy_router import build_toy_router


def test_total_loss_backward():
    router = build_toy_router()
    x = router.init_variables()
    loss = router.total_loss(x, w_wl=1.0, w_cong=1.0, w_conn=1.0, w_flow=1.0)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0


def test_augmented_lagrangian_backward():
    router = build_toy_router()
    x = router.init_variables()
    lam = torch.zeros(len(router.rrg.phys_list), device=router.device)
    loss = router.augmented_lagrangian(x, lam, rho=1.0, w_flow=1.0)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
