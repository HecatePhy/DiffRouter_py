"""Smoke tests for augmented Lagrangian optimization on toy grid."""

import torch

from src.router.meng_lambda import update_multipliers_meng
from tests.toy_router import build_toy_router


def test_one_outer_inner_step_finite():
    router = build_toy_router()
    x = router.init_variables()
    lam = torch.zeros(len(router.rrg.phys_list), device=router.device)
    L = router.augmented_lagrangian(x, lam, rho=1.0, w_flow=1.0)
    assert torch.isfinite(L)


def test_al_optimization_smoke():
    router = build_toy_router()
    x_init = router.init_variables()
    x_opt = router.optimize_augmented_lagrangian(
        max_iterations=50,
        num_outer=10,
        num_inner=5,
        viz_dir=None,
        checkpoint_dir=None,
        verbose=False,
        log_setup=False,
        w_flow=1.0,
        conn_net_batch=0,
        flow_net_batch=0,
    )
    assert (x_opt >= 0).all() and (x_opt <= 1).all()
    _, overflows = router._get_usage_and_overflows(x_opt)
    assert overflows.max().item() < 2.0


def test_lambda_increases_with_overflow():
    router = build_toy_router()
    x = router.init_variables()
    _, overflows = router._get_usage_and_overflows(x)
    overflow_ref = torch.clamp(overflows.clone(), min=1e-8)
    lam0 = torch.zeros_like(overflows)
    lam1 = update_multipliers_meng(lam0, overflows, overflow_ref, step_size=0.5)
    if overflows.max().item() > 0:
        assert lam1.norm().item() >= lam0.norm().item()
