#!/usr/bin/env python3
"""End-to-end AL smoke test on toy 3x3 INT tile grid."""

import torch

from tests.toy_router import build_toy_router


def test_toy_al_end_to_end():
    device = torch.device("cpu")
    router = build_toy_router(device)
    x_opt = router.optimize_augmented_lagrangian(
        max_iterations=100,
        num_outer=20,
        num_inner=5,
        viz_dir=None,
        checkpoint_dir=None,
        verbose=False,
        log_setup=False,
        w_flow=1.0,
        conn_net_batch=0,
        flow_net_batch=0,
    )
    _, overflows = router._get_usage_and_overflows(x_opt)
    conn = router.connectivity_loss_effective_resistance(x_opt).item()
    flow = router.flow_conservation_loss(x_opt).item()
    wl = router.wirelength_loss(x_opt).item()
    assert overflows.max().item() < 1.0
    assert conn < 10.0
    assert flow < 10.0
    assert wl > 0


if __name__ == "__main__":
    test_toy_al_end_to_end()
    print("PASS: toy AL test")
