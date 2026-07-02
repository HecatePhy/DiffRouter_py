"""Tests for effective-resistance connectivity loss."""

import torch

from src.router.connectivity import effective_resistance_loss_for_net
from tests.toy_router import build_toy_router


def test_disconnected_high_loss():
    router = build_toy_router()
    x = torch.zeros(router.num_vars, device=router.device)
    loss = router.connectivity_loss_effective_resistance(x)
    assert loss.item() > 0


def test_uniform_path_lower_loss():
    router = build_toy_router()
    x_zero = torch.zeros(router.num_vars, device=router.device)
    x_path = torch.zeros(router.num_vars, device=router.device)
    directed = router.rrg.directed_edges
    edge_list = router.net_edge_indices[0]
    for k, de_idx in enumerate(edge_list):
        u, v = directed[de_idx]
        if (u, v) in [(0, 1), (1, 2), (2, 5), (5, 8)]:
            x_path[k] = 1.0
    loss_zero = router.connectivity_loss_effective_resistance(x_zero).item()
    loss_path = router.connectivity_loss_effective_resistance(x_path).item()
    assert loss_path < loss_zero


def test_cg_vs_solve_agree():
    router = build_toy_router()
    x = router.init_variables()
    loss_solve = router.connectivity_loss_effective_resistance(x, solver="solve").item()
    loss_cg = router.connectivity_loss_effective_resistance(x, solver="cg").item()
    assert abs(loss_solve - loss_cg) < 0.05 * max(loss_solve, 1e-6)


def test_connectivity_unit_api():
    router = build_toy_router()
    x_slice = torch.ones(router.num_vars, device=router.device) * 0.3
    loss = effective_resistance_loss_for_net(
        x_slice,
        router.net_edge_indices[0],
        router.rrg.directed_edges,
        router.net_src_tile[0],
        router.net_sink_tiles[0],
        router.device,
        solver="solve",
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0
