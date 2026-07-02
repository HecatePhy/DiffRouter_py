"""Tests for undirected edge mode (one variable per physical edge, no flow term)."""

import torch

from src.router.route_extractor import RouteExtractor
from tests.toy_router import build_toy_router


def test_undirected_half_variables():
    directed = build_toy_router(edge_mode="directed")
    undirected = build_toy_router(edge_mode="undirected")
    assert undirected.num_vars == directed.num_vars // 2
    assert undirected.edge_mode == "undirected"


def test_flow_conservation_disabled():
    router = build_toy_router(edge_mode="undirected")
    x = router.init_variables()
    flow = router.flow_conservation_loss(x)
    assert flow.item() == 0.0


def test_undirected_losses_finite():
    router = build_toy_router(edge_mode="undirected")
    x = router.init_variables()
    wl = router.wirelength_loss(x)
    conn = router.connectivity_loss_effective_resistance(x)
    lam = torch.zeros(len(router.rrg.phys_list), device=router.device)
    al = router.augmented_lagrangian(x, lam, rho=1.0, w_flow=0.0)
    assert torch.isfinite(wl).all()
    assert torch.isfinite(conn).all()
    assert torch.isfinite(al).all()


def test_undirected_autograd():
    router = build_toy_router(edge_mode="undirected")
    x = router.init_variables()
    loss = router.total_loss(x, w_wl=1.0, w_cong=1.0, w_conn=1.0, w_flow=0.0)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0


def test_undirected_route_extraction():
    router = build_toy_router(edge_mode="undirected")
    x = router.init_variables()
    x.data.fill_(0.5)
    cong_map = router.get_congestion_map(x)
    extractor = RouteExtractor(threshold=0.01)
    paths = extractor.extract(router, x, cong_map)
    assert len(paths) == 1
    path = paths[0]
    assert path[0] == 0
    assert path[-1] == 8
    assert len(path) >= 2
