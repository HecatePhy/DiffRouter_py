"""Tests for contest-weighted wirelength loss."""

import torch

from tests.toy_router import build_toy_router


def test_wirelength_equals_weighted_sum():
    router = build_toy_router()
    x = torch.ones(router.num_vars, device=router.device, requires_grad=True)
    expected = (x * router._directed_wl[router.net_edge_indices[0]]).sum()
    assert torch.allclose(router.wirelength_loss(x), expected)


def test_changing_wl_changes_loss():
    router = build_toy_router()
    x = torch.ones(router.num_vars, device=router.device)
    base = router.wirelength_loss(x).item()

    edges = router.rrg.phys_edges
    boosted = {e: 10 for e in edges}
    router2 = build_toy_router(wl_scores=boosted)
    boosted_loss = router2.wirelength_loss(x).item()
    assert boosted_loss > base


def test_wirelength_grad_nonzero():
    router = build_toy_router()
    x = router.init_variables()
    loss = router.wirelength_loss(x)
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_fallback_wl_one_without_features():
    device = torch.device("cpu")
    tiles = [(0, 0, None, "INT_0_0", True), (0, 1, None, "INT_0_1", True)]
    coord_to_idx = {(0, 0): 0, (0, 1): 1}
    edges = [(0, 1)]
    tile_graph = {"tiles": tiles, "coord_to_idx": coord_to_idx, "edges": edges}
    from src.rrg.rrg import RRG

    rrg = RRG(tile_graph, {(0, 1): 1}, device)
    assert all(rrg.phys_edge_wl_score[e] == 1 for e in rrg.phys_edges)
