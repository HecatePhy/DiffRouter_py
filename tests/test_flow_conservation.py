"""Tests for per-node flow conservation penalty."""

import torch

from src.router.flow_conservation import (
    compute_node_imbalances,
    flow_conservation_loss_for_net,
    flow_conservation_penalty,
)
from tests.toy_router import build_toy_router


def _path_flow_on_toy(router, path_nodes):
    """Set x=1 on directed edges following path_nodes (0->1->...->8)."""
    x = torch.zeros(router.num_vars, device=router.device)
    edge_list = router.net_edge_indices[0]
    directed = router.rrg.directed_edges
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        for k, de_idx in enumerate(edge_list):
            if directed[de_idx] == (u, v):
                x[k] = 1.0
                break
    return x


def test_zero_penalty_for_feasible_path_flow():
    router = build_toy_router()
    path = [0, 1, 2, 5, 8]
    x = _path_flow_on_toy(router, path)
    loss = router.flow_conservation_loss(x)
    assert loss.item() < 1e-6


def test_positive_penalty_when_imbalanced():
    router = build_toy_router()
    x = torch.ones(router.num_vars, device=router.device) * 0.5
    loss = router.flow_conservation_loss(x)
    assert loss.item() > 0


def test_node_imbalance_on_global_indices():
    device = torch.device("cpu")
    directed_edges = [(0, 1), (1, 2), (1, 3)]
    edge_list = [0, 1, 2]
    # K=2 sinks: source must emit 2 units (1 per sink branch)
    x_slice = torch.tensor([2.0, 1.0, 1.0], device=device)
    imbalance = compute_node_imbalances(
        x_slice, edge_list, directed_edges, src=0, sinks=[2, 3], num_tiles=4, device=device
    )
    assert imbalance.shape == (4,)
    assert imbalance.abs().max().item() < 1e-6


def test_multi_sink_demands():
    device = torch.device("cpu")
    directed_edges = [(0, 1), (1, 2), (1, 3)]
    edge_list = [0, 1, 2]
    x_slice = torch.tensor([1.0, 0.5, 0.5], device=device)
    loss = flow_conservation_loss_for_net(
        x_slice,
        edge_list,
        directed_edges,
        src=0,
        sinks=[2, 3],
        num_tiles=4,
        device=device,
    )
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_penalty_is_sum_over_nodes():
    imbalance = torch.tensor([0.0, 1.0, -2.0, 0.0])
    assert flow_conservation_penalty(imbalance).item() == 5.0


def test_flow_grad_nonzero():
    router = build_toy_router()
    x = router.init_variables()
    loss = router.flow_conservation_loss(x)
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0
