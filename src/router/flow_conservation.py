"""Per-node flow conservation (Kirchhoff) penalty on global INT tile indices."""

from typing import List, Tuple

import torch


def compute_node_imbalances(
    x_slice: torch.Tensor,
    edge_list: List[int],
    directed_edges: List[Tuple[int, int]],
    src: int,
    sinks: List[int],
    num_tiles: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Kirchhoff imbalance at each global INT tile node v:

        imbalance[v] = in[v] - out[v] - demand[v]

    Multi-sink unit-flow demands: source = -K, each sink = +1, transit = 0
    (K = number of sinks). Indexed by global tile id (0 .. num_tiles-1).
    """
    dtype = x_slice.dtype
    in_flow = torch.zeros(num_tiles, device=device, dtype=dtype)
    out_flow = torch.zeros(num_tiles, device=device, dtype=dtype)

    for k, de_idx in enumerate(edge_list):
        u, v = directed_edges[de_idx]
        val = x_slice[k]
        out_flow[u] = out_flow[u] + val
        in_flow[v] = in_flow[v] + val

    demand = torch.zeros(num_tiles, device=device, dtype=dtype)
    if 0 <= src < num_tiles and sinks:
        demand[src] = -float(len(sinks))
    for sink in sinks:
        if 0 <= sink < num_tiles:
            demand[sink] = demand[sink] + 1.0

    return in_flow - out_flow - demand


def flow_conservation_penalty(imbalance: torch.Tensor) -> torch.Tensor:
    """Soft penalty: sum of squared per-node imbalances."""
    return (imbalance ** 2).sum()


def flow_conservation_loss_for_net(
    x_slice: torch.Tensor,
    edge_list: List[int],
    directed_edges: List[Tuple[int, int]],
    src: int,
    sinks: List[int],
    num_tiles: int,
    device: torch.device,
) -> torch.Tensor:
    """One net's contribution: Σ_v imbalance[v]² over global tile nodes."""
    if x_slice.numel() == 0 or not sinks:
        return torch.tensor(0.0, device=device, dtype=x_slice.dtype)

    imbalance = compute_node_imbalances(
        x_slice, edge_list, directed_edges, src, sinks, num_tiles, device
    )
    return flow_conservation_penalty(imbalance)
