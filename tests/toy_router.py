"""Shared 3x3 toy GlobalRouter fixture for unit tests."""

import torch
import torch.nn as nn

from src.rrg.rrg import RRG
from src.router.global_router import GlobalRouter


class _ToyDesign:
    def getNets(self):
        return []


def build_toy_router(device=None, wl_scores=None, edge_mode="directed"):
    """Build a minimal GlobalRouter on a 3x3 INT grid (src=0, sink=8)."""
    device = device or torch.device("cpu")
    tiles = [(r, c, None, "INT_%d_%d" % (r, c), True) for r in range(3) for c in range(3)]
    coord_to_idx = {(r, c): r * 3 + c for r in range(3) for c in range(3)}
    edges = []
    for r in range(3):
        for c in range(3):
            idx = coord_to_idx[(r, c)]
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in coord_to_idx:
                    nidx = coord_to_idx[(nr, nc)]
                    edges.append((min(idx, nidx), max(idx, nidx)))
    edges = sorted(set(edges))
    cap = {(a, b): 2 for a, b in edges}
    edge_dist = {(a, b): 1 for a, b in edges}
    if wl_scores is None:
        edge_wl_score = {(a, b): (3 if a % 2 == 0 else 1) for a, b in edges}
    else:
        edge_wl_score = wl_scores
    tile_graph = {
        "tiles": tiles,
        "coord_to_idx": coord_to_idx,
        "edges": edges,
        "edge_dist": edge_dist,
        "edge_wl_score": edge_wl_score,
    }

    router = GlobalRouter.__new__(GlobalRouter)
    nn.Module.__init__(router)
    router.design = _ToyDesign()
    router.device = device
    router.min_fanout = 0
    router.expansion_ratio = 0.0
    router.connectivity_solver = "solve"
    router.conn_net_batch = 0
    router.flow_net_batch = 0
    router.edge_mode = edge_mode
    router.device_rows = 3
    router.device_cols = 3
    router.coord_to_int = None
    router.rrg = RRG(tile_graph, cap, device, edge_mode=edge_mode)
    router._build_int_tile_prefix_sum()

    if edge_mode == "undirected":
        edge_indices = router.rrg.get_phys_edges_in_bbox(0, 2, 0, 2)
    else:
        edge_indices = router.rrg.get_edges_in_bbox(0, 2, 0, 2)
    router.nets = [None]
    router.net_src_tile = [0]
    router.net_sink_tiles = [[8]]
    router.net_fanout = [1]
    router.net_edge_indices = [edge_indices]
    router.net_bbox = [(0, 2, 0, 2)]
    router.num_nets = 1
    router._var_offset = [0, len(edge_indices)]
    router.num_vars = router._var_offset[-1]
    router._net_edge_tensors = [
        torch.tensor(edge_indices, device=device, dtype=torch.long)
    ]
    router._phys_capacity_tensor = router.rrg._phys_capacity_tensor.to(device)
    if edge_mode == "directed":
        router._phys_de_indices_gpu = [t.to(device) for t in router.rrg._phys_de_indices]
    else:
        router._phys_de_indices_gpu = []
    router._init_edge_weight_tensors()
    return router
