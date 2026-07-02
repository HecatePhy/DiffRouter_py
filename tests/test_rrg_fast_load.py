"""Tests for unified RRG and fast tensor load."""

import torch

from src.rrg.rrg import RRG


def _toy_tile_graph():
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
    return {
        "tiles": tiles,
        "coord_to_idx": coord_to_idx,
        "edges": edges,
        "edge_dist": {(a, b): 1 for a, b in edges},
        "edge_wl_score": {(a, b): 1 for a, b in edges},
    }, cap


def test_undirected_skips_directed_view():
    tg, cap = _toy_tile_graph()
    rrg = RRG(tg, cap, edge_mode="undirected")
    assert rrg.num_directed_edges == 0
    assert rrg.edge_mode == "undirected"
    assert len(rrg.phys_list) == len(tg["edges"])
    phys = rrg.get_phys_edges_in_bbox(0, 2, 0, 2)
    assert len(phys) == len(tg["edges"])


def test_directed_and_undirected_phys_bbox_agree():
    tg, cap = _toy_tile_graph()
    directed = RRG(tg, cap, edge_mode="directed")
    undirected = RRG(tg, cap, edge_mode="undirected")
    for bbox in [(0, 2, 0, 2), (0, 1, 0, 1)]:
        de = directed.get_edges_in_bbox(*bbox)
        phys_from_de = sorted({directed.phys_id_of_directed[i] for i in de})
        phys = undirected.get_phys_edges_in_bbox(*bbox)
        assert phys == phys_from_de


def test_from_tensors_matches_tile_graph():
    tg, cap = _toy_tile_graph()
    slow = RRG(tg, cap, edge_mode="directed")
    edge_arr = torch.tensor(tg["edges"], dtype=torch.long)
    cap_arr = torch.tensor([cap[e] for e in tg["edges"]], dtype=torch.float32)
    wl_arr = torch.tensor([1] * len(tg["edges"]), dtype=torch.long)
    tile_arr = torch.tensor(
        [[t[0], t[1], int(t[4])] for t in tg["tiles"]],
        dtype=torch.long,
    )
    data = {
        "format_version": 2,
        "device_rows": 3,
        "device_cols": 3,
        "tiles": tile_arr,
        "edges": edge_arr,
        "edge_capacities": cap_arr,
        "edge_wl_scores": wl_arr,
    }
    fast = RRG.from_tensors(data, edge_mode="directed")
    assert fast.num_directed_edges == slow.num_directed_edges
    assert len(fast.phys_list) == len(slow.phys_list)
    assert fast.get_edges_in_bbox(0, 2, 0, 2) == slow.get_edges_in_bbox(0, 2, 0, 2)
