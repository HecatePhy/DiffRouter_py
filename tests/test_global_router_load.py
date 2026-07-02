"""Tests for GlobalRouter.load() two-file path."""

import torch

from src.router.global_router import GlobalRouter
from src.router.net_index import NetIndex
from tests.toy_router import build_toy_router


class _MockPin:
    def __init__(self, row, col):
        self._row = row
        self._col = col

    def getTile(self):
        return self

    def getRow(self):
        return self._row

    def getColumn(self):
        return self._col


class _MockNet:
    def __init__(self, name, src, sinks):
        self._name = name
        self._src = _MockPin(*src)
        self._sinks = [_MockPin(r, c) for r, c in sinks]

    def getName(self):
        return self._name

    def getSource(self):
        return self._src

    def getSinkPins(self):
        return self._sinks

    def getPins(self):
        return [self._src] + self._sinks

    def isStaticNet(self):
        return False


class _MockDesign:
    def __init__(self, nets):
        self._nets = nets

    def getNets(self):
        return self._nets


def _write_toy_rrg_pt(path):
    tg, cap = _toy_graph_and_cap()
    edge_arr = torch.tensor(tg["edges"], dtype=torch.long)
    cap_arr = torch.tensor([cap[e] for e in tg["edges"]], dtype=torch.float32)
    wl_arr = torch.tensor([1] * len(tg["edges"]), dtype=torch.long)
    tile_arr = torch.tensor(
        [[t[0], t[1], int(t[4])] for t in tg["tiles"]],
        dtype=torch.long,
    )
    torch.save(
        {
            "format_version": 2,
            "device_rows": 3,
            "device_cols": 3,
            "tiles": tile_arr,
            "edges": edge_arr,
            "edge_capacities": cap_arr,
            "edge_wl_scores": wl_arr,
        },
        path,
    )


def _toy_graph_and_cap():
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
    tg = {
        "tiles": tiles,
        "coord_to_idx": coord_to_idx,
        "edges": edges,
        "edge_dist": {(a, b): 1 for a, b in edges},
        "edge_wl_score": {(a, b): 1 for a, b in edges},
    }
    return tg, cap


def test_global_router_load(tmp_path):
    rrg_path = tmp_path / "toy_rrg.pt"
    index_path = tmp_path / "net_index.pt"
    _write_toy_rrg_pt(str(rrg_path))

    toy = build_toy_router(edge_mode="directed")
    design = _MockDesign([_MockNet("n0", (0, 0), [(0, 2), (2, 2)])])
    index = NetIndex.build(
        design,
        toy.rrg,
        device_rows=3,
        device_cols=3,
        coord_to_int=None,
        min_fanout=0,
        expansion_ratio=0.0,
        edge_mode="directed",
        rrg_fingerprint="",
    )
    index.save(str(index_path))

    router = GlobalRouter.load(
        str(rrg_path),
        str(index_path),
        device=torch.device("cpu"),
        edge_mode="directed",
        verbose=False,
    )
    assert router.design is None
    assert router.num_nets == index.num_nets
    assert router.num_vars == index.num_vars
    x = router.init_variables()
    loss = router.total_loss(x, w_wl=1.0, w_cong=1.0, w_conn=1.0, w_flow=1.0)
    assert torch.isfinite(loss).all()


def test_attach_design():
    router = build_toy_router()
    router.net_names = ["n0"]
    router.nets = [None]
    design = _MockDesign([_MockNet("n0", (0, 0), [(0, 2)])])
    router.attach_design(design)
    assert router.nets[0] is not None
    assert router.design is design
