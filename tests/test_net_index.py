"""Tests for net index cache build/load."""

import os

import torch

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


def test_net_index_round_trip(tmp_path):
    router = build_toy_router(edge_mode="undirected")
    design = _MockDesign([
        _MockNet("n0", (0, 0), [(0, 2), (2, 2)]),
    ])
    index = NetIndex.build(
        design,
        router.rrg,
        device_rows=3,
        device_cols=3,
        coord_to_int=None,
        min_fanout=0,
        expansion_ratio=0.0,
        edge_mode="undirected",
        rrg_fingerprint="rrg",
        design_fingerprint="design",
    )
    path = tmp_path / "net_index.pt"
    index.save(str(path))
    loaded = NetIndex.load(str(path))
    assert loaded.num_nets == index.num_nets
    assert loaded.num_vars == index.num_vars
    assert loaded.var_offset.tolist() == index.var_offset.tolist()
    assert loaded.net_names == index.net_names


def test_net_index_build_from_json():
    router = build_toy_router(edge_mode="undirected")
    stub_json = {
        "format_version": 2,
        "route_filter": "stubs",
        "expansion_ratio": 0.0,
        "min_fanout": 0,
        "nets": [
            {
                "name": "n0",
                "src_int_idx": 0,
                "sink_int_idxs": [2],
                "bbox": [0, 2, 0, 2],
                "fanout": 1,
            }
        ],
    }
    index = NetIndex.build_from_json(
        stub_json,
        router.rrg,
        edge_mode="undirected",
        verbose=False,
    )
    assert index.num_nets == 1
    assert index.route_filter == "stubs"
    assert index.net_names == ["n0"]
    assert index.net_src_tile.tolist() == [0]
    assert index.net_sink_tiles[0].tolist() == [2]
    assert index.net_bbox.tolist() == [[0, 2, 0, 2]]


def test_net_index_apply_to_router():
    router = build_toy_router(edge_mode="directed")
    design = _MockDesign([
        _MockNet("n0", (0, 0), [(0, 2), (2, 2)]),
    ])
    index = NetIndex.build(
        design,
        router.rrg,
        device_rows=3,
        device_cols=3,
        coord_to_int=None,
        min_fanout=0,
        expansion_ratio=0.0,
        edge_mode="directed",
    )
    target = build_toy_router(edge_mode="directed")
    index.apply_to_router(target, design=design)
    assert target.num_nets == index.num_nets
    assert target.num_vars == index.num_vars
    assert len(target.nets) == 1
