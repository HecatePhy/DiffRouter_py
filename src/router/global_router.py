"""Differentiable global router (GlobalRouter)."""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.rrg.rrg import RRG
from src.router.connectivity import (
    effective_resistance_loss_batched,
    effective_resistance_loss_for_net,
    effective_resistance_loss_for_net_undirected,
)
from src.router.flow_conservation import flow_conservation_loss_for_net


def get_net_tiles(net: Any) -> Set[Tuple[int, int]]:
    tiles = set()
    for pin in net.getPins():
        tile = pin.getTile()
        if tile is not None:
            tiles.add((int(tile.getRow()), int(tile.getColumn())))
    return tiles


def plot_bbox_distribution(
    tile_cols: List[int],
    tile_rows: List[int],
    areas: List[int],
    int_tiles: Optional[List[int]] = None,
    output_path: str = "results/checkpoint/bbox_distribution.png",
) -> None:
    import matplotlib.pyplot as plt
    import os

    n = len(tile_cols)
    if n == 0:
        return
    n_plots = 4 if int_tiles else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    axes[0].hist(tile_cols, bins=min(50, max(10, n // 20)), edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("tile_col")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Bbox Width (tile_col)")

    axes[1].hist(tile_rows, bins=min(50, max(10, n // 20)), edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("tile_row")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Bbox Height (tile_row)")

    axes[2].hist(areas, bins=min(50, max(10, n // 20)), edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Area")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Bbox Area")

    if int_tiles is not None:
        axes[3].hist(int_tiles, bins=min(50, max(10, n // 20)), edgecolor="black", alpha=0.7)
        axes[3].set_xlabel("INT tiles")
        axes[3].set_ylabel("Count")
        axes[3].set_title("INT Tiles Covered")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bbox distribution plot saved to: {output_path}")


class GlobalRouter(nn.Module):
    """Differentiable global router with per-net bbox-scoped edge variables."""

    def __init__(self):
        super().__init__()
        self.design = None
        self.device = torch.device("cpu")
        self.edge_mode = "directed"
        self.min_fanout = 0
        self.expansion_ratio = 0.1
        self.max_nets = None
        self.connectivity_solver = "solve"
        self.conn_net_batch = 0
        self.flow_net_batch = 0
        self.coord_to_int = None
        self.net_names: List[str] = []
        self.nets: List[Any] = []

    @staticmethod
    def _compiled_cache_path(net_index_path: str, edge_mode: str) -> str:
        base = net_index_path[:-3] if net_index_path.endswith(".pt") else net_index_path
        return f"{base}.{edge_mode}.compiled.pt"

    @classmethod
    def load(
        cls,
        rrg_path: str,
        net_index_path: str,
        device: torch.device = None,
        edge_mode: str = "directed",
        verbose: bool = True,
        use_compiled: bool = True,
    ) -> "GlobalRouter":
        """Fast path: load pre-built RRG + net index (no Java design).

        With use_compiled, a one-time compiled cache (big contiguous tensors) is
        written next to the net index; subsequent loads skip the ~140s net-index
        deserialize + flat-array build and reload in ~15-20s.
        """
        import os as _os
        from src.load_design import load_rrg_fast
        from src.router.net_index import NetIndex

        if edge_mode not in ("directed", "undirected"):
            raise ValueError(f"edge_mode must be 'directed' or 'undirected', got {edge_mode!r}")

        compiled_path = cls._compiled_cache_path(net_index_path, edge_mode)
        if (
            use_compiled
            and _os.path.isfile(compiled_path)
            and _os.path.isfile(net_index_path)
            and _os.path.getmtime(compiled_path) >= _os.path.getmtime(net_index_path)
        ):
            try:
                return cls.load_compiled(
                    rrg_path, compiled_path, device=device,
                    edge_mode=edge_mode, verbose=verbose,
                )
            except Exception as exc:  # noqa: BLE001 - fall back to full build
                if verbose:
                    print(f"    Compiled cache load failed ({exc}); rebuilding")

        router = cls()
        router.device = device or torch.device("cpu")
        router.edge_mode = edge_mode

        if verbose:
            print(f"    Loading RRG from: {rrg_path}")
        rrg, device_rows, device_cols, coord_to_int, fmt = load_rrg_fast(
            rrg_path, edge_mode=edge_mode, device=router.device
        )
        if verbose and fmt < 2:
            print(
                "    Warning: RRG format_version < 2 (slow load). "
                "Re-pack with: python scripts/ExtractRRG.py --from-json ... -o ..."
            )
        if verbose:
            print(
                f"    RRG loaded: {rrg.num_tiles} tiles, "
                f"{len(rrg.phys_list)} phys edges, edge_mode={edge_mode}"
            )

        if verbose:
            print(f"    Loading net index: {net_index_path}")
        net_index = NetIndex.load(net_index_path)
        net_index.validate_for_load(rrg_path, edge_mode)
        router.min_fanout = net_index.min_fanout
        router.expansion_ratio = net_index.expansion_ratio
        router.max_nets = net_index.max_nets

        router._init_from_rrg_and_index(
            rrg, device_rows, device_cols, coord_to_int, net_index
        )
        if verbose:
            print(
                f"    Loaded: {router.num_nets} nets, "
                f"{router.num_vars} variables (edge_mode={edge_mode})"
            )
        if use_compiled:
            try:
                router.save_compiled(compiled_path)
                if verbose:
                    print(f"    Saved compiled cache -> {compiled_path}")
            except Exception as exc:  # noqa: BLE001 - caching is best-effort
                if verbose:
                    print(f"    save_compiled failed (non-fatal): {exc}")
        return router

    @classmethod
    def from_device(
        cls,
        design: Any,
        device_obj: Any,
        min_fanout: int = 0,
        expansion_ratio: float = 0.1,
        max_nets: Optional[int] = None,
        device: torch.device = None,
        rrg_log_path: Optional[str] = None,
        edge_mode: str = "directed",
        verbose: bool = True,
    ) -> "GlobalRouter":
        """Dev path: build RRG from RapidWright device and live net list (no net index)."""
        from src.load_design import get_tile_graph, get_tile_edge_capacities

        if edge_mode not in ("directed", "undirected"):
            raise ValueError(f"edge_mode must be 'directed' or 'undirected', got {edge_mode!r}")

        router = cls()
        router.design = design
        router.device = device or torch.device("cpu")
        router.edge_mode = edge_mode
        router.min_fanout = min_fanout
        router.expansion_ratio = expansion_ratio
        router.max_nets = max_nets

        if verbose:
            print("    Building tile graph from device...")
        tile_graph = get_tile_graph(device_obj)
        if verbose:
            print("    Computing edge capacities (wire-based, includes jump)...")
        edge_capacities, wire_edges = get_tile_edge_capacities(device_obj, tile_graph)
        tile_graph = dict(tile_graph)
        tile_graph["edges"] = sorted(set(tile_graph["edges"]) | wire_edges)
        device_rows = device_obj.getRows()
        device_cols = device_obj.getColumns()
        if verbose:
            print(f"    Building RRG (edge_mode={edge_mode})...")
        rrg = RRG(tile_graph, edge_capacities, router.device, edge_mode=edge_mode)
        router.coord_to_int = tile_graph.get("coord_to_int")

        if rrg_log_path:
            if verbose:
                print(f"    Writing RRG capacity log to: {rrg_log_path}")
            rrg.write_capacity_log(rrg_log_path)

        router.rrg = rrg
        router.device_rows = device_rows
        router.device_cols = device_cols
        router._build_int_tile_prefix_sum()

        if verbose:
            print(f"    Building net list live (min_fanout={min_fanout})...")
        router._build_net_list_from_design(design)
        router._finish_net_setup()
        if verbose:
            print(f"    Nets to route: {router.num_nets}, total variables: {router.num_vars}")
        return router

    @classmethod
    def load_live(
        cls,
        design: Any,
        rrg_path: str,
        device: torch.device = None,
        edge_mode: str = "directed",
        min_fanout: int = 0,
        expansion_ratio: float = 0.1,
        max_nets: Optional[int] = None,
        verbose: bool = True,
    ) -> "GlobalRouter":
        """Debug path: load RRG and build net list from design (no net index file)."""
        from src.load_design import load_rrg_fast

        router = cls()
        router.design = design
        router.device = device or torch.device("cpu")
        router.edge_mode = edge_mode
        router.min_fanout = min_fanout
        router.expansion_ratio = expansion_ratio
        router.max_nets = max_nets

        if verbose:
            print(f"    Loading RRG from: {rrg_path}")
        rrg, device_rows, device_cols, coord_to_int, _fmt = load_rrg_fast(
            rrg_path, edge_mode=edge_mode, device=router.device
        )
        router.rrg = rrg
        router.device_rows = device_rows
        router.device_cols = device_cols
        router.coord_to_int = coord_to_int
        router._build_int_tile_prefix_sum()
        if verbose:
            print(f"    Building net list live (min_fanout={min_fanout})...")
        router._build_net_list_from_design(design)
        router._finish_net_setup()
        router.nets = list(router.nets)
        if verbose:
            print(f"    Nets to route: {router.num_nets}, total variables: {router.num_vars}")
        return router

    def save_compiled(self, path: str) -> None:
        """Serialize the built runtime state as a few big contiguous tensors.

        Skips re-deserializing the 274K-tensor net index and re-running the
        per-net torch.unique loop in _build_flat_arrays on the next load (which
        together dominate GlobalRouter.load ~140s). Reload via load_compiled.
        """
        import os as _os
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        sink_sizes = [len(s) for s in self.net_sink_tiles]
        sink_off = torch.zeros(len(sink_sizes) + 1, dtype=torch.long)
        if sink_sizes:
            sink_off[1:] = torch.cumsum(torch.tensor(sink_sizes, dtype=torch.long), 0)
        sink_flat = torch.tensor(
            [t for s in self.net_sink_tiles for t in s], dtype=torch.long
        ) if sink_sizes else torch.zeros(0, dtype=torch.long)
        conn = None
        if self._conn is not None:
            conn = {
                "src_flat": self._conn["src_flat"].cpu(),
                "sink_flat": self._conn["sink_flat"].cpu(),
                "col_id": self._conn["col_id"].cpu(),
                "num_nodes": int(self._conn["num_nodes"]),
                "num_cols": int(self._conn["num_cols"]),
            }
        state = {
            "compiled_format": 1,
            "edge_mode": self.edge_mode,
            "num_nets": self.num_nets,
            "num_vars": self.num_vars,
            "num_global_edges": self._num_global_edges,
            "total_local_nodes": self._total_local_nodes,
            "device_rows": self.device_rows,
            "device_cols": self.device_cols,
            "flat_edge_idx": self._flat_edge_idx.cpu(),
            "flat_wl": self._flat_wl.cpu(),
            "flat_u": self._flat_u.cpu(),
            "flat_v": self._flat_v.cpu(),
            "flow_demand": self._flow_demand.cpu(),
            "d2p": None if self._d2p is None else self._d2p.cpu(),
            "phys_capacity": self._phys_capacity_tensor.cpu(),
            "var_offset": torch.tensor(self._var_offset, dtype=torch.long),
            "node_offset": torch.tensor(self._node_offset, dtype=torch.long),
            "net_src_tile": torch.tensor(self.net_src_tile, dtype=torch.long),
            "net_fanout": torch.tensor(self.net_fanout, dtype=torch.long),
            "net_bbox": torch.tensor(self.net_bbox, dtype=torch.long),
            "sink_flat": sink_flat,
            "sink_off": sink_off,
            "net_names": self.net_names,
            "conn": conn,
        }
        torch.save(state, path)

    @classmethod
    def load_compiled(
        cls,
        rrg_path: str,
        compiled_path: str,
        device: torch.device = None,
        edge_mode: str = "directed",
        verbose: bool = True,
    ) -> "GlobalRouter":
        """Fast reload from a save_compiled() cache (skips NetIndex + flat build)."""
        from src.load_design import load_rrg_fast

        router = cls()
        router.device = device or torch.device("cpu")
        router.edge_mode = edge_mode
        dev = router.device

        rrg, drows, dcols, c2i, _fmt = load_rrg_fast(
            rrg_path, edge_mode=edge_mode, device=dev
        )
        router.rrg = rrg
        router.device_rows = drows
        router.device_cols = dcols
        router.coord_to_int = c2i
        router._build_int_tile_prefix_sum()

        st = torch.load(compiled_path, map_location="cpu")
        if st.get("edge_mode") != edge_mode:
            raise ValueError(
                f"compiled cache edge_mode={st.get('edge_mode')} != {edge_mode}"
            )
        router.num_nets = int(st["num_nets"])
        router.num_vars = int(st["num_vars"])
        router._num_global_edges = int(st["num_global_edges"])
        router._total_local_nodes = int(st["total_local_nodes"])
        router._flat_edge_idx = st["flat_edge_idx"].to(dev)
        router._flat_wl = st["flat_wl"].to(dev)
        router._flat_u = st["flat_u"].to(dev)
        router._flat_v = st["flat_v"].to(dev)
        router._flow_demand = st["flow_demand"].to(dev)
        router._d2p = None if st["d2p"] is None else st["d2p"].to(dev)
        router._phys_capacity_tensor = st["phys_capacity"].to(dev)
        router._idx_dtype = router._flat_edge_idx.dtype
        router._var_offset = st["var_offset"].tolist()
        router._node_offset = st["node_offset"].tolist()
        router.net_src_tile = st["net_src_tile"].tolist()
        router.net_fanout = st["net_fanout"].tolist()
        router.net_bbox = [tuple(r) for r in st["net_bbox"].tolist()]
        so = st["sink_off"].tolist()
        sf = st["sink_flat"].tolist()
        router.net_sink_tiles = [sf[so[i]:so[i + 1]] for i in range(router.num_nets)]
        router.net_names = list(st["net_names"])
        router.nets = [None] * router.num_nets
        if st["conn"] is None:
            router._conn = None
        else:
            c = st["conn"]
            router._conn = {
                "flat_u": router._flat_u,
                "flat_v": router._flat_v,
                "src_flat": c["src_flat"].to(dev),
                "sink_flat": c["sink_flat"].to(dev),
                "col_id": c["col_id"].to(dev),
                "num_nodes": int(c["num_nodes"]),
                "num_cols": int(c["num_cols"]),
            }
        # Per-net edge views for extraction: slices of the flat array (no compute).
        vo = router._var_offset
        fe = router._flat_edge_idx
        router.net_edge_indices = [fe[vo[i]:vo[i + 1]] for i in range(router.num_nets)]
        router._net_edge_tensors = router.net_edge_indices
        if verbose:
            print(
                f"    Loaded compiled: {router.num_nets} nets, "
                f"{router.num_vars} variables (edge_mode={edge_mode})"
            )
        return router

    def attach_design(self, design: Any) -> None:
        """Attach Java net objects by name (required before detailed routing)."""
        from src.router.net_index import _net_name

        name_to_net: Dict[str, Any] = {}
        for net in design.getNets():
            name_to_net[_net_name(net)] = net
        nets: List[Any] = []
        missing = 0
        for name in self.net_names:
            net = name_to_net.get(name)
            if net is None:
                missing += 1
                nets.append(None)
            else:
                nets.append(net)
        if missing:
            print(f"  Warning: {missing} net(s) from index not found in design")
        self.nets = nets
        self.design = design

    def _init_from_rrg_and_index(
        self,
        rrg: RRG,
        device_rows: int,
        device_cols: int,
        coord_to_int: Optional[dict],
        net_index: Any,
    ) -> None:
        self.rrg = rrg
        self.device_rows = device_rows
        self.device_cols = device_cols
        self.coord_to_int = coord_to_int
        self._build_int_tile_prefix_sum()
        net_index.apply_to_router(self, design=None)

        caps = self.rrg._phys_capacity_tensor.to(self.device)
        self._phys_capacity_tensor = caps
        self._init_edge_weight_tensors()

    def _build_net_list_from_design(self, design: Any) -> None:
        self.net_names = []
        self.nets = []
        self.net_src_tile = []
        self.net_sink_tiles = []
        self.net_fanout = []
        self.net_edge_indices = []
        self.net_bbox = []

        pin_to_idx = self.coord_to_int if self.coord_to_int is not None else self.rrg.coord_to_idx
        for net in design.getNets():
            if net.getSource() is None and not net.isStaticNet():
                continue
            sink_pins = list(net.getSinkPins())
            fanout = len(sink_pins)
            if fanout <= self.min_fanout:
                continue

            tiles = get_net_tiles(net)
            if len(tiles) < 2:
                continue

            rows = [t[0] for t in tiles]
            cols = [t[1] for t in tiles]
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)

            w = max(1, max_col - min_col)
            h = max(1, max_row - min_row)
            exp_w = max(1, int(w * self.expansion_ratio))
            exp_h = max(1, int(h * self.expansion_ratio))
            min_col = max(0, min_col - exp_w)
            max_col = min(self.device_cols, max_col + exp_w)
            min_row = max(0, min_row - exp_h)
            max_row = min(self.device_rows, max_row + exp_h)

            edge_indices = (
                self.rrg.get_phys_edges_in_bbox(min_col, max_col, min_row, max_row)
                if self.edge_mode == "undirected"
                else self.rrg.get_edges_in_bbox(min_col, max_col, min_row, max_row)
            )
            if not edge_indices:
                continue

            src_pin = net.getSource()
            src_tile = src_pin.getTile() if src_pin else None
            if src_tile is None:
                continue
            src_idx = pin_to_idx.get((int(src_tile.getRow()), int(src_tile.getColumn())))
            if src_idx is None:
                continue

            sink_idxs = []
            for pin in sink_pins:
                t = pin.getTile()
                if t is None:
                    continue
                idx = pin_to_idx.get((int(t.getRow()), int(t.getColumn())))
                if idx is not None:
                    sink_idxs.append(idx)
            if not sink_idxs:
                continue

            name = net.getName()
            self.net_names.append(str(name) if name is not None else "")
            self.nets.append(net)
            self.net_src_tile.append(src_idx)
            self.net_sink_tiles.append(sink_idxs)
            self.net_fanout.append(fanout)
            self.net_edge_indices.append(edge_indices)
            self.net_bbox.append((min_col, max_col, min_row, max_row))

            if self.max_nets is not None and len(self.nets) >= self.max_nets:
                break

        self.num_nets = len(self.nets)
        self._var_offset = [0]
        for ei in self.net_edge_indices:
            self._var_offset.append(self._var_offset[-1] + len(ei))
        self.num_vars = self._var_offset[-1]

    def _finish_net_setup(self) -> None:
        n_glob = (
            self.rrg.num_directed_edges
            if self.edge_mode == "directed"
            else len(self.rrg.phys_list)
        )
        idx_dtype = torch.int32 if n_glob < (1 << 31) else torch.long
        self._idx_dtype = idx_dtype
        self._net_edge_tensors = []
        for edge_list in self.net_edge_indices:
            self._net_edge_tensors.append(
                torch.tensor(edge_list, device=self.device, dtype=idx_dtype)
            )
        caps = self.rrg._phys_capacity_tensor.to(self.device)
        self._phys_capacity_tensor = caps
        self._init_edge_weight_tensors()

    def _init_edge_weight_tensors(self) -> None:
        """Precompute per-edge WL scores, then all flattened loss tensors."""
        if self.edge_mode == "undirected":
            self._directed_wl = None
            edge_wl = self.rrg._phys_wl_score_tensor.to(self.device)
        else:
            pid = torch.tensor(self.rrg.phys_id_of_directed, dtype=torch.long)
            self._directed_wl = self.rrg._phys_wl_score_tensor[pid].to(self.device)
            edge_wl = self._directed_wl
        self._build_flat_arrays(edge_wl)

    def _build_flat_arrays(self, edge_wl: torch.Tensor) -> None:
        """Flatten all per-net structures into single contiguous tensors.

        Built once at setup so every loss term is a handful of vectorized ops:
        - _flat_edge_idx [num_vars]: global (directed or phys) edge id per var
        - _flat_wl       [num_vars]: WL score per var
        - _flat_u/_flat_v [num_vars]: flattened *local* node ids (per-net node
          blocks laid out consecutively, offsets in _node_offset)
        - _flow_demand [total_local_nodes]: Kirchhoff demand vector
        - _conn: batched connectivity RHS layout (one column per (net, sink))
        - _d2p [num_directed_edges]: directed edge -> phys edge id (directed)
        """
        dev = self.device
        idx_dtype = getattr(self, "_idx_dtype", torch.long)
        if self.edge_mode == "directed":
            endpoints = torch.tensor(self.rrg.directed_edges, dtype=torch.long)
            self._num_global_edges = self.rrg.num_directed_edges
            self._d2p = torch.tensor(
                self.rrg.phys_id_of_directed, dtype=idx_dtype, device=dev
            )
        else:
            endpoints = torch.tensor(self.rrg.phys_list, dtype=torch.long)
            self._num_global_edges = len(self.rrg.phys_list)
            self._d2p = None

        if self.num_nets > 0 and self.num_vars > 0:
            flat_edge = torch.cat([t.cpu() for t in self._net_edge_tensors])
        else:
            flat_edge = torch.zeros(0, dtype=idx_dtype)
        self._flat_edge_idx = flat_edge.to(dev, dtype=idx_dtype)
        self._flat_wl = edge_wl[self._flat_edge_idx]

        flat_u_l: List[torch.Tensor] = []
        flat_v_l: List[torch.Tensor] = []
        demand_idx_l: List[torch.Tensor] = []
        demand_val_l: List[torch.Tensor] = []
        src_flat_l: List[torch.Tensor] = []
        sink_flat_l: List[torch.Tensor] = []
        col_l: List[torch.Tensor] = []
        node_offset = [0]
        node_base = 0
        total_cols = 0

        for i in range(self.num_nets):
            e = self._net_edge_tensors[i].cpu()
            if e.numel() == 0:
                node_offset.append(node_base)
                continue
            uv = endpoints[e]
            nodes, inv = torch.unique(uv, return_inverse=True)
            n = int(nodes.numel())
            flat_u_l.append(inv[:, 0] + node_base)
            flat_v_l.append(inv[:, 1] + node_base)

            src = int(self.net_src_tile[i])
            sinks = torch.tensor(self.net_sink_tiles[i], dtype=torch.long)
            spos = int(torch.searchsorted(nodes, torch.tensor(src, dtype=torch.long)))
            src_in = spos < n and int(nodes[spos]) == src
            if sinks.numel() > 0:
                pos = torch.searchsorted(nodes, sinks).clamp(max=n - 1)
                ok = nodes[pos] == sinks
            else:
                pos = sinks
                ok = torch.zeros(0, dtype=torch.bool)

            if src_in and sinks.numel() > 0:
                demand_idx_l.append(torch.tensor([node_base + spos]))
                demand_val_l.append(
                    torch.tensor([-float(sinks.numel())], dtype=torch.float32)
                )
            if bool(ok.any()):
                sink_pos = pos[ok] + node_base
                demand_idx_l.append(sink_pos)
                demand_val_l.append(torch.ones(sink_pos.numel(), dtype=torch.float32))
                if src_in and n >= 2:
                    k = int(ok.sum())
                    src_flat_l.append(
                        torch.full((k,), node_base + spos, dtype=torch.long)
                    )
                    sink_flat_l.append(sink_pos)
                    col_l.append(
                        torch.arange(total_cols, total_cols + k, dtype=torch.long)
                    )
                    total_cols += k

            node_base += n
            node_offset.append(node_base)

        self._total_local_nodes = node_base
        self._node_offset = node_offset
        empty = torch.zeros(0, dtype=idx_dtype, device=dev)
        self._flat_u = torch.cat(flat_u_l).to(dev, dtype=idx_dtype) if flat_u_l else empty
        self._flat_v = torch.cat(flat_v_l).to(dev, dtype=idx_dtype) if flat_v_l else empty

        demand = torch.zeros(max(node_base, 1), dtype=torch.float32)
        if demand_idx_l:
            demand.index_add_(0, torch.cat(demand_idx_l), torch.cat(demand_val_l))
        self._flow_demand = demand[:node_base].to(dev)

        if total_cols > 0:
            self._conn = {
                "flat_u": self._flat_u,
                "flat_v": self._flat_v,
                "src_flat": torch.cat(src_flat_l).to(dev, dtype=idx_dtype),
                "sink_flat": torch.cat(sink_flat_l).to(dev, dtype=idx_dtype),
                "col_id": torch.cat(col_l).to(dev, dtype=idx_dtype),
                "num_nodes": node_base,
                "num_cols": total_cols,
            }
        else:
            self._conn = None

    def _build_int_tile_prefix_sum(self) -> None:
        grid = np.zeros((self.device_rows, self.device_cols), dtype=np.int64)
        for idx, tile in enumerate(self.rrg.tiles):
            row, col = tile[0], tile[1]
            is_int = tile[4] if len(tile) > 4 else True
            if is_int and 0 <= row < self.device_rows and 0 <= col < self.device_cols:
                grid[row, col] = 1
        self._int_sat = np.cumsum(np.cumsum(grid, axis=0), axis=1)

    def _count_int_tiles_in_bbox(
        self, min_col: int, max_col: int, min_row: int, max_row: int
    ) -> int:
        r0, r1 = max(0, min_row), min(self.device_rows - 1, max_row)
        c0, c1 = max(0, min_col), min(self.device_cols - 1, max_col)
        if r0 > r1 or c0 > c1:
            return 0
        sat = self._int_sat
        return int(
            sat[r1, c1]
            - (sat[r0 - 1, c1] if r0 > 0 else 0)
            - (sat[r1, c0 - 1] if c0 > 0 else 0)
            + (sat[r0 - 1, c0 - 1] if r0 > 0 and c0 > 0 else 0)
        )

    def get_bbox_size_distribution(self) -> Dict[str, Any]:
        if not self.net_bbox:
            return {}
        widths, heights, areas, int_counts = [], [], [], []
        for min_col, max_col, min_row, max_row in self.net_bbox:
            w = max_col - min_col + 1
            h = max_row - min_row + 1
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            int_counts.append(self._count_int_tiles_in_bbox(min_col, max_col, min_row, max_row))
        widths = sorted(widths)
        heights = sorted(heights)
        areas = sorted(areas)
        int_counts_sorted = sorted(int_counts)
        n = len(widths)
        return {
            "min_width": widths[0],
            "max_width": widths[-1],
            "median_width": widths[n // 2],
            "mean_width": sum(widths) / n,
            "min_height": heights[0],
            "max_height": heights[-1],
            "median_height": heights[n // 2],
            "mean_height": sum(heights) / n,
            "min_area": areas[0],
            "max_area": areas[-1],
            "median_area": areas[n // 2],
            "mean_area": sum(areas) / n,
            "p10_area": areas[int(n * 0.1)] if n >= 10 else areas[0],
            "p90_area": areas[int(n * 0.9)] if n >= 10 else areas[-1],
            "min_int_tiles": int_counts_sorted[0],
            "max_int_tiles": int_counts_sorted[-1],
            "median_int_tiles": int_counts_sorted[n // 2],
            "mean_int_tiles": sum(int_counts) / n,
            "p10_int_tiles": int_counts_sorted[int(n * 0.1)] if n >= 10 else int_counts_sorted[0],
            "p90_int_tiles": int_counts_sorted[int(n * 0.9)] if n >= 10 else int_counts_sorted[-1],
        }

    def print_bbox_size_distribution(self) -> None:
        stats = self.get_bbox_size_distribution()
        if not stats:
            print("No nets to report bbox distribution.")
            return
        print("Bounding box size distribution:")
        print(
            f"  Width:      min={stats['min_width']} max={stats['max_width']} "
            f"median={stats['median_width']:.0f} mean={stats['mean_width']:.1f}"
        )
        print(
            f"  Height:     min={stats['min_height']} max={stats['max_height']} "
            f"median={stats['median_height']:.0f} mean={stats['mean_height']:.1f}"
        )
        print(
            f"  Area:       min={stats['min_area']} max={stats['max_area']} "
            f"median={stats['median_area']:.0f} mean={stats['mean_area']:.1f} "
            f"p10={stats['p10_area']} p90={stats['p90_area']}"
        )
        print(
            f"  INT tiles:  min={stats['min_int_tiles']} max={stats['max_int_tiles']} "
            f"median={stats['median_int_tiles']:.0f} mean={stats['mean_int_tiles']:.1f} "
            f"p10={stats['p10_int_tiles']} p90={stats['p90_int_tiles']}"
        )

    def init_variables(self) -> torch.Tensor:
        offsets = torch.tensor(self._var_offset, dtype=torch.long)
        sizes = offsets[1:] - offsets[:-1]
        nonzero = sizes.clamp_min(1)
        x = torch.repeat_interleave(0.1 / nonzero.to(torch.float32), sizes).to(self.device)
        return x.requires_grad_(True)

    def wirelength_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_vars == 0:
            return torch.tensor(0.0, device=self.device, dtype=x.dtype)
        return (x * self._flat_wl).sum()

    def discretization_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Penalize fractional x: sum x*(1-x), minimized at x in {0,1}.

        Drives the relaxed solution to commit to discrete paths instead of
        spreading flow diffusely across many weak edges (which stays congested
        and is not cleanly routable). Typically annealed up over iterations.
        """
        if self.num_vars == 0:
            return torch.tensor(0.0, device=self.device, dtype=x.dtype)
        return (x * (1.0 - x)).sum()

    def flow_conservation_loss(
        self,
        x: torch.Tensor,
        net_batch: int = 0,
    ) -> torch.Tensor:
        if self.edge_mode == "undirected":
            return torch.tensor(0.0, device=self.device, dtype=x.dtype)

        if net_batch > 0 and net_batch < self.num_nets:
            return self._flow_conservation_loss_sampled(x, net_batch)

        if self._total_local_nodes == 0:
            return torch.tensor(0.0, device=self.device, dtype=x.dtype)
        flow = torch.zeros(self._total_local_nodes, device=self.device, dtype=x.dtype)
        flow = flow.index_add(0, self._flat_v, x).index_add(0, self._flat_u, -x)
        imbalance = flow - self._flow_demand.to(x.dtype)
        return (imbalance * imbalance).sum()

    def _flow_conservation_loss_sampled(self, x: torch.Tensor, net_batch: int) -> torch.Tensor:
        """Legacy per-net subsampled flow loss (net_batch > 0)."""
        loss = torch.tensor(0.0, device=self.device, dtype=x.dtype)
        net_indices = torch.randperm(self.num_nets, device=self.device)[:net_batch].tolist()
        for i in net_indices:
            start, end = self._var_offset[i], self._var_offset[i + 1]
            loss = loss + flow_conservation_loss_for_net(
                x[start:end],
                self.net_edge_indices[i],
                self.rrg.directed_edges,
                self.net_src_tile[i],
                self.net_sink_tiles[i],
                self.rrg.num_tiles,
                self.device,
            )
        return loss * (self.num_nets / net_batch)

    def _get_usage_and_overflows(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        usage = torch.zeros(self._num_global_edges, device=self.device, dtype=x.dtype)
        if self.num_vars > 0:
            usage = usage.scatter_add(0, self._flat_edge_idx, x)

        if self.edge_mode == "undirected":
            overflows = torch.relu(usage - self._phys_capacity_tensor)
            return usage, overflows

        num_phys = len(self.rrg.phys_list)
        phys_usage = torch.zeros(num_phys, device=self.device, dtype=x.dtype)
        phys_usage = phys_usage.index_add(0, self._d2p, usage)
        overflows = torch.relu(phys_usage - self._phys_capacity_tensor)
        return usage, overflows

    def congestion_loss(self, x: torch.Tensor, penalty: str = "soft") -> torch.Tensor:
        _, overflows = self._get_usage_and_overflows(x)
        return overflows.sum()

    def connectivity_loss_effective_resistance(
        self,
        x: torch.Tensor,
        eps: float = 1e-6,
        solver: str = "cg",
        net_batch: int = 0,
    ) -> torch.Tensor:
        return self._connectivity_grouped_or_loop(x, solver, eps, net_batch)

    def _build_supersink_conn(self) -> Optional[dict]:
        """B1: one connectivity column per net (source -> merged super-sink).

        Column RHS = {source: +1, each sink: -1/k}. Collapses num_cols from ~5/net
        to 1/net (~5x fewer CG columns/groups), still pulling the whole net toward
        its sinks. Uses the grouped solver's general-RHS path.
        """
        if self._conn is None:
            return None
        dev = self.device
        conn = self._conn
        no = self._grouped_no
        src = conn["src_flat"].long()
        sink = conn["sink_flat"].long()
        col_net = torch.searchsorted(no, src, right=True) - 1
        _nets, counts = torch.unique_consecutive(col_net, return_counts=True)
        num_super = int(_nets.numel())
        offsets = torch.cumsum(counts, 0) - counts       # first orig col per net
        src_super = src[offsets]                          # source node per super-col
        super_col_of_orig = torch.repeat_interleave(
            torch.arange(num_super, device=dev), counts)
        sink_val = (-1.0 / counts.to(torch.float32))[super_col_of_orig]
        rhs_node = torch.cat([src_super, sink])
        rhs_col = torch.cat([torch.arange(num_super, device=dev), super_col_of_orig])
        rhs_val = torch.cat([torch.ones(num_super, device=dev), sink_val])
        order = torch.argsort(rhs_col, stable=True)
        rhs_node, rhs_col, rhs_val = rhs_node[order], rhs_col[order], rhs_val[order]
        rhs_off = torch.zeros(num_super + 1, dtype=torch.long, device=dev)
        rhs_off[1:] = torch.cumsum(1 + counts, 0)
        return {
            "flat_u": conn["flat_u"], "flat_v": conn["flat_v"],
            "src_flat": src_super, "sink_flat": src_super,  # sink_flat unused (general)
            "col_id": torch.arange(num_super, device=dev),
            "num_nodes": conn["num_nodes"], "num_cols": num_super,
            "rhs_node": rhs_node, "rhs_col": rhs_col,
            "rhs_val": rhs_val, "rhs_off": rhs_off,
        }

    def _grouped_conn(self) -> Optional[dict]:
        """_conn, optionally super-sink (B1) or capping columns to conn_max_sinks."""
        if self._conn is None:
            return None
        if getattr(self, "conn_super_sink", False):
            if getattr(self, "_conn_super", None) is None:
                self._conn_super = self._build_supersink_conn()
            return self._conn_super
        cap = int(getattr(self, "conn_max_sinks", 0) or 0)
        if cap <= 0:
            return self._conn
        if getattr(self, "_conn_capped_k", None) == cap:
            return self._conn_capped
        dev = self.device
        no = self._grouped_no
        src = self._conn["src_flat"].long()
        col_net = torch.searchsorted(no, src, right=True) - 1
        net_ncol = torch.bincount(col_net, minlength=self.num_nets)
        net_c_off = torch.zeros(self.num_nets + 1, dtype=torch.long, device=dev)
        net_c_off[1:] = torch.cumsum(net_ncol, 0)
        rank = torch.arange(src.numel(), device=dev) - net_c_off[col_net]
        m = (rank < cap).nonzero(as_tuple=True)[0]
        self._conn_capped = {
            "flat_u": self._conn["flat_u"],
            "flat_v": self._conn["flat_v"],
            "src_flat": self._conn["src_flat"][m],
            "sink_flat": self._conn["sink_flat"][m],
            "col_id": torch.arange(m.numel(), device=dev),
            "num_nodes": self._conn["num_nodes"],
            "num_cols": int(m.numel()),
        }
        self._conn_capped_k = cap
        return self._conn_capped

    def _connectivity_grouped_or_loop(
        self, x: torch.Tensor, solver: str, eps: float, net_batch: int
    ) -> torch.Tensor:
        if solver == "grouped":
            from src.router.connectivity_grouped import effective_resistance_loss_grouped
            if getattr(self, "_grouped_vo", None) is None:
                self._grouped_vo = torch.tensor(
                    self._var_offset, dtype=torch.long, device=self.device)
                self._grouped_no = torch.tensor(
                    self._node_offset, dtype=torch.long, device=self.device)
                self._grouped_gcache = {}
                # warm-start cache persists across AL iterations (enable via conn_warm_start)
                self._grouped_ws = {} if getattr(self, "conn_warm_start", True) else None
            conn = self._grouped_conn()
            return effective_resistance_loss_grouped(
                x,
                conn,
                self._grouped_vo,
                self._grouped_no,
                eps=eps,
                cg_max_iter=getattr(self, "conn_cg_max_iter", 100),
                cg_tol=getattr(self, "conn_cg_tol", 1e-5),
                col_chunk=getattr(self, "conn_col_chunk", 128),
                precond=getattr(self, "conn_precond", "none"),
                ws_cache=self._grouped_ws,
                _mg_devices=getattr(self, "conn_mg_devices", None),
                _group_cache=self._grouped_gcache,
            )

        if solver == "cg":
            return effective_resistance_loss_batched(
                x,
                self._conn,
                eps=eps,
                cg_max_iter=getattr(self, "conn_cg_max_iter", 100),
                cg_tol=getattr(self, "conn_cg_tol", 1e-5),
                col_chunk=getattr(self, "conn_col_chunk", 8),
                edge_chunk=getattr(self, "conn_edge_chunk", 0),
            )

        loss = torch.tensor(0.0, device=self.device, dtype=x.dtype)
        net_indices = range(self.num_nets)
        if net_batch > 0 and net_batch < self.num_nets:
            net_indices = torch.randperm(self.num_nets, device=self.device)[:net_batch].tolist()

        for i in net_indices:
            start, end = self._var_offset[i], self._var_offset[i + 1]
            if self.edge_mode == "undirected":
                loss = loss + effective_resistance_loss_for_net_undirected(
                    x[start:end],
                    self.net_edge_indices[i],
                    self.rrg.phys_endpoints,
                    self.net_src_tile[i],
                    self.net_sink_tiles[i],
                    self.device,
                    eps=eps,
                    solver=solver,
                )
            else:
                loss = loss + effective_resistance_loss_for_net(
                    x[start:end],
                    self.net_edge_indices[i],
                    self.rrg.directed_edges,
                    self.net_src_tile[i],
                    self.net_sink_tiles[i],
                    self.device,
                    eps=eps,
                    solver=solver,
                )
        if net_batch > 0 and net_batch < self.num_nets:
            loss = loss * (self.num_nets / net_batch)
        return loss

    def total_loss(
        self,
        x: torch.Tensor,
        w_wl: float = 1.0,
        w_cong: float = 1.0,
        w_conn: float = 1.0,
        w_flow: float = 1.0,
        connectivity: str = "effective_resistance",
        connectivity_solver: str = "cg",
        conn_net_batch: int = 0,
        flow_net_batch: int = 0,
    ) -> torch.Tensor:
        conn = self.connectivity_loss_effective_resistance(
            x, solver=connectivity_solver, net_batch=conn_net_batch
        )
        flow = self.flow_conservation_loss(x, net_batch=flow_net_batch)
        return (
            w_wl * self.wirelength_loss(x)
            + w_cong * self.congestion_loss(x)
            + w_conn * conn
            + w_flow * flow
        )

    def augmented_lagrangian(
        self,
        x: torch.Tensor,
        lam: torch.Tensor,
        rho: float,
        w_wl: float = 1.0,
        w_conn: float = 1.0,
        w_flow: float = 1.0,
        w_disc: float = 0.0,
        connectivity: str = "effective_resistance",
        connectivity_solver: str = "cg",
        conn_net_batch: int = 0,
        flow_net_batch: int = 0,
    ) -> torch.Tensor:
        wl = self.wirelength_loss(x)
        flow = self.flow_conservation_loss(x, net_batch=flow_net_batch)
        _, overflows = self._get_usage_and_overflows(x)
        penalty_linear = (lam * overflows).sum()
        penalty_quad = (rho / 2) * (overflows ** 2).sum()
        total = w_wl * wl + w_flow * flow + penalty_linear + penalty_quad
        # Skip the expensive connectivity CG entirely when its weight is 0 (A1/A2:
        # connectivity-every-K-iters and freeze-after-convergence set w_conn=0).
        if w_conn != 0.0:
            conn = self.connectivity_loss_effective_resistance(
                x, solver=connectivity_solver, net_batch=conn_net_batch
            )
            total = total + w_conn * conn
        if w_disc != 0.0:
            total = total + w_disc * self.discretization_loss(x)
        return total

    def optimize_augmented_lagrangian(self, **kwargs) -> torch.Tensor:
        from src.router.augmented_lagrangian import optimize_augmented_lagrangian

        return optimize_augmented_lagrangian(self, **kwargs)

    def get_congestion_map(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            usage, _ = self._get_usage_and_overflows(x)
            if self.edge_mode == "undirected":
                phys_usage = usage
            else:
                num_phys = len(self.rrg.phys_list)
                phys_usage = torch.zeros(num_phys, device=self.device, dtype=usage.dtype)
                phys_usage = phys_usage.index_add(0, self._d2p, usage)
            caps = self._phys_capacity_tensor.clamp_min(1e-12)
            cong = (phys_usage / caps).cpu().numpy()

        endpoints = np.asarray(self.rrg.phys_list, dtype=np.int64).reshape(-1, 2)
        tile_cong = np.zeros(self.rrg.num_tiles, dtype=np.float32)
        np.maximum.at(tile_cong, endpoints[:, 0], cong)
        np.maximum.at(tile_cong, endpoints[:, 1], cong)

        grid = np.zeros((self.device_rows, self.device_cols), dtype=np.float32)
        for idx, (row, col, *_) in enumerate(self.rrg.tiles):
            if 0 <= row < self.device_rows and 0 <= col < self.device_cols:
                grid[row, col] = tile_cong[idx]
        return grid

    def print_info(self) -> None:
        mode = self.edge_mode
        print(
            f"GlobalRouter: {self.num_nets} nets, edge_mode={mode}, "
            f"{len(self.rrg.phys_list)} phys edges"
        )
        if mode == "directed":
            print(f"  Directed edges: {self.rrg.num_directed_edges}")
        print(f"  Total variables: {self.num_vars}")
        print(f"  RRG tiles: {self.rrg.num_tiles}")
