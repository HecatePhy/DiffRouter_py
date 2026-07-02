"""Differentiable global router (GlobalRouter)."""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.rrg.rrg import RRG
from src.router.connectivity import (
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

    @classmethod
    def load(
        cls,
        rrg_path: str,
        net_index_path: str,
        device: torch.device = None,
        edge_mode: str = "directed",
        verbose: bool = True,
    ) -> "GlobalRouter":
        """Fast path: load pre-built RRG + net index (no Java design)."""
        from src.load_design import load_rrg_fast
        from src.router.net_index import NetIndex

        if edge_mode not in ("directed", "undirected"):
            raise ValueError(f"edge_mode must be 'directed' or 'undirected', got {edge_mode!r}")

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
        if self.edge_mode == "directed":
            self._phys_de_indices_gpu = [
                t.to(self.device) for t in self.rrg._phys_de_indices
            ]
        else:
            self._phys_de_indices_gpu = None
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
        self._net_edge_tensors = []
        for edge_list in self.net_edge_indices:
            self._net_edge_tensors.append(
                torch.tensor(edge_list, device=self.device, dtype=torch.long)
            )
        caps = self.rrg._phys_capacity_tensor.to(self.device)
        self._phys_capacity_tensor = caps
        if self.edge_mode == "directed":
            self._phys_de_indices_gpu = [
                t.to(self.device) for t in self.rrg._phys_de_indices
            ]
        else:
            self._phys_de_indices_gpu = None
        self._init_edge_weight_tensors()

    def _init_edge_weight_tensors(self) -> None:
        """Precompute per-edge WL scores and per-net slices."""
        if self.edge_mode == "undirected":
            self._directed_wl = None
            phys_wl = self.rrg._phys_wl_score_tensor.to(self.device)
            self._net_wl_tensors = []
            for edge_list in self.net_edge_indices:
                if not edge_list:
                    self._net_wl_tensors.append(
                        torch.zeros(0, device=self.device, dtype=torch.float32)
                    )
                else:
                    idx = torch.tensor(edge_list, device=self.device, dtype=torch.long)
                    self._net_wl_tensors.append(phys_wl[idx])
            return

        directed_wl = torch.ones(
            self.rrg.num_directed_edges,
            device=self.device,
            dtype=torch.float32,
        )
        for de_idx, phys in enumerate(self.rrg.phys_edge_of_directed):
            directed_wl[de_idx] = float(self.rrg.phys_edge_wl_score.get(phys, 1))
        self._directed_wl = directed_wl
        self._net_wl_tensors: List[torch.Tensor] = []
        for edge_list in self.net_edge_indices:
            if not edge_list:
                self._net_wl_tensors.append(
                    torch.zeros(0, device=self.device, dtype=torch.float32)
                )
            else:
                idx = torch.tensor(edge_list, device=self.device, dtype=torch.long)
                self._net_wl_tensors.append(directed_wl[idx])

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
        x = torch.zeros(self.num_vars, device=self.device, dtype=torch.float32)
        for i in range(self.num_nets):
            start, end = self._var_offset[i], self._var_offset[i + 1]
            x[start:end] = 0.1 / (end - start)
        return x.requires_grad_(True)

    def wirelength_loss(self, x: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device, dtype=x.dtype)
        for i in range(self.num_nets):
            start, end = self._var_offset[i], self._var_offset[i + 1]
            if end <= start:
                continue
            loss = loss + (x[start:end] * self._net_wl_tensors[i]).sum()
        return loss

    def flow_conservation_loss(
        self,
        x: torch.Tensor,
        net_batch: int = 0,
    ) -> torch.Tensor:
        if self.edge_mode == "undirected":
            return torch.tensor(0.0, device=self.device, dtype=x.dtype)

        loss = torch.tensor(0.0, device=self.device, dtype=x.dtype)
        net_indices = range(self.num_nets)
        if net_batch > 0 and net_batch < self.num_nets:
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
        if net_batch > 0 and net_batch < self.num_nets:
            loss = loss * (self.num_nets / net_batch)
        return loss

    def _get_usage_and_overflows(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.edge_mode == "undirected":
            num_phys = len(self.rrg.phys_list)
            usage = torch.zeros(num_phys, device=self.device, dtype=x.dtype)
            for i in range(self.num_nets):
                start, end = self._var_offset[i], self._var_offset[i + 1]
                indices = self._net_edge_tensors[i]
                if indices.numel() == 0:
                    continue
                usage.scatter_add_(0, indices, x[start:end])
            overflows = torch.relu(usage - self._phys_capacity_tensor)
            return usage, overflows

        usage = torch.zeros(
            self.rrg.num_directed_edges,
            device=self.device,
            dtype=x.dtype,
        )
        for i in range(self.num_nets):
            start, end = self._var_offset[i], self._var_offset[i + 1]
            indices = self._net_edge_tensors[i]
            if indices.numel() == 0:
                continue
            usage.scatter_add_(0, indices, x[start:end])

        overflows = []
        for de_idxs, cap in zip(self._phys_de_indices_gpu, self._phys_capacity_tensor):
            total = usage[de_idxs].sum()
            overflows.append(torch.relu(total - cap))
        return usage, torch.stack(overflows) if overflows else torch.zeros(0, device=self.device)

    def congestion_loss(self, x: torch.Tensor, penalty: str = "soft") -> torch.Tensor:
        _, overflows = self._get_usage_and_overflows(x)
        return overflows.sum()

    def connectivity_loss_effective_resistance(
        self,
        x: torch.Tensor,
        eps: float = 1e-6,
        solver: str = "solve",
        net_batch: int = 0,
    ) -> torch.Tensor:
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
        connectivity_solver: str = "solve",
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
        connectivity: str = "effective_resistance",
        connectivity_solver: str = "solve",
        conn_net_batch: int = 0,
        flow_net_batch: int = 0,
    ) -> torch.Tensor:
        wl = self.wirelength_loss(x)
        conn = self.connectivity_loss_effective_resistance(
            x, solver=connectivity_solver, net_batch=conn_net_batch
        )
        flow = self.flow_conservation_loss(x, net_batch=flow_net_batch)
        _, overflows = self._get_usage_and_overflows(x)
        penalty_linear = (lam * overflows).sum()
        penalty_quad = (rho / 2) * (overflows ** 2).sum()
        return w_wl * wl + w_conn * conn + w_flow * flow + penalty_linear + penalty_quad

    def optimize_augmented_lagrangian(self, **kwargs) -> torch.Tensor:
        from src.router.augmented_lagrangian import optimize_augmented_lagrangian

        return optimize_augmented_lagrangian(self, **kwargs)

    def get_congestion_map(self, x: torch.Tensor) -> np.ndarray:
        usage, _ = self._get_usage_and_overflows(x)
        usage_np = usage.detach().cpu().numpy()

        tile_cong = np.zeros(self.rrg.num_tiles, dtype=np.float32)
        if self.edge_mode == "undirected":
            for phys_id, phys in enumerate(self.rrg.phys_list):
                cap = float(self._phys_capacity_tensor[phys_id].item())
                flow = float(usage_np[phys_id])
                cong = flow / cap if cap > 0 else 0.0
                a, b = phys
                tile_cong[a] = max(tile_cong[a], cong)
                tile_cong[b] = max(tile_cong[b], cong)
        else:
            for phys, de_idxs in self.rrg.phys_to_directed.items():
                cap = self.rrg.phys_capacity[phys]
                flow = sum(usage_np[j] for j in de_idxs)
                cong = flow / cap if cap > 0 else 0.0
                a, b = phys
                tile_cong[a] = max(tile_cong[a], cong)
                tile_cong[b] = max(tile_cong[b], cong)

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
