"""Tile-level routing resource graph (phys + optional directed view)."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class RRG:
    """
    Routing Resource Graph as a PyTorch-friendly sparse structure.

    Physical INT edges are always stored. In directed mode each phys edge is
    duplicated into two directed arcs for Kirchhoff flow; undirected mode skips
    that view entirely.
    """

    def __init__(
        self,
        tile_graph: dict,
        edge_capacities: Optional[Dict[Tuple[int, int], int]] = None,
        device: torch.device = None,
        edge_mode: str = "directed",
    ):
        if edge_mode not in ("directed", "undirected"):
            raise ValueError(f"edge_mode must be 'directed' or 'undirected', got {edge_mode!r}")
        self.device = device or torch.device("cpu")
        self.edge_mode = edge_mode
        self.coord_to_idx = tile_graph["coord_to_idx"]
        self.tiles = tile_graph["tiles"]
        self.num_tiles = len(self.tiles)
        self._build_device_coord_index()
        undir_edges = tile_graph["edges"]
        edge_dist = tile_graph.get("edge_dist", {})
        edge_wl = tile_graph.get("edge_wl_score", {})
        self._build_phys_core(undir_edges, edge_capacities, edge_dist, edge_wl)
        if edge_mode == "directed":
            self._build_directed_view(undir_edges, edge_capacities)
        else:
            self._init_undirected_sentinels()
        self._bbox_edge_cache: Dict[Tuple[int, int, int, int], List[int]] = {}
        self._bbox_phys_cache: Dict[Tuple[int, int, int, int], List[int]] = {}

    @classmethod
    def from_tensors(
        cls,
        data: dict,
        device: torch.device = None,
        edge_mode: str = "directed",
    ) -> "RRG":
        """Fast construct from v2 .pt tensor payload (no Python tile/edge lists)."""
        if edge_mode not in ("directed", "undirected"):
            raise ValueError(f"edge_mode must be 'directed' or 'undirected', got {edge_mode!r}")
        self = cls.__new__(cls)
        self.device = device or torch.device("cpu")
        self.edge_mode = edge_mode

        tile_arr = data["tiles"]
        edge_arr = data["edges"]
        cap_arr = data["edge_capacities"]
        wl_arr = data.get("edge_wl_scores")
        dist_arr = data.get("edge_distances")

        n_tiles = tile_arr.shape[0]
        self.num_tiles = n_tiles
        rows = tile_arr[:, 0].tolist()
        cols = tile_arr[:, 1].tolist()
        is_int = tile_arr[:, 2].tolist()
        self.tiles = [
            (int(rows[i]), int(cols[i]), None, "", bool(is_int[i]))
            for i in range(n_tiles)
        ]
        self._build_device_coord_index()

        self.coord_to_idx: Dict[Tuple[int, int], int] = {}
        if "int_interchange" in data:
            ic = data["int_interchange"]
            for idx in range(n_tiles):
                self.coord_to_idx[(int(ic[idx, 0].item()), int(ic[idx, 1].item()))] = idx
        else:
            for idx in range(n_tiles):
                self.coord_to_idx[(int(rows[idx]), int(cols[idx]))] = idx

        edge_dist: Dict[Tuple[int, int], int] = {}
        edge_wl: Dict[Tuple[int, int], int] = {}
        undir_edges: List[Tuple[int, int]] = []
        edge_capacities: Dict[Tuple[int, int], int] = {}
        ea = edge_arr.cpu().numpy()
        ca = cap_arr.cpu().numpy()
        for i in range(edge_arr.shape[0]):
            a, b = int(ea[i, 0]), int(ea[i, 1])
            phys = (min(a, b), max(a, b))
            undir_edges.append(phys)
            edge_capacities[phys] = int(ca[i])
            if wl_arr is not None:
                edge_wl[phys] = int(wl_arr[i].item())
            elif dist_arr is not None:
                edge_dist[phys] = int(dist_arr[i].item())
            else:
                edge_wl[phys] = 1

        self._build_phys_core(undir_edges, edge_capacities, edge_dist, edge_wl)
        if edge_mode == "directed":
            self._build_directed_view(undir_edges, edge_capacities)
        else:
            self._init_undirected_sentinels()
        self._bbox_edge_cache = {}
        self._bbox_phys_cache = {}
        return self

    def _init_undirected_sentinels(self) -> None:
        self.directed_edges: List[Tuple[int, int]] = []
        self.phys_edge_of_directed: List[Tuple[int, int]] = []
        self.directed_capacity: List[float] = []
        self.num_directed_edges = 0
        self.phys_to_directed: Dict[Tuple[int, int], List[int]] = {}
        self.out_edges: Dict[int, List[Tuple[int, int]]] = {}
        self.phys_id_of_directed: List[int] = []
        self._phys_de_indices: List[torch.Tensor] = []

    def _build_phys_core(
        self,
        undir_edges: List[Tuple[int, int]],
        edge_capacities: Optional[Dict[Tuple[int, int], int]],
        edge_dist: Dict[Tuple[int, int], int],
        edge_wl: Dict[Tuple[int, int], int],
    ) -> None:
        seen: Dict[Tuple[int, int], int] = {}
        phys_list: List[Tuple[int, int]] = []
        caps: List[float] = []
        wl_scores: List[float] = []
        self.phys_capacity: Dict[Tuple[int, int], float] = {}
        self.phys_edge_dist: Dict[Tuple[int, int], int] = {}
        self.phys_edge_wl_score: Dict[Tuple[int, int], int] = {}
        self.phys_adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(self.num_tiles)}

        for idx_a, idx_b in undir_edges:
            phys = (min(idx_a, idx_b), max(idx_a, idx_b))
            if phys in seen:
                continue
            pid = len(phys_list)
            seen[phys] = pid
            phys_list.append(phys)
            cap = float(edge_capacities.get(phys, 1)) if edge_capacities else 1.0
            caps.append(cap)
            wl = int(edge_wl.get(phys, edge_dist.get(phys, 1) if edge_dist else 1))
            wl_scores.append(float(wl))
            self.phys_capacity[phys] = cap
            self.phys_edge_dist[phys] = int(edge_dist.get(phys, 1))
            self.phys_edge_wl_score[phys] = wl
            a, b = phys
            self.phys_adj[a].append((b, pid))
            self.phys_adj[b].append((a, pid))

        self.phys_list = phys_list
        self.phys_edges = phys_list
        self._phys_to_id = {phys: i for i, phys in enumerate(phys_list)}
        self._phys_capacity_tensor = torch.tensor(caps, dtype=torch.float32)
        self._phys_wl_score_tensor = torch.tensor(wl_scores, dtype=torch.float32)

    def _build_directed_view(
        self,
        undir_edges: List[Tuple[int, int]],
        edge_capacities: Optional[Dict[Tuple[int, int], int]],
    ) -> None:
        self.directed_edges = []
        self.phys_edge_of_directed = []
        self.directed_capacity = []
        for idx_a, idx_b in undir_edges:
            phys = (min(idx_a, idx_b), max(idx_a, idx_b))
            cap = float(edge_capacities.get(phys, 1)) if edge_capacities else 1.0
            self.directed_edges.append((idx_a, idx_b))
            self.phys_edge_of_directed.append(phys)
            self.directed_capacity.append(cap)
            self.directed_edges.append((idx_b, idx_a))
            self.phys_edge_of_directed.append(phys)
            self.directed_capacity.append(cap)

        self.num_directed_edges = len(self.directed_edges)
        self.phys_to_directed: Dict[Tuple[int, int], List[int]] = {}
        for de_idx, phys in enumerate(self.phys_edge_of_directed):
            self.phys_to_directed.setdefault(phys, []).append(de_idx)

        self.out_edges = {i: [] for i in range(self.num_tiles)}
        for de_idx, (src, dst) in enumerate(self.directed_edges):
            self.out_edges[src].append((dst, de_idx))

        self.phys_id_of_directed = [
            self._phys_to_id[phys] for phys in self.phys_edge_of_directed
        ]
        self._phys_de_indices = []
        caps = []
        wl_scores = []
        for phys in self.phys_list:
            de_idxs = self.phys_to_directed[phys]
            self._phys_de_indices.append(torch.tensor(de_idxs, dtype=torch.long))
            caps.append(self.phys_capacity[phys])
            wl_scores.append(float(self.phys_edge_wl_score[phys]))
        self._phys_capacity_tensor = torch.tensor(caps, dtype=torch.float32)
        self._phys_wl_score_tensor = torch.tensor(wl_scores, dtype=torch.float32)

    def write_capacity_log(self, log_path: str) -> None:
        with open(log_path, "w") as f:
            f.write(f"# RRG Edge Capacity Log (edge_mode={self.edge_mode})\n")
            f.write(f"# num_tiles: {self.num_tiles}\n")
            f.write(f"# num_directed_edges: {self.num_directed_edges}\n")
            f.write(f"# num_physical_edges: {len(self.phys_edges)}\n\n")
            if self.edge_mode == "directed":
                f.write(
                    "# directed_edge_idx  src_tile  dst_tile  src_row  src_col  "
                    "dst_row  dst_col  phys_edge  capacity  dist  wl_score\n"
                )
                f.write("#" + "-" * 100 + "\n")
                for de_idx, (src, dst) in enumerate(self.directed_edges):
                    phys = self.phys_edge_of_directed[de_idx]
                    cap = self.directed_capacity[de_idx]
                    src_row, src_col = self.tiles[src][0], self.tiles[src][1]
                    dst_row, dst_col = self.tiles[dst][0], self.tiles[dst][1]
                    f.write(
                        f"{de_idx:8d}  {src:6d}  {dst:6d}  {src_row:6d}  {src_col:6d}  "
                        f"{dst_row:6d}  {dst_col:6d}  {phys}  {cap:.0f}  "
                        f"{self.phys_edge_dist.get(phys, 0)}  "
                        f"{self.phys_edge_wl_score.get(phys, 0)}\n"
                    )
            f.write("\n# Physical edge summary (phys_edge -> capacity, dist, wl_score)\n")
            for phys in self.phys_list:
                cap = self.phys_capacity[phys]
                nd = len(self.phys_to_directed.get(phys, []))
                f.write(
                    f"  {phys}  capacity={cap:.0f}  dist={self.phys_edge_dist.get(phys, 0)}  "
                    f"wl={self.phys_edge_wl_score.get(phys, 0)}  "
                    f"directed_edges={nd}\n"
                )
        print(f"RRG capacity log saved to: {log_path}")

    def _build_device_coord_index(self) -> None:
        """Map device grid (row, col) -> tile list index (used for bbox/corridor)."""
        self.device_coord_to_idx: Dict[Tuple[int, int], int] = {}
        for idx, t in enumerate(self.tiles):
            self.device_coord_to_idx[(int(t[0]), int(t[1]))] = idx

    def tile_coords(self, idx: int) -> Tuple[int, int]:
        """Interchange grid (row, col) for INT tile index."""
        t = self.tiles[idx]
        return int(t[0]), int(t[1])

    def get_edges_in_corridor(
        self,
        src_idx: int,
        sink_idxs: List[int],
        half_width: int = 2,
    ) -> List[int]:
        """L-shaped corridors src->each sink (H band at src row + V band at sink col)."""
        if self.edge_mode != "directed":
            raise RuntimeError("get_edges_in_corridor requires edge_mode='directed'")
        sr, sc = self.tile_coords(src_idx)
        seen: set = set()
        result: List[int] = []

        def add_rect(mc0: int, mc1: int, mr0: int, mr1: int) -> None:
            for de in self.get_edges_in_bbox(mc0, mc1, mr0, mr1):
                if de not in seen:
                    seen.add(de)
                    result.append(de)

        for sink_idx in sink_idxs:
            kr, kc = self.tile_coords(int(sink_idx))
            add_rect(min(sc, kc), max(sc, kc), sr - half_width, sr + half_width)
            add_rect(kc - half_width, kc + half_width, min(sr, kr), max(sr, kr))
        return result

    def get_phys_edges_in_corridor(
        self,
        src_idx: int,
        sink_idxs: List[int],
        half_width: int = 2,
    ) -> List[int]:
        """Undirected phys-edge ids in L-shaped corridors."""
        sr, sc = self.tile_coords(src_idx)
        seen: set = set()
        result: List[int] = []

        def add_rect(mc0: int, mc1: int, mr0: int, mr1: int) -> None:
            for pe in self.get_phys_edges_in_bbox(mc0, mc1, mr0, mr1):
                if pe not in seen:
                    seen.add(pe)
                    result.append(pe)

        for sink_idx in sink_idxs:
            kr, kc = self.tile_coords(int(sink_idx))
            add_rect(min(sc, kc), max(sc, kc), sr - half_width, sr + half_width)
            add_rect(kc - half_width, kc + half_width, min(sr, kr), max(sr, kr))
        return sorted(result)

    def _bbox_tile_set(
        self, min_col: int, max_col: int, min_row: int, max_row: int
    ) -> set:
        bbox_tiles = set()
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                idx = self.device_coord_to_idx.get((row, col))
                if idx is not None:
                    bbox_tiles.add(idx)
        return bbox_tiles

    def get_edges_in_bbox(
        self,
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int,
    ) -> List[int]:
        if self.edge_mode != "directed":
            raise RuntimeError("get_edges_in_bbox requires edge_mode='directed'")
        key = (min_col, max_col, min_row, max_row)
        cached = self._bbox_edge_cache.get(key)
        if cached is not None:
            return cached

        bbox_tiles = self._bbox_tile_set(min_col, max_col, min_row, max_row)
        edge_indices = []
        for src in bbox_tiles:
            for dst, de_idx in self.out_edges[src]:
                if dst in bbox_tiles:
                    edge_indices.append(de_idx)

        self._bbox_edge_cache[key] = edge_indices
        return edge_indices

    def phys_endpoints(self, phys_id: int) -> Tuple[int, int]:
        """Return tile indices (u, v) for physical edge phys_id."""
        return self.phys_list[phys_id]

    def get_phys_edges_in_bbox(
        self,
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int,
    ) -> List[int]:
        """Unique physical edge ids with both endpoints inside the bbox."""
        key = (min_col, max_col, min_row, max_row)
        cached = self._bbox_phys_cache.get(key)
        if cached is not None:
            return cached

        bbox_tiles = self._bbox_tile_set(min_col, max_col, min_row, max_row)
        phys_ids = set()
        for tile in bbox_tiles:
            for neighbor, phys_id in self.phys_adj[tile]:
                if neighbor in bbox_tiles:
                    phys_ids.add(phys_id)
        result = sorted(phys_ids)
        self._bbox_phys_cache[key] = result
        return result


DirectedRRG = RRG
