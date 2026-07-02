"""Extract discrete INT-tile paths from continuous global routing solution."""

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class RouteExtractor:
    """Convert relaxed edge flows x into discrete tile paths per net."""

    def __init__(
        self,
        threshold: float = 0.01,
        eps: float = 1e-6,
    ):
        self.threshold = threshold
        self.eps = eps

    @staticmethod
    def _edge_endpoints(router, edge_idx: int) -> Tuple[int, int]:
        if getattr(router, "edge_mode", "directed") == "undirected":
            return router.rrg.phys_endpoints(edge_idx)
        return router.rrg.directed_edges[edge_idx]

    def extract(
        self,
        router,
        x: torch.Tensor,
        congestion_map: Optional[np.ndarray] = None,
    ) -> Dict[int, List[int]]:
        x_cpu = x.detach().cpu()
        paths: Dict[int, List[int]] = {}
        for net_idx in range(router.num_nets):
            paths[net_idx] = self.extract_one(router, x_cpu, net_idx, congestion_map)
        return paths

    def extract_one(
        self,
        router,
        x: torch.Tensor,
        net_idx: int,
        congestion_map: Optional[np.ndarray] = None,
    ) -> List[int]:
        src = router.net_src_tile[net_idx]
        sinks = router.net_sink_tiles[net_idx]
        start, end = router._var_offset[net_idx], router._var_offset[net_idx + 1]
        edge_list = router.net_edge_indices[net_idx]
        x_slice = x[start:end]

        adj: Dict[int, List[Tuple[int, float, int]]] = {}
        for k, edge_idx in enumerate(edge_list):
            u, v = self._edge_endpoints(router, edge_idx)
            w = float(x_slice[k].item())
            if w < self.threshold:
                continue
            adj.setdefault(u, []).append((v, w, edge_idx))
            adj.setdefault(v, []).append((u, w, edge_idx))

        terminals = {src} | set(sinks)
        path_tiles: Set[int] = {src}
        for sink in sinks:
            segment = self._shortest_path(
                router, adj, src, sink, congestion_map, net_idx
            )
            path_tiles.update(segment)

        if len(path_tiles) < len(terminals):
            for sink in sinks:
                if sink not in path_tiles:
                    fallback = self._dijkstra_fallback(
                        router, net_idx, src, sink, x_slice, edge_list, congestion_map
                    )
                    path_tiles.update(fallback)

        return self._build_tile_sequence(router, adj, src, path_tiles, sinks)

    def _edge_cost(
        self,
        weight: float,
        router,
        tile: int,
        congestion_map: Optional[np.ndarray],
        net_idx: int,
    ) -> float:
        base = 1.0 / (weight + self.eps)
        if congestion_map is not None:
            row, col = router.rrg.tiles[tile][0], router.rrg.tiles[tile][1]
            if 0 <= row < congestion_map.shape[0] and 0 <= col < congestion_map.shape[1]:
                base *= 1.0 + float(congestion_map[row, col])
        return base

    def _shortest_path(
        self,
        router,
        adj: Dict[int, List[Tuple[int, float, int]]],
        src: int,
        dst: int,
        congestion_map: Optional[np.ndarray],
        net_idx: int,
    ) -> List[int]:
        if src == dst:
            return [src]
        dist = {src: 0.0}
        prev: Dict[int, int] = {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, float("inf")):
                continue
            for v, w, _ in adj.get(u, []):
                cost = self._edge_cost(w, router, v, congestion_map, net_idx)
                nd = d + cost
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if dst not in prev and dst != src:
            return self._dijkstra_on_full_bbox(router, net_idx, src, dst, congestion_map)
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _dijkstra_fallback(
        self,
        router,
        net_idx: int,
        src: int,
        dst: int,
        x_slice: torch.Tensor,
        edge_list: List[int],
        congestion_map: Optional[np.ndarray],
    ) -> List[int]:
        adj: Dict[int, List[Tuple[int, float]]] = {}
        for k, de_idx in enumerate(edge_list):
            u, v = self._edge_endpoints(router, de_idx)
            w = float(x_slice[k].item()) + self.eps
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, []).append((u, w))
        return self._dijkstra(router, adj, src, dst, congestion_map, net_idx)

    def _dijkstra_on_full_bbox(
        self,
        router,
        net_idx: int,
        src: int,
        dst: int,
        congestion_map: Optional[np.ndarray],
    ) -> List[int]:
        edge_list = router.net_edge_indices[net_idx]
        adj: Dict[int, List[Tuple[int, float]]] = {}
        for edge_idx in edge_list:
            u, v = self._edge_endpoints(router, edge_idx)
            w = 1.0
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, []).append((u, w))
        return self._dijkstra(router, adj, src, dst, congestion_map, net_idx)

    def _dijkstra(
        self,
        router,
        adj: Dict[int, List[Tuple[int, float]]],
        src: int,
        dst: int,
        congestion_map: Optional[np.ndarray],
        net_idx: int,
    ) -> List[int]:
        dist = {src: 0.0}
        prev: Dict[int, int] = {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, float("inf")):
                continue
            for v, w in adj.get(u, []):
                cost = self._edge_cost(w, router, v, congestion_map, net_idx)
                nd = d + cost
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if dst not in prev and dst != src:
            return [src, dst]
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _build_tile_sequence(
        self,
        router,
        adj: Dict[int, List[Tuple[int, float, int]]],
        src: int,
        tiles: Set[int],
        sinks: List[int],
    ) -> List[int]:
        """Order tiles as a walk from src covering all terminals (greedy Steiner-like)."""
        remaining = set(sinks)
        sequence = [src]
        current = src
        visited = {src}

        while remaining or len(visited) < len(tiles):
            targets = remaining if remaining else tiles - visited
            if not targets:
                break
            best_path = None
            best_len = float("inf")
            for t in targets:
                path = self._shortest_path(router, adj, current, t, None, -1)
                if len(path) < best_len:
                    best_len = len(path)
                    best_path = path
            if not best_path or len(best_path) < 2:
                break
            for tile in best_path[1:]:
                if tile not in visited:
                    sequence.append(tile)
                    visited.add(tile)
            current = best_path[-1]
            if current in remaining:
                remaining.discard(current)

        return sequence

    def paths_to_bbox(
        self,
        router,
        paths: Dict[int, List[int]],
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """Tight bbox (min_col, max_col, min_row, max_row) from tile path."""
        bboxes = {}
        for net_idx, tiles in paths.items():
            if not tiles:
                continue
            rows = [router.rrg.tiles[t][0] for t in tiles]
            cols = [router.rrg.tiles[t][1] for t in tiles]
            bboxes[net_idx] = (min(cols), max(cols), min(rows), max(rows))
        return bboxes
