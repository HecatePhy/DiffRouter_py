"""Extract discrete INT-tile paths from continuous global routing solution."""

import heapq
import os
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
        workers: int = 0,
    ) -> Dict[int, List[int]]:
        """Extract discrete paths for all nets.

        workers=1 forces serial; workers>1 or 0 (auto) runs a fork-based process
        pool -- nets are independent, so this scales near-linearly across cores.
        """
        if workers == 1:
            x_cpu = x.detach().cpu()
            return {
                i: self.extract_one(router, x_cpu, i, congestion_map)
                for i in range(router.num_nets)
            }
        return extract_paths_parallel(
            router, x, threshold=self.threshold, workers=workers,
            congestion_map=congestion_map,
        )

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
        if torch.is_tensor(edge_list):
            edge_list = edge_list.tolist()
        x_slice = x[start:end]

        adj: Dict[int, List[Tuple[int, float, int]]] = {}
        for k, edge_idx in enumerate(edge_list):
            u, v = self._edge_endpoints(router, edge_idx)
            w = float(x_slice[k].item())
            if w < self.threshold:
                continue
            adj.setdefault(u, []).append((v, w, edge_idx))
            adj.setdefault(v, []).append((u, w, edge_idx))

        # Routing tree = union of src->sink shortest paths. Return its tiles in a
        # stable order (src first, then each path), deduplicated. This IS the tree;
        # no need for the old greedy walk that ran a Dijkstra per remaining tile
        # until every above-threshold tile was ordered -- that was O(tiles^2 *
        # Dijkstra) per net and hung on large post-optimization nets.
        ordered: List[int] = []
        seen: Set[int] = set()

        def _add(tiles: List[int]) -> None:
            for t in tiles:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)

        _add([src])
        for sink in sinks:
            segment = self._shortest_path(
                router, adj, src, sink, congestion_map, net_idx
            )
            _add(segment)

        missing = [s for s in sinks if s not in seen]
        for sink in missing:
            fallback = self._dijkstra_fallback(
                router, net_idx, src, sink, x_slice, edge_list, congestion_map
            )
            _add(fallback)

        return ordered

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
        if torch.is_tensor(edge_list):
            edge_list = edge_list.tolist()
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


# --- Parallel extraction (nets are independent) --------------------------------
# Workers share the read-only router/x/extractor via copy-on-write fork, set as
# module globals before the pool is created.
_MP_ROUTER = None
_MP_X = None
_MP_EX = None
_MP_CONG = None


def _mp_extract_range(rng: Tuple[int, int]) -> Dict[int, List[int]]:
    lo, hi = rng
    return {
        i: _MP_EX.extract_one(_MP_ROUTER, _MP_X, i, _MP_CONG)
        for i in range(lo, hi)
    }


def extract_paths_parallel(
    router,
    x: torch.Tensor,
    threshold: float = 0.01,
    workers: int = 0,
    congestion_map: Optional[np.ndarray] = None,
) -> Dict[int, List[int]]:
    """Extract all nets across a fork-based process pool. workers<=0 -> auto."""
    import multiprocessing as mp

    global _MP_ROUTER, _MP_X, _MP_EX, _MP_CONG
    if workers <= 0:
        workers = min(64, os.cpu_count() or 1)
    ex = RouteExtractor(threshold=threshold)
    x_cpu = x.detach().cpu()
    N = router.num_nets
    if workers == 1 or N == 0:
        return {i: ex.extract_one(router, x_cpu, i, congestion_map) for i in range(N)}

    _MP_ROUTER, _MP_X, _MP_EX, _MP_CONG = router, x_cpu, ex, congestion_map
    nchunks = workers * 4
    step = (N + nchunks - 1) // nchunks
    ranges = [(i, min(i + step, N)) for i in range(0, N, step)]
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    paths: Dict[int, List[int]] = {}
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for part in pool.imap_unordered(_mp_extract_range, ranges):
                paths.update(part)
    finally:
        torch.set_num_threads(prev_threads)
        _MP_ROUTER = _MP_X = _MP_EX = _MP_CONG = None
    return paths
