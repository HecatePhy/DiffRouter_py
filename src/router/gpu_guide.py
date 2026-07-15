"""GPU batched shortest-path route guides (Potter GR guidance).

Every net's subgraph is a disjoint block of the flattened local-node space
(`_flat_u`/`_flat_v`), so one `scatter_reduce(amin)` relaxes ALL nets' edges at
once. Bellman-Ford to convergence (routes are short -> tens of rounds), then a
vectorised backward walk reconstructs every path. ~100x faster than per-net
Dijkstra, with the same paths.

Edge cost = 1/(x + eps) with eps=1e-6 (same scale as the CPU Dijkstra extractor):
cheap where the global route put flow, so paths follow the relaxed solution. No thresholding, so low-flow nets still get a sensible
path (the CPU extractor needed a separate unit-weight fallback for those).
"""

import time
from typing import Optional, Tuple

import torch


def compute_guide_paths(
    router,
    x: torch.Tensor,
    eps: float = 1e-6,   # match the CPU Dijkstra extractor's cost scale
    max_rounds: int = 400,
    max_walk: int = 4000,
    verbose: bool = True,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Return (net_ids, tile_ids): deduped (net, global tile) pairs, sorted by net."""
    conn = router._conn
    if conn is None or conn.get("num_cols", 0) == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    dev = x.device
    fu = conn["flat_u"].long().to(dev)
    fv = conn["flat_v"].long().to(dev)
    n_nodes = int(conn["num_nodes"])

    # local -> global tile id (every local node is an endpoint of some edge)
    de = torch.tensor(router.rrg.directed_edges, dtype=torch.long, device=dev)
    guv = de[router._flat_edge_idx.long().to(dev)]
    l2g = torch.zeros(n_nodes, dtype=torch.long, device=dev)
    l2g[fu] = guv[:, 0]
    l2g[fv] = guv[:, 1]

    node_off = torch.tensor(router._node_offset, dtype=torch.long, device=dev)
    src_col = conn["src_flat"].long().to(dev)
    sink_col = conn["sink_flat"].long().to(dev)
    col_net = torch.searchsorted(node_off, src_col, right=True) - 1

    cost = 1.0 / (x.to(dev).clamp_min(0) + eps)

    t = time.time()
    dist = torch.full((n_nodes,), float("inf"), device=dev)
    dist[src_col] = 0.0
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        new = dist.clone()
        new.scatter_reduce_(0, fv, dist[fu] + cost, reduce="amin")
        new.scatter_reduce_(0, fu, dist[fv] + cost, reduce="amin")
        if torch.equal(new, dist):
            break
        dist = new
    if verbose:
        print(f"    [guide] SSSP {rounds} rounds in {time.time()-t:.1f}s", flush=True)

    # predecessor: smallest u with dist[u] + cost == dist[v]
    tol = 1e-6
    big = torch.full((n_nodes,), n_nodes, dtype=torch.long, device=dev)
    ok_v = (dist[fu] + cost - dist[fv]).abs() <= tol * dist[fv].clamp_min(1.0)
    ok_u = (dist[fv] + cost - dist[fu]).abs() <= tol * dist[fu].clamp_min(1.0)
    big.scatter_reduce_(0, fv[ok_v], fu[ok_v], reduce="amin")
    big.scatter_reduce_(0, fu[ok_u], fv[ok_u], reduce="amin")
    pred = torch.where(big < n_nodes, big, torch.full_like(big, -1))
    pred[src_col] = -1

    # backward walk from every sink at once
    cur = sink_col.clone()
    alive = torch.ones_like(cur, dtype=torch.bool)
    rows = [cur.clone()]
    for _ in range(max_walk):
        nxt = torch.where(alive, pred[cur], torch.full_like(cur, -1))
        alive = alive & (nxt >= 0)
        if not bool(alive.any()):
            break
        cur = torch.where(alive, nxt, cur)
        rows.append(torch.where(alive, cur, torch.full_like(cur, -1)))
    walk = torch.stack(rows, 0)

    steps, ncols = walk.shape
    valid = walk >= 0
    nets_f = col_net.unsqueeze(0).expand(steps, ncols)[valid]
    tiles_f = l2g[walk[valid].clamp_min(0)]
    nets_f = torch.cat([nets_f, col_net])          # include each net's source tile
    tiles_f = torch.cat([tiles_f, l2g[src_col]])
    ntiles = router.rrg.num_tiles
    key = torch.unique(nets_f * ntiles + tiles_f)
    return (key // ntiles), (key % ntiles)


def write_guide(router, net_ids, tile_ids, out_path: str, verbose: bool = True) -> int:
    """Write the Potter guide file: `<net_name> <ntiles> row,col row,col ...`."""
    import numpy as np

    nets_u = net_ids.cpu().numpy()
    tiles_u = tile_ids.cpu().numpy()
    tiles_xy = router.rrg.tiles
    names = router.net_names
    bounds = np.searchsorted(nets_u, np.arange(router.num_nets + 1))
    n_written = 0
    tot = 0
    with open(out_path, "w") as f:
        for i in range(router.num_nets):
            a, b = bounds[i], bounds[i + 1]
            if b <= a:
                continue
            nm = names[i] if i < len(names) else str(i)
            rc = [(tiles_xy[t][0], tiles_xy[t][1]) for t in tiles_u[a:b]]
            f.write(f"{nm} {len(rc)} " + " ".join(f"{r},{c}" for r, c in rc) + "\n")
            n_written += 1
            tot += len(rc)
    if verbose:
        print(f"    [guide] {n_written} nets, {tot} tiles "
              f"({tot/max(1,n_written):.1f} tiles/net) -> {out_path}", flush=True)
    return n_written


def export_guide(router, x: torch.Tensor, out_path: str, verbose: bool = True) -> int:
    """Compute + write the guide from an already-loaded router (no reload)."""
    t = time.time()
    net_ids, tile_ids = compute_guide_paths(router, x, verbose=verbose)
    n = write_guide(router, net_ids, tile_ids, out_path, verbose=verbose)
    if verbose:
        print(f"    [guide] total {time.time()-t:.1f}s", flush=True)
    return n
