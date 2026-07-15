#!/usr/bin/env python3
"""GPU batched shortest-path guide export (all nets at once).

The per-net Dijkstra extractor spends ~54 ms/net in Python (heapq + dict adjacency);
even at 64 processes that is ~4 min for 274K nets. But every net's subgraph occupies a
*disjoint block* of the flattened local-node space (`_flat_u`/`_flat_v`), so a single
`scatter_reduce(amin)` relaxes ALL nets' edges simultaneously -- the same trick
eval_connectivity.py uses to label-propagate every net in ~1 s.

So: Bellman-Ford (min-plus relaxation) over the whole flat edge array, iterated until
no distance changes (routes are short, so this converges in tens of rounds), then a
vectorised backward walk from every sink at once to reconstruct paths.

Edge cost = 1/(x + eps): high-flow edges are cheap, so paths follow the global route.
No thresholding -- low-x nets still get a sensible path (the CPU extractor needed a
separate unit-weight fallback for those).
"""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch


def batched_sssp(router, x, dev, eps=1e-6, max_rounds=400, verbose=True):
    """Return (dist, pred, local2global, src_local_per_net, sinks_local, sink_net)."""
    conn = router._conn
    fu = conn["flat_u"].long().to(dev)
    fv = conn["flat_v"].long().to(dev)
    n_nodes = conn["num_nodes"]

    # local -> global tile id: every local node is an endpoint of some edge
    de = torch.tensor(router.rrg.directed_edges, dtype=torch.long, device=dev)
    guv = de[router._flat_edge_idx.long().to(dev)]          # [num_vars, 2] global tiles
    local2global = torch.zeros(n_nodes, dtype=torch.long, device=dev)
    local2global[fu] = guv[:, 0]
    local2global[fv] = guv[:, 1]

    # per-net source / sinks (columns are (net, sink) pairs)
    node_off = torch.tensor(router._node_offset, dtype=torch.long, device=dev)
    src_col = conn["src_flat"].long().to(dev)
    sink_col = conn["sink_flat"].long().to(dev)
    col_net = torch.searchsorted(node_off, src_col, right=True) - 1

    w = x.to(dev).clamp_min(0)
    cost = 1.0 / (w + 1e-3)          # cheap where the global route put flow

    INF = torch.tensor(float("inf"), device=dev)
    dist = torch.full((n_nodes,), float("inf"), device=dev)
    dist[src_col] = 0.0              # all nets' sources at once

    t = time.time()
    for r in range(max_rounds):
        # relax both directions (the extractor treats corridor edges as undirected)
        cand_v = dist[fu] + cost
        cand_u = dist[fv] + cost
        new = dist.clone()
        new.scatter_reduce_(0, fv, cand_v, reduce="amin")
        new.scatter_reduce_(0, fu, cand_u, reduce="amin")
        if torch.equal(new, dist):
            break
        dist = new
    if verbose:
        print(f"[sssp] {r+1} relaxation rounds in {time.time()-t:.1f}s", flush=True)

    # predecessor: for edge (u,v), u is a valid pred of v iff dist[u]+cost == dist[v].
    # pick the smallest such u (deterministic) via scatter_reduce amin.
    t = time.time()
    pred = torch.full((n_nodes,), -1, dtype=torch.long, device=dev)
    big = torch.full((n_nodes,), n_nodes, dtype=torch.long, device=dev)
    tol = 1e-6
    ok_v = (dist[fu] + cost - dist[fv]).abs() <= tol * dist[fv].clamp_min(1.0)
    ok_u = (dist[fv] + cost - dist[fu]).abs() <= tol * dist[fu].clamp_min(1.0)
    big.scatter_reduce_(0, fv[ok_v], fu[ok_v], reduce="amin")
    big.scatter_reduce_(0, fu[ok_u], fv[ok_u], reduce="amin")
    pred = torch.where(big < n_nodes, big, pred)
    pred[src_col] = -1               # sources terminate the walk
    if verbose:
        print(f"[pred] {time.time()-t:.1f}s", flush=True)
    return dist, pred, local2global, src_col, sink_col, col_net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--testcase", default="boom_soc_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-walk", type=int, default=4000)
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    dev = torch.device(f"cuda:{args.gpu}")
    ni = default_net_index_path("./data/", args.testcase, args.rrg, "directed", 0, 0.1,
                                route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t0 = time.time()
    router = GlobalRouter.load(args.rrg, ni, device=dev, edge_mode="directed", verbose=False)
    print(f"[load] {time.time()-t0:.1f}s nets={router.num_nets}", flush=True)
    obj = torch.load(args.x, map_location=dev)
    x = (obj["x"] if isinstance(obj, dict) else obj).to(dev)

    t_sp = time.time()
    dist, pred, l2g, src_col, sink_col, col_net = batched_sssp(router, x, dev)

    # backward walk from EVERY sink at once: cur[i] steps toward its source each round.
    t = time.time()
    cur = sink_col.clone()
    alive = torch.ones_like(cur, dtype=torch.bool)
    rows = [cur.clone()]
    for _ in range(args.max_walk):
        nxt = torch.where(alive, pred[cur], torch.full_like(cur, -1))
        alive = alive & (nxt >= 0)
        if not bool(alive.any()):
            break
        cur = torch.where(alive, nxt, cur)
        rows.append(torch.where(alive, cur, torch.full_like(cur, -1)))
    walk = torch.stack(rows, 0)                     # [steps, num_cols], -1 = done
    print(f"[walk] {walk.shape[0]} steps in {time.time()-t:.1f}s", flush=True)

    # (net, tile) pairs from the walk, deduped
    t = time.time()
    steps, ncols = walk.shape
    net_of_col = col_net.unsqueeze(0).expand(steps, ncols)
    valid = walk >= 0
    nets_f = net_of_col[valid]
    tiles_f = l2g[walk[valid].clamp_min(0)]
    # every net also includes its source tile
    nets_f = torch.cat([nets_f, col_net])
    tiles_f = torch.cat([tiles_f, l2g[src_col]])
    ntiles = router.rrg.num_tiles
    key = torch.unique(nets_f * ntiles + tiles_f)
    nets_u = (key // ntiles).cpu().numpy()
    tiles_u = (key % ntiles).cpu().numpy()
    print(f"[collect] {time.time()-t:.1f}s  ({len(nets_u)} (net,tile) pairs)", flush=True)
    print(f"[GPU shortest-path total] {time.time()-t_sp:.1f}s", flush=True)

    t = time.time()
    import numpy as np
    tiles_xy = router.rrg.tiles
    names = router.net_names
    bounds = np.searchsorted(nets_u, np.arange(router.num_nets + 1))
    n_written = 0
    tot = 0
    with open(args.out, "w") as f:
        for i in range(router.num_nets):
            a, b = bounds[i], bounds[i + 1]
            if b <= a:
                continue
            nm = names[i] if i < len(names) else str(i)
            rc = [(tiles_xy[t2][0], tiles_xy[t2][1]) for t2 in tiles_u[a:b]]
            f.write(f"{nm} {len(rc)} " + " ".join(f"{r},{c}" for r, c in rc) + "\n")
            n_written += 1
            tot += len(rc)
    print(f"[write] {time.time()-t:.1f}s -> {args.out}", flush=True)
    print(f"[guide] {n_written} nets, {tot} tiles ({tot/max(1,n_written):.1f} tiles/net)", flush=True)
    print(f"[TOTAL] {time.time()-t0:.1f}s", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
