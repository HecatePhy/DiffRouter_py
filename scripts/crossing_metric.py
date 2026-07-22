#!/usr/bin/env python3
"""Did the optimizer actually REROUTE, or just rescale the initial paths?

The whole-session finding was Jaccard 1.0: the x>0.5 support after optimization was
identical to the shortest-path init's, in every config -- the optimizer never moved a
net. This script measures, against that same init, whether a run finally produced
movement:

  support Jaccard   : |final_support ∩ init_support| / |union|   (1.0 = never moved)
  edges added/dropped: entered/left the x>0.5 support
  crossings          : per net, init-path edges whose final x fell BELOW a non-init
                       edge of the same net -- i.e. an alternative overtook the incumbent.
                       This is the differentiable analogue of a reroute decision.
  extracted Jaccard  : same, but on the actually-extracted shortest paths (what the guide
                       encodes) rather than the raw x>0.5 mask.

Usage: crossing_metric.py --x run/global_x.pt [--x2 other/global_x.pt ...] [--gpu 0]
"""
import argparse
import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.chdir(_root)

import torch


def extracted_support(router, x, dev, eps=1e-6):
    """The x>0.5-independent 'what gets extracted' support: aligned tree edges of the
    min-cost (cost=wl/(x+eps)) shortest paths -- exactly what the guide walk encodes."""
    conn = router._conn
    fu = conn["flat_u"].long().to(dev)
    fv = conn["flat_v"].long().to(dev)
    n_nodes = int(conn["num_nodes"])
    src = conn["src_flat"].long().to(dev)
    sink = conn["sink_flat"].long().to(dev)
    xd = x.to(dev).clamp_min(0)
    wl = getattr(router, "_flat_wl", None)
    cost = (wl.to(dev, xd.dtype) / (xd + eps)) if wl is not None else 1.0 / (xd + eps)
    dist = torch.full((n_nodes,), float("inf"), device=dev)
    dist[src] = 0.0
    for _ in range(400):
        new = dist.clone()
        new.scatter_reduce_(0, fv, dist[fu] + cost, reduce="amin")
        new.scatter_reduce_(0, fu, dist[fv] + cost, reduce="amin")
        if torch.equal(new, dist):
            break
        dist = new
    tol = 1e-6
    big = torch.full((n_nodes,), n_nodes, dtype=torch.long, device=dev)
    okv = (dist[fu] + cost - dist[fv]).abs() <= tol * dist[fv].clamp_min(1.0)
    oku = (dist[fv] + cost - dist[fu]).abs() <= tol * dist[fu].clamp_min(1.0)
    big.scatter_reduce_(0, fv[okv], fu[okv], reduce="amin")
    big.scatter_reduce_(0, fu[oku], fv[oku], reduce="amin")
    pred = torch.where(big < n_nodes, big, torch.full_like(big, -1))
    pred[src] = -1
    visited = torch.zeros(n_nodes, dtype=torch.bool, device=dev)
    visited[sink] = True
    cur = sink.clone()
    alive = torch.ones_like(cur, dtype=torch.bool)
    for _ in range(4000):
        nxt = torch.where(alive, pred[cur], torch.full_like(cur, -1))
        alive = alive & (nxt >= 0)
        if not bool(alive.any()):
            break
        cur = torch.where(alive, nxt, cur)
        visited[cur[alive]] = True
    return (pred[fv] == fu) & visited[fv]        # aligned-orientation tree edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testcase", default="boom_soc_v2")
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--x", action="append", required=True, help="global_x.pt (repeatable)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--off-path", type=float, default=0.01)
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    from src.router.gpu_guide import shortest_path_x

    dev = torch.device(f"cuda:{args.gpu}")
    ni = default_net_index_path("./data/", args.testcase, args.rrg, "directed", 0, 0.1,
                                route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t = time.time()
    r = GlobalRouter.load(args.rrg, ni, device=dev, edge_mode="directed", verbose=False)
    print(f"[load] {time.time()-t:.1f}s  nets={r.num_nets} vars={r.num_vars}", flush=True)

    vo = torch.tensor(r._var_offset, dtype=torch.long, device=dev)
    counts = (vo[1:] - vo[:-1]).clamp_min(1)
    var_net = torch.repeat_interleave(torch.arange(len(counts), device=dev), counts)

    init = shortest_path_x(r, dev, on_path=1.0, off_path=args.off_path, verbose=False) > 0.5
    init_ext = extracted_support(r, torch.where(init, torch.ones(r.num_vars, device=dev),
                                                torch.full((r.num_vars,), args.off_path, device=dev)), dev)
    print(f"init |x>0.5|={int(init.sum())}  |extracted|={int(init_ext.sum())}\n", flush=True)
    print(f"{'run':34s} {'Jacc(x>.5)':>11s} {'added':>8s} {'dropped':>8s} "
          f"{'crossings':>10s} {'Jacc(extr)':>11s}", flush=True)
    print("-" * 88, flush=True)

    for xp in args.x:
        o = torch.load(xp, map_location=dev)
        x = (o["x"] if isinstance(o, dict) else o).to(dev).float()
        s = x > 0.5
        inter = int((s & init).sum()); uni = int((s | init).sum())
        jac = inter / max(uni, 1)
        added = int((s & ~init).sum()); dropped = int((~s & init).sum())
        # crossings: per net, does any non-init edge exceed the min final-x over that
        # net's init-path edges? (an alternative overtook the weakest incumbent)
        NEG = torch.tensor(float("inf"), device=dev)
        init_x = torch.where(init, x, NEG)
        min_init = torch.full((r.num_nets,), float("inf"), device=dev)
        min_init.scatter_reduce_(0, var_net, init_x, reduce="amin", include_self=True)
        noninit_x = torch.where(~init, x, torch.tensor(-1.0, device=dev))
        max_noninit = torch.full((r.num_nets,), -1.0, device=dev)
        max_noninit.scatter_reduce_(0, var_net, noninit_x, reduce="amax", include_self=True)
        crossings = int(((max_noninit > min_init) & torch.isfinite(min_init)).sum())
        ext = extracted_support(r, x, dev)
        ei = int((ext & init_ext).sum()); eu = int((ext | init_ext).sum())
        jce = ei / max(eu, 1)
        name = os.path.relpath(xp)[-33:]
        print(f"{name:34s} {jac:11.5f} {added:8d} {dropped:8d} {crossings:10d} {jce:11.5f}",
              flush=True)
        del x, o
        torch.cuda.empty_cache()
    print("\ncrossings>0 and Jacc<1 => the optimizer finally MOVED nets. "
          "Jacc(extr)<1 => that movement changes the extracted guide.", flush=True)


if __name__ == "__main__":
    main()
