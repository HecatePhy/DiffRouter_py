#!/usr/bin/env python3
"""Fast guide export: per-net above-threshold tile set via GPU scatter (no
Dijkstra extraction). For Potter's SOFT guide the whole corridor is a fine
constraint, so this replaces the ~4-min per-net extraction with ~seconds.
"""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--testcase", default="boom_soc_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="Absolute x threshold (lower bound on kept edges)")
    ap.add_argument("--rel", type=float, default=0.3,
                    help="Per-net relative threshold: keep edges with x >= rel * max_x(net). "
                         "Scale-invariant per net -> tighter corridor AND every net covered "
                         "(each net's strongest edge always passes). 0 disables.")
    ap.add_argument("--gpu", type=int, default=0)
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

    t = time.time()
    de = torch.tensor(router.rrg.directed_edges, dtype=torch.long, device=dev)  # [E,2]
    eid = router._flat_edge_idx.long()
    guv = de[eid]                                             # [num_vars,2] global tiles
    vo = torch.tensor(router._var_offset, dtype=torch.long, device=dev)
    net_of_var = torch.repeat_interleave(torch.arange(router.num_nets, device=dev),
                                         vo[1:] - vo[:-1])
    ntiles = router.rrg.num_tiles
    # Per-net relative threshold: an absolute cutoff both over-includes (diffuse nets
    # keep their whole corridor) and drops nets entirely (nets whose x never clears it).
    # Scaling by each net's own max fixes both.
    m = x >= args.threshold
    if args.rel > 0:
        net_max = torch.zeros(router.num_nets, device=dev, dtype=x.dtype)
        net_max.scatter_reduce_(0, net_of_var, x, reduce="amax")
        # Pure relative: no absolute floor. An absolute cutoff silently drops every net
        # whose x never clears it -- ~35% of nets here, since small nets init at
        # 0.1/|corridor| and the optimizer never pushes them up. Scaling by each net's
        # own max keeps every net that has any positive flow, and keeps only its
        # strongest edges (tight corridor).
        m = (x > 0) & (x >= args.rel * net_max[net_of_var])
    nv = net_of_var[m]
    tv = guv[m]
    net_rep = nv.repeat(2)
    tile_rep = torch.cat([tv[:, 0], tv[:, 1]])
    key = torch.unique(net_rep * ntiles + tile_rep)          # dedup (net,tile)
    nets_u = (key // ntiles).cpu()
    tiles_u = (key % ntiles).cpu()
    print(f"[compute] per-net tiles in {time.time()-t:.1f}s (GPU scatter)", flush=True)

    # group tiles by net (sorted by net since key sorted)
    tiles_xy = router.rrg.tiles
    names = router.net_names
    t = time.time()
    import numpy as np
    nets_np = nets_u.numpy(); tiles_np = tiles_u.numpy()
    # boundaries where net changes
    bounds = np.searchsorted(nets_np, np.arange(router.num_nets + 1))
    with open(args.out, "w") as f:
        for i in range(router.num_nets):
            a, b = bounds[i], bounds[i + 1]
            if b <= a:
                continue
            nm = names[i] if i < len(names) else str(i)
            rc = [(tiles_xy[t2][0], tiles_xy[t2][1]) for t2 in tiles_np[a:b]]
            f.write(f"{nm} {len(rc)} " + " ".join(f"{r},{c}" for r, c in rc) + "\n")
    print(f"[write] {time.time()-t:.1f}s -> {args.out}", flush=True)
    print(f"[TOTAL guide production] {time.time()-t0:.1f}s (vs ~4min Dijkstra extraction)", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
