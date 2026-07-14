#!/usr/bin/env python3
"""Test the fixed route extractor: time it and check connected-net count."""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="only extract first N nets (0=all)")
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    from src.router.route_extractor import RouteExtractor

    dev = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
    ni = default_net_index_path("./data/", "boom_soc_v2", args.rrg, "directed", 0, 0.1,
                                route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t0 = time.time()
    router = GlobalRouter.load(args.rrg, ni, device=dev, edge_mode="directed", verbose=True)
    print(f"[load] {time.time()-t0:.1f}s nets={router.num_nets}", flush=True)
    obj = torch.load(args.x, map_location="cpu")
    x = (obj["x"] if isinstance(obj, dict) else obj)

    ex = RouteExtractor(threshold=args.threshold)
    N = router.num_nets if args.limit <= 0 else min(args.limit, router.num_nets)
    t0 = time.time()
    x_cpu = x.detach().cpu()
    paths = {}
    for i in range(N):
        paths[i] = ex.extract_one(router, x_cpu, i, None)
        if (i + 1) % 50000 == 0:
            print(f"  extracted {i+1}/{N}  ({time.time()-t0:.1f}s)", flush=True)
    dt = time.time() - t0
    connected = sum(
        1 for i in range(N)
        if router.net_src_tile[i] in paths[i]
        and all(s in paths[i] for s in router.net_sink_tiles[i])
    )
    avg_tiles = sum(len(p) for p in paths.values()) / max(1, N)
    print(f"[extract] {N} nets in {dt:.1f}s ({1000*dt/N:.2f} ms/net)  "
          f"connected={connected} ({100*connected/N:.2f}%)  avg_path_tiles={avg_tiles:.1f}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
