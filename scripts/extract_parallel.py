#!/usr/bin/env python3
"""Parallel route extraction across CPU cores (nets are independent).

The router + x + extractor are set as module globals BEFORE the process pool is
created, so forked workers share them copy-on-write (extraction is read-only).
Each worker extracts a contiguous net-index range.
"""
import argparse, os, sys, time, json
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch
import multiprocessing as mp

_ROUTER = None
_X = None
_EX = None


def _worker(rng):
    lo, hi = rng
    out = {}
    for i in range(lo, hi):
        out[i] = _EX.extract_one(_ROUTER, _X, i, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--save-paths", default="")
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    from src.router.route_extractor import RouteExtractor

    ni = default_net_index_path("./data/", "boom_soc_v2", args.rrg, "directed", 0, 0.1,
                                route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t = time.time()
    router = GlobalRouter.load(args.rrg, ni, device=torch.device("cpu"),
                               edge_mode="directed", verbose=True)
    print(f"[load] {time.time()-t:.1f}s nets={router.num_nets}", flush=True)
    obj = torch.load(args.x, map_location="cpu")
    x = (obj["x"] if isinstance(obj, dict) else obj).detach().cpu()

    global _ROUTER, _X, _EX
    _ROUTER = router
    _X = x
    _EX = RouteExtractor(threshold=args.threshold)
    N = router.num_nets

    # contiguous ranges, a few per worker for load balancing
    nchunks = args.workers * 4
    step = (N + nchunks - 1) // nchunks
    ranges = [(i, min(i + step, N)) for i in range(0, N, step)]

    torch.set_num_threads(1)  # workers are the parallelism; avoid oversubscription
    t = time.time()
    paths = {}
    with mp.Pool(processes=args.workers) as pool:
        for part in pool.imap_unordered(_worker, ranges):
            paths.update(part)
    dt = time.time() - t

    connected = sum(
        1 for i in range(N)
        if router.net_src_tile[i] in paths[i]
        and all(s in paths[i] for s in router.net_sink_tiles[i])
    )
    sizes = [len(p) for p in paths.values()]
    print(f"[extract-parallel] {N} nets in {dt:.1f}s with {args.workers} workers "
          f"({1000*dt/N:.2f} ms/net effective)  connected={connected} "
          f"({100*connected/N:.2f}%)  mean_tiles={sum(sizes)/N:.1f}", flush=True)
    if args.save_paths:
        with open(args.save_paths, "w") as f:
            json.dump({str(k): v for k, v in paths.items()}, f)
        print(f"  saved -> {args.save_paths}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
