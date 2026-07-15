#!/usr/bin/env python3
"""Export a Potter GR route guide from a saved global_x.pt (GPU batched shortest paths).

Thin wrapper over src/router/gpu_guide.py -- the same code run_exp.py --guide-out uses
inline. Prefer --guide-out (no save/reload round-trip); use this to build a guide from a
checkpoint after the fact.
"""
import argparse, os, sys, time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.chdir(_root)

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="global_x.pt (or a checkpoint)")
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--testcase", default="boom_soc_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--edge-mode", default="directed")
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    from src.router.gpu_guide import export_guide

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ni = default_net_index_path("./data/", args.testcase, args.rrg, args.edge_mode, 0, 0.1,
                                route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t0 = time.time()
    router = GlobalRouter.load(args.rrg, ni, device=dev,
                               edge_mode=args.edge_mode, verbose=False)
    print(f"[load] {time.time()-t0:.1f}s nets={router.num_nets}", flush=True)

    obj = torch.load(args.x, map_location=dev)
    x = (obj["x"] if isinstance(obj, dict) else obj).to(dev)

    export_guide(router, x, args.out)
    print(f"[TOTAL] {time.time()-t0:.1f}s", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
