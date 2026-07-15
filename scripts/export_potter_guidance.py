#!/usr/bin/env python3
"""Export per-connection corridor bounding boxes as Potter GR guidance.

Potter (github.com/diriLin/Potter) computes each Connection's bbox from the net
pin centroid + source/sink tiles (src/db/netlist.cpp ~L455). That box drives both
the A* search bound and the PartitionTree parallel scheduler. This exports the
tighter DiffRouter corridor bbox per (net, sink) so Potter can override it and
get more parallelism + faster search. See docs/POTTER_INTEGRATION.md.

Output format (one line per connection, whitespace-separated):
    <net_name> <sink_row> <sink_col> <bb_min_col> <bb_min_row> <bb_max_col> <bb_max_row>
Coordinates are DiffRouter device grid (row, col). Potter maps tile X<-col,
Y<-row (adjust in the loader if the device orientation differs). A margin can be
added here or on the Potter side; over-tight guidance should fall back to the
pin bbox if detailed routing fails inside it.
"""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="global_x.pt (or checkpoint)")
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--testcase", default="boom_soc_v2")
    ap.add_argument("--out", default="",
                    help="output per-connection bbox guidance file (for Potter's PartitionTree). "
                         "WARNING: single-threaded per-net Dijkstra -- slow on large designs. "
                         "Omit unless you need bboxes; --guide-out is the A* route guide.")
    ap.add_argument("--guide-out", default="", help="write per-net route GUIDE (tile set) file (parallel)")
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--margin", type=int, default=1, help="tiles of slack added to each bbox")
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    from src.router.route_extractor import RouteExtractor

    ni = default_net_index_path("./data/", args.testcase, args.rrg, "directed", 0, 0.1,
                                route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t = time.time()
    router = GlobalRouter.load(args.rrg, ni, device=torch.device("cpu"),
                               edge_mode="directed", verbose=True)
    print(f"[load] {time.time()-t:.1f}s nets={router.num_nets}", flush=True)
    obj = torch.load(args.x, map_location="cpu")
    x = (obj["x"] if isinstance(obj, dict) else obj)

    if not args.out and not args.guide_out:
        ap.error("nothing to do: pass --guide-out (route guide) and/or --out (bboxes)")

    ex = RouteExtractor(threshold=args.threshold)
    conn_bboxes = None
    if args.out:
        t = time.time()
        conn_bboxes = ex.extract_connection_bboxes(router, x, None)
        print(f"[extract-conn-bbox] {time.time()-t:.0f}s", flush=True)

    # per-net route GUIDE (tile set) = the corridor the router should stay within.
    # This is the strong, digital-routing-style guidance (soft, with margin) that
    # prunes the detailed router's A* search. Reuse the same shortest-path tiles.
    if args.guide_out:
        tiles = router.rrg.tiles
        t = time.time()
        guide_paths = ex.extract(router, x, None, workers=0)  # per-net tile lists
        gsz = []
        with open(args.guide_out, "w") as gf:
            for net_idx, tlist in guide_paths.items():
                name = router.net_names[net_idx] if net_idx < len(router.net_names) else str(net_idx)
                rc = [(tiles[t2][0], tiles[t2][1]) for t2 in tlist]
                gsz.append(len(rc))
                cells = " ".join(f"{r},{c}" for r, c in rc)
                gf.write(f"{name} {len(rc)} {cells}\n")
        import numpy as _np
        print(f"[export-guide] {len(guide_paths)} net guides -> {args.guide_out} "
              f"(mean {_np.mean(gsz):.1f} tiles/net, {time.time()-t:.0f}s)", flush=True)

    if conn_bboxes is None:
        print("[done]", flush=True)
        return

    tiles = router.rrg.tiles
    m = args.margin
    R, C = router.device_rows, router.device_cols
    n_lines = 0
    areas_pin, areas_gd = [], []
    with open(args.out, "w") as f:
        for net_idx, conns in conn_bboxes.items():
            name = router.net_names[net_idx] if net_idx < len(router.net_names) else str(net_idx)
            for sink_tile, (mnc, mxc, mnr, mxr) in conns:
                sr, sc = tiles[sink_tile][0], tiles[sink_tile][1]
                mnc = max(0, mnc - m); mnr = max(0, mnr - m)
                mxc = min(C, mxc + m); mxr = min(R, mxr + m)
                f.write(f"{name} {sr} {sc} {mnc} {mnr} {mxc} {mxr}\n")
                n_lines += 1
                areas_gd.append((mxc - mnc + 1) * (mxr - mnr + 1))
    import numpy as np
    ag = np.array(areas_gd)
    # baseline pin-bbox area for comparison
    nb = np.array([(b[1]-b[0]+1)*(b[3]-b[2]+1) for b in router.net_bbox])
    print(f"[export] {n_lines} connection guidance lines -> {args.out}", flush=True)
    print(f"  guidance conn-bbox area: mean={ag.mean():.0f} median={int(np.median(ag))}", flush=True)
    print(f"  vs net pin-bbox area:    mean={nb.mean():.0f} median={int(np.median(nb))} "
          f"(guidance ~{nb.mean()/max(1,ag.mean()):.1f}x tighter)", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
