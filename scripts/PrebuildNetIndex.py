#!/usr/bin/env python3
"""Pre-build net index cache for a testcase (offline)."""

import argparse
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.chdir(_root)


def main():
    parser = argparse.ArgumentParser(description="Pre-build net index for GlobalRouter.load()")
    parser.add_argument("--testcase", required=True)
    parser.add_argument("--data", default="./data/")
    parser.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    parser.add_argument(
        "--device",
        default=None,
        help="FPGAIF device file (default: <data>/xcvu3p.device)",
    )
    parser.add_argument("--min-fanout", type=int, default=0)
    parser.add_argument("--expansion-ratio", type=float, default=0.1)
    parser.add_argument("--edge-mode", choices=["directed", "undirected"], default="directed")
    parser.add_argument("--max-nets", type=int, default=None)
    parser.add_argument("--net-index", default=None, help="Output path (default: auto under data/<testcase>/net_index/)")
    parser.add_argument(
        "--use-java",
        action="store_true",
        help="Use RapidWright/Java for all signal nets (legacy; slow)",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from src.router.net_index import build_and_save, default_net_index_path

    netlist_path = os.path.join(args.data, args.testcase, f"{args.testcase}.netlist")
    physical_path = os.path.join(args.data, args.testcase, f"{args.testcase}_unrouted.phys")
    device_path = args.device or os.path.join(args.data, "xcvu3p.device")
    route_filter = "all" if args.use_java else "stubs"
    cache_path = args.net_index or default_net_index_path(
        args.data,
        args.testcase,
        args.rrg,
        args.edge_mode,
        args.min_fanout,
        args.expansion_ratio,
        route_filter=route_filter,
    )

    print(f"Pre-building net index -> {cache_path}")
    if args.use_java:
        from src.load_design import load_design

        design = load_design(physical_path, netlist_path)
    else:
        design = None

    index = build_and_save(
        args.rrg,
        cache_path,
        physical_path,
        netlist_path=netlist_path,
        device_path=device_path,
        min_fanout=args.min_fanout,
        expansion_ratio=args.expansion_ratio,
        edge_mode=args.edge_mode,
        max_nets=args.max_nets,
        route_filter=route_filter,
        use_java=args.use_java,
        design=design,
        verbose=True,
    )
    print(f"Done: {index.num_nets} nets, {index.num_vars} variables")


if __name__ == "__main__":
    main()
