#!/usr/bin/env python3
"""Total routed wirelength of a physical netlist.

wa.py reports *critical-path* wirelength -- the contest's quality metric, but a single
path, so small differences between routers are within noise. This sums the wirelength of
every routed PIP across every net: a much more stable aggregate for comparing runs.

Reuses Potter's WirelengthAnalyzer.segment_to_wirelength, i.e. the same official PIP
wirelength table wa.py uses, so numbers are directly comparable.

Usage:
    python scripts/total_wirelength.py <routed>.phys [--potter <path to Potter>]
"""
import argparse
import gzip
import itertools
import os
import re
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phys", help="routed physical netlist")
    ap.add_argument("--potter", default=os.environ.get("POTTER", "third_party/Potter"),
                    help="Potter checkout (for its wirelength_analyzer)")
    ap.add_argument("--quiet", action="store_true", help="print only the total")
    args = ap.parse_args()

    phys_path = os.path.abspath(args.phys)
    potter = os.path.abspath(args.potter)
    wa_dir = os.path.join(potter, "wirelength_analyzer")
    if not os.path.isdir(wa_dir):
        sys.exit(f"ERROR: no wirelength_analyzer at {wa_dir}. "
                 f"Run potter/setup_potter.sh or pass --potter.")
    sys.path.insert(0, wa_dir)
    sys.path.insert(0, os.path.join(potter, "fpga-interchange-schema", "interchange"))
    os.chdir(wa_dir)   # wa.py/capnp resolve schema paths relative to themselves

    import capnp  # noqa: F401
    import PhysicalNetlist_capnp
    from wa import WirelengthAnalyzer
    from xcvup_device_data import xcvupDeviceData

    t0 = time.time()
    with gzip.open(phys_path, "rb") as f:
        raw = f.read()
    ctx = PhysicalNetlist_capnp.PhysNetlist.from_bytes(
        raw, traversal_limit_in_words=sys.maxsize, nesting_limit=2**20)
    phys = ctx.__enter__()

    # Build just enough of a WirelengthAnalyzer for segment_to_wirelength: it needs the
    # pip table, the tile-name machinery, its caches, and self.phys for the string list.
    wana = WirelengthAnalyzer.__new__(WirelengthAnalyzer)
    xcvup = xcvupDeviceData()
    wana.phys = phys
    wana.pips = xcvup.pips
    wana.tile_root_name_regex = xcvup.tile_root_name_regex
    wana.tile_types = xcvup.tile_types
    wana.pip_cache = {}
    wana.tile_cache = {}
    wana.site_pip_cache = {}

    total = 0
    n_nets = 0
    n_routed = 0
    n_skip_seg = 0
    skipped_tiles = set()
    per_net = []
    for net in phys.physNets:
        n_nets += 1
        wl = 0
        stack = list(net.sources) + list(net.stubs)
        while stack:
            b = stack.pop()
            try:
                wl += wana.segment_to_wirelength(b.routeSegment)
            except ValueError as exc:
                # Potter's PIP table does not cover clock-distribution tiles
                # (RCLK_RCLK_XIPHY / CLK_HDISTR). Those segments are on clock nets, not
                # signal wirelength, so skip them rather than dropping the whole design.
                n_skip_seg += 1
                msg = str(exc)
                if len(skipped_tiles) < 8:
                    skipped_tiles.add(msg.split(":", 1)[-1].strip().split(",")[0])
            stack.extend(b.branches)
        if wl:
            n_routed += 1
            total += wl
            per_net.append(wl)

    ctx.__exit__(None, None, None)
    dt = time.time() - t0
    if args.quiet:
        print(total)
        return
    per_net.sort()
    print(f"Total wirelength: {total}")
    print(f"  nets: {n_nets} ({n_routed} with routed PIPs)")
    if n_skip_seg:
        print(f"  skipped {n_skip_seg} segments with PIPs outside Potter's table "
              f"(clock-distribution tiles, e.g. {', '.join(sorted(skipped_tiles))})")
    if per_net:
        print(f"  per-net: mean={total/n_routed:.1f} p50={per_net[len(per_net)//2]} "
              f"p99={per_net[int(len(per_net)*0.99)]} max={per_net[-1]}")
    print(f"  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
