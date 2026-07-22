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
    # The Cap'n Proto schema (PhysicalNetlist.capnp) ships in Potter under two identical
    # paths depending on the checkout; use whichever exists so this works on any Potter.
    schema_dir = next(
        (d for d in (os.path.join(potter, "fpga-interchange-schema", "interchange"),
                     os.path.join(potter, "libs", "interchange", "definition"))
         if os.path.isfile(os.path.join(d, "PhysicalNetlist.capnp"))),
        None,
    )
    if schema_dir is None:
        sys.exit(
            "ERROR: PhysicalNetlist.capnp not found under Potter "
            f"({potter}/fpga-interchange-schema/interchange or "
            f"{potter}/libs/interchange/definition). Fetch Potter's interchange schema "
            "(re-run potter/setup_potter.sh), and `pip install pycapnp`.")
    sys.path.insert(0, schema_dir)
    os.chdir(wa_dir)   # wa.py/capnp resolve schema paths relative to themselves

    # Name the exact missing piece rather than a generic ImportError -- the wirelength
    # tools pull in several things (pycapnp, networkx, the capnp schema, Potter's local
    # xcvup_device_data), and "No module named X" tells you which to install/fix.
    try:
        import capnp  # noqa: F401  (pip install pycapnp)
        import PhysicalNetlist_capnp  # noqa: F401  (capnp schema at schema_dir, above)
        from wa import WirelengthAnalyzer  # also imports networkx
        from xcvup_device_data import xcvupDeviceData
    except ImportError as exc:
        name = getattr(exc, "name", None) or str(exc)
        hint = {
            "capnp": "pip install pycapnp",
            "networkx": "pip install networkx",
            "PhysicalNetlist_capnp": "capnp schema missing under Potter -- re-run "
                                     "potter/setup_potter.sh (creates fpga-interchange-schema/interchange)",
            "xcvup_device_data": f"expected in {wa_dir} -- your Potter checkout is incomplete",
        }.get(name, "install it into the SAME python env that runs this script")
        sys.exit(f"ERROR: cannot import '{name}' for the wirelength analyzer -> {hint}\n"
                 f"       (python: {sys.executable})")

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
