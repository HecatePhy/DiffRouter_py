"""Evaluate routing metrics for DiffRouter testcases."""

import argparse
import json
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from src.load_design import load_design
from src.Timer import Timer


def count_fanout_buckets(design) -> dict:
    net_ids = [2, 10, 20, 50, 100, 200, 500, 1000]
    net_dict = {k: 0 for k in net_ids}
    for net in design.getNets():
        if net.getSource() is None or net.isStaticNet():
            continue
        if not net.hasPIPs():
            num_pins = len(net.getSinkPins())
            placed = False
            for net_idx in net_ids:
                if num_pins <= net_idx:
                    net_dict[net_idx] += 1
                    placed = True
                    break
            if not placed:
                net_dict[1000] += 1
    return net_dict


def routing_metrics(design, label: str = "") -> dict:
    routed = 0
    unrouted = 0
    total_pips = 0
    wl_per_net = []

    for net in design.getNets():
        if net.getSource() is None or net.isStaticNet():
            continue
        if net.hasPIPs():
            routed += 1
            pips = len(list(net.getPIPs()))
            total_pips += pips
            wl_per_net.append(pips)
        else:
            unrouted += 1

    return {
        "label": label,
        "routed_nets": routed,
        "unrouted_nets": unrouted,
        "total_pips": total_pips,
        "mean_pips_per_routed_net": sum(wl_per_net) / len(wl_per_net) if wl_per_net else 0.0,
        "max_pips_per_net": max(wl_per_net) if wl_per_net else 0,
    }


def evaluate_testcase(testcase: str, data_prefix: str, results_prefix: str) -> dict:
    netlist_path = os.path.join(data_prefix, testcase, f"{testcase}.netlist")
    physical_unrouted = os.path.join(data_prefix, testcase, f"{testcase}_unrouted.phys")
    result_dir = os.path.join(results_prefix, testcase)
    routed_phys = os.path.join(result_dir, f"{testcase}_routed.phys")
    metrics_json = os.path.join(result_dir, "metrics.json")

    report = {"testcase": testcase}

    if os.path.isfile(physical_unrouted):
        with Timer(f"load_unrouted_{testcase}", logger=print):
            design_u = load_design(physical_unrouted, netlist_path)
        report["fanout_buckets_unrouted"] = count_fanout_buckets(design_u)
        report["unrouted_metrics"] = routing_metrics(design_u, "unrouted_input")

    if os.path.isfile(routed_phys):
        with Timer(f"load_routed_{testcase}", logger=print):
            design_r = load_design(routed_phys, netlist_path)
        report["routed_metrics"] = routing_metrics(design_r, "routed_output")

    if os.path.isfile(metrics_json):
        with open(metrics_json) as f:
            report["pipeline_metrics"] = json.load(f)

    return report


def main():
    parser = argparse.ArgumentParser(description="DiffRouter evaluation")
    parser.add_argument("--testcases", nargs="+", default=["boom_med_pb", "boom_soc_v2"])
    parser.add_argument("--data", default="./data/")
    parser.add_argument("--results", default="./results/")
    parser.add_argument("--out", default="results/eval_report.json")
    args = parser.parse_args()

    full_report = {}
    for tc in args.testcases:
        print(f"\n=== {tc} ===")
        full_report[tc] = evaluate_testcase(tc, args.data, args.results)
        if "routed_metrics" in full_report[tc]:
            m = full_report[tc]["routed_metrics"]
            print(f"  Routed: {m['routed_nets']}, Unrouted: {m['unrouted_nets']}, PIPs: {m['total_pips']}")
        if "fanout_buckets_unrouted" in full_report[tc]:
            print(f"  Fanout buckets: {full_report[tc]['fanout_buckets_unrouted']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nReport saved: {args.out}")


if __name__ == "__main__":
    main()
