"""Detailed routing via RapidWright PartialRouter (Tier A handoff)."""

from typing import Any, Dict, List, Optional


def _get_partial_router():
    from com.xilinx.rapidwright.rwroute import PartialRouter, RWRouteConfig
    return PartialRouter, RWRouteConfig


def _collect_pins_for_nets(design: Any, net_objects: List[Any]) -> List[Any]:
    pins = []
    for net in net_objects:
        src = net.getSource()
        if src is not None:
            pins.append(src)
        pins.extend(net.getSinkPins())
    return pins


def _net_name(net: Any) -> str:
    name = net.getName()
    return str(name) if name is not None else ""


def route_design_detailed(
    design: Any,
    global_router,
    global_net_objects: List[Any],
    paths: Optional[Dict[int, List[int]]] = None,
    soft_preserve: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Route nets using PartialRouter. Optionally restrict to nets from global router.

    Returns routing statistics dict.
    """
    rwroute_args = [
        "--fixBoundingBox",
        "--useUTurnNodes",
        "--nonTimingDriven",
        "--wirelengthWeight",
        "0.8",
        "--initialPresentCongestionFactor",
        "0.5",
        "--presentCongestionMultiplier",
        "2",
        "--maxIterations",
        "150",
    ]
    PartialRouter, RWRouteConfig = _get_partial_router()
    config = RWRouteConfig(rwroute_args)

    pins_to_route = _collect_pins_for_nets(design, global_net_objects)
    if verbose:
        print(f"  Detailed routing: {len(global_net_objects)} nets, {len(pins_to_route)} pins")

    if not pins_to_route:
        return {"routed_nets": 0, "unrouted_nets": 0, "total_pips": 0}

    PartialRouter.routeDesignPartialNonTimingDriven(
        design,
        config,
        pins_to_route,
        soft_preserve,
    )

    routed = 0
    unrouted = 0
    total_pips = 0
    global_names: set = {_net_name(n) for n in global_net_objects}

    for net in design.getNets():
        if net.getSource() is None or net.isStaticNet():
            continue
        if _net_name(net) not in global_names:
            continue
        if net.hasPIPs():
            routed += 1
            total_pips += len(list(net.getPIPs()))
        else:
            unrouted += 1

    stats = {
        "routed_nets": routed,
        "unrouted_nets": unrouted,
        "total_pips": total_pips,
        "pins_routed": len(pins_to_route),
    }
    if verbose:
        print(
            f"  Detailed route done: routed={routed} unrouted={unrouted} "
            f"total_pips={total_pips}"
        )
    return stats


def load_design_for_routing(phys_path: str, netlist_path: str) -> Any:
    from src.load_design import load_design

    return load_design(phys_path, netlist_path)
