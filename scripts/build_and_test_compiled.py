#!/usr/bin/env python3
"""Build a compiled router cache from the net index, then verify load_compiled
is fast AND bit-identical to the slow GlobalRouter.load path."""
import os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch

rrg_path = "data/rrg_xcvu3p_int.pt"
from src.router.global_router import GlobalRouter
from src.router.net_index import default_net_index_path

ni = default_net_index_path("./data/", "boom_soc_v2", rrg_path, "directed", 0, 0.1,
                            route_filter="stubs", edge_scope="corridor",
                            corridor_width=2, max_edges_per_net=50000)
compiled = ni.replace(".pt", ".compiled.pt")
dev = torch.device("cuda:0")

t = time.time()
r1 = GlobalRouter.load(rrg_path, ni, device=dev, edge_mode="directed", verbose=True)
print(f"[slow] GlobalRouter.load: {time.time()-t:.1f}s  vars={r1.num_vars}", flush=True)

t = time.time(); r1.save_compiled(compiled)
sz = os.path.getsize(compiled) / 1e9
print(f"[save] save_compiled: {time.time()-t:.1f}s  ({sz:.2f} GB) -> {compiled}", flush=True)

# fresh process would be cleaner, but validate in-process: reload compiled
t = time.time()
r2 = GlobalRouter.load_compiled(rrg_path, compiled, device=dev, edge_mode="directed", verbose=True)
print(f"[fast] load_compiled: {time.time()-t:.1f}s  vars={r2.num_vars}", flush=True)

# ---- validate identical ----
def eq(a, b):
    return bool(torch.equal(a.cpu(), b.cpu()))

checks = []
checks.append(("num_vars", r1.num_vars == r2.num_vars))
checks.append(("num_nets", r1.num_nets == r2.num_nets))
checks.append(("flat_edge_idx", eq(r1._flat_edge_idx, r2._flat_edge_idx)))
checks.append(("flat_wl", eq(r1._flat_wl, r2._flat_wl)))
checks.append(("flat_u", eq(r1._flat_u, r2._flat_u)))
checks.append(("flat_v", eq(r1._flat_v, r2._flat_v)))
checks.append(("flow_demand", eq(r1._flow_demand, r2._flow_demand)))
checks.append(("d2p", eq(r1._d2p, r2._d2p)))
checks.append(("phys_cap", eq(r1._phys_capacity_tensor, r2._phys_capacity_tensor)))
checks.append(("var_offset", r1._var_offset == r2._var_offset))
checks.append(("node_offset", r1._node_offset == r2._node_offset))
checks.append(("net_src_tile", r1.net_src_tile == r2.net_src_tile))
checks.append(("net_sink_tiles", r1.net_sink_tiles == r2.net_sink_tiles))
checks.append(("net_bbox", r1.net_bbox == r2.net_bbox))
checks.append(("conn_src", eq(r1._conn["src_flat"], r2._conn["src_flat"])))
checks.append(("conn_sink", eq(r1._conn["sink_flat"], r2._conn["sink_flat"])))
checks.append(("conn_col", eq(r1._conn["col_id"], r2._conn["col_id"])))
checks.append(("conn_numcols", r1._conn["num_cols"] == r2._conn["num_cols"]))
# loss equivalence
x = r1.init_variables()
wl1 = r1.wirelength_loss(x).item(); wl2 = r2.wirelength_loss(x).item()
checks.append(("wirelength_loss", abs(wl1 - wl2) < 1e-3))
_, ov1 = r1._get_usage_and_overflows(x); _, ov2 = r2._get_usage_and_overflows(x)
checks.append(("overflow", abs(ov1.sum().item() - ov2.sum().item()) < 1e-1))

print("\n[validate]", flush=True)
allok = True
for name, ok in checks:
    print(f"  {'OK ' if ok else 'FAIL'} {name}", flush=True)
    allok = allok and ok
print(f"\n[result] {'ALL IDENTICAL' if allok else 'MISMATCH!'}", flush=True)
print("[done]", flush=True)
