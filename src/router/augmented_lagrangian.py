"""Augmented Lagrangian optimizer for GlobalRouter."""

import os
from typing import Optional

import torch

from src.router.checkpoint import save_checkpoint
from src.router.meng_lambda import update_multipliers_meng


def _eg_step(x, state, eg_lr, eg_clip, eps=1e-12):
    """One exponentiated-gradient step with a per-net mass constraint (in place on x).

    x <- x * exp(-eg_lr * ghat), then renormalise each net's total x back to its initial
    mass. ghat is the raw gradient normalised by its per-net RMS, so eg_lr is a
    dimensionless step (~0.1-1) that behaves the same regardless of the loss's absolute
    scale -- important because the terms here span wl~7e4 to ER~1e5. The exponent is
    clipped to [-eg_clip, eg_clip] so one step moves any edge by at most a factor
    exp(eg_clip).

    Renormalising to the fixed net mass is what forbids the shrink-to-zero exploit: a
    hot edge's positive gradient shrinks it, and its net-mates GROW to compensate, so
    congestion is reduced by moving mass, not deleting it. Off-path edges must start > 0
    (multiplicative update can't lift an exact zero) -- see the EG setup warning.
    """
    var_net = state["var_net"]
    g = x.grad
    if g is None:
        return
    with torch.no_grad():
        nnet = state["net_mass0"].numel()
        gsq = torch.zeros(nnet, device=x.device, dtype=x.dtype)
        gsq.index_add_(0, var_net, g * g)
        rms = (gsq / state["count"]).sqrt().clamp_min(eps)
        ghat = g / rms[var_net]
        x.mul_(torch.exp((-eg_lr * ghat).clamp_(-eg_clip, eg_clip)))
        new_sum = torch.zeros(nnet, device=x.device, dtype=x.dtype)
        new_sum.index_add_(0, var_net, x)
        x.mul_((state["net_mass0"] / new_sum.clamp_min(eps))[var_net])
        x.clamp_(0.0, 1.0)
    x.grad = None


def optimize_augmented_lagrangian(
    router,
    x: Optional[torch.Tensor] = None,
    num_outer: int = 200,
    num_inner: int = 5,
    max_iterations: int = 1000,
    viz_interval: int = 50,
    viz_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 0,
    resume_path: Optional[str] = None,
    lr_x: float = 0.01,
    lr_lam: float = 0.1,
    rho: float = 1.0,
    rho_max: float = 100.0,
    rho_mult: float = 2.0,
    w_wl: float = 1.0,
    w_conn: float = 1.0,
    w_flow: float = 1.0,
    w_disc: float = 0.0,
    disc_ramp_outer: int = 0,
    conn_every: int = 1,
    conn_freeze_outer: int = 0,
    grad_balance: str = "off",
    balance_ratio: float = 1.0,
    balance_every: int = 1,
    optimizer_kind: str = "adam",
    eg_lr: float = 0.5,
    eg_clip: float = 1.0,
    lam_update: str = "meng",
    lam_mult_eta: float = 0.5,
    lam_floor: float = 0.0,
    lam_base: float = 1.0,
    early_stop_tol: float = 0.0,
    early_stop_patience: int = 3,
    overflow_stop_frac: float = 0.0,
    connectivity: str = "effective_resistance",
    connectivity_solver: str = "grouped",
    conn_net_batch: int = 0,
    flow_net_batch: int = 0,
    verbose: bool = True,
    log_setup: bool = True,
) -> torch.Tensor:
    router.connectivity_solver = connectivity_solver
    router.conn_net_batch = conn_net_batch
    router.flow_net_batch = flow_net_batch

    if log_setup:
        print("  [4a] Initializing x variables...")
    overflow_ref = None
    if resume_path and os.path.isfile(resume_path):
        from src.router.checkpoint import load_checkpoint
        ckpt = load_checkpoint(resume_path, router.device)
        x = ckpt["x"]
        lam = ckpt["lam"]
        rho = ckpt.get("rho", rho)
        total_iter = ckpt.get("total_iter", 0)
        start_outer = ckpt.get("outer", 0)
        if "overflow_ref" in ckpt:
            overflow_ref = ckpt["overflow_ref"].to(router.device)
        if log_setup:
            print(f"  [4a] Resumed from {resume_path} at iter {total_iter}")
    else:
        if x is None:
            x = router.init_variables()
        else:
            x = x.detach().clone().requires_grad_(True)
        # Multiplicative history needs a positive seed (0 * anything = 0 forever).
        lam_init = lam_base if lam_update == "mult" else 0.0
        lam = torch.full((len(router.rrg.phys_list),), lam_init,
                         device=router.device, dtype=x.dtype)
        total_iter = 0
        start_outer = 0

    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)

    if log_setup:
        print(f"  [4b] Created λ multipliers: {len(router.rrg.phys_list)} physical edges")

    # --- Optimizer setup ---------------------------------------------------------
    # 'adam' (default): unchanged additive update on the box [0,1]^E. Measured to never
    # reroute (Jaccard 1.0 support vs init) -- on a box, every congestion loss is
    # minimised by SHRINKING x, so the optimiser turns paths down instead of moving them.
    # 'eg': exponentiated-gradient / mirror descent with a PER-NET mass constraint. Each
    # net's total x is pinned to its initial value, so congestion can only be reduced by
    # MOVING mass off hot edges onto connected alternatives (the flow-conservation +
    # wirelength terms decide where it lands) -- the feasible set becomes a simplex per
    # net instead of a box, which is the structural change that makes rerouting possible.
    optimizer = None
    eg_state = None
    if optimizer_kind == "adam":
        optimizer = torch.optim.Adam([x], lr=lr_x)
        if log_setup:
            print(f"  [4c] Adam optimizer ready (lr_x={lr_x})")
    elif optimizer_kind == "eg":
        vo = torch.tensor(router._var_offset, dtype=torch.long, device=router.device)
        counts = (vo[1:] - vo[:-1]).clamp_min(1)
        var_net = torch.repeat_interleave(
            torch.arange(len(counts), device=router.device), counts)
        with torch.no_grad():
            net_mass0 = torch.zeros(len(counts), device=router.device, dtype=x.dtype)
            net_mass0.index_add_(0, var_net, x.detach())
            xmin = float(x.min())
        eg_state = {"var_net": var_net, "net_mass0": net_mass0,
                    "count": counts.to(x.dtype)}
        if log_setup:
            print(f"  [4c] EG optimizer ready (eg_lr={eg_lr}, clip={eg_clip}); "
                  f"per-net mass pinned to init")
        if xmin <= 0.0:
            print("  [4c] WARNING: init x has exact zeros -- EG is multiplicative, so "
                  "those stay 0 forever. Use --init-mode shortest_path --init-off-path "
                  "0.01 (or any >0) so off-path edges can receive mass.", flush=True)
    else:
        raise ValueError(f"unknown optimizer_kind={optimizer_kind!r}")

    gif_frames = []
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        if log_setup:
            print(f"  [4d] Viz dir ready: {viz_dir}")

    if log_setup:
        print("  [4e] Starting outer loop (Augmented Lagrangian)...")
        if early_stop_tol > 0:
            print(
                f"  [4e] Early stop armed: overflow improvement < "
                f"{early_stop_tol*100:.1f}% over {early_stop_patience} outers "
                f"(after rho reaches {rho_max})"
            )

    if log_setup and grad_balance != "off":
        print(f"  [4e] Gradient balancing: {grad_balance}, ratio={balance_ratio}, "
              f"every {balance_every} outer(s)")

    ovf_hist: list = []
    ovf_baseline = None
    w_conn_bal = w_conn
    for outer in range(start_outer, num_outer):
        # Gradient balancing (opt-in). Measured on boom_soc_v2: ||grad|| is 7.2e4 for
        # wirelength, 2.7e7 for the congestion penalty at rho_max -- but 1.3e13 to 3.2e14
        # for effective resistance, because dR/dw ~ 1/w^2 blows up at small x. So ER is
        # ~1e10x every other term and the AL optimises it alone. lambda cannot correct
        # this: the Meng update is normalised, so ||dlam|| == lr_lam per outer regardless
        # of violation, and multipliers weight the *constraint* anyway -- there is no
        # multiplier on an objective term. Rescaling w_conn so ||w_conn*grad_conn|| ==
        # balance_ratio * ||grad_wl|| puts every term within ~1e2 of the others, which is
        # the commensurate regime lambda/rho were designed for.
        if grad_balance == "conn" and outer % balance_every == 0:
            norms = router.term_grad_norms(x, connectivity_solver=connectivity_solver)
            g_wl = norms["wirelength"][1]
            g_cn = norms["connectivity"][1]
            if g_cn > 0:
                w_conn_bal = balance_ratio * g_wl / g_cn
                if verbose:
                    print(f"      [balance] ||g_wl||={g_wl:.3e} ||g_conn||={g_cn:.3e} "
                          f"-> w_conn={w_conn_bal:.3e}", flush=True)

        # Anneal discretization weight from 0 to w_disc over disc_ramp_outer outers
        # (establish connectivity first, then sharpen to discrete paths).
        if w_disc != 0.0 and disc_ramp_outer > 0:
            w_disc_eff = w_disc * min(1.0, outer / float(disc_ramp_outer))
        else:
            w_disc_eff = w_disc
        # A2: freeze connectivity after it has converged (conn_freeze_outer>0).
        conn_frozen = conn_freeze_outer > 0 and outer >= conn_freeze_outer
        for _inner in range(num_inner):
            if optimizer is not None:
                optimizer.zero_grad()
            else:
                x.grad = None
            # A1: evaluate the (expensive) connectivity term only every conn_every
            # inner steps; other steps optimize wirelength/congestion/flow only.
            include_conn = (not conn_frozen) and (total_iter % conn_every == 0)
            w_conn_step = w_conn_bal if include_conn else 0.0
            L_A = router.augmented_lagrangian(
                x,
                lam,
                rho,
                w_wl=w_wl,
                w_conn=w_conn_step,
                w_flow=w_flow,
                w_disc=w_disc_eff,
                connectivity=connectivity,
                connectivity_solver=connectivity_solver,
                conn_net_batch=conn_net_batch,
                flow_net_batch=flow_net_batch,
            )
            L_A.backward()
            if optimizer is not None:
                optimizer.step()
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)
            else:
                _eg_step(x, eg_state, eg_lr, eg_clip)

            total_iter += 1

            if viz_dir and total_iter % viz_interval == 0:
                if verbose:
                    print(f"      [viz] Capturing frame at iter {total_iter}")
                from PIL import Image
                from src.Visualizer import render_congestion_frame
                cong_grid = router.get_congestion_map(x)
                frame_arr = render_congestion_frame(
                    cong_grid,
                    title=f"Congestion (flow/capacity) @ iter {total_iter}",
                )
                gif_frames.append(Image.fromarray(frame_arr))

            if checkpoint_dir and save_every > 0 and total_iter % save_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_iter{total_iter:06d}.pt")
                extra = {"overflow_ref": overflow_ref.detach().cpu()} if overflow_ref is not None else None
                save_checkpoint(ckpt_path, x, lam, rho, total_iter, outer + 1, extra=extra)

            if total_iter >= max_iterations:
                if verbose:
                    print(f"      [stop] Reached max_iterations={max_iterations}")
                break

        if total_iter >= max_iterations:
            break

        with torch.no_grad():
            _, overflows = router._get_usage_and_overflows(x)
            if overflow_ref is None:
                overflow_ref = torch.clamp(overflows.clone(), min=1e-8)
            if lam_update == "mult":
                # Multiplicative (PathFinder-style) history: lam grows geometrically on
                # edges that stay over capacity, so a persistently-contested edge keeps
                # getting more expensive even after brief relief -- the ratchet the
                # normalised Meng update lacks (measured lam_max=0.0104, decorative).
                # lam <- lam * (1 + eta * relu(u/c - 1)), floored so it never dies.
                u = router._phys_usage(x)
                c = router._phys_capacity_tensor.clamp_min(1e-12)
                lam = torch.clamp(lam * (1.0 + lam_mult_eta * torch.relu(u / c - 1.0)),
                                  min=lam_floor)
            else:
                lam = update_multipliers_meng(lam, overflows, overflow_ref, lr_lam)

        if overflows.max().item() > 0.1 and rho < rho_max:
            rho = min(rho * rho_mult, rho_max)
            if verbose:
                print(f"      [rho] Increased to {rho:.2f}")

        # Early stop on AL convergence. Overflow (the constraint violation) is already
        # computed above for the lambda update, so this costs nothing. A fixed
        # max-iterations is design-size-dependent: it over-runs small designs and
        # under-runs large ones. Stopping when overflow stops improving adapts to both.
        #
        # Only arm once rho has reached rho_max: overflow legitimately RISES while the
        # penalty ramps (flow spreads before it is squeezed), so an earlier check would
        # fire on that transient.
        # Overflow-fraction early stop (EG mode): stop once relaxed overflow has fallen
        # below overflow_stop_frac of its FIRST-outer value. Measured on boom_soc_v2, the
        # EG rerouting is done (crossings plateaued, overflow -99%) by the outer where
        # overflow crosses ~1% of its initial value; iterations past that only sharpen
        # magnitudes the argmin guide ignores (validated: iter-25 vs iter-40 total WL
        # within Potter noise, mean -0.08%). Absolute-fraction, not relative-improvement:
        # overflow bounces near its floor, which fools an improvement<tol rule.
        if overflow_stop_frac > 0:
            # Measure HARD overflow (relu), not the training `overflows`: with
            # --congestion-mode soft the latter is a softplus that never approaches 0, so
            # it never crosses overflow_stop_frac of its initial value and the stop would
            # never fire. The convergence curve that set this threshold used hard overflow,
            # which drops ~99% as the reroute resolves.
            with torch.no_grad():
                hard_ovf = float(torch.relu(
                    router._phys_usage(x) - router._phys_capacity_tensor).sum().item())
            if ovf_baseline is None:
                ovf_baseline = hard_ovf        # first-outer value
            elif hard_ovf <= overflow_stop_frac * ovf_baseline:
                if verbose:
                    print(f"      [early-stop] outer {outer+1} (iter {total_iter}): "
                          f"overflow={hard_ovf:.4g} <= {overflow_stop_frac:.0%} of initial "
                          f"{ovf_baseline:.4g} -- rerouting converged")
                break

        if early_stop_tol > 0:
            ovf_hist.append(float(overflows.sum().item()))
            if rho >= rho_max and len(ovf_hist) > early_stop_patience:
                window = ovf_hist[-(early_stop_patience + 1):]
                base = max(window[0], 1e-12)
                improvement = (window[0] - window[-1]) / base
                if improvement < early_stop_tol:
                    if verbose:
                        print(
                            f"      [early-stop] outer {outer+1} (iter {total_iter}): "
                            f"overflow improved {improvement*100:.2f}% over last "
                            f"{early_stop_patience} outers (< {early_stop_tol*100:.1f}%); "
                            f"overflow_sum={window[-1]:.4g}"
                        )
                    break

        if verbose and (outer + 1) % 20 == 0:
            wl = router.wirelength_loss(x).item()
            conn = router.connectivity_loss_effective_resistance(
                x, solver=connectivity_solver, net_batch=conn_net_batch
            ).item()
            max_of = overflows.max().item()
            lam_min, lam_max = lam.min().item(), lam.max().item()
            lam_mean = lam.float().mean().item()
            lam_norm = lam.norm().item()
            if getattr(router, "edge_mode", "directed") == "undirected":
                print(
                    f"  [metrics] Outer {outer+1}/{num_outer} (iter {total_iter}): "
                    f"wl={wl:.2f} conn={conn:.2f} max_overflow={max_of:.4f} | "
                    f"λ: min={lam_min:.4f} max={lam_max:.4f} mean={lam_mean:.4f} ‖λ‖={lam_norm:.2f}"
                )
            else:
                flow = router.flow_conservation_loss(
                    x, net_batch=flow_net_batch
                ).item()
                print(
                    f"  [metrics] Outer {outer+1}/{num_outer} (iter {total_iter}): "
                    f"wl={wl:.2f} conn={conn:.2f} flow={flow:.2f} max_overflow={max_of:.4f} | "
                    f"λ: min={lam_min:.4f} max={lam_max:.4f} mean={lam_mean:.4f} ‖λ‖={lam_norm:.2f}"
                )

    if log_setup:
        print("  [4f] Optimization loop finished")

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_ckpt = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        extra = {"overflow_ref": overflow_ref.detach().cpu()} if overflow_ref is not None else None
        save_checkpoint(final_ckpt, x, lam, rho, total_iter, outer + 1, extra=extra)
        global_x_path = os.path.join(checkpoint_dir, "global_x.pt")
        torch.save(x.detach().cpu(), global_x_path)
        if log_setup:
            print(f"  [4f] Saved checkpoint: {final_ckpt}")

    if viz_dir and gif_frames:
        if log_setup:
            print(f"  [4g] Writing congestion evolution GIF ({len(gif_frames)} frames)...")
        gif_path = os.path.join(viz_dir, "congestion_evolution.gif")
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=200,
            loop=0,
        )
        if log_setup:
            print(f"  [4g] Saved: {gif_path}")

    return x.detach()
