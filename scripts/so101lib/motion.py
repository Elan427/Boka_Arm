"""Motion primitives for the SO-101 scripted pick sequence.

Every move is *streamed* the way HOME always was: an interpolated joint target
re-sent to the follower force=True (no throttle, no smoothing) at ~HW_STREAM_HZ,
so each servo's own acceleration-limited profile turns it into one continuous
motion rather than a staircase of setpoints. The sim is stepped alongside to stay
roughly real time. Progress shows in the in-place `hud` readout, not a log.
"""
import time

import mujoco
import numpy as np

from . import hud
from .config import HW_STREAM_HZ

# The sim steps at ~300 Hz; rendering every step just burns CPU. Push a frame to
# the viewer every Nth step/iteration instead -> plenty smooth.
SYNC_EVERY = 3

# Straight-line Cartesian moves are sampled into IK waypoints ~this far apart (m);
# the streamer interpolates joint-space between them every tick.
IK_WP_LEN = 0.04

# home() gives up when total joint error improves < STALL_DEG within STALL_SECS.
STALL_DEG = 1.0
STALL_SECS = 3.0


# --- low level ----------------------------------------------------------------

def _pace(sim, follower, t0):
    if follower.connected:
        dt = sim.m.opt.timestep - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


def _steps(sim, secs):
    return max(1, int(secs / sim.m.opt.timestep))


def _drive(sim, viewer, follower, ctrl, n_steps, contact_stop=None, on_step=None):
    """Hold a fixed ctrl for n_steps of sim (sim-only path). Returns "contact",
    True, or False (viewer closed)."""
    sim.d.ctrl[:6] = ctrl
    for i in range(n_steps):
        t0 = time.time()
        mujoco.mj_step(sim.m, sim.d)
        if on_step is not None:
            on_step()
        if i % SYNC_EVERY == 0 or i == n_steps - 1:
            viewer.sync()
        follower.send(ctrl)
        hud.update(follower, ctrl)
        if not viewer.is_running():
            return False
        if contact_stop is not None and contact_stop():
            viewer.sync()
            return "contact"
        _pace(sim, follower, t0)
    return True


def _stream(sim, viewer, follower, target_fn, secs, contact_stop=None, on_step=None,
            settle_at=None):
    """Play a moving target to the arm over `secs` wall-clock. target_fn(f), f in
    0..1, returns the 6-vector ctrl at that fraction of the move. On hardware the
    target is re-sent force=True every loop tick (~HW_STREAM_HZ) so the servos'
    own profile smooths it; the sim is micro-stepped to keep it near real time.
    Sim-only: just runs the interpolation over the equivalent number of steps.

    Returns True (done), "contact", or False (viewer closed).
    """
    ts = sim.m.opt.timestep
    hw = follower.connected
    n_steps = _steps(sim, secs)
    period = 1.0 / HW_STREAM_HZ
    t_start = time.time()
    prev = t_start
    i = 0
    while viewer.is_running():
        t0 = time.time()
        f = (t0 - t_start) / secs if hw else i / n_steps
        if f >= 1.0:
            break
        fs = f * f * (3.0 - 2.0 * f)  # smoothstep: ease in and out like HOME does
        ctrl = np.asarray(target_fn(fs), dtype=float)
        sim.d.ctrl[:6] = ctrl
        nsub = int(np.clip(round((t0 - prev) / ts), 1, 16)) if hw else 1
        prev = t0
        for _ in range(nsub):
            mujoco.mj_step(sim.m, sim.d)
            if on_step is not None:
                on_step()
            if contact_stop is not None and contact_stop():
                viewer.sync()
                return "contact"
        if i % SYNC_EVERY == 0 or not hw:
            viewer.sync()
        follower.send(ctrl, force=True)
        hud.update(follower, ctrl)
        i += 1
        if hw:
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    if not viewer.is_running():
        return False
    if settle_at is not None:
        settle(sim, viewer, follower, np.asarray(settle_at, dtype=float), on_step=on_step)
    return True


def glide(sim, viewer, follower, ctrl, secs=2.0, contact_stop=None, on_step=None,
          label=None):
    """Stream a *fixed* target (the HOME-style move) to a 6-vector ctrl over
    `secs`, then settle. Returns True / "contact" / False."""
    if label:
        hud.phase(label)
    ctrl = np.asarray(ctrl, dtype=float)
    return _stream(sim, viewer, follower, lambda _f: ctrl, secs,
                   contact_stop=contact_stop, on_step=on_step, settle_at=ctrl)


# --- home --------------------------------------------------------------------

def home(sim, viewer, follower, home_ctrl, tol_deg=3.0, timeout=45.0, secs=1.2,
         on_step=None):
    """Ramp sim + real arm to home_ctrl. Sim-only: a short hold. With hardware:
    ride lerobot's clamp toward the fixed target, watching for a stall.
    Returns (ok, err_rad|None)."""
    home_ctrl = np.asarray(home_ctrl, dtype=float)
    if not follower.connected:
        ok = _drive(sim, viewer, follower, home_ctrl, _steps(sim, secs),
                    on_step=on_step) is True
        return ok, None

    sim.d.ctrl[:6] = home_ctrl
    tol = np.radians(tol_deg)
    t_end = time.time() + timeout
    mark = None  # (time, total_err) at the last meaningful progress
    errs = None
    i = 0
    while time.time() < t_end and viewer.is_running():
        t0 = time.time()
        mujoco.mj_step(sim.m, sim.d)
        if on_step is not None:
            on_step()
        if i % SYNC_EVERY == 0:
            viewer.sync()
        i += 1
        follower.send(home_ctrl, force=True)
        hud.update(follower, home_ctrl)
        try:
            errs = np.abs(follower.read_joints_rad() - home_ctrl[:5])
        except Exception:
            errs = None
        if errs is not None:
            if errs.max() < tol:
                return True, errs
            # Stall = the TOTAL error over all joints has not shrunk by STALL_DEG
            # within STALL_SECS. Judging by the worst joint alone bails while the
            # other joints are still travelling; a tick count bails on a slow
            # start (servo acceleration ramp + gravity load on the first lift).
            now = time.time()
            total = errs.sum()
            if mark is None or mark[1] - total > np.radians(STALL_DEG):
                mark = (now, total)
            elif now - mark[0] > STALL_SECS:
                return False, errs
        _pace(sim, follower, t0)
    return False, errs


def settle(sim, viewer, follower, ctrl, tol_deg=1.5, timeout=3.0, on_step=None):
    """Keep commanding ctrl until the real arm's joints converge to it (or stop
    moving). No-op without hardware."""
    ctrl = np.asarray(ctrl, dtype=float)
    if not follower.connected:
        return
    tol = np.radians(tol_deg)
    t_end = time.time() + timeout
    prev = None
    i = 0
    while time.time() < t_end and viewer.is_running():
        t0 = time.time()
        sim.d.ctrl[:6] = ctrl
        mujoco.mj_step(sim.m, sim.d)
        if on_step is not None:
            on_step()
        if i % SYNC_EVERY == 0:
            viewer.sync()
        i += 1
        follower.send(ctrl, force=True)
        hud.update(follower, ctrl)
        try:
            q = follower.read_joints_rad()
        except Exception:
            q = None
        if q is not None:
            if np.max(np.abs(q - ctrl[:5])) < tol:
                return
            if prev is not None and np.max(np.abs(q - prev)) < np.radians(0.15):
                return
            prev = q
        _pace(sim, follower, t0)


# --- cartesian --------------------------------------------------------------

def _foreign_table_contact(sim):
    table, cube = sim.geom("table"), sim.geom("cube_geom")
    for c in sim.d.contact[: sim.d.ncon]:
        pair = {c.geom1, c.geom2}
        if table in pair and cube not in pair:
            return True
    return False


def goto(sim, viewer, follower, ctrl, secs=2.5, label=None, on_step=None):
    """Stream straight to a 6-vector ctrl over `secs`, then settle."""
    return glide(sim, viewer, follower, ctrl, secs=secs, on_step=on_step, label=label)


def move_line(sim, viewer, follower, ik, calib, desired_xyz, gripper_ctrl,
              secs=3.0, warm=None, label="MOVE", stop_on_contact=False,
              correct=True, on_step=None):
    """Straight-line Cartesian move of the TCP to desired_xyz (a base-frame point;
    calib.apply_target maps it to the model goal that lands the real TCP there).
    correct=False commands the model TCP to desired_xyz directly (for XY calib).

    The path is sampled into IK waypoints ~IK_WP_LEN apart; `_stream` then plays a
    joint-space-interpolated target across them over `secs`, force=True, so the
    arm moves like HOME does. Returns (q6, status): True / "contact" / False.
    """
    desired_xyz = np.asarray(desired_xyz, dtype=float)
    start = sim.tcp_pos()
    goal = calib.apply_target(desired_xyz) if correct else desired_xyz.copy()
    n = max(2, int(np.ceil(np.linalg.norm(goal - start) / IK_WP_LEN)) + 1)

    qs = []
    q = warm
    err = 0.0
    for k in range(n):
        wp = start + (k / (n - 1)) * (goal - start)
        q, err = ik.solve(wp, warm_start=None if q is None else np.asarray(q)[:5])
        qs.append(np.asarray(q, dtype=float)[:5].copy())
    qs = np.array(qs)

    def target_fn(f):
        x = f * (n - 1)
        k = min(int(x), n - 2)
        qi = qs[k] + (x - k) * (qs[k + 1] - qs[k])
        return np.array([*qi, gripper_ctrl])

    hud.phase(label)
    if err > 0.010:
        hud.note(f"IK short {err * 1000:.0f} mm -- best effort")
    cstop = (lambda: _foreign_table_contact(sim)) if stop_on_contact else None
    final = np.array([*qs[-1], gripper_ctrl])
    r = _stream(sim, viewer, follower, target_fn, secs, contact_stop=cstop,
                on_step=on_step, settle_at=final)
    if r is False:
        return final, False
    if r == "contact":
        hud.note(f"table contact z={sim.tcp_pos()[2]:+.3f} -- stopped")
        return final, "contact"
    return final, True


def set_gripper(sim, viewer, follower, q_arm, sim_gripper_ctrl, secs=2.5,
                label="GRIPPER", on_step=None):
    """Stream only the gripper (arm held at q_arm) to a sim gripper ctrl value."""
    ctrl = np.array([*np.asarray(q_arm, float)[:5], sim_gripper_ctrl], dtype=float)
    return glide(sim, viewer, follower, ctrl, secs=secs, on_step=on_step, label=label) is not False
