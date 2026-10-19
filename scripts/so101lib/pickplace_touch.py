"""Touch-aware pick-and-place: pickplace.run_cycle plus gripper feedback.

Same phases as pickplace.run_cycle, but GRASP is closed-loop on the two jaw flex
sensors (so101lib.touch) instead of a blind full close:

  GRASP    close the gripper gradually; stop the moment BOTH jaws feel the cube
           (or give up at fully closed -> the cube was missed)
  TIGHTEN  creep closed a little at a time until the weaker jaw reaches
           GRIP_FORCE_TARGET (units of the touch threshold) or SQUEEZE_MAX_STEPS
  LIFT/CARRY  verify the cube is still felt; abort and re-open if not
  RELEASE  open fully

run_cycle returns one of: "ok", "closed" (viewer closed), "missed" (never touched
the cube), "dropped" (lost contact while carrying).
"""
import time

import numpy as np

from . import grip, hud, motion
from .pickplace import GRASP_TIP_Z, HOVER_TIP_Z, LIFT_TIP_Z, dur

GRIP_FORCE_TARGET = 3.0     # weaker jaw's force, in touch-threshold units
SQUEEZE_STEP = 0.04         # fraction of the gripper range per TIGHTEN step
SQUEEZE_MAX_STEPS = 6
CLOSE_SECS = 3.5            # full-range close time; slow = gentle contact
DROP_GRACE = 0.25           # s of lost contact tolerated before "dropped"


def _close_until_touch(ctx, viewer, touch, q_arm, on_step):
    """Stream the gripper closed, stopping when both jaws touch. Returns
    (gripper_ctrl_at_stop, felt), or (None, None) if the viewer closed."""
    sim, follower = ctx["sim"], ctx["follower"]
    g_open, g_closed = ctx["g_open"], ctx["g_closed"]
    last = [g_open]

    def target(f):
        g = g_open + f * (g_closed - g_open)
        last[0] = g
        return np.array([*q_arm[:5], g])

    hud.phase("  GRASP (closing until both jaws feel it)")
    r = motion._stream(sim, viewer, follower, target, dur(CLOSE_SECS),
                       contact_stop=touch.both, on_step=on_step)
    if r is False:
        return None, None
    return last[0], r == "contact"


def _tighten(ctx, viewer, touch, q_arm, g, on_step, profile=None):
    """Squeeze to the grip force. With a learned GripProfile (train_grip.py) this is a
    closed loop on the sensor levels in counts (target and limit from the profile);
    without one, creep closed until the weaker jaw reaches GRIP_FORCE_TARGET.
    Returns the final gripper ctrl, or None if the viewer closed."""
    sim, follower = ctx["sim"], ctx["follower"]
    g_open, g_closed = ctx["g_open"], ctx["g_closed"]
    span = g_closed - g_open
    step = SQUEEZE_STEP * span
    lo, hi = min(g_open, g_closed), max(g_open, g_closed)
    if profile is not None:
        for _ in range(SQUEEZE_MAX_STEPS * 3):
            d = grip.squeeze_step(profile, touch.levels(), SQUEEZE_STEP)
            if d == 0.0:
                break
            g = float(np.clip(g + d * span, lo, hi))
            if not motion.set_gripper(sim, viewer, follower, q_arm, g, secs=dur(0.5),
                                      label="  TIGHTEN (learned grip force)", on_step=on_step):
                return None
        lv = touch.levels()
        hud.note(f"grip L{lv[0]:.0f}/{profile.target_level[0]:.0f} R{lv[1]:.0f}/{profile.target_level[1]:.0f}")
        return g
    for _ in range(SQUEEZE_MAX_STEPS):
        if min(touch.force()) >= GRIP_FORCE_TARGET:
            break
        g = float(np.clip(g + step, lo, hi))
        if not motion.set_gripper(sim, viewer, follower, q_arm, g, secs=dur(0.5),
                                  label="  TIGHTEN (grip force)", on_step=on_step):
            return None
    fl, fr = touch.force()
    hud.note(f"grip force L{fl:.1f} R{fr:.1f}")
    return g


class _Watch:
    """on_step wrapper that flags a drop once contact has been gone DROP_GRACE s."""

    def __init__(self, touch, inner):
        self.touch, self.inner = touch, inner
        self.lost_since = None
        self.dropped = False

    def __call__(self):
        self.inner()
        if self.touch.any():
            self.lost_since = None
            return
        now = time.time()
        self.lost_since = self.lost_since or now
        if now - self.lost_since > DROP_GRACE:
            self.dropped = True
            hud.note("CONTACT LOST")


def run_cycle(ctx, viewer, pick_xy, place_xy, touch, profile=None):
    sim, follower, ik, calib = ctx["sim"], ctx["follower"], ctx["ik"], ctx["calib"]
    g_open, g_closed = ctx["g_open"], ctx["g_closed"]
    table_z, cube_hold_z = calib.table_z, ctx["cube_hold_z"]
    use_touch = touch is not None and touch.available

    def at(xy, tip_z):
        return np.array([xy[0], xy[1], table_z + tip_z])

    hold = lambda: sim.hold_cube(pick_xy, cube_hold_z)

    q, s = motion.move_line(sim, viewer, follower, ik, calib, at(pick_xy, HOVER_TIP_Z),
                            g_open, secs=dur(3.0), on_step=hold, label="  HOVER over cube")
    if s is not True:
        return "closed"
    q, s = motion.move_line(sim, viewer, follower, ik, calib, at(pick_xy, GRASP_TIP_Z),
                            g_open, secs=dur(3.5), warm=q, on_step=hold,
                            label="  DESCEND to cube", stop_on_contact=True)
    if s is False:
        return "closed"

    grasp_q = q
    if use_touch:
        g, felt = _close_until_touch(ctx, viewer, touch, grasp_q, hold)
        if g is None:
            return "closed"
        if not felt:
            hud.note("no touch before fully closed -- cube missed")
            motion.set_gripper(sim, viewer, follower, grasp_q, g_open,
                               label="  OPEN (missed)", on_step=hold)
            return "missed"
        g_hold = _tighten(ctx, viewer, touch, grasp_q, g, hold, profile)
        if g_hold is None:
            return "closed"
    else:
        motion.set_gripper(sim, viewer, follower, grasp_q, g_closed, secs=dur(1.8),
                           label="  GRASP", on_step=hold)
        g_hold = g_closed

    # Virtual grasp for the sim cube (see pickplace.run_cycle); hardware holds for real.
    grip_offset = sim.cube_pos() - sim.tcp_pos()

    def carry():
        p = sim.tcp_pos() + grip_offset
        sim.hold_cube(p[:2], p[2])

    watch = _Watch(touch, carry) if use_touch else carry

    def lost():
        return use_touch and (watch.dropped or not touch.any())

    q, s = motion.move_line(sim, viewer, follower, ik, calib, at(pick_xy, LIFT_TIP_Z), g_hold,
                            secs=dur(3.0), warm=grasp_q, on_step=watch, label="  LIFT")
    if s is not True:
        return "closed"
    if lost():
        motion.set_gripper(sim, viewer, follower, q, g_open, label="  OPEN (dropped)", on_step=carry)
        return "dropped"

    q, s = motion.move_line(sim, viewer, follower, ik, calib, at(place_xy, LIFT_TIP_Z), g_hold,
                            secs=dur(4.5), warm=q, on_step=watch, label="  CARRY to drop spot")
    if s is not True:
        return "closed"
    if lost():
        motion.set_gripper(sim, viewer, follower, q, g_open, label="  OPEN (dropped)", on_step=carry)
        return "dropped"

    q, s = motion.move_line(sim, viewer, follower, ik, calib, at(place_xy, GRASP_TIP_Z + 0.005),
                            g_hold, secs=dur(3.0), warm=q, on_step=carry,
                            label="  LOWER to place")
    if s is False:
        return "closed"

    rest = lambda: sim.hold_cube(place_xy, cube_hold_z)
    motion.set_gripper(sim, viewer, follower, q, g_open, label="  RELEASE", on_step=rest)
    motion.goto(sim, viewer, follower, np.array([*q[:5], g_open]), secs=dur(1.0), on_step=rest)
    motion.move_line(sim, viewer, follower, ik, calib, at(place_xy, LIFT_TIP_Z), g_open,
                     secs=dur(2.5), warm=q, on_step=rest, label="  RETREAT")
    return "ok"
