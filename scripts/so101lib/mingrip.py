"""Find the lightest grip that still lifts the object.

The search, driven by the two jaw flex sensors (so101lib.touch):

  1. start with the jaws open and no sensor signal
  2. close slowly until the FIRST sensor signal appears -> first-contact position
  3. lift the arm a few cm and check the sensors still read (the object is held)
  4. if contact was lost (object dropped): put the arm back down, open, close again
     to first contact + one more STEP of jaw travel (default 5 % of the gripper
     range), and try the lift again
  5. repeat until a lift holds -> that jaw closure is the minimum grip

Steps 2-5 are `search`, which is hardware-agnostic (callables in, result out) so it
can be tested against a fake plant. `run` wires it to the arm, sim and sensors.
"""
from __future__ import annotations

import time
import types
from dataclasses import dataclass

import numpy as np

from . import forcegrip, hud, motion
from .pickplace import GRASP_TIP_Z, HOVER_TIP_Z, dur
from .pickplace_touch import DROP_GRACE, _Watch  # noqa: F401  (shared drop watcher)

STEP = 0.05          # extra jaw travel per retry, as a fraction of the full range
MAX_STEPS = 8        # retries beyond first contact (8 x 5 % = 40 % of the range)
LIFT_TIP_Z = 0.05    # tip height of the test lift (m above the table)
HOLD_SECS = 0.8      # how long the lifted object must stay felt
CLOSE_SECS = 4.0     # full-range close time while hunting first contact (slow = gentle)
SETTLE_SECS = 0.4    # let the servo and sensors settle after a closure change
DROP_FRAC = 0.5         # a lift is "dropping" when the mean sensor level falls below this
                        # fraction of its level just before the lift (or contact is lost)
DROP_POLLS = 12         # ... for this many consecutive polls (sim steps)
DROP_SECS = 0.04        # ... and, on real hardware, for at least this long
SERVO_SLOPE = 19000.0   # first guess: sensor counts per unit of jaw travel (adapts as it moves)
SERVO_MAX_STEPS = 30    # force-servo iterations before accepting the closest level reached
SERVO_MAX_DF = 0.01     # largest jaw step per iteration (fraction of full range)
SERVO_MIN_DF = 0.002    # smallest jaw step per iteration
SERVO_MAX_PAST = 0.30   # never close more than this past first contact
MAX_BLIND = 1.0      # give up if no first-contact signal by this fraction of the jaw travel


@dataclass
class Result:
    status: str                 # "found" | "no_contact" | "not_found" | "limit"
    travel: float = 0.0         # jaw travel fraction of the lightest holding grip
    levels: tuple = (0.0, 0.0)  # sensor levels (counts) while holding
    attempts: int = 0           # lifts tried
    contact_travel: float = 0.0  # jaw travel at first sensor signal
    achieved: float = 0.0       # level-mode: mean sensor level reached before the lift, % of full scale
    grip_levels: tuple = (0.0, 0.0)  # level-mode: left/right counts just before the lift


def search(find_contact, grip_to, lift_check, put_down, step=STEP, max_steps=MAX_STEPS):
    """find_contact() -> travel at first signal, or None.
    grip_to(f) -> False if a safety limit stopped the closure at travel f.
    lift_check() -> (held, levels).
    put_down(held) -> set the object down and (if not held) reopen the jaws."""
    f_c = find_contact()
    if f_c is None:
        return Result("no_contact")
    attempts = 0
    for k in range(max_steps + 1):
        f = min(1.0, f_c + k * step)
        if k > 0 and not grip_to(f):
            return Result("limit", f, attempts=attempts, contact_travel=f_c)
        attempts += 1
        held, levels = lift_check()
        if held:
            put_down(True)
            return Result("found", f, tuple(levels), attempts, f_c)
        put_down(False)
        if f >= 1.0:
            break
    return Result("not_found", attempts=attempts, contact_travel=f_c)


class ViewerClosed(Exception):
    pass


class _DropMonitor:
    """Polled every sim step during a lift (as move_line's stop_fn): trips as soon as
    the sensor value drops -- the mean level under DROP_FRAC of what it was when the
    lift began, or contact lost -- so the arm can go straight back down."""

    def __init__(self, touch, real_time):
        self.touch, self.real_time = touch, real_time
        self.ref = max(float(np.mean(touch.levels())), 1.0)
        self.count, self.t_bad, self.tripped = 0, None, False

    def __call__(self):
        lv = self.touch.levels()
        bad = float(np.mean(lv)) < DROP_FRAC * self.ref or not self.touch.any()
        if not bad:
            self.count, self.t_bad = 0, None
            return self.tripped
        self.count += 1
        self.t_bad = self.t_bad or time.time()
        if self.count >= DROP_POLLS and (not self.real_time or time.time() - self.t_bad >= DROP_SECS):
            self.tripped = True
        return self.tripped


def run(ctx, viewer, touch, pick_xy, step=STEP, max_steps=MAX_STEPS, max_current=120,
        require="both", limit_level=None, max_blind=MAX_BLIND, first_contact="both",
        extra=None, jolt_cm=0.0, level_pct=None, full_scale=600.0, session=None):
    """Run the search on the arm at pick_xy (the object sits there). Returns a Result.
    The first close gives up (status "no_contact") if no sensor signal appears within
    max_blind of the full jaw travel, so dead sensors never squeeze the object.
    first_contact="both" closes until BOTH jaw sensors read (the lift then starts with
    both already touching); "any" stops at the first sensor to read.

    extra=None runs the staircase search (retry with +step until a lift holds).
    extra=x runs ONE probe instead: close to first contact + x of jaw travel, lift,
    and return Result("held" | "dropped") -- one labelled attempt for the learner.
    level_pct=p runs ONE force-targeted probe: close to first contact, then servo the jaws
    until the mean sensor reading is p % of `full_scale` counts (backing off if first
    contact already overshoots it), lift, and return Result("held" | "dropped") with
    .achieved = the level actually reached. This is the building block of forcegrip.
    session=fn runs a CONTINUOUS grasp session: close to first contact once, then call
    fn(ctl) with ctl.attempt(phase, x_pct) -> forcegrip.Attempt. Each attempt sets the
    jaws to first contact + x_pct of the jaw travel (the jaws stay closed on the
    object -- no release, no re-descent), lifts a few cm while watching the sensors,
    and goes straight back down if the sensor value drops. fn's return value (e.g. a
    forcegrip.Outcome) is returned after the object is released.
    jolt_cm > 0 adds a lateral jolt (out, across, back; ~0.8 s per leg) at the top of
    the lift, so a marginal grip actually fails instead of surviving a gentle lift."""
    sim, follower, ik, calib = ctx["sim"], ctx["follower"], ctx["ik"], ctx["calib"]
    g_open, g_closed = ctx["g_open"], ctx["g_closed"]
    table_z, cube_hold_z = calib.table_z, ctx["cube_hold_z"]
    held_ok = touch.both if require == "both" else touch.any
    span = g_closed - g_open
    state = {"q": None}

    def at(z, dxy=(0.0, 0.0)):
        return np.array([pick_xy[0] + dxy[0], pick_xy[1] + dxy[1], table_z + z])

    hold = lambda: sim.hold_cube(pick_xy, cube_hold_z)

    def g_of(f):
        return g_open + f * span

    def check(r):
        if r is False:
            raise ViewerClosed()
        return r

    def move(z, g, secs, label, on_step, contact=False, dxy=(0.0, 0.0), stop_fn=None):
        q, s = motion.move_line(sim, viewer, follower, ik, calib, at(z, dxy), g, secs=dur(secs),
                                warm=state["q"], on_step=on_step, label=label,
                                stop_on_contact=contact, stop_fn=stop_fn)
        check(s)
        state["q"] = q
        return s

    def settle(g, on_step):
        check(motion.glide(sim, viewer, follower, np.array([*state["q"][:5], g]),
                           secs=SETTLE_SECS, on_step=on_step))

    def find_contact():
        move(HOVER_TIP_Z, g_open, 3.0, "  HOVER over object", hold)
        move(GRASP_TIP_Z, g_open, 3.5, "  DESCEND", hold, contact=True)
        last = [0.0]
        q_arm = state["q"]

        def target(frac):
            last[0] = frac * max_blind
            return np.array([*q_arm[:5], g_of(last[0])])

        hud.phase(f"  CLOSE until {'both sensors' if first_contact == 'both' else 'a sensor'} read")
        r = check(motion._stream(sim, viewer, follower, target, dur(CLOSE_SECS * max_blind),
                                 contact_stop=touch.both if first_contact == "both" else touch.any,
                                 on_step=hold))
        if r != "contact":
            motion.set_gripper(sim, viewer, follower, q_arm, g_open, label="  OPEN (no signal)",
                               on_step=hold)
            return None
        return last[0]

    def grip_to(f):
        hud.phase(f"  CLOSE to {f * 100:.0f}% travel")
        check(motion.set_gripper(sim, viewer, follower, state["q"], g_of(f), secs=dur(1.5),
                                 label=f"  CLOSE to {f * 100:.0f}% travel", on_step=hold))
        settle(g_of(f), hold)
        cur = follower.read_gripper_current() if follower.connected else None
        lv = np.asarray(touch.levels())
        if cur is not None and cur >= max_current:
            hud.note(f"gripper current {cur:.0f} >= {max_current:.0f} -- stopping search")
            return False
        if limit_level is not None and np.any(lv > np.asarray(limit_level)):
            hud.note("sensor limit level reached -- stopping search")
            return False
        return True

    def lift_check():
        g = g_of(grip_travel[0])
        grip_offset = sim.cube_pos() - sim.tcp_pos()   # virtual grasp for the sim cube

        def carry():
            p = sim.tcp_pos() + grip_offset
            sim.hold_cube(p[:2], p[2])

        watch = _Watch(touch, carry)
        move(LIFT_TIP_Z, g, 2.5, "  LIFT test", watch)
        check(motion.glide(sim, viewer, follower, np.array([*state["q"][:5], g]),
                           secs=HOLD_SECS, on_step=watch, label="  HOLD check"))
        if jolt_cm > 0 and not watch.dropped and held_ok():
            d = jolt_cm / 100.0
            for sgn in (1.0, -1.0, 0.0):   # out, across, back to centre
                move(LIFT_TIP_Z, g, 0.8, "  JOLT test", watch, dxy=(0.0, sgn * d))
        held = (not watch.dropped) and held_ok()
        lv = tuple(touch.levels())
        hud.note(f"{'HELD' if held else 'LOST'}  L{lv[0]:.0f} R{lv[1]:.0f}")
        return held, lv

    def put_down(held):
        g = g_of(grip_travel[0])
        if held:
            grip_offset = sim.cube_pos() - sim.tcp_pos()
            carry = lambda: (lambda p: sim.hold_cube(p[:2], p[2]))(sim.tcp_pos() + grip_offset)
            move(GRASP_TIP_Z + 0.005, g, 2.5, "  LOWER object back", carry)
            motion.set_gripper(sim, viewer, follower, state["q"], g_open, label="  RELEASE", on_step=hold)
            move(HOVER_TIP_Z, g_open, 2.5, "  RETREAT", hold)
        else:   # dropped: open where we are, go back down to the object
            check(motion.set_gripper(sim, viewer, follower, state["q"], g_open,
                                     label="  OPEN (dropped)", on_step=hold))
            move(GRASP_TIP_Z, g_open, 2.5, "  back DOWN to object", hold, contact=True)

    # grip_to / lift_check / put_down read the current closure from here; `search`
    # passes it to grip_to, and for k == 0 find_contact has already closed to it.
    grip_travel = [0.0]
    _grip_to, _find = grip_to, find_contact

    def find_contact_wrapped():
        f = _find()
        if f is not None:
            grip_travel[0] = f
        return f

    def grip_to_wrapped(f):
        grip_travel[0] = f
        return _grip_to(f)

    def probe(x):
        f_c = find_contact_wrapped()
        if f_c is None:
            return Result("no_contact")
        f = min(1.0, f_c + x)
        if x > 0 and not grip_to_wrapped(f):
            return Result("limit", f, attempts=1, contact_travel=f_c)
        held, levels = lift_check()
        put_down(held)
        return Result("held" if held else "dropped", f, tuple(levels), 1, f_c)

    def servo_level(target_pct):
        """Move the jaws (open or closed) until the mean sensor level is within tolerance
        of target_pct of full scale. Stops early if the reading crosses the target (the
        finest jaw step is coarser than the sensor's slope), or on a safety limit.
        Returns (ok, achieved_pct)."""
        target = target_pct / 100.0 * full_scale
        tol = max(0.02 * full_scale, 1.5 * float(np.max(touch.thresholds())))
        f_c = grip_travel[0]
        f, prev_err, slope = f_c, None, SERVO_SLOPE
        prev_f = prev_lv = None
        for _ in range(SERVO_MAX_STEPS):
            lv = float(np.mean(touch.levels()))
            err = target - lv
            if abs(err) <= tol or (prev_err is not None and err * prev_err < 0):
                break
            if prev_f is not None and abs(f - prev_f) > 1e-3 and abs(lv - prev_lv) > 5:
                slope = float(np.clip(0.5 * slope + 0.5 * abs((lv - prev_lv) / (f - prev_f)), 2000, 60000))
            prev_err, prev_f, prev_lv = err, f, lv
            df = float(np.clip(err / slope, -SERVO_MAX_DF, SERVO_MAX_DF))
            if abs(df) < SERVO_MIN_DF:
                df = SERVO_MIN_DF if df >= 0 else -SERVO_MIN_DF
            f = float(np.clip(f + df, 0.0, min(1.0, f_c + SERVO_MAX_PAST)))
            grip_travel[0] = f
            check(motion.set_gripper(sim, viewer, follower, state["q"], g_of(f), secs=dur(0.3),
                                     label=f"  SQUEEZE to {target_pct:.0f}% ({lv / full_scale * 100:.0f}% now)",
                                     on_step=hold))
            cur = follower.read_gripper_current() if follower.connected else None
            if cur is not None and cur >= max_current:
                hud.note(f"gripper current {cur:.0f} >= {max_current:.0f} -- stopping")
                return False, lv / full_scale * 100
            if limit_level is not None and np.any(np.asarray(touch.levels()) > np.asarray(limit_level)):
                hud.note("sensor limit level reached -- stopping")
                return False, lv / full_scale * 100
        settle(g_of(grip_travel[0]), hold)
        return True, float(np.mean(touch.levels())) / full_scale * 100

    def probe_level(target_pct):
        f_c = find_contact_wrapped()
        if f_c is None:
            return Result("no_contact")
        ok, ach = servo_level(target_pct)
        if not ok:
            return Result("limit", grip_travel[0], attempts=1, contact_travel=f_c, achieved=ach)
        pre = tuple(float(v) for v in touch.levels())
        held, levels = lift_check()
        put_down(held)
        return Result("held" if held else "dropped", grip_travel[0], tuple(levels), 1, f_c, ach, pre)

    def lift_attempt():
        """Lift a few cm; stop and report dropped the moment the sensor value drops."""
        g = g_of(grip_travel[0])
        grip_offset = sim.cube_pos() - sim.tcp_pos()   # virtual grasp for the sim cube

        def carry():
            p = sim.tcp_pos() + grip_offset
            sim.hold_cube(p[:2], p[2])

        mon = _DropMonitor(touch, follower.connected)
        move(LIFT_TIP_Z, g, 2.0, "  LIFT (watching sensors)", carry, stop_fn=mon)
        if not mon.tripped:
            check(motion.glide(sim, viewer, follower, np.array([*state["q"][:5], g]),
                               secs=HOLD_SECS, contact_stop=mon, on_step=carry, label="  HOLD check"))
        if not mon.tripped and jolt_cm > 0:
            d = jolt_cm / 100.0
            for sgn in (1.0, -1.0, 0.0):
                move(LIFT_TIP_Z, g, 0.8, "  JOLT test", carry, dxy=(0.0, sgn * d), stop_fn=mon)
                if mon.tripped:
                    break
        held = (not mon.tripped) and held_ok()
        lv = tuple(touch.levels())
        hud.note(f"{'HELD' if held else 'DROP detected'}  L{lv[0]:.0f} R{lv[1]:.0f}")
        return held, carry

    def lower_back(held, carry):
        g = g_of(grip_travel[0])
        if held:
            move(GRASP_TIP_Z + 0.005, g, 2.0, "  LOWER object back", carry)
        else:
            move(GRASP_TIP_Z, g, 2.0, "  back DOWN to object", hold, contact=True)

    def make_attempt(f_c):
        def attempt(phase, target_pct):
            t0 = time.time()
            f = min(1.0, f_c + target_pct / 100.0)
            if abs(f - grip_travel[0]) > 1e-4 and not grip_to_wrapped(f):
                raise forcegrip.Abort("limit")
            pre = tuple(float(v) for v in touch.levels())
            held, carry = lift_attempt()
            lower_back(held, carry)
            return forcegrip.Attempt(achieved=target_pct, held=held, left=pre[0], right=pre[1],
                                     travel=grip_travel[0], contact=f_c, t0=t0, t1=time.time())
        return attempt

    if session is not None:
        try:
            f_c = find_contact_wrapped()
            if f_c is None:
                return forcegrip.Outcome("no_contact")
            out = session(types.SimpleNamespace(f_c=f_c, attempt=make_attempt(f_c)))
            check(motion.set_gripper(sim, viewer, follower, state["q"], g_open,
                                     label="  RELEASE object", on_step=hold))
            move(HOVER_TIP_Z, g_open, 2.5, "  RETREAT", hold)
            return out
        except ViewerClosed:
            return forcegrip.Outcome("closed")

    try:
        if level_pct is not None:
            res = probe_level(level_pct)
        elif extra is not None:
            res = probe(extra)
        else:
            res = search(find_contact_wrapped, grip_to_wrapped, lift_check, put_down, step, max_steps)
        if res.status == "limit":   # stopped with the jaws still closed on the object
            check(motion.set_gripper(sim, viewer, follower, state["q"], g_open,
                                     label="  OPEN (limit)", on_step=hold))
            move(HOVER_TIP_Z, g_open, 2.5, "  RETREAT", hold)
        return res
    except ViewerClosed:
        return Result("closed")
