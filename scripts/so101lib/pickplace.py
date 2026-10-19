"""The SO-101 pick-and-place task, shared by the two entry-point scripts:

  * scripts/SimToArm.py      -- one pick-and-place, then stop
  * scripts/SimToArmLoop.py  -- pick/drop forever at random reachable spots

`run_cycle` is one full cycle (HOVER -> DESCEND -> GRASP -> LIFT -> CARRY ->
LOWER -> RELEASE -> RETREAT); the scripts wrap it with HOME moves and decide
where the cube goes. Heights are the REAL pinch-tip height above the calibrated
table; every XY target is mapped through calib.apply_target so the real TCP
lands where the phase asks.
"""
import logging

import numpy as np

from . import hud, motion, scene
from .calib import Calibration
from .config import ARM_JOINT_NAMES, CUBE_HALF
from .hardware import Follower
from .ik import SO101IK

# Motion speed multiplier. 1.0 = the original per-phase timing; higher is faster
# (phase durations scale by 1/SPEED). Slower = smoother on hardware. Bump this,
# MAX_RELATIVE_TARGET and HW_SEND_HZ together and gently -- SPEED 2.0 + 6 deg/tick
# tripped the shoulder_pan overload latch mid-GRASP.
SPEED = 1.0

# Heights are the REAL pinch-tip height above the calibrated table (metres).
# Short arm: at a typical cube position the tip only reaches ~10 cm up.
HOVER_TIP_Z = 0.075     # tip height when hovering, before the descent
LIFT_TIP_Z = 0.075      # tip height to lift the grasped cube to
GRASP_TIP_Z = CUBE_HALF  # tip at the cube's mid-height to grasp
# Extra height when lowering the carried cube to release. The table is only flat
# to ~5-7 mm across the reachable area, so lowering to the full grasp height can
# press the cube into the table while the arm holds still -> overload latch.
# Dropping it from ~1 cm is harmless.
PLACE_CLEARANCE = 0.012

# Random drop targets (SimToArmLoop) are sampled in the base frame: polar radius
# (m) and angle (deg from +x). The IK reachability check trims the unreachable.
DROP_RADIUS = (0.15, 0.24)
DROP_ANGLE_DEG = (5.0, 85.0)
DROP_MIN_MOVE = 0.06     # a new drop must be at least this far from the pick spot


class _DropClampWarning(logging.Filter):
    def filter(self, record):
        return "had to be clamped to be safe" not in record.getMessage()


def silence_clamp_warnings():
    logging.getLogger().addFilter(_DropClampWarning())


def dur(secs):
    """Phase duration after the global SPEED scaling."""
    return secs / SPEED


# --- setup -----------------------------------------------------------------

def build(hardware_enabled):
    """Load calib + sim + IK, open the follower if asked. Returns a dict with
    everything run_cycle and the scripts need."""
    calib = Calibration.load()
    sim = scene.load()
    m = sim.m
    g_open = m.actuator_ctrlrange[5][1]
    g_closed = m.actuator_ctrlrange[5][0]
    ik = SO101IK()
    follower = Follower(calib, (g_closed, g_open))

    if hardware_enabled:
        follower.connect()
        print("\n!! REAL HARDWARE WILL MOVE !!")
        print("The arm is limp -- hand-position it roughly at HOME first so the first")
        print("move is a short ramp, not a long clamped crawl.")
        input("Clear the workspace, then ENTER to start (Ctrl+C to abort): ")

    return dict(calib=calib, sim=sim, ik=ik, follower=follower,
                m=m, d=sim.d, g_open=g_open, g_closed=g_closed,
                home=np.array([0.0, 0.0, 0.0, 0.0, 0.0, g_open]),
                cube_hold_z=calib.table_z + CUBE_HALF)


def go_home(ctx, viewer, on_step=None, prompt_on_fail=True):
    """Ramp sim + real arm to HOME. Returns True if it landed cleanly."""
    hud.phase("HOME")
    ok, errs = motion.home(ctx["sim"], viewer, ctx["follower"], ctx["home"], on_step=on_step)
    if not ok and prompt_on_fail:
        hud.done()
        print("did not reach HOME cleanly -- Ctrl+C now if the arm is stuck.")
        if errs is not None:
            worst = ", ".join(f"{n} {np.degrees(e):.1f}deg"
                              for n, e in zip(ARM_JOINT_NAMES, errs))
            print(f"  remaining error: {worst}")
        try:
            print("  motor        torque  status  load  temp  volt(0.1V)  raw_pos")
            for name, r in ctx["follower"].diagnose():
                f = lambda k: "--" if r[k] is None else r[k]
                print(f"  {name:<12} {f('Torque_Enable'):>6}  {f('Status'):>6}  "
                      f"{f('Present_Load'):>4}  {f('Present_Temperature'):>4}  "
                      f"{f('Present_Voltage'):>10}  {f('Present_Position'):>7}")
        except Exception as e:
            print(f"  (diagnose failed: {e})")
        input("ENTER to continue anyway: ")
    return ok


# --- random drop target --------------------------------------------------

def _reachable(ik, calib, xy, tip_z, tol=0.008):
    goal = calib.apply_target([xy[0], xy[1], calib.table_z + tip_z])
    _, err = ik.solve(goal)
    return err < tol


def random_table_xy(ik, calib, rng, avoid_xy, tries=300):
    """Sample a base-frame XY the arm can both hover over and reach down to,
    kept clear of avoid_xy. Raises if nothing lands after `tries` samples."""
    avoid_xy = np.asarray(avoid_xy)
    for _ in range(tries):
        r = rng.uniform(*DROP_RADIUS)
        th = np.radians(rng.uniform(*DROP_ANGLE_DEG))
        xy = np.array([r * np.cos(th), r * np.sin(th)])
        if np.linalg.norm(xy - avoid_xy) < DROP_MIN_MOVE:
            continue
        if _reachable(ik, calib, xy, HOVER_TIP_Z) and _reachable(ik, calib, xy, GRASP_TIP_Z):
            return xy
    raise RuntimeError("could not sample a reachable table point -- widen DROP_RADIUS/ANGLE")


# --- one pick-and-place cycle ------------------------------------------------

def run_cycle(ctx, viewer, pick_xy, place_xy):
    """Pick the cube at pick_xy and drop it at place_xy. Returns True if the
    cycle finished, False if the viewer was closed mid-cycle."""
    sim, follower, ik, calib = ctx["sim"], ctx["follower"], ctx["ik"], ctx["calib"]
    g_open, g_closed = ctx["g_open"], ctx["g_closed"]
    table_z = calib.table_z
    cube_hold_z = ctx["cube_hold_z"]

    def at(xy, tip_z):
        return np.array([xy[0], xy[1], table_z + tip_z])

    hold = lambda: sim.hold_cube(pick_xy, cube_hold_z)

    q, s = motion.move_line(sim, viewer, follower, ik, calib,
                            at(pick_xy, HOVER_TIP_Z), g_open, secs=dur(3.0),
                            on_step=hold, label="  HOVER over cube")
    if s is not True:
        return False

    q, s = motion.move_line(sim, viewer, follower, ik, calib,
                            at(pick_xy, GRASP_TIP_Z), g_open, secs=dur(3.5),
                            warm=q, on_step=hold, label="  DESCEND to cube",
                            stop_on_contact=True)
    if s is False:
        return False

    grasp_q = q
    motion.set_gripper(sim, viewer, follower, grasp_q, g_closed, secs=dur(1.8),
                       label="  GRASP", on_step=hold)
    # The sim's proxy jaw pads make point contact and drop the cube on any
    # non-vertical move, so once "grasped" the cube is carried by pinning it to
    # the TCP (a virtual grasp). The real gripper holds for real; the follower
    # only ever sees joint targets, so hardware is unaffected.
    grip_offset = sim.cube_pos() - sim.tcp_pos()

    def carry():
        p = sim.tcp_pos() + grip_offset
        sim.hold_cube(p[:2], p[2])

    q, s = motion.move_line(sim, viewer, follower, ik, calib,
                            at(pick_xy, LIFT_TIP_Z), g_closed, secs=dur(3.0),
                            warm=grasp_q, on_step=carry, label="  LIFT")
    if s is not True:
        return False

    q, s = motion.move_line(sim, viewer, follower, ik, calib,
                            at(place_xy, LIFT_TIP_Z), g_closed, secs=dur(4.5),
                            warm=q, on_step=carry, label="  CARRY to drop spot")
    if s is not True:
        return False

    q, s = motion.move_line(sim, viewer, follower, ik, calib,
                            at(place_xy, GRASP_TIP_Z + PLACE_CLEARANCE), g_closed, secs=dur(3.0),
                            warm=q, on_step=carry, label="  LOWER to place")
    if s is False:
        return False

    rest = lambda: sim.hold_cube(place_xy, cube_hold_z)
    motion.set_gripper(sim, viewer, follower, q, g_open, label="  RELEASE", on_step=rest)
    motion.goto(sim, viewer, follower, np.array([*q[:5], g_open]), secs=dur(1.0),
                on_step=rest)
    motion.move_line(sim, viewer, follower, ik, calib,
                     at(place_xy, LIFT_TIP_Z), g_open, secs=dur(2.5),
                     warm=q, on_step=rest, label="  RETREAT")
    return True
