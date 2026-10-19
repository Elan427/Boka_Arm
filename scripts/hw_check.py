"""Verify the sim -> real joint convention for the SO-101.

SimToArm.py assumes sim radians map to real motor degrees with a per-joint sign
and offset (config/so101_calib.json: joint_sign, joint_offset_deg):

    real_deg = joint_sign[i] * degrees(q_rad[i]) + joint_offset_deg[i]

This script calibrates both halves of that map:

  1. joint_offset_deg -- torque is released and you hand-pose the real arm to
     match the sim's HOME posture in the viewer; the raw motor readings at that
     pose become the offsets, so a commanded sim-HOME reproduces it. lerobot's
     own calibration zeroes each joint at the *middle of its range*, which is
     NOT the sim HOME keyframe, so without this step a commanded HOME drives the
     arm into a random (often unreachable) posture and every check below starts
     from a mismatched pose. Do this step unless the offsets are already good.
  2. joint_sign -- a known move is commanded on each joint, on the sim and the
     real arm together, and you confirm they moved the same way; the sign is
     flipped when they did not.
  3. gripper open/closed direction.

Torque is ENABLED for the sign/gripper checks: the real arm moves. Keep the
workspace clear and a hand on Ctrl+C.

Run: python scripts/hw_check.py
"""
import logging
import time

import mujoco
import mujoco.viewer
import numpy as np

from so101lib import Calibration, Follower, scene
from so101lib.config import ARM_JOINT_NAMES

TEST_ANGLE_RAD = 0.25
RAMP_SECONDS = 2.0
HOME_TOL_DEG = 3.0        # "close enough" to HOME to start the checks
HOME_TIMEOUT_S = 45.0     # give up ramping to HOME after this
STALL_WINDOW = 40         # loops (~2 s) of no progress => stalled


def _quiet_relative_clamp_warning():
    """lerobot logs a WARNING on every send whose target is >max_relative_target
    from the present position. That is expected during any multi-second ramp
    (it IS the ramp), so drop just that one message -- everything else stays.
    """
    msg = "had to be clamped to be safe"

    class _Filter(logging.Filter):
        def filter(self, record):
            return msg not in record.getMessage()

    root = logging.getLogger()
    root.addFilter(_Filter())
    for h in root.handlers:
        h.addFilter(_Filter())


def _ask_yes(prompt, default_no=True):
    ans = input(prompt).strip().lower()
    if not ans:
        return not default_no
    return ans.startswith("y")


def _mirror_move(viewer, sim, follower, ctrl_from, ctrl_to, secs, hold=0.0):
    """Interpolate ctrl from ctrl_from to ctrl_to over `secs` wall-clock seconds,
    then keep commanding ctrl_to for a further `hold` seconds, stepping the sim
    and mirroring every step to the real arm.
    """
    ctrl_from = np.asarray(ctrl_from, dtype=float)
    ctrl_to = np.asarray(ctrl_to, dtype=float)
    steps = max(1, int(secs / sim.m.opt.timestep))
    hold_steps = max(0, int(hold / sim.m.opt.timestep))
    for s in range(steps + 1 + hold_steps):
        a = min(1.0, s / steps)
        ctrl = (1 - a) * ctrl_from + a * ctrl_to
        sim.d.ctrl[:6] = ctrl
        t0 = time.time()
        mujoco.mj_step(sim.m, sim.d)
        viewer.sync()
        follower.send(ctrl)
        dt = sim.m.opt.timestep - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


def _settle(viewer, sim, follower, ctrl, tol_deg=1.5, timeout=6.0):
    """Hold `ctrl` until the real arm's joints converge to it -- or stop moving --
    so a "same direction?" question is never asked while the arm is still moving.
    The real position servos lag the commanded ramp under load (wrist/elbow
    especially), and the per-tick clamp stretches a large move out further.
    """
    ctrl = np.asarray(ctrl, dtype=float)
    tol = np.radians(tol_deg)
    t_end = time.time() + timeout
    prev = None
    while time.time() < t_end and viewer.is_running():
        loop_t = time.time()
        sim.d.ctrl[:6] = ctrl
        mujoco.mj_step(sim.m, sim.d)
        viewer.sync()
        follower.send(ctrl, force=True)
        try:
            q = follower.read_joints_rad()
        except Exception:
            q = None
        if q is not None:
            if np.max(np.abs(q - ctrl[:5])) < tol:
                return
            if prev is not None and np.max(np.abs(q - prev)) < np.radians(0.15):
                return  # no longer moving -- as settled as it will get
            prev = q
        dt = 1.0 / 20 - (time.time() - loop_t)
        if dt > 0:
            time.sleep(dt)


def capture_offsets(calib, follower, viewer, sim, home):
    """Set joint_offset_deg so a commanded sim-HOME reproduces the real arm's
    HOME pose. Torque is released; you hand-pose the arm to match the sim.
    """
    sim.d.ctrl[:6] = home
    for _ in range(200):
        mujoco.mj_step(sim.m, sim.d)
        viewer.sync()

    follower.disable_torque(ARM_JOINT_NAMES)
    print("\nArm torque released. Hand-pose the REAL arm so every joint matches")
    print("the sim's HOME posture shown in the viewer (arm straight, joints centred).")
    input("Press ENTER when the real arm matches the sim HOME: ")

    raw = follower.read_raw_deg()
    offsets = [float(raw[name]) for name in ARM_JOINT_NAMES]
    calib.joint_offset_deg = offsets
    print(f"      joint_offset_deg <- {[round(o, 1) for o in offsets]}")

    follower.enable_torque(ARM_JOINT_NAMES)
    print("      torque re-enabled -- the arm now holds HOME.\n")


def home_arm(viewer, sim, follower, home):
    """Drive sim and real arm to HOME. The real arm rides lerobot's own per-tick
    clamp toward the fixed HOME target; a single send() would only move each
    motor MAX_RELATIVE_TARGET degrees. Returns (reached, errs_rad|None).

    Detects a stall (present position stops changing while still far from HOME)
    and bails instead of spinning for the full timeout.
    """
    sim.d.ctrl[:6] = home
    tol = np.radians(HOME_TOL_DEG)
    t_end = time.time() + HOME_TIMEOUT_S
    prev_err = None
    stalled = 0
    errs = None
    while time.time() < t_end and viewer.is_running():
        loop_t = time.time()
        mujoco.mj_step(sim.m, sim.d)
        viewer.sync()
        follower.send(home, force=True)
        try:
            errs = np.abs(follower.read_joints_rad() - home[:5])
        except Exception:
            errs = None
        if errs is not None:
            err = float(errs.max())
            if err < tol:
                return True, errs
            if prev_err is not None and prev_err - err < np.radians(0.05):
                stalled += 1
            else:
                stalled = 0
            prev_err = err
            if stalled >= STALL_WINDOW:
                return False, errs
        dt = 1.0 / 20 - (time.time() - loop_t)
        if dt > 0:
            time.sleep(dt)
    return False, errs


def main():
    _quiet_relative_clamp_warning()
    calib = Calibration.load()
    sim = scene.load()
    g_closed = sim.m.actuator_ctrlrange[5][0]
    g_open = sim.m.actuator_ctrlrange[5][1]
    home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, g_open])

    follower = Follower(calib, (g_closed, g_open))
    follower.connect()

    ranges = getattr(follower._robot, "calibration", {}) or {}
    if ranges:
        print("lerobot motor ranges (raw ticks):")
        for name in ARM_JOINT_NAMES + ["gripper"]:
            c = ranges.get(name)
            if c is not None:
                print(f"  {name:14s} min={c.range_min:5d} max={c.range_max:5d} homing={c.homing_offset}")
        print()

    offsets_zero = all(o == 0.0 for o in calib.joint_offset_deg)
    if offsets_zero:
        print("joint_offset_deg is all zeros: a commanded HOME will target raw motor")
        print("zeros, which is usually NOT the sim HOME posture. Capturing offsets is")
        print("strongly recommended.\n")

    try:
        with mujoco.viewer.launch_passive(sim.m, sim.d) as viewer:
            if _ask_yes("Capture joint_offset_deg now? (torque releases so you can "
                        "hand-pose the arm to the sim HOME) [y/n]: ",
                        default_no=not offsets_zero):
                capture_offsets(calib, follower, viewer, sim, home)
            else:
                print(f"Keeping joint_offset_deg = {calib.joint_offset_deg}")

            print("\nRamping the real arm to HOME (watch it, hand on Ctrl+C)...")
            reached, errs = home_arm(viewer, sim, follower, home)
            if errs is not None:
                print("  per-joint HOME error (deg): " + ", ".join(
                    f"{n}={np.degrees(e):+.1f}" for n, e in zip(ARM_JOINT_NAMES, errs)))
            if not reached:
                print("  WARNING: arm did not reach HOME (stall or joint limit).")
                if offsets_zero:
                    print("  Most likely cause: joint_offset_deg is all zeros. Ctrl+C, re-run,")
                    print("  and do the 'Capture joint_offset_deg' step this time.")
                if not _ask_yes("  continue with the checks anyway? [y/N]: "):
                    return

            input("Arm at HOME on sim and real. Press ENTER to begin joint checks: ")

            for i, name in enumerate(ARM_JOINT_NAMES):
                if not viewer.is_running():
                    break
                target = home.copy()
                target[i] = calib.joint_sign[i] * TEST_ANGLE_RAD
                delta_deg = calib.joint_sign[i] * np.degrees(TEST_ANGLE_RAD)
                print(f"\n[{i + 1}/5] {name}: sim moving {delta_deg:+.0f} deg")
                _mirror_move(viewer, sim, follower, home, target, RAMP_SECONDS)
                _settle(viewer, sim, follower, target)
                same = _ask_yes(f"      did the REAL {name} move the SAME direction as the sim? [y/n]: ")
                if not same:
                    calib.joint_sign[i] = -calib.joint_sign[i]
                    print(f"      flipped joint_sign[{i}] -> {calib.joint_sign[i]}")
                _mirror_move(viewer, sim, follower, target, home, RAMP_SECONDS)
                _settle(viewer, sim, follower, home)

            # Gripper: sweep OPEN<->CLOSED, let the user dial in the real
            # end-stop percentages until it matches (handles a gripper whose
            # calibrated range is rotated/scaled off the sim's). Commands stop
            # ~8% short of the extremes and hold only briefly, so a miscalibrated
            # end does not stall the motor into an overload fault.
            g_mid = 0.5 * (g_open + g_closed)

            def _safe(ctrl):
                return ctrl + 0.08 * (g_mid - ctrl)

            while viewer.is_running():
                for g_ctrl, word, other in ((g_open, "OPEN", g_closed),
                                            (g_closed, "CLOSED", g_open)):
                    if not viewer.is_running():
                        break
                    pct = (calib.gripper_open_pct if word == "OPEN"
                           else calib.gripper_closed_pct)
                    print(f"\n[gripper] commanding {word} (pct {pct:.0f})")
                    _mirror_move(viewer, sim, follower,
                                 [0, 0, 0, 0, 0, _safe(other)], [0, 0, 0, 0, 0, _safe(g_ctrl)],
                                 2.0, hold=0.4)
                _mirror_move(viewer, sim, follower,
                             [0, 0, 0, 0, 0, _safe(g_closed)], [0, 0, 0, 0, 0, g_mid], 0.6)
                if not viewer.is_running():
                    break
                if _ask_yes("      does the real gripper match the sim OPEN and CLOSED? [y/n]: "):
                    break
                raw = input("      'm' measure end-stops by hand, 's' swap, "
                            "'<open> <closed>' set pct directly, ENTER to leave as-is: ").strip()
                if not raw:
                    break
                if raw.lower() == "m":
                    follower.disable_torque(["gripper"])
                    print("      gripper torque released.")
                    input("      hold the jaw where the sim shows CLOSED, then ENTER: ")
                    c = float(follower.read_raw_deg()["gripper"])
                    input("      hold the jaw where the sim shows OPEN, then ENTER: ")
                    o = float(follower.read_raw_deg()["gripper"])
                    follower.enable_torque(["gripper"])
                    calib.gripper_open_pct, calib.gripper_closed_pct = o, c
                elif raw.lower() == "s":
                    calib.gripper_open_pct, calib.gripper_closed_pct = (
                        calib.gripper_closed_pct, calib.gripper_open_pct)
                else:
                    try:
                        o, c = (float(v) for v in raw.replace(",", " ").split())
                        calib.gripper_open_pct, calib.gripper_closed_pct = o, c
                    except ValueError:
                        print("      (could not parse -- unchanged)")
                print(f"      gripper pct -> open={calib.gripper_open_pct} "
                      f"closed={calib.gripper_closed_pct}")
    finally:
        follower.disconnect()

    calib.notes = "joint offset/sign + gripper direction from hw_check"
    calib.save()
    print(f"\njoint_sign       = {calib.joint_sign}")
    print(f"joint_offset_deg = {calib.joint_offset_deg}")


if __name__ == "__main__":
    main()
