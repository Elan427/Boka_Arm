"""Fit the base-frame XY correction for the real SO-101.

The sim IK is accurate to ~1 mm, so when the real gripper lands to the side of
where a phase asked, the sim->real map is rotated/shifted -- mostly a
shoulder_pan offset error (rotates the whole workspace) plus a constant shift.

This drives the closed tip to a spread of known MODEL points on the table, you
measure where the tip actually is, and it fits

    real_xy ~= Rot(xy_rot_deg) @ model_xy + tcp_xy_bias

Both go in config/so101_calib.json and are undone by calib.apply_target, so
after this every phase's XY lands where it asks.

Prereqs: hw_check.py (+ solve_joint_calib.py) and touch_off.py done -- FK and
table_z must be right.

Run: python scripts/calibrate_xy.py
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

from so101lib import Calibration, Follower, SO101IK, motion, scene

# MODEL-space TCP targets to visit, spread wide for a good rotation moment arm.
TARGETS_XY = [
    (0.16, 0.14), (0.16, -0.14), (0.24, 0.00),
    (0.12, 0.10), (0.12, -0.10), (0.20, 0.13),
]
HOVER_Z = 0.06  # above table_z, model space


def main():
    calib = Calibration.load()
    sim = scene.load()
    g_closed = sim.m.actuator_ctrlrange[5][0]
    g_open = sim.m.actuator_ctrlrange[5][1]
    ik = SO101IK()
    follower = Follower(calib, (g_closed, g_open))
    follower.connect()
    print("\n!! REAL HARDWARE WILL MOVE !!")
    print("Hand-position the arm roughly at HOME (it is limp).")
    input("Clear the workspace, then ENTER (Ctrl+C to abort): ")

    home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, g_open])
    tz = calib.table_z
    model_xy, real_xy = [], []

    try:
        with mujoco.viewer.launch_passive(sim.m, sim.d) as viewer:
            print("\nHOME")
            motion.home(sim, viewer, follower, home)

            print("Closing gripper...")
            for _ in range(40):
                follower.send([0, 0, 0, 0, 0, g_closed], force=True)
                time.sleep(0.05)

            q = None
            for i, (tx, ty) in enumerate(TARGETS_XY, 1):
                if not viewer.is_running():
                    break
                q, _ = motion.move_line(sim, viewer, follower, ik, calib,
                                        [tx, ty, tz + HOVER_Z], g_closed, warm=q,
                                        correct=False, label=f"[{i}/{len(TARGETS_XY)}] over ({tx:+.2f}, {ty:+.2f})")
                q, _ = motion.move_line(sim, viewer, follower, ik, calib,
                                        [tx, ty, tz + 0.006], g_closed, secs=2.5, warm=q,
                                        correct=False, label="  down to just above the table")
                got = sim.tcp_pos()[:2]
                raw = input(f"  measure the real tip (x y) from the base in mm "
                            f"[model ({got[0] * 1000:.0f} {got[1] * 1000:.0f})]: ").strip()
                motion.move_line(sim, viewer, follower, ik, calib,
                                 [tx, ty, tz + HOVER_Z], g_closed, secs=2.0, warm=q,
                                 correct=False, label="  up")
                if not raw:
                    print("  skipped")
                    continue
                try:
                    mx, my = (float(v) / 1000.0 for v in raw.replace(",", " ").split())
                except ValueError:
                    print("  unparsable -- skipped")
                    continue
                model_xy.append(got)
                real_xy.append([mx, my])

            print("\nHOME")
            motion.home(sim, viewer, follower, home)
    finally:
        follower.disconnect()

    if len(model_xy) < 3:
        print(f"\nOnly {len(model_xy)} points -- need >=3. Nothing written.")
        return

    rot_deg, bias, rms = Calibration.fit_xy_correction(model_xy, real_xy)
    print("\n" + "=" * 60)
    print(f"points used     : {len(model_xy)}")
    print(f"xy_rot_deg      : {calib.xy_rot_deg:+.2f}  ->  {rot_deg:+.2f}")
    print(f"tcp_xy_bias (mm): {np.round(np.array(calib.tcp_xy_bias) * 1000, 1).tolist()}  ->  "
          f"{np.round(np.array(bias) * 1000, 1).tolist()}")
    print(f"fit residual    : {rms * 1000:.1f} mm rms")
    if rms > 0.008:
        print("WARNING: residual > 8 mm -- measurements noisy, or the error is not a "
              "rigid transform (a joint offset is still wrong). Re-measure carefully.")

    if input("\nWrite xy_rot_deg + tcp_xy_bias? [y/n]: ").strip().lower().startswith("y"):
        calib.xy_rot_deg = rot_deg
        calib.tcp_xy_bias = bias
        calib.notes = f"xy correction from {len(model_xy)}-point calibrate_xy"
        calib.save()
    else:
        print("Not written.")


if __name__ == "__main__":
    main()
