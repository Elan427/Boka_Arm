"""Table-Z touch-off calibration for the real SO-101.

The arm base sits on the same table as the work object, so the table height in
the base frame is a fixed constant. This script measures where the model's TCP
site actually sits when the real closed pinch point rests on that table, and
writes it to config/so101_calib.json as `table_z`. SimToArm.py then descends to
`table_z + cube_half + offset` instead of blindly to the cube centre.

Procedure:
  1. The gripper closes, then arm torque is released so you can hand-guide it.
  2. For each of N samples, move the closed pinch tip so it just touches the
     table (use the paper-drag trick: slide paper under the tips, stop when it
     drags), then press ENTER.
  3. FK on each captured pose gives the model TCP Z; their mean is table_z.

Prereq: run scripts/hw_check.py (and scripts/solve_joint_calib.py if needed) so
joint_sign/offset are correct -- the FK here depends on them. A large spread in
the per-sample Z means they are not. XY correction is a separate step:
scripts/calibrate_xy.py.

Run: python scripts/touch_off.py
"""
import time

import numpy as np

from so101lib import Calibration, Follower, fk, scene
from so101lib.config import ARM_JOINT_NAMES

N_SAMPLES = 5
SPREAD_WARN_M = 0.003  # per-sample Z std above this => joint calibration is off


def main():
    calib = Calibration.load()
    sim = scene.load()
    g_closed = sim.m.actuator_ctrlrange[5][0]
    g_open = sim.m.actuator_ctrlrange[5][1]

    follower = Follower(calib, (g_closed, g_open))
    follower.connect()

    try:
        print("Closing gripper...")
        # Ramp: lerobot's per-tick clamp (MAX_RELATIVE_TARGET) only lets one send
        # move the gripper a few units, so a single send would not fully close it.
        for _ in range(40):
            follower.send([0, 0, 0, 0, 0, g_closed], force=True)
            time.sleep(0.05)
        input("Confirm the gripper is closed, then press ENTER to release arm torque: ")
        follower.disable_torque(ARM_JOINT_NAMES)
        print("Arm torque released -- you can now hand-guide it.\n")

        q_samples = []
        for k in range(N_SAMPLES):
            input(f"[{k + 1}/{N_SAMPLES}] Touch the closed pinch tip to the table "
                  f"(vary the posture), then press ENTER: ")
            q = follower.read_joints_rad()
            p = fk(q)["pos"]
            q_samples.append(q)
            print(f"      model TCP = [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")
    finally:
        follower.disconnect()

    zs = np.array([fk(q)["pos"][2] for q in q_samples])
    table_z = float(zs.mean())
    print("\n" + "=" * 60)
    print(f"per-sample TCP z (m): {np.round(zs, 4).tolist()}")
    print(f"table_z = mean = {table_z:+.4f} m   (std {zs.std() * 1000:.2f} mm)")
    if zs.std() > SPREAD_WARN_M:
        print(f"WARNING: spread > {SPREAD_WARN_M * 1000:.0f} mm -- run "
              f"scripts/solve_joint_calib.py, a pitch-joint sign is likely wrong.")

    calib.table_z = table_z
    calib.notes = f"table_z from {N_SAMPLES}-point touch-off"
    calib.save()
    print(f"\ndescend_z() is now {calib.descend_z():+.4f} m")


if __name__ == "__main__":
    main()
