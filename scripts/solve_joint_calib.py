"""Resolve the pitch-joint signs (shoulder_lift, elbow_flex, wrist_flex) from
table touches.

hw_check's visual "same direction?" check is easy to get wrong, and a flipped
sign on a pitch joint makes FK compute the arm bending the wrong way -- which is
exactly what a big touch_off Z spread and a nonsense table_z mean.

This uses a hard geometric fact instead: every closed-pinch-tip touch is on the
same plane. Hand-guide the tip to the table at several very different postures;
for each of the 8 sign combinations of the three pitch joints, FK the recorded
motor angles and score how *flat* the touch points come out. The right combo
wins by a wide margin (wrong ones are 10x worse).

Signs are all this resolves. joint_offset_deg is taken as-is from the calib file
(from hw_check's HOME capture) -- table touches do not contain enough
information to pin 5 continuous offsets, only the discrete signs. shoulder_pan
(yaw) and wrist_roll barely move the TCP vertically, so their sign is left
untouched; check those with hw_check's visual test.

Run:  python scripts/solve_joint_calib.py
Raw poses are saved to config/so101_touch_poses.json.
"""
import json
import time
from itertools import product

import numpy as np

from so101lib import Calibration, Follower, fk, paths, scene
from so101lib.config import ARM_JOINT_NAMES

N_SAMPLES = 8
PITCH = [1, 2, 3]  # shoulder_lift, elbow_flex, wrist_flex
POSES_JSON = paths.CALIB_JSON.parent / "so101_touch_poses.json"


def _touch_zs(poses_deg, sign, offset):
    """FK TCP z for every captured pose under a candidate sign vector."""
    q = np.radians((poses_deg - offset) / sign)  # (N, 5) sim radians
    return np.array([fk(row)["pos"][2] for row in q])


def collect(follower, g_closed):
    print("Closing gripper...")
    for _ in range(40):
        follower.send([0, 0, 0, 0, 0, g_closed], force=True)
        time.sleep(0.05)
    input("Confirm the gripper is closed, then ENTER to release arm torque: ")
    follower.disable_torque(ARM_JOINT_NAMES)
    print("Arm torque released -- hand-guide it.\n")

    poses = []
    for k in range(N_SAMPLES):
        input(f"[{k + 1}/{N_SAMPLES}] Closed pinch tip touching the table "
              f"(very different posture each time), then ENTER: ")
        raw = follower.read_raw_deg()
        poses.append([float(raw[n]) for n in ARM_JOINT_NAMES])
        print(f"      raw deg = {[round(v, 1) for v in poses[-1]]}")
    return np.array(poses)


def score_combos(poses, calib):
    sign0 = np.array(calib.joint_sign, dtype=float)
    offset = np.array(calib.joint_offset_deg, dtype=float)
    rows = []
    for combo in product((1.0, -1.0), repeat=len(PITCH)):
        sign = sign0.copy()
        sign[PITCH] = combo
        zs = _touch_zs(poses, sign, offset)
        rows.append({"combo": [int(c) for c in combo], "sign": sign,
                     "flat": float(zs.std()), "zmean": float(zs.mean())})
    rows.sort(key=lambda r: r["flat"])
    return rows


def main():
    calib = Calibration.load()
    if all(o == 0.0 for o in calib.joint_offset_deg):
        print("joint_offset_deg is all zeros -- run hw_check.py's offset capture "
              "first, this needs it.")
        return

    sim = scene.load()
    g_closed = sim.m.actuator_ctrlrange[5][0]
    g_open = sim.m.actuator_ctrlrange[5][1]
    follower = Follower(calib, (g_closed, g_open))
    follower.connect()
    try:
        poses = collect(follower, g_closed)
    finally:
        follower.disconnect()
    POSES_JSON.write_text(json.dumps(poses.tolist(), indent=2) + "\n")
    print(f"Saved raw poses to {POSES_JSON}\n")

    rows = score_combos(poses, calib)
    print("pitch signs [shoulder_lift, elbow_flex, wrist_flex] -- flattest first:")
    for r in rows:
        print(f"  {str(r['combo']):14s} flatness {r['flat'] * 1000:6.1f} mm   "
              f"z-mean {r['zmean'] * 1000:+7.1f} mm")

    best, second = rows[0], rows[1]
    print(f"\nbest pitch signs: {best['combo']}  "
          f"(flatness {best['flat'] * 1000:.1f} mm vs next {second['flat'] * 1000:.1f} mm)")
    if best["flat"] > 0.008:
        print("WARNING: best combo still >8 mm non-flat -- touches were sloppy, or "
              "shoulder_pan/wrist_roll sign is wrong too. Recollect with more "
              "posture variety (high/low, near/far, elbow up vs down).")
    if second["flat"] < best["flat"] * 2.0:
        print("WARNING: top two combos are close -- not a confident pick.")
    if abs(best["zmean"]) > 0.015:
        print(f"\nNOTE: touch plane sits at z = {best['zmean'] * 1000:+.0f} mm, not ~0 "
              f"(base plane). Signs are right, but joint_offset_deg is off by roughly\n"
              f"that much of arm droop -- redo hw_check's HOME capture carefully "
              f"(watch the viewer) for a clean approach angle and table_z.")

    new_sign = [int(s) for s in best["sign"]]
    table_z = max(0.0, best["zmean"])
    print(f"\n  joint_sign   {calib.joint_sign}  ->  {new_sign}")
    print(f"  table_z      {calib.table_z:+.4f}  ->  {table_z:+.4f}")
    if input("\nWrite joint_sign + table_z? [y/n]: ").strip().lower().startswith("y"):
        calib.joint_sign = new_sign
        calib.table_z = table_z
        calib.notes = f"pitch signs + table_z from solve_joint_calib ({N_SAMPLES} touches)"
        calib.save()
    else:
        print("Not written.")


if __name__ == "__main__":
    main()
