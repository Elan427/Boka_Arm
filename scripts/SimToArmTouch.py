"""SO-101 sim -> real: pick-and-place with gripper touch feedback.

Like SimToArmLoop.py (pick the cube, drop it at a random reachable spot, repeat),
but the gripper has two flex sensors (one per jaw, on an Arduino -- see
firmware/gripper_touch and so101lib/touch.py) and the grasp is closed-loop:

  * the gripper closes until BOTH jaws feel the cube, then tightens to a target
    grip force instead of blindly closing all the way
  * a missed grasp (fully closed, no touch) is detected, the gripper re-opens, and
    the cycle retries from HOME
  * losing contact while lifting/carrying aborts the carry instead of "placing"
    nothing

Run scripts/train_grip.py once per object to learn the grip force (config/grip_profile.json).
Flash firmware/gripper_touch to the Arduino first. Keep the jaws open and
untouched while the sensors tare at start-up.

Run:  python scripts/SimToArmTouch.py [--port COM3] [--once] [--no-sensors]
      python scripts/touch_monitor.py        # wiring check, no arm needed
"""
import argparse

import mujoco
import mujoco.viewer
import numpy as np

from so101lib import grip, hud, pickplace, pickplace_touch
from so101lib.touch import TouchSensors

HARDWARE_ENABLED = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3", help="Arduino serial port")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="extra sensor smoothing: weight of the newest sample (0 = off, 0.1 = heavy)")
    ap.add_argument("--once", action="store_true", help="one successful cycle, then stop")
    ap.add_argument("--no-sensors", action="store_true",
                    help="run without touch feedback (falls back to a blind close)")
    args = ap.parse_args()

    touch = None
    if not args.no_sensors:
        touch = TouchSensors(args.port, smooth=args.smooth)
        touch.start()

    pickplace.silence_clamp_warnings()
    ctx = pickplace.build(HARDWARE_ENABLED)
    sim, follower, ik, calib = ctx["sim"], ctx["follower"], ctx["ik"], ctx["calib"]
    rng = np.random.default_rng()
    pick_xy = np.array(calib.cube_xy)

    try:
        if touch is not None:
            input("Jaws open and untouched, then ENTER to tare the touch sensors: ")
            base, noise = touch.tare()
            print(f"touch baseline {np.round(base, 1).tolist()}  noise {np.round(noise, 1).tolist()}  "
                  f"on-threshold {np.round(touch.thresholds(), 1).tolist()} counts")

        profile = grip.GripProfile.load()
        if profile is None:
            print("No grip profile -- using default force target. Run scripts/train_grip.py to learn one.")
        else:
            print(f"Grip profile '{profile.object}': hold {np.round(profile.target_level).tolist()} counts, "
                  f"limit {np.round(profile.limit_level).tolist()} (max seen {np.round(profile.sat_level).tolist()})")

        with mujoco.viewer.launch_passive(ctx["m"], ctx["d"]) as viewer:
            sim.set_cube_pose(pick_xy, ctx["cube_hold_z"])
            viewer.sync()
            pickplace.go_home(ctx, viewer)

            cycle = 0
            while viewer.is_running():
                cycle += 1
                place_xy = pickplace.random_table_xy(ik, calib, rng, pick_xy)
                hud.context(f"cycle {cycle}   pick [{pick_xy[0]:+.2f} {pick_xy[1]:+.2f}] "
                            f"-> drop [{place_xy[0]:+.2f} {place_xy[1]:+.2f}]")
                sim.hold_cube(pick_xy, ctx["cube_hold_z"])
                viewer.sync()

                result = pickplace_touch.run_cycle(ctx, viewer, pick_xy, place_xy, touch, profile)
                if result == "closed":
                    break
                if result == "ok":
                    pickplace.go_home(ctx, viewer,
                                      on_step=lambda: sim.hold_cube(place_xy, ctx["cube_hold_z"]))
                    pick_xy = place_xy
                    if args.once:
                        break
                else:
                    hud.done()
                    print(f"cycle {cycle}: {result} -- going HOME, retrying the same pick")
                    pickplace.go_home(ctx, viewer,
                                      on_step=lambda: sim.hold_cube(pick_xy, ctx["cube_hold_z"]))
    except KeyboardInterrupt:
        pass
    finally:
        hud.done()
        print("\nStopped.")
        follower.disconnect()
        if touch is not None:
            touch.stop()


if __name__ == "__main__":
    main()
