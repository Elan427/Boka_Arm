"""Find the lightest grip that still lifts the object, using the jaw flex sensors.

Starts with the jaws open and no sensor signal, closes slowly until the first
signal appears, then lifts the arm a few cm and checks the sensors still read. If
the object was lost it goes back down, reopens, closes 5 % more of the gripper
travel than last time and lifts again -- repeating until a lift holds. That closure
(and the sensor levels it produced) is the minimum grip; it is saved into
config/grip_profile.json as the hold target (x --margin) for SimToArmTouch.py.

Put the object at the pick spot (config/so101_calib.json "cube_xy"). Flash
firmware/gripper_touch, wire both sensors, and check them with touch_monitor.py first.
Keep the jaws open and untouched while the sensors tare.

Run: python scripts/find_min_grip.py [--step 0.05] [--max-steps 8] [--max-current 120]
                                     [--require both|any] [--margin 1.25] [--object cube]
"""
import argparse

import mujoco
import mujoco.viewer
import numpy as np

from so101lib import grip, hud, mingrip, pickplace
from so101lib.touch import TouchSensors

HARDWARE_ENABLED = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3", help="Arduino serial port")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="extra sensor smoothing: weight of the newest sample (0 = off, 0.1 = heavy)")
    ap.add_argument("--step", type=float, default=mingrip.STEP,
                    help="extra jaw travel per retry, fraction of the gripper range (0.05 = 5%%)")
    ap.add_argument("--max-steps", type=int, default=mingrip.MAX_STEPS)
    ap.add_argument("--max-current", type=float, default=120,
                    help="stop closing at this gripper current (raw units, ~6.5 mA each)")
    ap.add_argument("--first-contact", choices=["both", "any"], default="both",
                    help="close until BOTH jaw sensors read (default) or until the first one does")
    ap.add_argument("--max-blind", type=float, default=mingrip.MAX_BLIND,
                    help="give up if no sensor signal by this fraction of the jaw travel")
    ap.add_argument("--require", choices=["both", "any"], default="both",
                    help="which jaw sensors must still read after the lift to count as held")
    ap.add_argument("--margin", type=float, default=1.25,
                    help="saved hold target = margin x the sensor levels of the lightest grip")
    ap.add_argument("--object", default="cube")
    args = ap.parse_args()

    touch = TouchSensors(args.port, smooth=args.smooth)
    touch.start()
    pickplace.silence_clamp_warnings()
    ctx = pickplace.build(HARDWARE_ENABLED)
    sim, follower, calib = ctx["sim"], ctx["follower"], ctx["calib"]
    pick_xy = np.array(calib.cube_xy)
    profile = grip.GripProfile.load()
    result = None

    try:
        input("Jaws open and untouched, then ENTER to tare the touch sensors: ")
        base, noise = touch.tare()
        print(f"touch baseline {np.round(base, 1).tolist()}  noise {np.round(noise, 2).tolist()}  "
              f"on-threshold {np.round(touch.thresholds(), 1).tolist()} counts")
        issues = touch.problems()
        if issues:
            print("\nTOUCH SENSORS NOT READY -- the grip search could not detect a squeeze:")
            for m in issues:
                print("  -", m)
            print("Fix the wiring (run touch_monitor.py), then retry. Nothing was moved.")
            return
        print("Checking the sensors respond:")
        if not all(touch.response_check()):
            print("A sensor did not respond to a press -- check its wiring. Nothing was moved.")
            return
        limit = profile.limit_level if profile and any(profile.limit_level) else None

        with mujoco.viewer.launch_passive(ctx["m"], ctx["d"]) as viewer:
            sim.set_cube_pose(pick_xy, ctx["cube_hold_z"])
            viewer.sync()
            pickplace.go_home(ctx, viewer)
            hud.context(f"min-grip search at [{pick_xy[0]:+.2f} {pick_xy[1]:+.2f}]  "
                        f"step {args.step * 100:.0f}%")
            result = mingrip.run(ctx, viewer, touch, pick_xy, step=args.step,
                                 max_steps=args.max_steps, max_current=args.max_current,
                                 require=args.require, limit_level=limit,
                                 max_blind=args.max_blind,
                                 first_contact=args.first_contact)
            hud.done()
            if result.status != "closed":
                pickplace.go_home(ctx, viewer,
                                  on_step=lambda: sim.hold_cube(pick_xy, ctx["cube_hold_z"]))
    except KeyboardInterrupt:
        pass
    finally:
        hud.done()
        follower.disconnect()
        touch.stop()

    if result is None:
        print("\nStopped.")
        return
    print("\n" + "=" * 60)
    if result.status == "found":
        print(f"lightest holding grip: {result.travel * 100:.0f}% of jaw travel "
              f"({(result.travel - result.contact_travel) * 100:.0f}% past first contact at "
              f"{result.contact_travel * 100:.0f}%)")
        print(f"sensor levels while holding: L {result.levels[0]:.0f}  R {result.levels[1]:.0f} counts   "
              f"({result.attempts} lift attempt{'s' if result.attempts != 1 else ''})")
        if input("Save as the hold target in config/grip_profile.json? [y/n]: ").strip().lower().startswith("y"):
            p = grip.apply_min_hold(profile, result.levels, result.travel,
                                    result.contact_travel, result.attempts, args.margin)
            p.object = args.object
            p.save()
    else:
        why = {"no_contact": "no sensor signal before the jaws fully closed -- check wiring, sensor "
                             "placement and that the object is between the jaws",
               "not_found": f"the object was lost on every lift up to full closure",
               "limit": "stopped at the motor-current / sensor limit before a lift held; raise "
                        "--max-current carefully or retrain with train_grip.py",
               "closed": "viewer closed"}[result.status]
        print(f"No result: {why}.")


if __name__ == "__main__":
    main()
