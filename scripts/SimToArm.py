"""SO-101 sim -> real: a single scripted pick-and-place.

HOME -> HOVER over the cube -> DESCEND (stops on table contact) -> GRASP ->
LIFT -> CARRY to the place spot -> LOWER -> RELEASE -> RETREAT -> HOME, then
stop. For the continuous "pick it up and drop it somewhere random, forever"
version, run scripts/SimToArmLoop.py instead.

The cube starts at calib.cube_xy and is placed at PLACE_XY (default: its mirror
across the base x-axis). Heights are the REAL pinch-tip height above the
calibrated table; every XY target is mapped through calib.apply_target so the
real TCP lands where the phase asks. Progress shows as the in-place joint-angle
block from so101lib/hud.py -- no scrolling phase log.

Speed / smoothness: pickplace.SPEED scales phase durations; the HW_* knobs in
so101lib/config.py (send rate, per-joint acceleration, command smoothing) shape
how the real arm tracks.

Calibration order:  hw_check.py -> solve_joint_calib.py (if needed) ->
touch_off.py -> calibrate_xy.py.  With no calib file the sim-only defaults run.

Run: python scripts/SimToArm.py
"""
import mujoco
import mujoco.viewer
import numpy as np

from so101lib import hud, pickplace

HARDWARE_ENABLED = True

PLACE_XY = None  # None -> mirror of cube_xy across the base x-axis


def main():
    pickplace.silence_clamp_warnings()
    ctx = pickplace.build(HARDWARE_ENABLED)
    sim, follower = ctx["sim"], ctx["follower"]

    pick_xy = np.array(ctx["calib"].cube_xy)
    place_xy = np.array(PLACE_XY) if PLACE_XY is not None else np.array([pick_xy[0], -pick_xy[1]])
    done = False

    try:
        with mujoco.viewer.launch_passive(ctx["m"], ctx["d"]) as viewer:
            sim.set_cube_pose(pick_xy, ctx["cube_hold_z"])
            viewer.sync()

            pickplace.go_home(ctx, viewer)

            hud.context(f"pick [{pick_xy[0]:+.2f} {pick_xy[1]:+.2f}] "
                        f"-> drop [{place_xy[0]:+.2f} {place_xy[1]:+.2f}]")
            done = pickplace.run_cycle(ctx, viewer, pick_xy, place_xy)
            if done:
                pickplace.go_home(ctx, viewer,
                                  on_step=lambda: sim.hold_cube(place_xy, ctx["cube_hold_z"]))
    except KeyboardInterrupt:
        pass
    except ConnectionError as e:
        hud.done()
        print(f"\nLost the servo bus: {e}")
        print("A motor probably latched an overload -- power-cycle the arm before rerunning.")
    finally:
        hud.done()
        print("\nDone." if done else "\nStopped.")
        follower.disconnect()


if __name__ == "__main__":
    main()
