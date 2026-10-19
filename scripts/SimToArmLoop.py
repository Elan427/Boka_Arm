"""SO-101 sim -> real: a continuous pick-and-place loop.

Each cycle: HOME -> HOVER over the cube -> DESCEND (stops on table contact) ->
GRASP -> LIFT -> CARRY to a random reachable table spot -> LOWER -> RELEASE ->
RETREAT -> HOME. The next cycle picks the cube up from wherever it was just
dropped and moves it somewhere new. Runs until you close the viewer or Ctrl+C.

For a single pick-and-place that stops after one cycle, run scripts/SimToArm.py.

Drop targets are sampled in the base frame (pickplace.DROP_RADIUS /
DROP_ANGLE_DEG) and filtered by an IK reachability check at both hover and grasp
height. Heights are the REAL pinch-tip height above the calibrated table; every
XY target is mapped through calib.apply_target so the real TCP lands where the
phase asks.

Speed / smoothness: pickplace.SPEED scales phase durations; the HW_* knobs in
so101lib/config.py (send rate, per-joint acceleration, command smoothing) shape
how the real arm tracks. Progress shows as the in-place joint-angle block from
so101lib/hud.py -- no scrolling phase log.

Calibration order:  hw_check.py -> solve_joint_calib.py (if needed) ->
touch_off.py -> calibrate_xy.py.  With no calib file the sim-only defaults run.

Run: python scripts/SimToArmLoop.py
"""
import mujoco
import mujoco.viewer
import numpy as np


from so101lib import hud, pickplace

HARDWARE_ENABLED = True


def main():
    pickplace.silence_clamp_warnings()
    ctx = pickplace.build(HARDWARE_ENABLED)
    sim, follower, ik, calib = ctx["sim"], ctx["follower"], ctx["ik"], ctx["calib"]
    rng = np.random.default_rng()

    pick_xy = np.array(calib.cube_xy)

    try:
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

                if not pickplace.run_cycle(ctx, viewer, pick_xy, place_xy):
                    break

                pickplace.go_home(ctx, viewer,
                                  on_step=lambda: sim.hold_cube(place_xy, ctx["cube_hold_z"]))
                pick_xy = place_xy
    except KeyboardInterrupt:
        pass
    finally:
        hud.done()
        print("\nStopped.")
        follower.disconnect()


if __name__ == "__main__":
    main()
