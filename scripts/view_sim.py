"""Watch the grasp sequence live in the MuJoCo viewer.

Same reach -> grasp -> lift sequence as grasp_test.py, but rendered
interactively in real time instead of running headless.
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)

home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
reach_pose = np.array([-0.3, -0.8, 0.5, -0.3, 0.0, 0.0])


def run_phase(viewer, ctrl, n_steps, label):
    print(label)
    d.ctrl[:6] = ctrl
    for _ in range(n_steps):
        step_start = time.time()
        mujoco.mj_step(m, d)
        viewer.sync()
        dt = m.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
        if not viewer.is_running():
            return False
    return True


with mujoco.viewer.launch_passive(m, d) as viewer:
    d.ctrl[:] = home
    if not run_phase(viewer, home, 1000, "Phase 0: HOME"):
        raise SystemExit

    ctrl = reach_pose.copy()
    ctrl[5] = 0  # gripper open
    if not run_phase(viewer, ctrl, 2000, "Phase 1: REACH ABOVE CUBE"):
        raise SystemExit

    ctrl[5] = m.actuator_ctrlrange[5][0]  # gripper closed
    if not run_phase(viewer, ctrl, 2000, "Phase 2: CLOSE GRIPPER"):
        raise SystemExit

    lift_pose = reach_pose.copy()
    lift_pose[1] -= 0.5
    ctrl = lift_pose.copy()
    ctrl[5] = m.actuator_ctrlrange[5][0]
    if not run_phase(viewer, ctrl, 2000, "Phase 3: LIFT 10cm"):
        raise SystemExit

    print("Sequence complete. Close the viewer window to exit.")
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)
        viewer.sync()
        dt = m.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
