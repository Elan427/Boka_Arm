"""Verify cube visibility in both cameras at grasp time."""
import numpy as np
import mujoco
from PIL import Image
from pathlib import Path

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)

Path("out").mkdir(exist_ok=True)

# Grasp position (we'll approximate it; mink will refine)
# For now, use a pose that should roughly reach the cube
d.qpos[:6] = np.array([-0.1, -0.9, 0.4, -0.3, 0.0, 0.0])
d.ctrl[:6] = d.qpos[:6]

# Settle kinematics
mujoco.mj_kinematics(m, d)

tcp_pos = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")]
cube_pos = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")]

print("CAMERA VISIBILITY CHECK AT GRASP POSE")
print("=" * 80)
print(f"TCP position:  {np.round(tcp_pos, 4)}")
print(f"Cube position: {np.round(cube_pos, 4)}")
print(f"TCP-to-cube distance: {np.linalg.norm(tcp_pos - cube_pos):.4f} m\n")

with mujoco.Renderer(m, 480, 640) as r:
    for cam in ["top", "wrist"]:
        r.update_scene(d, camera=cam)
        img = Image.fromarray(r.render())
        img.save(f"out/grasp_check_{cam}.png")
        print(f"✓ {cam:8s} camera -> out/grasp_check_{cam}.png")

print("\nInspect both images:")
print("  top   camera: Is the cube and gripper both visible?")
print("  wrist camera: Is the cube in the approach space (not occluded)?")
