"""Find joint angles to reach the cube."""
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)

tcp_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
cube_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")

# Cube location
d.ctrl[:] = 0
for _ in range(500):
    mujoco.mj_step(m, d)
cube_pos = d.xpos[cube_body_id].copy()
print(f"Cube position: {np.round(cube_pos, 4)}")
print(f"Cube xy: {np.round(cube_pos[:2], 4)}, z: {cube_pos[2]:.4f}")

# Try a series of reaching poses
print("\nScanning joint angles for reach to cube...\n")

best_error = float('inf')
best_pose = None

# Sweep through reasonable joint ranges
for s_pan in np.linspace(-0.5, 0.5, 5):  # shoulder_pan
    for s_lift in np.linspace(-1.5, -0.3, 7):  # shoulder_lift (lower = reach down)
        for e_flex in np.linspace(-0.5, 0.8, 5):  # elbow_flex
            for w_flex in np.linspace(-0.5, 0.5, 5):  # wrist_flex
                pose = np.array([s_pan, s_lift, e_flex, w_flex, 0.0, 0.0])
                
                d.ctrl[:] = pose
                for _ in range(1000):
                    mujoco.mj_step(m, d)
                
                tcp_pos = d.site_xpos[tcp_id].copy()
                error = np.linalg.norm(tcp_pos[:2] - cube_pos[:2])  # xy error
                z_error = abs(tcp_pos[2] - cube_pos[2])  # z error
                
                # Prefer poses that are at cube height and close in xy
                total_error = error + z_error
                
                if total_error < best_error and z_error < 0.02:  # z should be within 2cm
                    best_error = total_error
                    best_pose = pose
                    if total_error < 0.05:  # Found a good one
                        print(f"GOOD: pose={np.round(pose, 3)}")
                        print(f"      tcp={np.round(tcp_pos, 4)}, xy_err={error:.4f}, z_err={z_error:.4f}\n")

if best_pose is not None:
    print("\n" + "=" * 80)
    print("BEST POSE FOUND:")
    print(f"  Joint angles: {np.round(best_pose, 3)}")
    d.ctrl[:] = best_pose
    for _ in range(1000):
        mujoco.mj_step(m, d)
    tcp_pos = d.site_xpos[tcp_id].copy()
    print(f"  TCP position: {np.round(tcp_pos, 4)}")
    print(f"  Error: {best_error:.4f} m")
else:
    print("No good pose found")
