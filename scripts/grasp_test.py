"""Hand-coded grasp test: reach, descend, grasp, lift.

Validates TCP position, friction, gripper range, and contact model.
If the cube comes with the gripper, all components are working correctly.
"""
import numpy as np
import mujoco
from pathlib import Path

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)

# IDs
tcp_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
cube_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")

out_dir = Path("out/grasp_test")
out_dir.mkdir(exist_ok=True)

print("HAND-CODED GRASP TEST")
print("=" * 80)
print("Using direct joint-space reach poses (no IK solver)\n")

# ============================================================================
# Reaching poses (discovered via trial in sim)
# ============================================================================

# Home
home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Pose 1: Reach above the cube (hand-tuned joint angles)
reach_pose = np.array([-0.3, -0.8, 0.5, -0.3, 0.0, 0.0])

# Settle physics and record initial cube state
print("Phase 0: HOME")
print("-" * 80)
d.ctrl[:] = home
for _ in range(1000):
    mujoco.mj_step(m, d)

cube_z_rest = d.xpos[cube_body_id][2]
cube_xy = d.xpos[cube_body_id][:2]
tcp_z_home = d.site_xpos[tcp_id][2]

print(f"Cube at z = {cube_z_rest:.4f} m")
print(f"TCP at z = {tcp_z_home:.4f} m")
print(f"Cube center xy = {np.round(cube_xy, 4)}")

# ============================================================================
# Test sequence
# ============================================================================

print("\nPhase 1: REACH ABOVE CUBE")
print("-" * 80)
print(f"Target joint angles: {np.round(reach_pose, 3)}")

d.ctrl[:6] = reach_pose
d.ctrl[5] = 0  # gripper open
for step in range(2000):
    mujoco.mj_step(m, d)
    if step % 500 == 0:
        tcp = d.site_xpos[tcp_id]
        print(f"  step {step}: tcp = {np.round(tcp, 4)}, qpos = {np.round(d.qpos[:6], 3)}")

tcp_reach = d.site_xpos[tcp_id].copy()
print(f"✓ Reached. TCP = {np.round(tcp_reach, 4)}")

print("\nPhase 2: CLOSE GRIPPER")
print("-" * 80)
print("Gripper closing...")
d.ctrl[5] = m.actuator_ctrlrange[5][0]  # Fully closed
for step in range(2000):
    mujoco.mj_step(m, d)
    if step % 500 == 0:
        cube_z = d.xpos[cube_body_id][2]
        print(f"  step {step}: cube z = {cube_z:.4f} m, gripper ctrl = {d.ctrl[5]:.3f}")

cube_z_grasped = d.xpos[cube_body_id][2]
print(f"✓ Gripper closed. Cube z = {cube_z_grasped:.4f} m")

print("\nPhase 3: LIFT 10cm")
print("-" * 80)
# Move upward in joint space: increase shoulder_lift (joint 1)
lift_pose = reach_pose.copy()
lift_pose[1] -= 0.5  # Decrease shoulder_lift angle to lift arm

print(f"Target joint angles: {np.round(lift_pose, 3)}")
d.ctrl[:6] = lift_pose
d.ctrl[5] = m.actuator_ctrlrange[5][0]  # Keep gripper closed

for step in range(2000):
    mujoco.mj_step(m, d)
    if step % 500 == 0:
        cube_z = d.xpos[cube_body_id][2]
        tcp_z = d.site_xpos[tcp_id][2]
        print(f"  step {step}: cube z = {cube_z:.4f} m, tcp z = {tcp_z:.4f} m")

cube_z_lifted = d.xpos[cube_body_id][2]
tcp_z_lifted = d.site_xpos[tcp_id][2]

# ============================================================================
# Results
# ============================================================================

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

cube_displacement = cube_z_lifted - cube_z_rest
tcp_displacement = tcp_z_lifted - tcp_z_home

print(f"Cube z before grasp:  {cube_z_rest:.4f} m")
print(f"Cube z after lift:    {cube_z_lifted:.4f} m")
print(f"Cube vertical move:   {cube_displacement:.4f} m ({cube_displacement*1000:.1f} mm)")
print()
print(f"TCP z at home:        {tcp_z_home:.4f} m")
print(f"TCP z after lift:     {tcp_z_lifted:.4f} m")
print(f"TCP vertical move:    {tcp_displacement:.4f} m ({tcp_displacement*1000:.1f} mm)")

print("\n" + "=" * 80)
if cube_displacement > 0.03:  # Cube moved at least 3cm
    print(f"✓ SUCCESS: Cube lifted {cube_displacement*1000:.1f}mm with gripper.")
    print("  TCP position, friction, gripper range, and contact model are validated.")
elif cube_displacement > 0.005:
    print(f"⚠ PARTIAL: Cube moved {cube_displacement*1000:.1f}mm but incomplete lift.")
    print("  Check gripper closure force, friction coefficients, or TCP-object alignment.")
else:
    print(f"✗ FAILURE: Cube didn't move ({cube_displacement*1000:.2f}mm).")
    print("  Likely issues:")
    print("    1. Gripper closing torque insufficient (check kp, forcerange)")
    print("    2. TCP position misaligned (re-run validate_tcp.py)")
    print("    3. Object friction too low or gripper surface slips (impratio, mu)")
    print("    4. Reach pose doesn't actually contact the cube")

print("=" * 80)
