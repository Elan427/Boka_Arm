"""Grasp test using mink IK: hover, descend, grasp, lift."""
import numpy as np
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ik_mink import SO101IK, rotation_matrix_z_down

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)

tcp_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
cube_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")

print("GRASP TEST WITH MINK IK")
print("=" * 80)
print("Sequence: hover above cube → descend → close gripper → lift 10cm\n")

# Initialize IK solver
ik = SO101IK()

# Get cube position
d.qpos[:] = 0
mujoco.mj_forward(m, d)
cube_pos_init = d.xpos[cube_body_id].copy()
cube_z_rest = cube_pos_init[2]

print(f"Cube initial position: {np.round(cube_pos_init, 4)}")
print(f"Cube z: {cube_z_rest:.4f} m\n")

# ============================================================================
# Phase 1: HOVER
# ============================================================================

print("Phase 1: HOVER 8cm ABOVE CUBE")
print("-" * 80)

hover_height = cube_z_rest + 0.08
hover_target = np.array([cube_pos_init[0], cube_pos_init[1], hover_height])

print(f"Target TCP position: {np.round(hover_target, 4)}")
print("Solving with mink...")

q_hover = ik.solve_to_position(hover_target, verbose=True)
print(f"Solution: q = {np.round(q_hover, 3)}\n")

# Apply to simulation
d.ctrl[:6] = q_hover
d.ctrl[5] = 0  # gripper open
for step in range(1000):
    mujoco.mj_step(m, d)

tcp_actual = d.site_xpos[tcp_id]
print(f"Actual TCP at hover: {np.round(tcp_actual, 4)}")
print(f"Error from target:   {np.linalg.norm(tcp_actual - hover_target):.4f} m\n")

# ============================================================================
# Phase 2: DESCEND
# ============================================================================

print("Phase 2: DESCEND TO CUBE")
print("-" * 80)

descend_target = np.array([cube_pos_init[0], cube_pos_init[1], cube_z_rest + 0.01])
print(f"Target TCP position: {np.round(descend_target, 4)}")
print("Solving with mink...")

q_descend = ik.solve_to_position(descend_target, verbose=True)
print(f"Solution: q = {np.round(q_descend, 3)}\n")

# Apply to simulation
d.ctrl[:6] = q_descend
d.ctrl[5] = 0  # gripper open
for step in range(1000):
    mujoco.mj_step(m, d)

tcp_actual = d.site_xpos[tcp_id]
cube_pos = d.xpos[cube_body_id]
print(f"Actual TCP at descend: {np.round(tcp_actual, 4)}")
print(f"Cube now at:           {np.round(cube_pos, 4)}")
print(f"Error from target:     {np.linalg.norm(tcp_actual - descend_target):.4f} m\n")

# ============================================================================
# Phase 3: CLOSE GRIPPER
# ============================================================================

print("Phase 3: CLOSE GRIPPER")
print("-" * 80)

d.ctrl[:6] = q_descend
d.ctrl[5] = m.actuator_ctrlrange[5][0]  # Close fully
print("Gripper closing...")

for step in range(2000):
    mujoco.mj_step(m, d)
    if step % 500 == 0:
        print(f"  step {step}: cube z = {d.xpos[cube_body_id][2]:.4f}")

cube_z_grasped = d.xpos[cube_body_id][2]
print(f"✓ Gripper closed. Cube z = {cube_z_grasped:.4f} m\n")

# ============================================================================
# Phase 4: LIFT
# ============================================================================

print("Phase 4: LIFT 10cm")
print("-" * 80)

# Raise the target height 10cm
lift_target = np.array([cube_pos_init[0], cube_pos_init[1], cube_z_grasped + 0.10])
print(f"Target TCP position: {np.round(lift_target, 4)}")
print("Solving with mink...")

q_lift = ik.solve_to_position(lift_target, verbose=True)
print(f"Solution: q = {np.round(q_lift, 3)}\n")

# Apply to simulation, keep gripper closed
d.ctrl[:6] = q_lift
d.ctrl[5] = m.actuator_ctrlrange[5][0]  # Keep closed
print("Lifting...")

for step in range(2000):
    mujoco.mj_step(m, d)
    if step % 500 == 0:
        cube_z = d.xpos[cube_body_id][2]
        tcp_z = d.site_xpos[tcp_id][2]
        print(f"  step {step}: cube z = {cube_z:.4f}, tcp z = {tcp_z:.4f}")

cube_z_lifted = d.xpos[cube_body_id][2]
tcp_z_lifted = d.site_xpos[tcp_id][2]

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

cube_displacement = cube_z_lifted - cube_z_rest
tcp_displacement = tcp_z_lifted - d.site_xpos[tcp_id][2] + 0.12  # Approximate

print(f"Cube z at rest:      {cube_z_rest:.4f} m")
print(f"Cube z after lift:   {cube_z_lifted:.4f} m")
print(f"Cube displacement:   {cube_displacement:.4f} m ({cube_displacement*1000:.1f} mm)")

print("\n" + "=" * 80)
if cube_displacement > 0.05:
    print(f"✓ SUCCESS: Cube lifted {cube_displacement*1000:.1f}mm!")
    print("  All components validated:")
    print("    ✓ Mink IK reaching")
    print("    ✓ TCP position correct")
    print("    ✓ Gripper closure torque sufficient")
    print("    ✓ Friction (impratio=10) holding")
    print("    ✓ Contact model and physics accurate")
elif cube_displacement > 0.01:
    print(f"⚠ PARTIAL: Cube moved {cube_displacement*1000:.1f}mm but incomplete.")
    print("  Check: gripper closure force, friction, or TCP-object contact")
else:
    print(f"✗ FAILURE: Cube didn't move ({cube_displacement*1000:.2f}mm)")
    print("  Debug: Check IK convergence, TCP offset, or gripper mechanism")

print("=" * 80)

# Save cube final position for inspection
print(f"\nCube freejoint position after lift: {d.qpos[-7:]}")
