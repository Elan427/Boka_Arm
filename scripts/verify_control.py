"""Verify closed-loop position control: data.ctrl -> data.qpos convergence."""
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)

# Test waypoints: small, safe movements from home
# Joint ranges: shoulder_pan [-1.92, 1.92], shoulder_lift [-1.75, 1.75],
#               elbow_flex [-1.69, 1.69], wrist_flex [-1.66, 1.66],
#               wrist_roll [-2.74, 2.84], gripper [-0.17, 1.75]
waypoints = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),      # Home
    np.array([0.2, -0.3, 0.1, 0.1, 0.0, 0.5]),      # Pose 1
    np.array([-0.2, -0.5, -0.1, -0.1, 0.2, 0.8]),   # Pose 2 (fixed: gripper in valid range)
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),       # Back to home
]

physics_steps_per_ctrl = 10
ctrl_steps_per_waypoint = 30  # 100 ms at 30 Hz
tolerance = 0.02  # radians (~1.1°)

print("CLOSED-LOOP CONTROL VERIFICATION")
print(f"Testing position servo: data.ctrl[:6] -> data.qpos[:6] convergence\n")

for wp_idx, target_q in enumerate(waypoints):
    print(f"Waypoint {wp_idx}: target = {np.round(target_q, 3)}")
    
    # Command the target
    d.ctrl[:6] = target_q
    
    # Simulate for 100 ms (30 control steps × 10 physics steps × 3.33 ms)
    for step in range(ctrl_steps_per_waypoint * physics_steps_per_ctrl):
        mujoco.mj_step(m, d)
    
    # Check convergence
    achieved_q = d.qpos[:6].copy()
    error = np.abs(achieved_q - target_q)
    max_error = np.max(error)
    all_converged = np.all(error < tolerance)
    
    status = "✓ PASS" if all_converged else "✗ FAIL"
    print(f"  achieved = {np.round(achieved_q, 3)}")
    print(f"  max error = {max_error:.4f} rad ({np.degrees(max_error):.2f}°) {status}")
    
    if not all_converged:
        print(f"  Joints not converged:")
        for i in range(6):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if error[i] >= tolerance:
                print(f"    {name}: {error[i]:.4f} rad")
    print()

print("\nVerification complete.")
print("If all waypoints show PASS, control contract is confirmed: data.ctrl is joint angle (radians)")
