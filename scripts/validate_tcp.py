"""Validate TCP stability under wrist_roll sweep."""
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)

sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

print("TCP STABILITY UNDER WRIST_ROLL SWEEP")
print("=" * 80)
print("Pure wrist_roll rotation should cause minimal TCP motion (stable gripper geometry)")
print("Large x-motion would indicate asymmetric jaw offset\n")

tcp_positions = []
roll_angles = np.linspace(-2.5, 2.5, 11)

for wr in roll_angles:
    d.ctrl[:] = 0
    d.ctrl[4] = wr  # wrist_roll is actuator 4
    
    # Settle physics
    for _ in range(2000):
        mujoco.mj_step(m, d)
    
    tcp_pos = d.site_xpos[sid].copy()
    flange_pos = d.site_xpos[gid].copy()
    tcp_positions.append(tcp_pos)
    
    print(f"roll={wr:+.1f}rad  tcp={np.round(tcp_pos, 4)}  flange={np.round(flange_pos, 4)}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

tcp_array = np.array(tcp_positions)
tcp_range = np.max(tcp_array, axis=0) - np.min(tcp_array, axis=0)

print(f"TCP position range (max - min):")
print(f"  x: {tcp_range[0]:.4f} m ({tcp_range[0]*1000:.2f} mm)")
print(f"  y: {tcp_range[1]:.4f} m ({tcp_range[1]*1000:.2f} mm)")
print(f"  z: {tcp_range[2]:.4f} m ({tcp_range[2]*1000:.2f} mm)")

# Check if x motion is suspiciously large
if tcp_range[0] > 0.003:  # > 3mm is suspicious for a roll-only axis
    print(f"\n⚠ WARNING: X-motion under wrist_roll is {tcp_range[0]*1000:.2f}mm (expected <3mm)")
    print("  This suggests asymmetric jaw offset or incorrect centroid averaging")
    print("  Re-examine tune_tcp.py output and jaw geometry")
else:
    print(f"\n✓ X-motion is stable: {tcp_range[0]*1000:.2f}mm (acceptable)")

# Y and Z should be very small (just roll axis)
if tcp_range[1] < 0.001 and tcp_range[2] < 0.001:
    print(f"✓ Y and Z motion are negligible (roll is pure rotation about gripper)")
else:
    print(f"⚠ Y or Z motion larger than expected")
