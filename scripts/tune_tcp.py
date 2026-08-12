"""Compute TCP site position from closed gripper geometry."""
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)

# Close the gripper and settle
d.ctrl[:] = 0
d.ctrl[5] = m.actuator_ctrlrange[5][0]  # Close gripper fully
print("Closing gripper and settling physics...")
for _ in range(2000):
    mujoco.mj_step(m, d)

def gid(n):
    """Get geom ID by name."""
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)

def bid(n):
    """Get body ID by name."""
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

print("\nGEOMETRY IN JAW BODIES (world positions, closed gripper):")
print("=" * 80)

# Map body names to their geoms
jaw_bodies = {"gripper": bid("gripper"), "moving_jaw_so101_v1": bid("moving_jaw_so101_v1")}
tip_geoms = {}

for body_name, body_id in jaw_bodies.items():
    print(f"\n{body_name} (body_id={body_id}):")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == body_id:
            geom_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            world_pos = d.geom_xpos[g].copy()
            geom_type = m.geom_type[g]
            if geom_name is None:
                geom_name = f"<unnamed_geom_{g}>"
            print(f"  [{g}] {geom_name:32s} type={geom_type} pos={np.round(world_pos, 4)}")
            if not geom_name.startswith("<unnamed"):
                tip_geoms[geom_name] = (g, world_pos)

print("\n" + "=" * 80)
print("COMPUTE PINCH POINT")
print("=" * 80)

# Since geoms are unnamed meshes, compute centroid of each jaw body's geometry
gripper_geoms = []
moving_jaw_geoms = []

for g in range(m.ngeom):
    if m.geom_bodyid[g] == bid("gripper"):
        gripper_geoms.append(d.geom_xpos[g].copy())
    elif m.geom_bodyid[g] == bid("moving_jaw_so101_v1"):
        moving_jaw_geoms.append(d.geom_xpos[g].copy())

if gripper_geoms and moving_jaw_geoms:
    # Use centroid of each jaw's geometry as the contact point
    fixed_jaw_pos = np.mean(gripper_geoms, axis=0)
    moving_jaw_pos = np.mean(moving_jaw_geoms, axis=0)
    
    print(f"Fixed jaw (gripper body) centroid:      {np.round(fixed_jaw_pos, 4)}")
    print(f"Moving jaw (moving_jaw_so101_v1) centroid: {np.round(moving_jaw_pos, 4)}")
    
    # Pinch point is midpoint between the two
    pinch_world = 0.5 * (fixed_jaw_pos + moving_jaw_pos)
    print(f"Pinch point (world):                   {np.round(pinch_world, 4)}")
    
    # Transform to gripper body frame
    gripper_body_id = bid("gripper")
    R = d.xmat[gripper_body_id].reshape(3, 3)
    p_gripper = d.xpos[gripper_body_id]
    
    # local = R.T @ (pinch_world - p_gripper)
    tcp_local = R.T @ (pinch_world - p_gripper)
    print(f"\nGripper body frame:")
    print(f"  position: {np.round(p_gripper, 4)}")
    print(f"  rotation: 3x3 matrix")
    
    print(f"\n*** TCP LOCAL POSITION ***")
    print(f"pos=\"{tcp_local[0]:.4f} {tcp_local[1]:.4f} {tcp_local[2]:.4f}\"")
    print(f"\nUpdate so101_new_calib.xml, gripper body:")
    print(f"  <site name=\"tcp\" pos=\"{tcp_local[0]:.4f} {tcp_local[1]:.4f} {tcp_local[2]:.4f}\" size=\"0.004\" rgba=\"0 1 0 0.6\"/>")
else:
    print("ERROR: Could not find geoms in both jaw bodies.")
