import numpy as np, mujoco, itertools

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")

d.qpos[:] = 0
mujoco.mj_kinematics(m, d)
print("base body world pos:", np.round(d.xpos[bid], 4))
print("tcp at zero pose   :", np.round(d.site_xpos[sid], 4))

lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
pts = []
for combo in itertools.product(*[np.linspace(lo[i], hi[i], 7) for i in range(5)]):
    d.qpos[:5] = combo
    mujoco.mj_kinematics(m, d)
    pts.append(d.site_xpos[sid].copy())
pts = np.array(pts)

for ax, name in enumerate("xyz"):
    print(f"{name}: [{pts[:,ax].min():+.4f}, {pts[:,ax].max():+.4f}]")

for tgt in [np.array([0.15, 0.15, 0.05]), np.array([0.15, 0.15, 0.015])]:
    print(f"closest TCP to {tgt}: {np.linalg.norm(pts - tgt, axis=1).min():.4f}")
