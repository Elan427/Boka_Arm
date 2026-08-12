"""Inspect SO-ARM100 MuJoCo model contract."""
from pathlib import Path
import numpy as np
import mujoco

XML = Path("assets/so101/scene_lift_cube.xml")
m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print(f"timestep {m.opt.timestep}  nq {m.nq}  nu {m.nu}  nbody {m.nbody}\n")

print("JOINTS")
for i in range(m.njnt):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    lo, hi = m.jnt_range[i]
    print(f"  {i} {n:20s} limited={bool(m.jnt_limited[i])} range=[{lo:+.3f}, {hi:+.3f}]")

print("\nACTUATORS")
for i in range(m.nu):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    lo, hi = m.actuator_ctrlrange[i]
    print(f"  {i} {n:20s} ctrlrange=[{lo:+.3f}, {hi:+.3f}] gear={m.actuator_gear[i][0]:.2f}")

print("\nSITES")
for i in range(m.nsite):
    print(f"  {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)}")

print("\nBODIES")
for i in range(m.nbody):
    print(f"  {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)}  mass={m.body_mass[i]:.4f}")

print("\nCAMERAS")
for i in range(m.ncam):
    print(f"  {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)}")

# Detailed diagnostics
TRN = {0: "JOINT", 1: "JOINTINPARENT", 2: "SLIDERCRANK", 3: "TENDON", 4: "SITE", 5: "BODY"}
GAIN = {0: "FIXED", 1: "AFFINE", 2: "MUSCLE", 3: "USER"}
BIAS = {0: "NONE", 1: "AFFINE", 2: "MUSCLE", 3: "USER"}

print("\nACTUATOR TYPE")
for i in range(m.nu):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {n:16s} trn={TRN[m.actuator_trntype[i]]:6s} gain={GAIN[m.actuator_gaintype[i]]:6s} "
          f"bias={BIAS[m.actuator_biastype[i]]:6s} kp={m.actuator_gainprm[i][0]:.1f} "
          f"biasprm={m.actuator_biasprm[i][:3]}")

print("\nSITE WORLD POS (home)")
for i in range(m.nsite):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  {n:16s} {np.round(d.site_xpos[i], 4)}")

print("\nJOINT DAMPING / ARMATURE")
for i in range(m.njnt):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  {n:16s} damping={m.dof_damping[i]:.4f} armature={m.dof_armature[i]:.5f} "
          f"frictionloss={m.dof_frictionloss[i]:.4f}")
