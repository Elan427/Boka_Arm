"""MuJoCo model/data loading + name lookups for the lift-cube scene.

scene_lift_cube.xml has nq > 6: the cube's freejoint adds 7 DOF after the 6 arm
joints. qpos layout is [0:6] arm, [6:9] cube xyz, [9:13] cube quat (wxyz).
"""
from dataclasses import dataclass

import mujoco

from . import paths

_ARM_QPOS = slice(0, 6)
_CUBE_XYZ_QPOS = slice(6, 9)
_CUBE_QUAT_QPOS = slice(9, 13)


@dataclass
class Sim:
    """Bundle of (model, data) plus cached name->id lookups."""

    m: mujoco.MjModel
    d: mujoco.MjData

    def _id(self, objtype, name):
        i = mujoco.mj_name2id(self.m, objtype, name)
        if i < 0:
            raise KeyError(f"no {objtype!r} named {name!r} in model")
        return i

    def site(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_SITE, name)

    def body(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_BODY, name)

    def geom(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_GEOM, name)

    # --- cube helpers ---------------------------------------------------
    def set_cube_pose(self, xy, z, quat=(1, 0, 0, 0)):
        """Reset arm to home and place the cube at (xy, z), then mj_forward."""
        self.d.qpos[_ARM_QPOS] = 0
        self.d.qpos[_CUBE_XYZ_QPOS] = [xy[0], xy[1], z]
        self.d.qpos[_CUBE_QUAT_QPOS] = quat
        mujoco.mj_forward(self.m, self.d)

    def hold_cube(self, xy, z, quat=(1, 0, 0, 0)):
        """Pin the cube at (xy, z) for this step -- position + zero velocity,
        arm untouched. Call every step to kinematically hold it (e.g. so the
        gripper can descend onto it despite a model TCP-site height offset)."""
        self.d.qpos[_CUBE_XYZ_QPOS] = [xy[0], xy[1], z]
        self.d.qpos[_CUBE_QUAT_QPOS] = quat
        self.d.qvel[6:12] = 0

    def cube_pos(self):
        return self.d.xpos[self.body("cube")].copy()

    def tcp_pos(self):
        return self.d.site_xpos[self.site("tcp")].copy()


def load(model_path=None):
    """Load the lift-cube scene. Returns a Sim."""
    path = str(model_path or paths.SCENE_XML)
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return Sim(m, d)
