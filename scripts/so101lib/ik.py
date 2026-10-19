"""Mink-based inverse kinematics wrapper for SO-101, plus a forward-kinematics
helper for the arm.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from . import paths


class MinkTransform:
    """Simple transform wrapper for mink compatibility."""

    def __init__(self, se3_matrix):
        """Create from 4x4 SE3 matrix."""
        self.matrix = se3_matrix.copy()
        rotation_matrix = se3_matrix[:3, :3]
        position = se3_matrix[:3, 3]

        # Convert rotation matrix to quaternion (xyzw then convert to wxyz)
        rot = R.from_matrix(rotation_matrix)
        quat_xyzw = rot.as_quat()  # Returns [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        self.wxyz_xyz = np.concatenate([quat_wxyz, position])

    def copy(self):
        """Return a copy of this transform."""
        return MinkTransform(self.matrix)


class SO101IK:
    """Inverse kinematics solver for SO-101 using mink."""

    def __init__(self, model_path=None):
        import mink

        self.mink = mink
        self.m = mujoco.MjModel.from_xml_path(str(model_path or paths.SCENE_XML))
        self.cfg = mink.Configuration(self.m)

        # Set home posture as null-space target (full DOF: arm + cube freejoint)
        self.home_q_full = np.zeros(self.m.nq)
        self.home_q_full[:6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # cube freejoint defaults to zero (home at origin)

        # TCP task: servo the tcp site to a target pose
        self.tcp_task = mink.FrameTask(
            frame_name="tcp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.3,
            lm_damping=1.0,
        )

        # Posture task: gently bias the null space toward home. Low cost so it
        # does not fight the position target (1e-2 left ~2 mm of residual).
        self.posture_task = mink.PostureTask(self.m, cost=1e-3)
        self.posture_task.set_target(self.home_q_full)

        self.tasks = [self.tcp_task, self.posture_task]

    def _fk_tcp_pos(self):
        """TCP site world position at the solver's current configuration."""
        self.cfg.update(self.cfg.q)
        return self.cfg.data.site_xpos[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")].copy()

    def solve_to_pose(self, target_se3, max_iters=80, dt=0.01, tol=1e-3,
                      warm_start=None, verbose=False):
        """Solve IK to a target SE3 pose. Iterates until the TCP position error
        is under `tol` (m) or `max_iters` is hit.

        Returns (q6, pos_err_m). q6 = [pan, lift, elbow, wrist_flex, wrist_roll,
        gripper]; the gripper DOF is untouched and should be overwritten by the
        caller.
        """
        q0 = self.home_q_full.copy()
        if warm_start is not None:
            q0[:len(warm_start)] = warm_start
        self.cfg.q[:] = q0

        target = target_se3 if not isinstance(target_se3, np.ndarray) else MinkTransform(target_se3)
        self.tcp_task.set_target(target)
        goal = np.asarray(target_se3)[:3, 3] if isinstance(target_se3, np.ndarray) else None

        err = np.inf
        for i in range(max_iters):
            try:
                v = self.mink.solve_ik(self.cfg, self.tasks, dt, solver="daqp", damping=1e-3)
            except RuntimeError as e:
                if verbose:
                    print(f"  solver stopped at iter {i}: {e}")
                break
            self.cfg.integrate_inplace(v, dt)
            if goal is not None:
                err = float(np.linalg.norm(self._fk_tcp_pos() - goal))
                if err < tol:
                    break
        if verbose:
            print(f"  IK: {i + 1} iters, pos err {err * 1000:.2f} mm")
        return self.cfg.q[:6].copy(), err

    def solve(self, target_pos_world, orientation=None, warm_start=None, verbose=False):
        """Position IK with the default top-down grasp orientation.
        Returns (q6, pos_err_m)."""
        rotation = np.eye(3) if orientation is None else orientation
        se3 = np.eye(4)
        se3[:3, :3] = rotation
        se3[:3, 3] = np.asarray(target_pos_world, dtype=float)
        return self.solve_to_pose(se3, warm_start=warm_start, verbose=verbose)

    def solve_to_position(self, target_pos_world, orientation=None, verbose=False):
        """Back-compat: returns just q6 (see solve() for the error too).

        Default orientation is a top-down grasp: the TCP approach axis is the
        gripper body's local -z, so mapping local +z to world +z (identity)
        points it straight down.
        """
        return self.solve(target_pos_world, orientation, verbose=verbose)[0]


_fk_model = None
_fk_data = None
_fk_tcp_sid = None


def fk(q5, model_path=None):
    """Forward kinematics for the arm: 5 joint angles (radians) -> TCP world pose.

    Uses a private model/data so it never disturbs a caller's simulation.

    Returns:
        dict with "pos" (xyz, shape (3,)) and "mat" (3x3 rotation).
    """
    global _fk_model, _fk_data, _fk_tcp_sid
    if _fk_model is None:
        _fk_model = mujoco.MjModel.from_xml_path(str(model_path or paths.SCENE_XML))
        _fk_data = mujoco.MjData(_fk_model)
        _fk_tcp_sid = mujoco.mj_name2id(_fk_model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    _fk_data.qpos[:] = 0
    _fk_data.qpos[:5] = q5
    mujoco.mj_forward(_fk_model, _fk_data)
    return {
        "pos": _fk_data.site_xpos[_fk_tcp_sid].copy(),
        "mat": _fk_data.site_xmat[_fk_tcp_sid].reshape(3, 3).copy(),
    }
