"""Mink-based inverse kinematics wrapper for SO-101."""
import numpy as np
import mujoco
import mink
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace

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
    
    def __init__(self, model_path="assets/so101/scene_lift_cube.xml"):
        self.m = mujoco.MjModel.from_xml_path(model_path)
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
        
        # Posture task: keep arm near home in null space
        self.posture_task = mink.PostureTask(self.m, cost=1e-2)
        self.posture_task.set_target(self.home_q_full)
        
        self.tasks = [self.tcp_task, self.posture_task]
    
    def solve_to_pose(self, target_se3, max_iters=50, dt=0.01, verbose=False):
        """Solve IK to reach target SE3 pose.
        
        Args:
            target_se3: 4x4 SE3 matrix (position + rotation)
            max_iters: max iterations for the solver
            dt: integration timestep
            verbose: print convergence info
        
        Returns:
            q_sol: 6-DOF joint angles [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        self.cfg.q[:] = self.home_q_full.copy()
        
        # Convert numpy SE3 to MinkTransform
        if isinstance(target_se3, np.ndarray):
            target = MinkTransform(target_se3)
        else:
            target = target_se3
        
        self.tcp_task.set_target(target)
        
        for iter_num in range(max_iters):
            try:
                v = mink.solve_ik(self.cfg, self.tasks, dt, solver="daqp", damping=1e-3)
                self.cfg.integrate_inplace(v, dt)
                
                if verbose and iter_num % 10 == 0:
                    print(f"  iter {iter_num:3d}: solving...")
            except RuntimeError as e:
                if verbose:
                    print(f"  Solver warning at iter {iter_num}: {e}")
                break
        
        return self.cfg.q[:6].copy()
    
    def solve_to_position(self, target_pos_world, orientation=None, verbose=False):
        """Convenience: solve IK to reach a target position with optional orientation.
        
        Args:
            target_pos_world: [x, y, z] target position in world frame
            orientation: None for default (z-down), or 3x3 rotation matrix
        
        Returns:
            q_sol: 6-DOF joint angles
        """
        # Default orientation: Z-axis points down (for top-down grasp)
        if orientation is None:
            # TCP x-axis forward, y-axis right, z-axis down (into table)
            R = np.array([
                [1,  0,  0],  # x forward
                [0, -1,  0],  # y right (flipped for table frame)
                [0,  0, -1],  # z down
            ])
        else:
            R = orientation
        
        # Build SE3 as numpy matrix for conversion
        target_se3_np = np.eye(4)
        target_se3_np[:3, :3] = R
        target_se3_np[:3, 3] = target_pos_world
        
        return self.solve_to_pose(target_se3_np, verbose=verbose)


def se3_from_rotation_translation(R, p):
    """Build 4x4 SE3 from 3x3 rotation and 3D position."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rotation_matrix_z_down():
    """Rotation matrix for top-down grasp: z-axis points down."""
    return np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ])
