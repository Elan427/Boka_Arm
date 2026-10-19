"""Static constants shared across every SO-101 script.

Anything that varies with the physical rig (table height, joint sign, real cube
placement) lives in the calibration file instead -- see so101lib.calib.
"""
import numpy as np

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT]

# --- Real-hardware mirroring ----------------------------------------------
HARDWARE_PORT = "COM4"
HARDWARE_ID = "my_so101_follower"
HW_SEND_HZ = 25  # setpoint rate to the arm (wall-clock, independent of sim speed).
# 30+ once coincided with a bus dropout ("no status packet") on this rig; 25 is
# the current ceiling. More setpoints/s = smoother, if the bus stays reliable.

# Exponential smoothing on the commanded ARM joint angles, applied in
# Follower.send at the setpoint rate: cmd += HW_CMD_SMOOTHING * (target - cmd).
# 1.0 = off (send the raw IK target). Only used on the *throttled* (non-force)
# send path; the pick sequence now streams every move force=True (see below), so
# this mainly matters to other scripts that call Follower.send directly.
HW_CMD_SMOOTHING = 0.4

# Every move in the pick sequence is streamed the way HOME always was: a target
# re-sent force=True (no throttle, no EMA) as fast as this rate allows, letting
# each servo's own acceleration-limited profile (HW_ARM_ACCELERATION) carry it --
# one continuous motion instead of a 25 Hz staircase. This is the loop rate for
# that stream; lerobot's send_action already reads Present_Position each call
# (~5-15 ms), so the real ceiling is lower. Match it to what HOME feels like.
HW_STREAM_HZ = 80

# Feetech STS3215 position-loop gains written at connect time. lerobot's follower
# default (P=16, I=0, D=32) is tuned for smooth teleop replay, not for holding
# the arm's own weight -- with no integral term, elbow_flex/wrist_flex sag under
# gravity and never climb to target ("no torque"). More P + a little I fixes it.
HW_POSITION_PID = (32, 12, 16)  # (P, I, D)

# Joints that get I=0 (P/D only) at connect. They carry no gravity load, so the
# integral term buys nothing -- it just winds up while the joint holds still
# against friction/backlash or a cube pressed on the table, until the STS3215
# overload latch trips. shoulder_pan latched this way during GRASP and RELEASE.
HW_NO_INTEGRAL_JOINTS = ["shoulder_pan", "wrist_roll"]

# Feetech "Acceleration" register (addr 41), written to the arm joints at connect.
# 0 = the servo slews to each new Goal_Position as fast as it can (snappy, and the
# reason a stream of position setpoints looks like a staircase -- sprint, hold,
# sprint). A small nonzero value ramps each step in/out so the ~20 Hz setpoint
# stream blends into continuous motion. Units are ~100 steps/s^2; 10-40 is a
# gentle glide, higher tracks tighter. Gripper keeps 0 so the grasp stays crisp.
HW_ARM_ACCELERATION = 16

# Per-tick safety clamp (lerobot's max_relative_target): caps how far a single
# sync_write can move a motor from its last reported position, so a wrong sign or
# a bad IK solution ramps into a hard stop instead of snapping the real arm
# there. Degrees for the arm joints, 0-100 units for the gripper.
# Governs the real arm's top speed: max_relative_target * HW_SEND_HZ deg/s
# (4 * 25 = 100 deg/s). Raise this and HW_SEND_HZ together to go faster, but keep
# it modest: a fast move into a hard stop (bad IK, or a descend that jams the
# table because stop_on_contact only sees *sim* contact) trips the STS3215
# overload latch, and shoulder_pan has latched at 6 deg / 30 Hz + SPEED 2.0.
MAX_RELATIVE_TARGET = {name: 4 for name in ARM_JOINT_NAMES}
MAX_RELATIVE_TARGET[GRIPPER_JOINT] = 6.0

# --- Task geometry ------------------------------------------------------
CUBE_HALF = 0.015  # half-edge of the 3cm cube in scene_lift_cube.xml

# How far above the table the pinch point stops on descend. The fixed/moving pad
# proxies are only ~1.5cm apart when closed, so there is almost no vertical slack
# -- verified in sim, 0.002 still grasps (~96mm lift) but 0.004 fails outright.
DESCEND_HEIGHT_OFFSET = 0.0002

HOME_Q = np.zeros(5)  # arm home posture (radians), gripper handled separately
