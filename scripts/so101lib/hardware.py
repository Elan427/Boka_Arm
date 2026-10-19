"""Thin wrapper over lerobot's SOFollower: sim<->real unit conversion, the
~20 Hz wall-clock send throttle, and torque control for hand-guided calibration.

All conversions go through a Calibration, so a flipped joint or an inverted
gripper is a one-line config change, not a code edit.
"""
import time

import numpy as np

from .calib import Calibration
from .config import (
    ARM_JOINT_NAMES, GRIPPER_JOINT, HARDWARE_ID, HARDWARE_PORT,
    HW_ARM_ACCELERATION, HW_CMD_SMOOTHING, HW_NO_INTEGRAL_JOINTS, HW_POSITION_PID,
    HW_SEND_HZ,
    MAX_RELATIVE_TARGET,
)


class Follower:
    def __init__(self, calib: Calibration, gripper_ctrlrange, port=HARDWARE_PORT, robot_id=HARDWARE_ID):
        self.calib = calib
        self.g_closed, self.g_open = gripper_ctrlrange  # sim actuator ctrl values
        self.port = port
        self.robot_id = robot_id
        self._robot = None
        self._last_send = 0.0
        self._cmd_ema = None  # smoothed arm command, see send()

    # --- lifecycle ------------------------------------------------------
    def connect(self):
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        print(f"Connecting to follower on {self.port} (id={self.robot_id})...")
        p, i, d = HW_POSITION_PID
        cfg = SOFollowerRobotConfig(
            port=self.port, id=self.robot_id, max_relative_target=MAX_RELATIVE_TARGET,
            position_p_coefficient=p, position_i_coefficient=i, position_d_coefficient=d,
        )
        self._robot = SOFollower(cfg)
        self._robot.connect(calibrate=True)
        # The gripper works against hard stops; an integral term winds up there
        # and trips the overload latch. Keep it P/D only -- the arm joints, which
        # need I to beat gravity droop, keep the configured gains.
        # Same for the joints with no gravity load (see HW_NO_INTEGRAL_JOINTS).
        for name in [GRIPPER_JOINT, *HW_NO_INTEGRAL_JOINTS]:
            try:
                self._robot.bus.write("I_Coefficient", name, 0)
            except Exception as e:
                print(f"  (could not clear {name} I term: {e})")
        # Ramp each arm-joint step so the 20 Hz setpoint stream reads as
        # continuous motion instead of a sprint-hold staircase.
        for name in ARM_JOINT_NAMES:
            try:
                self._robot.bus.write("Acceleration", name, HW_ARM_ACCELERATION)
            except Exception as e:
                print(f"  (could not set {name} acceleration: {e})")
        print("Follower connected.\n")

    def disconnect(self):
        if self._robot is None:
            return
        try:
            self._robot.disconnect()  # releases torque (disable_torque_on_disconnect=True)
            print("Follower disconnected (torque released).")
        except Exception as e:
            print(f"WARNING: motor fault on disconnect ({e}).")
            self._force_release()
        finally:
            self._robot = None

    def _force_release(self):
        """A motor latched an overload fault and refused torque-disable. Relieve
        the commanded stall, release every motor we can, and close the port
        without crashing. A latched overload may still need a power cycle."""
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            return
        try:
            present = bus.sync_read("Present_Position")
            for name, pos in present.items():
                try:
                    bus.write("Goal_Position", name, pos)  # stop pushing
                except Exception:
                    pass
            time.sleep(0.4)
        except Exception:
            pass
        released = []
        for name in ARM_JOINT_NAMES + [GRIPPER_JOINT]:
            for _ in range(3):
                try:
                    bus.disable_torque([name])
                    released.append(name)
                    break
                except Exception:
                    time.sleep(0.3)
        try:
            bus.disconnect(False)
        except Exception:
            pass
        stuck = [n for n in ARM_JOINT_NAMES + [GRIPPER_JOINT] if n not in released]
        if stuck:
            print(f"  still latched: {stuck} -- power-cycle the arm to clear the overload.")
        else:
            print("  released all motors after clearing the fault.")

    @property
    def connected(self):
        return self._robot is not None

    # --- torque (for hand-guided calibration) -------------------------
    def disable_torque(self, joints=None):
        self._robot.bus.disable_torque(joints)

    def enable_torque(self, joints=None):
        self._robot.bus.enable_torque(joints)

    def diagnose(self):
        """Per-motor health snapshot: torque on?, hardware-error bits, load,
        temperature, voltage. Reads each register on its own so one dead motor
        does not hide the rest."""
        bus = self._robot.bus
        rows = []
        for name in ARM_JOINT_NAMES + [GRIPPER_JOINT]:
            row = {}
            for reg in ("Torque_Enable", "Status", "Present_Load",
                        "Present_Temperature", "Present_Voltage", "Present_Position"):
                try:
                    row[reg] = bus.read(reg, name, normalize=False)
                except Exception:
                    row[reg] = None
            rows.append((name, row))
        return rows

    # --- io -----------------------------------------------------------
    def read_joints_rad(self):
        """Present arm position as 5 joint angles in radians (sim convention)."""
        pos = self._robot.bus.sync_read("Present_Position")
        deg5 = np.array([pos[name] for name in ARM_JOINT_NAMES], dtype=float)
        return self.calib.deg_to_rad(deg5)

    def read_gripper_current(self):
        """Gripper motor current in raw servo units (~6.5 mA each), or None."""
        try:
            return float(self._robot.bus.read("Present_Current", GRIPPER_JOINT, normalize=False))
        except Exception:
            return None

    def read_raw_deg(self):
        """Raw present position dict, degrees/percent, keyed by motor name."""
        return dict(self._robot.bus.sync_read("Present_Position"))

    def _gripper_pct(self, sim_ctrl):
        span = self.g_open - self.g_closed
        frac = 0.0 if span == 0 else (sim_ctrl - self.g_closed) / span
        c, o = self.calib.gripper_closed_pct, self.calib.gripper_open_pct
        return float(np.clip(c + frac * (o - c), 0, 100))

    def _action(self, ctrl6):
        deg5 = self.calib.rad_to_deg(ctrl6[:5])
        action = {f"{name}.pos": float(deg5[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
        action[f"{GRIPPER_JOINT}.pos"] = self._gripper_pct(ctrl6[5])
        return action

    def send(self, ctrl6, force=False):
        """Mirror a sim ctrl vector [5 arm radians, gripper sim-ctrl] to the arm.

        Throttled to HW_SEND_HZ by wall-clock time so sim speed does not change
        how fast the real motors are commanded. The arm angles are run through an
        exponential smoother (HW_CMD_SMOOTHING) so the setpoint stream eases
        rather than steps. force=True bypasses both the throttle and the smoother
        (used by home/settle/calibration, which want the exact pose).
        """
        if not self.connected:
            return
        now = time.time()
        if not force and now - self._last_send < 1.0 / HW_SEND_HZ:
            return
        self._last_send = now
        ctrl6 = np.asarray(ctrl6, dtype=float).copy()
        a = HW_CMD_SMOOTHING
        if force or self._cmd_ema is None or not (0.0 < a < 1.0):
            self._cmd_ema = ctrl6.copy()
        else:
            self._cmd_ema[:5] += a * (ctrl6[:5] - self._cmd_ema[:5])
            self._cmd_ema[5] = ctrl6[5]  # gripper is not smoothed
            ctrl6 = self._cmd_ema.copy()
        self._robot.send_action(self._action(ctrl6))
