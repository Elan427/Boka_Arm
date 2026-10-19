"""Physical-rig calibration: the handful of numbers that differ between the sim
model and the real SO-101 on its table.

Written by scripts/hw_check.py (joint sign/offset, gripper direction) and
scripts/touch_off.py (table_z, tcp_xy_bias). Consumed by scripts/SimToArm.py.

Kept as a plain tracked JSON file (config/so101_calib.json) so it is versioned
with the rest of the generated artifacts. Every field has a safe default, so all
scripts run with no file present -- load() just prints a warning.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from dataclasses import dataclass, field

import numpy as np

from . import paths
from .config import CUBE_HALF, DESCEND_HEIGHT_OFFSET


@dataclass
class Calibration:
    # World Z (model frame) of the TCP site when the real pinch point rests on
    # the table. The table plane is nominally z=0; this absorbs the constant
    # TCP-site offset error without editing the model XML.
    table_z: float = 0.0
    # Base-frame XY error between where the model thinks the TCP is and where it
    # physically is, as a rigid transform: real_xy ~= Rot(xy_rot_deg) @ model_xy
    # + tcp_xy_bias. xy_rot_deg absorbs a shoulder_pan offset error (rotates the
    # whole workspace); tcp_xy_bias absorbs a constant shift. Both are fitted by
    # scripts/calibrate_xy.py and undone by apply_target().
    tcp_xy_bias: list[float] = field(default_factory=lambda: [0.0, 0.0])
    xy_rot_deg: float = 0.0
    # Where the real cube actually sits, measured from the base (metres).
    cube_xy: list[float] = field(default_factory=lambda: [0.15, 0.15])
    # Sim(radians)->real(degrees) convention, per arm joint.
    #   real_deg = joint_sign[i] * degrees(q_rad[i]) + joint_offset_deg[i]
    joint_sign: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    joint_offset_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # lerobot gripper units: 0..100. Which end is open/closed on this rig.
    gripper_open_pct: float = 100.0
    gripper_closed_pct: float = 0.0
    updated: str = ""
    notes: str = ""

    # --- io ----------------------------------------------------------------
    @classmethod
    def load(cls, path=None) -> "Calibration":
        path = path or paths.CALIB_JSON
        if not path.exists():
            print(f"WARNING: no calib file at {path} -- using defaults (sim-only values).")
            return cls()
        data = json.loads(path.read_text())
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path=None) -> None:
        path = path or paths.CALIB_JSON
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated = _dt.datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")
        print(f"Wrote {path}")

    # --- use -------------------------------------------------------------
    def _rot(self, deg):
        t = np.radians(deg)
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s], [s, c]])

    def apply_target(self, pos):
        """Map a desired base-frame TCP point to the model target that, once the
        rig error is undone, puts the real TCP there: invert the fitted
        rotation + translation, then clamp Z to the table."""
        pos = np.asarray(pos, dtype=float).copy()
        xy = self._rot(-self.xy_rot_deg) @ (pos[:2] - np.asarray(self.tcp_xy_bias))
        pos[0], pos[1] = xy
        pos[2] = max(pos[2], self.table_z)
        return pos

    def predict_real_xy(self, model_xy):
        """Forward: where the real TCP lands when the model TCP is at model_xy."""
        model_xy = np.asarray(model_xy, dtype=float)
        return self._rot(self.xy_rot_deg) @ model_xy + np.asarray(self.tcp_xy_bias)

    @staticmethod
    def fit_xy_correction(model_xys, real_xys):
        """Rigid 2D fit real ~= Rot(theta) @ model + t. Returns (rot_deg, [tx, ty],
        rms_residual_m)."""
        m = np.asarray(model_xys, dtype=float)
        r = np.asarray(real_xys, dtype=float)
        mc, rc = m.mean(0), r.mean(0)
        h = (m - mc).T @ (r - rc)
        theta = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        t = rc - rot @ mc
        resid = r - (m @ rot.T + t)
        rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
        return float(np.degrees(theta)), [float(t[0]), float(t[1])], rms

    def descend_z(self):
        """Target Z for the descend phase: just above the table surface."""
        return self.table_z + CUBE_HALF + DESCEND_HEIGHT_OFFSET

    def rad_to_deg(self, q5):
        """Sim arm angles (radians) -> real motor targets (degrees)."""
        q5 = np.asarray(q5, dtype=float)
        return np.asarray(self.joint_sign, dtype=float) * np.degrees(q5) + np.asarray(
            self.joint_offset_deg, dtype=float
        )

    def deg_to_rad(self, deg5):
        """Real motor readings (degrees) -> sim arm angles (radians)."""
        deg5 = np.asarray(deg5, dtype=float)
        return np.radians(
            (deg5 - np.asarray(self.joint_offset_deg, dtype=float))
            / np.asarray(self.joint_sign, dtype=float)
        )
