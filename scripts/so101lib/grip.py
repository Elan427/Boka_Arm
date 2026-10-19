"""Self-calibrating grip force: learn how hard this gripper can squeeze before the
jaw flex sensors saturate, then hold a safe fraction of that.

`train_trial` closes the gripper in small steps on a real object while watching the
two sensor levels (counts away from baseline, see so101lib.touch) and the gripper
motor current, and stops on the first of:

  plateau   a sensor stopped rising although the jaw kept closing -> it saturated;
            the level reached is that sensor's maximum reading
  current   motor current hit the limit (protects the servo and the object)
  travel    the jaw went MAX_OVERTRAVEL past first contact
  closed    the jaw reached fully closed

`fit_profile` combines several trials (median, so one bad trial does not matter)
into a GripProfile: per-sensor saturation level, contact position, sensitivity
(counts per unit of jaw travel) and the force target/limit the controller uses.
The profile is saved to config/grip_profile.json.

Everything here is hardware-agnostic: the caller supplies callables, so the logic
runs against a fake plant in tests.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import paths

STEP = 0.01             # jaw travel per step, as a fraction of full open->closed range
DWELL = 0.12            # s to let the jaw and sensors settle after each step
MAX_OVERTRAVEL = 0.20   # travel allowed past first contact
PLATEAU_RISE = 3.0      # counts: a rise smaller than this over PLATEAU_STEPS = flat
PLATEAU_STEPS = 3
TARGET_FRAC = 0.6       # grip target as a fraction of the learned saturation level
LIMIT_FRAC = 0.85       # never command more than this fraction of saturation

PROFILE_JSON = paths.CONFIG_DIR / "grip_profile.json"


@dataclass
class Trial:
    """One close-on-object run. `curve` rows are (travel, left, right, current)."""
    curve: list = field(default_factory=list)
    contact_travel: float | None = None
    max_level: tuple = (0.0, 0.0)
    stop: str = ""
    peak_current: float = 0.0


def train_trial(set_travel, read_levels, read_current, on_threshold, *, max_current=120,
                step=STEP, dwell=DWELL, sleep=time.sleep, progress=None) -> Trial:
    """Close from open (travel 0) toward closed (travel 1) on an object.

    set_travel(f)      command the jaw to travel fraction f in 0..1 (0 open, 1 closed)
    read_levels()      (left, right) counts away from the tared baseline
    read_current()     gripper motor current in raw servo units, or None if unreadable
    on_threshold       (left, right) touch-detect levels, counts
    """
    t = Trial()
    on = np.asarray(on_threshold, dtype=float)
    f = 0.0
    hist = []  # (left, right) after each step since contact
    while True:
        set_travel(f)
        sleep(dwell)
        lv = np.asarray(read_levels(), dtype=float)
        cur = read_current()
        cur = 0.0 if cur is None else float(cur)
        t.curve.append((f, lv[0], lv[1], cur))
        t.peak_current = max(t.peak_current, cur)
        t.max_level = (max(t.max_level[0], lv[0]), max(t.max_level[1], lv[1]))
        if progress:
            progress(f, lv, cur)

        if t.contact_travel is None and np.all(lv > on):
            t.contact_travel = f
        if t.contact_travel is not None:
            hist.append(lv)
            if len(hist) > PLATEAU_STEPS:
                rise = np.asarray(hist[-1]) - np.asarray(hist[-1 - PLATEAU_STEPS])
                if np.all(rise < PLATEAU_RISE):
                    t.stop = "plateau"
                    break
            if f - t.contact_travel >= MAX_OVERTRAVEL:
                t.stop = "travel"
                break
        if cur >= max_current:
            t.stop = "current"
            break
        if f >= 1.0:
            t.stop = "closed"
            break
        f = min(1.0, f + step)
    return t


@dataclass
class GripProfile:
    # Per-sensor [left, right], in counts away from baseline.
    sat_level: list = field(default_factory=lambda: [0.0, 0.0])      # learned maximum reading
    target_level: list = field(default_factory=lambda: [0.0, 0.0])   # what the controller holds
    limit_level: list = field(default_factory=lambda: [0.0, 0.0])    # never exceed
    sensitivity: list = field(default_factory=lambda: [1.0, 1.0])    # counts per unit jaw travel
    contact_travel: float = 0.0       # jaw travel fraction where both jaws first touch
    peak_current: float = 0.0
    trials: int = 0
    stops: list = field(default_factory=list)
    object: str = ""
    updated: str = ""
    # Filled by find_min_grip.py: the lightest squeeze that still lifted the object.
    min_hold_travel: float = 0.0                                      # jaw travel fraction used
    min_hold_level: list = field(default_factory=lambda: [0.0, 0.0])  # sensor levels while holding
    min_hold_attempts: int = 0

    @classmethod
    def load(cls, path=None):
        path = path or PROFILE_JSON
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path=None):
        path = path or PROFILE_JSON
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated = _dt.datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")
        print(f"Wrote {path}")


def fit_profile(trials, on_threshold, obj="", target_frac=TARGET_FRAC, limit_frac=LIMIT_FRAC):
    """Combine trials into a GripProfile. Trials that never made contact are skipped."""
    good = [t for t in trials if t.contact_travel is not None]
    if not good:
        raise RuntimeError("no trial made contact with the object -- nothing to learn")
    on = np.asarray(on_threshold, dtype=float)
    sat = np.median([t.max_level for t in good], axis=0)

    slopes = []
    for t in good:
        c = np.array(t.curve)
        for i in (0, 1):
            # slope over the part of the squeeze from contact up to half of saturation
            sel = (c[:, 0] >= t.contact_travel) & (c[:, 1 + i] <= 0.5 * t.max_level[i])
            sel &= c[:, 1 + i] > 0
            if sel.sum() >= 3:
                slopes.append((i, np.polyfit(c[sel, 0], c[sel, 1 + i], 1)[0]))
    sens = [float(np.median([s for j, s in slopes if j == i] or [1.0])) for i in (0, 1)]

    p = GripProfile()
    p.sat_level = [float(v) for v in sat]
    p.target_level = [float(max(target_frac * s, 2 * o)) for s, o in zip(sat, on)]
    p.limit_level = [float(max(limit_frac * s, 3 * o)) for s, o in zip(sat, on)]
    p.sensitivity = [max(s, 1.0) for s in sens]
    p.contact_travel = float(np.median([t.contact_travel for t in good]))
    p.peak_current = float(max(t.peak_current for t in good))
    p.trials = len(good)
    p.stops = [t.stop for t in good]
    p.object = obj
    return p


def combine_min_holds(results):
    """Reduce several mingrip.Result("found") runs to one conservative grip: the
    largest closure any run needed, the highest sensor levels seen holding, the
    median first-contact position. Returns (travel, levels, contact_travel, attempts)."""
    found = [r for r in results if r.status == "found"]
    if not found:
        raise RuntimeError("no run held the object -- nothing to learn")
    travel = max(r.travel for r in found)
    levels = np.max([r.levels for r in found], axis=0)
    contact = float(np.median([r.contact_travel for r in found]))
    return travel, levels, contact, sum(r.attempts for r in results)


def apply_min_hold(profile, levels, travel, contact_travel, attempts, margin=1.25):
    """Record the result of a minimum-grip search in `profile` (created if None) and
    make it the controller's hold target. Returns the profile.

    The target is margin x the sensor levels that lifted the object, but never above
    what the sensors can read: 0.9 x the trained saturation level if train history
    exists, otherwise the measured levels themselves (the sensors top out near the
    ADC ceiling, so scaling a reading that is already near it would give a target the
    controller could never reach and it would squeeze until its step limit)."""
    p = profile or GripProfile()
    lv = np.asarray(levels, dtype=float)
    p.min_hold_travel = float(travel)
    p.min_hold_level = [float(v) for v in lv]
    p.min_hold_attempts = int(attempts)
    p.contact_travel = float(contact_travel)
    if any(p.sat_level):
        tgt = np.minimum(margin * lv, 0.9 * np.asarray(p.sat_level))
        tgt = np.maximum(tgt, lv)          # never below the grip that actually held
    else:
        tgt = lv
    p.target_level = [float(v) for v in tgt]
    p.limit_level = [float(v) for v in 1.3 * tgt]
    # Rough counts-per-travel from this search; the sensors rise over a few % of travel.
    run = max(travel - contact_travel, 0.05)
    p.sensitivity = [float(max(v / run, 1.0)) for v in lv]
    return p


def squeeze_step(profile, levels, max_step):
    """One closed-loop grip-force update. Returns the change in jaw travel
    (fraction): >0 tighten, <0 back off, 0 hold.

    Tightens until the jaws' mean level reaches target, at most max_step per call
    and in proportion to the remaining error, using the learned sensitivity. Backs off immediately if either jaw exceeds its limit level."""
    lv = np.asarray(levels, dtype=float)
    tgt = np.asarray(profile.target_level)
    lim = np.asarray(profile.limit_level)
    sens = np.asarray(profile.sensitivity)
    over = lv - lim
    if np.any(over > 0):
        i = int(np.argmax(over))
        return -float(min(max_step, over[i] / sens[i]))
    err = (tgt - lv) / sens           # travel still needed per jaw
    need = float(np.mean(err))        # average of the two jaws; the limit check above guards each
    if need <= 0.002:
        return 0.0
    return float(min(max_step, need))
