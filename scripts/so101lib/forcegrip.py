"""Find the minimum grip FORCE that still picks the object up -- a force staircase.

The controlled quantity is the flex-sensor reading (a force proxy), expressed as a
percentage of the sensor's full scale. One "attempt" = squeeze until the mean sensor
reading reaches the target, lift, and see whether the object stays held.

  1. ASCENT   start at 0 % (no squeeze -- it will not hold). Raise the target by `up`
              (5 %) after every failure until a lift holds.
  2. DESCENT  from the first hold, lower the target by `down` (1 %) while the lift
              still holds. Stop at the first failure, or when the gripper cannot
              settle on a lower reading than before (the sensor is steeper than the
              gripper's finest step -- resolution limit).
  3. CONFIRM  the answer must hold `confirm` times in a row. If the lowest hold fails
              on re-test, bump the target up `bump` (e.g. 2 %) and confirm again, so a
              lucky single hold never becomes the answer.

`staircase` is hardware-agnostic: the caller supplies `try_level(phase, target) ->
Attempt` (raising `Abort` to stop), so the logic runs against a fake plant in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class Abort(Exception):
    """Raised by try_level to stop the search (viewer closed, no contact, a limit)."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Attempt:
    phase: str = ""           # "up" | "down" | "confirm" | "bump"
    target: float = 0.0       # requested level, % of full scale
    achieved: float = 0.0     # level actually reached before the lift, % of full scale
    held: bool = False
    left: float = 0.0         # sensor counts while holding / at the lift
    right: float = 0.0
    travel: float = 0.0       # jaw travel fraction at the lift
    contact: float = 0.0      # jaw travel fraction at first contact
    t0: float = 0.0
    t1: float = 0.0


@dataclass
class Outcome:
    status: str                       # "found" | "never_held" | "max_attempts" | abort reason
    level: float = 0.0                # lowest confirmed holding level, % of full scale
    best: Attempt | None = None       # the last holding attempt at that level
    attempts: list = field(default_factory=list)
    floor: bool = False               # descent stopped on the resolution limit, not a failure


def staircase(try_level, up=5.0, down=1.0, bump=2.0, confirm=2, max_level=100.0,
              max_attempts=40, log=lambda *_: None) -> Outcome:
    atts: list[Attempt] = []

    def go(phase, target):
        if len(atts) >= max_attempts:
            raise Abort("max_attempts")
        a = try_level(phase, target)
        a.phase, a.target = phase, target
        atts.append(a)
        log(a)
        return a

    try:
        # 1. ascent: 0 %, 5 %, 10 % ... until a lift holds
        target = 0.0
        while True:
            a = go("up", target)
            if a.held:
                break
            target += up
            if target > max_level:
                return Outcome("never_held", attempts=atts)
        best, streak, floor = a, 1, False

        # 2. descent: lower by `down` while it still holds
        while True:
            t = best.achieved - down
            if t < 0:
                floor = True
                break
            a = go("down", t)
            if a.achieved > best.achieved - down / 2:   # could not settle any lower
                floor = True
                break
            if a.held:
                best, streak = a, 1
            else:
                break

        # 3. confirm; on a failed re-test bump up and start the streak again
        level = best.achieved
        while streak < confirm:
            a = go("confirm" if streak else "bump", level)
            if a.held:
                best, streak = a, streak + 1
            else:
                level = max(level, a.achieved) + bump
                streak = 0
                if level > max_level:
                    return Outcome("never_held", attempts=atts)
        return Outcome("found", best.achieved, best, atts, floor)
    except Abort as e:
        return Outcome(e.reason, attempts=atts)
