"""A single in-place terminal readout of the arm's commanded joint angles.

The motion primitives push updates here instead of printing a per-phase log, so
nothing scrolls: the block below is rewritten in place as the arm moves.

    hud.context("cycle 3")            # left tag, persists across phases
    hud.phase("CARRY to drop spot")   # current phase
    hud.update(follower, ctrl6)       # redraw with the latest command
    hud.done()                        # drop the cursor below the block
"""
import sys
import time

import numpy as np

from .config import ARM_JOINT_NAMES

_context = ""
_phase = ""
_note = ""
_prev_lines = 0
_last = 0.0
_enabled = True


def enable(on=True):
    global _enabled
    _enabled = on


def context(text=""):
    global _context
    _context = str(text)


def phase(text="", note=""):
    global _phase, _note
    _phase = str(text).strip()
    _note = str(note)


def note(text=""):
    global _note
    _note = str(text)


def _bar(v, lo=-135.0, hi=135.0, w=25):
    mid = w // 2
    c = int(round((np.clip(v, lo, hi) - lo) / (hi - lo) * (w - 1)))
    cells = [" "] * w
    cells[mid] = "|"
    for i in range(min(mid, c), max(mid, c) + 1):
        cells[i] = "#"
    return "[" + "".join(cells) + "]"


def _lines(deg5, gripper):
    head = f"  {_context}   {_phase}".rstrip()
    if _note:
        head += f"   [{_note}]"
    rows = [head, ""]
    for name, v in zip(ARM_JOINT_NAMES, deg5):
        rows.append(f"    {name:<13}{v:+8.2f} deg  {_bar(v)}")
    rows.append(f"    {'gripper':<13}{gripper:+8.2f}")
    return rows


def _render(deg5, gripper, force=False, period=0.05):
    global _prev_lines, _last
    if not _enabled:
        return
    now = time.time()
    if not force and now - _last < period:
        return
    _last = now
    rows = _lines(deg5, gripper)
    buf = []
    if _prev_lines:
        buf.append(f"\x1b[{_prev_lines}A")
    for r in rows:
        buf.append("\r\x1b[2K" + r + "\n")
    sys.stdout.write("".join(buf))
    sys.stdout.flush()
    _prev_lines = len(rows)


def update(follower, ctrl6, force=False):
    """Redraw the block from a sim ctrl vector (5 arm radians + gripper)."""
    ctrl6 = np.asarray(ctrl6, dtype=float)
    deg5 = follower.calib.rad_to_deg(ctrl6[:5])
    _render(deg5, float(ctrl6[5]), force=force)


def done():
    """Move the cursor below the block so normal prints / input() are clean."""
    global _prev_lines
    if _prev_lines:
        sys.stdout.write("\n")
        sys.stdout.flush()
    _prev_lines = 0
