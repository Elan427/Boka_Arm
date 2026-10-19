"""Learn the lightest reliable grip from experience (Bayesian active learning).

Each attempt squeezes the object by x (extra jaw travel beyond first contact, as a
fraction of the full gripper range), lifts and jolts it, and records whether it was
held. We model

        P(hold | x) = lapse + (1 - 2 lapse) * sigmoid((x - x0) / s)

with unknown threshold x0 (the squeeze at which holding becomes 50 % likely) and
sharpness s, and keep a full posterior over (x0, s) on a grid. After every attempt:

  * the posterior is updated from ALL stored attempts (they persist in
    config/grip_history.json, so it keeps learning across runs and sessions)
  * the next x to try is drawn by Thompson sampling, aimed at the squeeze where holding
    would become reliable (P = TARGET_P) -- so it explores where it matters and stops
    wasting attempts on grips that are clearly too loose or far too tight
  * `recommend()` returns the smallest x whose posterior-mean hold probability reaches
    TARGET_P, i.e. the learned lightest reliable grip, with an honest uncertainty.

The `lapse` term keeps one freak drop or one lucky hold from dominating the fit.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import paths

HISTORY_JSON = paths.CONFIG_DIR / "grip_history.json"

X_MAX = 0.40        # never try more than this much extra travel past first contact
TARGET_P = 0.95     # reliability the recommended grip must reach
LAPSE = 0.03        # chance an attempt's outcome is a fluke either way
_X0 = np.linspace(0.0, X_MAX, 81)                  # candidate thresholds
_S = np.array([0.004, 0.008, 0.015, 0.03, 0.06])   # candidate sharpnesses
_XS = np.linspace(0.0, X_MAX, 161)                 # grid for the P(hold | x) curve


def _p(x, x0, s):
    z = 1.0 / (1.0 + np.exp(-(x - x0) / s))
    return LAPSE + (1.0 - 2.0 * LAPSE) * z


class GripLearner:
    def __init__(self, attempts=None):
        self.attempts = list(attempts or [])        # [{"x": float, "held": bool, ...}]

    # --- model ------------------------------------------------------------------
    def posterior(self):
        """Weights over the (x0, s) grid, shape (len(_X0), len(_S)), summing to 1."""
        x0, s = np.meshgrid(_X0, _S, indexing="ij")
        logp = -0.5 * ((x0 - 0.10) / 0.15) ** 2           # weak prior: threshold near 10 %
        for a in self.attempts:
            p = _p(a["x"], x0, s)
            logp += np.log(p if a["held"] else 1.0 - p)
        logp -= logp.max()
        w = np.exp(logp)
        return w / w.sum()

    def curve(self, xs=_XS):
        """Posterior-mean P(hold) at each x in xs."""
        w = self.posterior()
        x0, s = np.meshgrid(_X0, _S, indexing="ij")
        return np.array([float((w * _p(x, x0, s)).sum()) for x in xs])

    def threshold(self):
        """Posterior mean and std of the 50 % threshold x0."""
        w = self.posterior().sum(axis=1)
        m = float((w * _X0).sum())
        return m, float(np.sqrt((w * (_X0 - m) ** 2).sum()))

    # --- decisions ---------------------------------------------------------------
    def choose(self, rng=None):
        """Next squeeze to try: Thompson-sample (x0, s), then aim at its TARGET_P point."""
        rng = rng or np.random.default_rng()
        w = self.posterior().ravel()
        i = rng.choice(w.size, p=w)
        x0 = _X0[i // len(_S)]
        s = _S[i % len(_S)]
        z = np.log(TARGET_P / (1.0 - TARGET_P))
        return float(np.clip(x0 + z * s, 0.0, X_MAX))

    def recommend(self, target=TARGET_P):
        """Smallest squeeze with posterior-mean hold probability >= target, or None."""
        c = self.curve()
        ok = np.where(c >= target)[0]
        return float(_XS[ok[0]]) if ok.size else None

    def add(self, x, held, **extra):
        self.attempts.append({"x": float(x), "held": bool(held), "t": time.time(), **extra})

    def converged(self, window=3, tol=0.02):
        """Recommendation has moved < tol (2 % travel) over the last `window` attempts (needs >= 6)."""
        if len(self.attempts) < max(6, window + 1):
            return False
        recs = []
        for k in range(window + 1):
            sub = GripLearner(self.attempts[: len(self.attempts) - k])
            r = sub.recommend()
            if r is None:
                return False
            recs.append(r)
        return max(recs) - min(recs) < tol


# --- persistence (per object) -------------------------------------------------------

def load_history(obj, path=None):
    path = path or HISTORY_JSON
    if not path.exists():
        return []
    return json.loads(path.read_text()).get(obj, [])


def save_history(obj, attempts, path=None):
    path = path or HISTORY_JSON
    data = json.loads(path.read_text()) if path.exists() else {}
    data[obj] = attempts
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
