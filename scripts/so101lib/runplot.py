"""Plots for a grip-learning run: sensor traces with each attempt marked, the learned
P(hold | squeeze) curve, and how the recommendation evolved. Saved as a PNG."""
import numpy as np

from .learn import TARGET_P, X_MAX, GripLearner

RED, BLUE, GREEN, AMBER, GREY = "#d1495b", "#2e86ab", "#3a9d5d", "#e0a030", "#8a93a0"


def save(path, touch_log, attempts, title="", new_from=0):
    """touch_log: iterable of (t, A0, A1, A0s, A1s). attempts: learner attempts, each
    with x, held and (for ones from this run) t0/t1 wall-clock bounds. The first
    `new_from` attempts came from earlier runs and have no trace to show."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = np.array(list(touch_log), dtype=float)
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax = fig.add_subplot(gs[0, :])
    if len(log):
        t = log[:, 0] - log[0, 0]
        ax.plot(t, log[:, 1], color=RED, lw=0.6, alpha=0.35)
        ax.plot(t, log[:, 2], color=BLUE, lw=0.6, alpha=0.35)
        ax.plot(t, log[:, 3], color=RED, lw=1.8, label="A0 left")
        ax.plot(t, log[:, 4], color=BLUE, lw=1.8, label="A1 right")
        for a in attempts[new_from:]:
            if "t0" not in a:
                continue
            c = GREEN if a["held"] else RED
            ax.axvspan(a["t0"] - log[0, 0], a["t1"] - log[0, 0], color=c, alpha=0.12)
            ax.text((a["t0"] + a["t1"]) / 2 - log[0, 0], ax.get_ylim()[1] * 0.97,
                    f"+{a['x'] * 100:.0f}%\n{'held' if a['held'] else 'DROPPED'}",
                    ha="center", va="top", fontsize=8, color=c)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sensor (ADC counts)")
    ax.set_title(title or "Gripper sensors during the learning run")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    L = GripLearner(attempts)
    xs = np.linspace(0.0, X_MAX, 161)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(xs * 100, L.curve(xs), color=BLUE, lw=2, label="learned P(hold)")
    ax2.axhline(TARGET_P, color=AMBER, ls="--", lw=1, label=f"target {TARGET_P:.0%}")
    rng = np.random.default_rng(0)
    for a in attempts:
        ax2.scatter(a["x"] * 100, float(a["held"]) + rng.uniform(-0.03, 0.03),
                    color=GREEN if a["held"] else RED, s=26, zorder=3)
    rec = L.recommend()
    if rec is not None:
        ax2.axvline(rec * 100, color=GREEN, lw=1.5, label=f"recommended +{rec * 100:.0f}%")
    ax2.set_xlabel("squeeze past first contact (% of jaw travel)")
    ax2.set_ylabel("P(object held)")
    ax2.set_ylim(-0.08, 1.08)
    ax2.set_title(f"What it learned ({len(attempts)} attempts)")
    ax2.legend(fontsize=8, loc="center right")
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 1])
    ns, recs, mus, sds = [], [], [], []
    for n in range(1, len(attempts) + 1):
        sub = GripLearner(attempts[:n])
        r = sub.recommend()
        m, sd = sub.threshold()
        ns.append(n)
        recs.append(np.nan if r is None else r * 100)
        mus.append(m * 100)
        sds.append(sd * 100)
    ns, mus, sds = np.array(ns), np.array(mus), np.array(sds)
    ax3.fill_between(ns, mus - sds, mus + sds, color=BLUE, alpha=0.2, label="50% threshold ± 1 sd")
    ax3.plot(ns, mus, color=BLUE, lw=1.5)
    ax3.plot(ns, recs, color=GREEN, lw=2, marker="o", ms=3, label="recommended grip")
    ax3.set_xlabel("attempt number")
    ax3.set_ylabel("squeeze past first contact (%)")
    ax3.set_title("Learning curve: estimate settling as data accumulates")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_staircase(path, touch_log, attempts, outcome, full_scale, title="",
                   unit="% of sensor full scale"):
    """Plots for a force-staircase run: sensor traces with every attempt marked, the
    staircase itself (target and achieved level per attempt, colored held / dropped),
    and the left / right sensor counts at each lift."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = np.array(list(touch_log), dtype=float)
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax = fig.add_subplot(gs[0, :])
    if len(log):
        t0 = log[0, 0]
        ax.plot(log[:, 0] - t0, log[:, 1], color=RED, lw=0.6, alpha=0.35)
        ax.plot(log[:, 0] - t0, log[:, 2], color=BLUE, lw=0.6, alpha=0.35)
        ax.plot(log[:, 0] - t0, log[:, 3], color=RED, lw=1.8, label="A0 left")
        ax.plot(log[:, 0] - t0, log[:, 4], color=BLUE, lw=1.8, label="A1 right")
        top = max(float(log[:, 3:5].max()), 50.0) * 1.15
        ax.set_ylim(0, top)
        for a in attempts:
            c = GREEN if a.held else RED
            ax.axvspan(a.t0 - t0, a.t1 - t0, color=c, alpha=0.12)
            ax.text((a.t0 + a.t1) / 2 - t0, top * 0.98,
                    f"{a.phase}\n{a.target:.0f}%\n{'held' if a.held else 'DROP'}",
                    ha="center", va="top", fontsize=7, color=c)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sensor (ADC counts)")
    ax.set_title(title or "Force staircase: sensors during the run")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    n = np.arange(1, len(attempts) + 1)
    tgt = np.array([a.target for a in attempts])
    ach = np.array([a.achieved for a in attempts])
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(n, tgt, color=GREY, lw=1.2, ls="--", label="target")
    ax2.plot(n, ach, color=BLUE, lw=1, alpha=0.6)
    for i, a in enumerate(attempts):
        ax2.scatter(i + 1, a.achieved, s=46, zorder=3, color=GREEN if a.held else RED,
                    marker="o" if a.phase in ("up", "down") else "D")
        ax2.annotate(a.phase[0], (i + 1, a.achieved), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=7, color=GREY)
    if outcome.status == "found":
        ax2.axhline(outcome.level, color=GREEN, lw=1.5, label=f"minimum grip {outcome.level:.1f}%")
    ax2.set_xlabel("attempt  (u = ascent, d = descent, c = confirm, b = bump)")
    ax2.set_ylabel(f"grip level ({unit})")
    ax2.set_title("The staircase: green = object held, red = dropped")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 1])
    w = 0.38
    ax3.bar(n - w / 2, [a.left for a in attempts], w, color=RED, label="A0 left")
    ax3.bar(n + w / 2, [a.right for a in attempts], w, color=BLUE, label="A1 right")
    ax3.set_xlabel("attempt")
    ax3.set_ylabel("sensor counts at the lift")
    ax3.set_title("Left / right sensor counts at each lift")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
