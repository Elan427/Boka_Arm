"""Learn the minimum grip FORCE that still picks the object up.

Default method -- a continuous grasp session driven by the force staircase
(so101lib/mingrip.py + so101lib/forcegrip.py). The jaws close on the object ONCE, at the
lowest sensor value that counts as touch, and then never let go until the end:
  1. ASCENT   lift a few cm right at first contact. If the sensor value drops (the object
              slips), go straight back down and close the jaws --up-step (5 %) more of
              their travel, then lift again -- until a lift holds.
  2. DESCENT  once it holds, lower the object back, open the jaws by --down-step (1 %) and
              lift again, while it still holds; stop at the first failure.
  3. CONFIRM  the answer must hold --confirm times in a row; if its re-test fails, close
              --bump (2 %) more and confirm again.
The result is the lowest confirmed holding grip (extra jaw travel past first contact).
(--control level instead servos each attempt to a target sensor READING in % of
--full-scale, re-gripping from scratch every time.) Optionally (--method bayes) a
Bayesian model of P(hold | extra jaw travel) instead -- see so101lib/learn.py.

It saves the result as the hold target in config/grip_profile.json (after you confirm),
every attempt to config/grip_history.json, and a plot of the run to out/.

Safety: the first close gives up if neither jaw reads by --max-blind of the travel;
every squeeze is stopped by --max-current and the profile's sensor limit; the jaws
never close more than 30 % past first contact; the object is dropped (~5 cm) on every
failed attempt, so use a soft object. Support the arm: it goes limp when the script
exits. Wire both sensors, check them with touch_monitor.py / plot_sensors.py, and keep
your hands off them during the tare.

Run: python scripts/train_grip.py [--object ball] [--smooth 0.3] [--jolt-cm 4]
         [--up-step 5] [--down-step 1] [--bump 2] [--confirm 2] [--full-scale 600]
         [--max-attempts 40] [--max-current 120] [--max-blind 1.0]
         [--first-contact both|any] [--require both|any] [--no-plot]
         [--method bayes --trials 12 --until-converged --fresh]
"""
import argparse
import time
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np

from so101lib import forcegrip, grip, hud, learn, liveplot, mingrip, paths, pickplace, runplot
from so101lib.touch import TouchSensors

HARDWARE_ENABLED = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3", help="Arduino serial port")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="extra sensor smoothing: weight of the newest sample (0 = off, 0.1 = heavy)")
    ap.add_argument("--method", choices=["staircase", "bayes"], default="staircase")
    ap.add_argument("--control", choices=["travel", "level"], default="travel",
                    help="travel: stay gripped, adjust jaw closure between lifts (default); "
                         "level: servo each attempt to a sensor reading")
    ap.add_argument("--up-step", type=float, default=5.0,
                    help="ascent step: %% of jaw travel (travel) or of sensor full scale (level)")
    ap.add_argument("--down-step", type=float, default=1.0, help="descent step, same unit")
    ap.add_argument("--bump", type=float, default=2.0,
                    help="step back up when the lowest hold fails its re-test, same unit")
    ap.add_argument("--confirm", type=int, default=2, help="consecutive holds that confirm the answer")
    ap.add_argument("--full-scale", type=float, default=600.0,
                    help="sensor counts that count as 100%% (your sensors top out near 630)")
    ap.add_argument("--max-attempts", type=int, default=40)
    ap.add_argument("--jolt-cm", type=float, default=None,
                    help="sideways jolt at the top of the lift, cm (default: 0 staircase, 4 bayes)")
    ap.add_argument("--trials", type=int, default=12, help="bayes: attempts to run this session")
    ap.add_argument("--until-converged", action="store_true", help="bayes: stop once settled")
    ap.add_argument("--max-x", type=float, default=learn.X_MAX,
                    help="bayes: largest extra squeeze, fraction of the jaw travel")
    ap.add_argument("--max-current", type=float, default=120,
                    help="stop closing at this gripper current (raw units, ~6.5 mA each)")
    ap.add_argument("--first-contact", choices=["both", "any"], default="both",
                    help="close until BOTH jaw sensors read (default) or until the first one does")
    ap.add_argument("--max-blind", type=float, default=mingrip.MAX_BLIND,
                    help="give up if no first-contact signal by this fraction of the jaw travel")
    ap.add_argument("--require", choices=["both", "any"], default="both",
                    help="which jaw sensors must still read after the lift to count as held")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="saved hold target = margin x the sensor levels of the learned grip")
    ap.add_argument("--object", default="cube", help="history/profile name (e.g. ball)")
    ap.add_argument("--fresh", action="store_true", help="bayes: ignore stored history")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG saved at the end")
    ap.add_argument("--no-live", action="store_true", help="skip the live sensor window")
    args = ap.parse_args()
    jolt = args.jolt_cm if args.jolt_cm is not None else (4.0 if args.method == "bayes" else 0.0)

    touch = TouchSensors(args.port, smooth=args.smooth)
    touch.start()
    live = None
    if not args.no_live:
        live = liveplot.LivePlot(touch)
        live.start()
    pickplace.silence_clamp_warnings()
    ctx = pickplace.build(HARDWARE_ENABLED)
    sim, follower, calib = ctx["sim"], ctx["follower"], ctx["calib"]
    pick_xy = np.array(calib.cube_xy)
    profile = grip.GripProfile.load()
    result = None

    try:
        input("Jaws open and untouched, then ENTER to tare the touch sensors: ")
        base, noise = touch.tare()
        print(f"touch baseline {np.round(base, 1).tolist()}  noise {np.round(noise, 2).tolist()}  "
              f"on-threshold {np.round(touch.thresholds(), 1).tolist()} counts")
        issues = touch.problems()
        if issues:
            print("\nTOUCH SENSORS NOT READY -- the grip search could not detect a squeeze:")
            for m in issues:
                print("  -", m)
            print("Fix the wiring (run touch_monitor.py), then retry. Nothing was moved.")
            if live:
                input("Live plot is still open -- ENTER to close it and exit: ")
            return
        print("Checking the sensors respond:")
        if not all(touch.response_check()):
            print("A sensor did not respond to a press -- check its wiring. Nothing was moved.")
            return
        if follower.read_gripper_current() is None:
            print("WARNING: cannot read gripper current, so the current limit is NOT protecting "
                  "the object. Only the sensor limit, --max-blind and the 30%-past-contact cap apply.")
            if not input("Continue anyway? [y/n]: ").strip().lower().startswith("y"):
                return
        limit = profile.limit_level if profile and any(profile.limit_level) else None
        common = dict(max_current=args.max_current, require=args.require, limit_level=limit,
                      max_blind=args.max_blind, first_contact=args.first_contact, jolt_cm=jolt)

        with mujoco.viewer.launch_passive(ctx["m"], ctx["d"]) as viewer:
            sim.set_cube_pose(pick_xy, ctx["cube_hold_z"])
            viewer.sync()
            pickplace.go_home(ctx, viewer)
            if args.method == "staircase":
                result = run_staircase(args, ctx, viewer, touch, pick_xy, common)
            else:
                result = run_bayes(args, ctx, viewer, touch, pick_xy, common)
            if result.get("stop") != "closed":
                pickplace.go_home(ctx, viewer,
                                  on_step=lambda: sim.hold_cube(pick_xy, ctx["cube_hold_z"]))
    except KeyboardInterrupt:
        pass
    finally:
        hud.done()
        follower.disconnect()
        if live:
            live.stop()
        touch.stop()

    if not result or not result.get("attempts"):
        print("\nNo attempts completed." + (f" Reason: {result['stop']}." if result and result.get("stop") else ""))
        return
    report(args, touch, result, profile)


# --- force staircase ---------------------------------------------------------------

def run_staircase(args, ctx, viewer, touch, pick_xy, common):
    state = {"n": 0}
    if args.control == "travel":
        return _run_continuous(args, ctx, viewer, touch, pick_xy, common, state)

    def try_level(phase, target):
        state["n"] += 1
        hud.context(f"attempt {state['n']}   {phase}   target {target:.0f}% of full scale")
        t0 = time.time()
        res = mingrip.run(ctx, viewer, touch, pick_xy, level_pct=target,
                          full_scale=args.full_scale, **common)
        t1 = time.time()
        hud.done()
        if res.status not in ("held", "dropped"):
            print(f"attempt {state['n']}: {res.status} -- stopping")
            raise forcegrip.Abort(res.status)
        a = forcegrip.Attempt(achieved=res.achieved, held=res.status == "held",
                              left=res.grip_levels[0], right=res.grip_levels[1],
                              travel=res.travel, contact=res.contact_travel, t0=t0, t1=t1)
        print(f"attempt {state['n']:2d} {phase:<8} target {target:5.1f}%  reached {res.achieved:5.1f}%  "
              f"(L{a.left:.0f} R{a.right:.0f})  ->  {'HELD' if a.held else 'dropped'}")
        return a

    out = forcegrip.staircase(try_level, up=args.up_step, down=args.down_step, bump=args.bump,
                              confirm=args.confirm, max_attempts=args.max_attempts)
    learn.save_history(args.object + "__force", [
        {"phase": a.phase, "target": a.target, "achieved": a.achieved, "held": a.held,
         "left": a.left, "right": a.right, "travel": a.travel, "contact": a.contact}
        for a in out.attempts])
    return {"method": "staircase", "outcome": out, "attempts": out.attempts,
            "stop": out.status if out.status not in ("found", "never_held", "max_attempts") else None}


def _run_continuous(args, ctx, viewer, touch, pick_xy, common, state):
    def session(ctl):
        def try_level(phase, target):
            state["n"] += 1
            hud.context(f"attempt {state['n']}   {phase}   grip +{target:.0f}% past contact")
            a = ctl.attempt(phase, target)
            hud.done()
            print(f"attempt {state['n']:2d} {phase:<8} grip +{target:4.1f}% past contact   "
                  f"(L{a.left:.0f} R{a.right:.0f})  ->  {'HELD' if a.held else 'dropped'}")
            return a

        return forcegrip.staircase(try_level, up=args.up_step, down=args.down_step,
                                   bump=args.bump, confirm=args.confirm,
                                   max_level=args.max_x * 100.0, max_attempts=args.max_attempts)

    out = mingrip.run(ctx, viewer, touch, pick_xy, session=session, **common)
    learn.save_history(args.object + "__travel", [
        {"phase": a.phase, "target": a.target, "held": a.held, "left": a.left,
         "right": a.right, "travel": a.travel, "contact": a.contact} for a in out.attempts])
    return {"method": "staircase", "outcome": out, "attempts": out.attempts,
            "stop": out.status if out.status not in ("found", "never_held", "max_attempts") else None}


# --- bayesian travel model (optional) -------------------------------------------------

def run_bayes(args, ctx, viewer, touch, pick_xy, common):
    history = [] if args.fresh else learn.load_history(args.object)
    learner = learn.GripLearner(history)
    rng = np.random.default_rng()
    stop = None
    for k in range(1, args.trials + 1):
        x = min(learner.choose(rng), args.max_x)
        est = learner.recommend()
        hud.context(f"attempt {k}/{args.trials}   squeeze +{x * 100:.0f}%   "
                    f"estimate {('+%.0f%%' % (est * 100)) if est is not None else '?'}")
        t0 = time.time()
        res = mingrip.run(ctx, viewer, touch, pick_xy, extra=x, **common)
        t1 = time.time()
        hud.done()
        if res.status not in ("held", "dropped"):
            stop = res.status
            print(f"attempt {k}: {res.status} -- stopping")
            break
        held = res.status == "held"
        learner.add(x, held, t0=t0, t1=t1, contact=res.contact_travel, travel=res.travel,
                    left=float(res.levels[0]), right=float(res.levels[1]), jolt_cm=common["jolt_cm"])
        learn.save_history(args.object, [{k2: a[k2] for k2 in
                                          ("x", "held", "t", "contact", "travel", "left", "right", "jolt_cm")
                                          if k2 in a} for a in learner.attempts])
        rec = learner.recommend()
        m, sd = learner.threshold()
        print(f"attempt {k}: +{x * 100:4.1f}% squeeze -> {'HELD   ' if held else 'DROPPED'}  "
              f"L{res.levels[0]:.0f} R{res.levels[1]:.0f}   |  50% point {m * 100:.1f}% +/- {sd * 100:.1f}"
              f"   recommended {('+%.0f%%' % (rec * 100)) if rec is not None else 'not yet'}")
        if args.until_converged and learner.converged():
            print("recommendation has settled -- stopping early")
            break
    return {"method": "bayes", "learner": learner, "attempts": learner.attempts,
            "n_prior": len(history), "stop": stop}


# --- report + save -----------------------------------------------------------------------

def report(args, touch, result, profile):
    print("\n" + "=" * 60)
    png = None
    if not args.no_plot:
        paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
        png = paths.OUT_DIR / f"grip_learning_{datetime.now():%Y%m%d_%H%M%S}.png"

    if result["method"] == "staircase":
        out = result["outcome"]
        n = len(out.attempts)
        held_n = sum(a.held for a in out.attempts)
        print(f"{n} attempts: {held_n} held, {n - held_n} dropped")
        if png:
            runplot.save_staircase(png, touch.log, out.attempts, out, args.full_scale,
                                   title=f"Grip staircase: {args.object}",
                                   unit=("% of jaw travel past first contact"
                                         if args.control == "travel" else "% of sensor full scale"))
            print(f"Plot -> {png}")
        if out.status != "found":
            why = {"never_held": "the object was never held, even at the largest allowed squeeze",
                   "max_attempts": "ran out of attempts (--max-attempts)",
                   "no_contact": "no sensor signal before the safety cutoff",
                   "limit": "stopped at the motor-current / sensor limit",
                   "closed": "viewer closed"}.get(out.status, out.status)
            print(f"No result: {why}.")
            return
        best = out.best
        near = [a for a in out.attempts if a.held and abs(a.achieved - out.level) <= 1.5]
        lv = np.median([[a.left, a.right] for a in near], axis=0)
        what = (f"+{out.level:.1f}% of jaw travel past first contact" if args.control == "travel"
                else f"sensor level {out.level:.1f}% of full scale")
        print(f"MINIMUM GRIP: {what} (L {lv[0]:.0f} / R {lv[1]:.0f} counts), held {len(near)} time(s)"
              + ("  [resolution limit: the gripper can't settle lower]" if out.floor else ""))
        contact, travel, n_att = best.contact, best.travel, n
    else:
        learner, n_prior = result["learner"], result["n_prior"]
        rec = learner.recommend()
        m, sd = learner.threshold()
        print(f"{len(learner.attempts)} attempts in total ({len(learner.attempts) - n_prior} new)")
        print(f"50% hold point: +{m * 100:.1f}% +/- {sd * 100:.1f}% travel past first contact")
        if png:
            runplot.save(png, touch.log, learner.attempts, new_from=n_prior,
                         title=f"Grip learning: {args.object}")
            print(f"Plot -> {png}")
        if rec is None:
            print("No squeeze reached 95% hold probability yet -- run more attempts.")
            return
        contact = float(np.median([a["contact"] for a in learner.attempts if "contact" in a] or [0.0]))
        held_lv = [[a["left"], a["right"]] for a in learner.attempts if a["held"] and "left" in a]
        lv = np.max(held_lv, axis=0) if held_lv else np.zeros(2)
        travel, n_att = contact + rec, len(learner.attempts)
        print(f"LEARNED GRIP: +{rec * 100:.0f}% of jaw travel past first contact "
              f"(L {lv[0]:.0f} / R {lv[1]:.0f} counts)")

    if input("Save as the hold target in config/grip_profile.json? [y/n]: ").strip().lower().startswith("y"):
        p = grip.apply_min_hold(profile, lv, travel, contact, n_att, args.margin)
        p.object = args.object
        p.save()
    else:
        print("Not saved.")


if __name__ == "__main__":
    main()
