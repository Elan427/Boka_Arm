"""Live plot of the two gripper flex sensors (A0 left, A1 right).

Each sensor shows two traces: a faint one (mains-boxcar filtered only) and a bold
one with the extra smoothing applied. Dashed lines are the touch thresholds from the
start-up tare. Tkinter only -- nothing to install.

Keys:  + / -   less / more smoothing      t   re-tare (hands off the sensors)
       y       toggle y-axis 0-700 / 0-1023   p   pause   q   quit

Run: python scripts/plot_sensors.py [--port COM3] [--smooth 0.3] [--window 10]
"""
import argparse
import time
import tkinter as tk
from collections import deque

from so101lib.touch import TouchSensors

W, H = 980, 520
ML, MR, MT, MB = 62, 20, 44, 40
COLORS = {"A0": "#d1495b", "A1": "#2e86ab"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--smooth", type=float, default=0.3,
                    help="extra smoothing, weight of the newest sample (0 = off, 0.05 = heavy)")
    ap.add_argument("--window", type=float, default=10.0, help="seconds of history shown")
    ap.add_argument("--selftest", action="store_true", help="run 4 s hidden and report")
    args = ap.parse_args()

    touch = TouchSensors(args.port, smooth=args.smooth)
    touch.start()
    print("Taring -- keep your hands off the sensors...")
    touch.tare()

    root = tk.Tk()
    root.title("SO-101 gripper sensors")
    cv = tk.Canvas(root, width=W, height=H, bg="#14171c", highlightthickness=0)
    cv.pack()
    hist = deque()
    st = {"ymax": 700.0, "paused": False, "frames": 0}

    def px(t, now):
        return ML + (1.0 - (now - t) / args.window) * (W - ML - MR)

    def py(v):
        return H - MB - min(max(v, 0.0), st["ymax"]) / st["ymax"] * (H - MT - MB)

    def line(points, color, width, dash=None):
        if len(points) >= 2:
            cv.create_line(*[c for p in points for c in p], fill=color, width=width,
                           dash=dash, smooth=False)

    def draw():
        now = time.time()
        if not st["paused"]:
            u, s = touch.unsmoothed(), touch.raw()
            hist.append((now, u[0], u[1], s[0], s[1]))
        while hist and now - hist[0][0] > args.window:
            hist.popleft()
        cv.delete("all")
        for k in range(0, int(st["ymax"]) + 1, 100):       # grid + y labels
            cv.create_line(ML, py(k), W - MR, py(k), fill="#262b33")
            cv.create_text(ML - 8, py(k), text=str(k), fill="#8a93a0", anchor="e", font=("Segoe UI", 9))
        for sec in range(int(args.window) + 1):
            x = px(now - sec, now)
            cv.create_line(x, MT, x, H - MB, fill="#1c2027")
            cv.create_text(x, H - MB + 14, text=f"-{sec}s" if sec else "now", fill="#8a93a0",
                           font=("Segoe UI", 9))
        base, thr = touch._base, touch.thresholds()
        for i, name in enumerate(("A0", "A1")):             # threshold lines
            y = py(base[i] + thr[i])
            cv.create_line(ML, y, W - MR, y, fill=COLORS[name], dash=(4, 4))
        for i, name in enumerate(("A0", "A1")):             # traces
            line([(px(r[0], now), py(r[1 + i])) for r in hist], COLORS[name], 1)
            line([(px(r[0], now), py(r[3 + i])) for r in hist], COLORS[name], 3)
        s = touch.raw()
        c = touch.contact()
        a = touch.smooth
        cv.create_text(ML, 14, anchor="w", fill="#e8ebf0", font=("Segoe UI", 12, "bold"),
                       text=f"A0 left {s[0]:5.0f} {'TOUCH' if c[0] else '     '}    "
                            f"A1 right {s[1]:5.0f} {'TOUCH' if c[1] else '     '}")
        cv.create_text(ML, 32, anchor="w", fill="#8a93a0", font=("Segoe UI", 9),
                       text=f"smoothing {'off' if not 0 < a < 1 else f'alpha {a:.2f}'}   "
                            f"[+/-] smoothing  [t] tare  [y] axis  [p] pause  [q] quit"
                            f"{'   PAUSED' if st['paused'] else ''}")
        cv.create_text(W - MR, 14, anchor="e", fill=COLORS["A0"], text="A0", font=("Segoe UI", 11, "bold"))
        cv.create_text(W - MR - 28, 14, anchor="e", fill=COLORS["A1"], text="A1", font=("Segoe UI", 11, "bold"))
        st["frames"] += 1
        root.after(16, draw)

    def key(e):
        k = e.keysym.lower()
        a = touch.smooth if 0 < touch.smooth < 1 else 1.0    # 1.0 = off
        if k in ("plus", "equal", "kp_add"):                 # more smoothing
            touch.smooth = max(0.02, a * 0.7)
        elif k in ("minus", "kp_subtract"):                  # less smoothing
            touch.smooth = 0.0 if a / 0.7 >= 0.95 else a / 0.7
        elif k == "t":
            touch.tare()
        elif k == "y":
            st["ymax"] = 1023.0 if st["ymax"] < 1000 else 700.0
        elif k == "p":
            st["paused"] = not st["paused"]
        elif k in ("q", "escape"):
            root.destroy()

    def close():
        touch.stop()
        root.destroy()

    root.bind("<Key>", key)
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(16, draw)
    if args.selftest:
        root.withdraw()
        root.after(4000, root.quit)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        touch.stop()
    if args.selftest:
        print(f"selftest: {st['frames']} frames drawn, {len(hist)} samples in window, "
              f"last A0/A1 = {touch.raw()}")


if __name__ == "__main__":
    main()
