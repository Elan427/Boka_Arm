"""Live plot of the two gripper flex sensors, running in a background thread so it can
share the TouchSensors (and so the serial port) with a script that is doing other work.

    plot = LivePlot(touch); plot.start()
    ...
    plot.stop()

Same look as scripts/plot_sensors.py: faint trace = mains-boxcar only, bold = with the
extra smoothing, dashed = touch threshold from the last tare. Keys: y axis, p pause.
The Tk window lives entirely in its own thread; nothing here touches the sensor state.
"""
import threading
import time
from collections import deque

W, H = 980, 420
ML, MR, MT, MB = 62, 20, 36, 36
COLORS = {"A0": "#d1495b", "A1": "#2e86ab"}


class LivePlot:
    def __init__(self, touch, window=10.0, title="SO-101 gripper sensors"):
        self.touch, self.window, self.title = touch, window, title
        self._thread = None
        self._stop = threading.Event()
        self._ready = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(5.0)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        try:
            self._loop()
        except Exception as e:           # a plot problem must never take down a hardware run
            print(f"(live plot stopped: {e})")
        finally:
            self._ready.set()

    def _loop(self):
        import tkinter as tk

        touch, window = self.touch, self.window
        root = tk.Tk()
        root.title(self.title)
        cv = tk.Canvas(root, width=W, height=H, bg="#14171c", highlightthickness=0)
        cv.pack()
        hist = deque()
        st = {"ymax": 700.0, "paused": False}

        def px(t, now):
            return ML + (1.0 - (now - t) / window) * (W - ML - MR)

        def py(v):
            return H - MB - min(max(v, 0.0), st["ymax"]) / st["ymax"] * (H - MT - MB)

        def line(points, color, width, dash=None):
            if len(points) >= 2:
                cv.create_line(*[c for p in points for c in p], fill=color, width=width, dash=dash)

        def draw():
            if self._stop.is_set():
                root.destroy()
                return
            now = time.time()
            s, u = touch.raw(), touch.unsmoothed()
            if not st["paused"]:
                hist.append((now, u[0], u[1], s[0], s[1]))
            while hist and now - hist[0][0] > window:
                hist.popleft()
            cv.delete("all")
            for k in range(0, int(st["ymax"]) + 1, 100):
                cv.create_line(ML, py(k), W - MR, py(k), fill="#262b33")
                cv.create_text(ML - 8, py(k), text=str(k), fill="#8a93a0", anchor="e",
                               font=("Segoe UI", 9))
            for sec in range(int(window) + 1):
                x = px(now - sec, now)
                cv.create_line(x, MT, x, H - MB, fill="#1c2027")
                cv.create_text(x, H - MB + 14, text=f"-{sec}s" if sec else "now", fill="#8a93a0",
                               font=("Segoe UI", 9))
            base, thr = touch._base, touch.thresholds()
            for i, name in enumerate(("A0", "A1")):
                y = py(base[i] + thr[i])
                cv.create_line(ML, y, W - MR, y, fill=COLORS[name], dash=(4, 4))
            for i, name in enumerate(("A0", "A1")):
                line([(px(r[0], now), py(r[1 + i])) for r in hist], COLORS[name], 1)
                line([(px(r[0], now), py(r[3 + i])) for r in hist], COLORS[name], 3)
            cv.create_text(ML, 14, anchor="w", fill="#e8ebf0", font=("Segoe UI", 12, "bold"),
                           text=f"A0 left {s[0]:5.0f}    A1 right {s[1]:5.0f}"
                                f"{'    PAUSED' if st['paused'] else ''}    "
                                f"[y] axis  [p] pause")
            cv.create_text(W - MR, 14, anchor="e", fill=COLORS["A0"], text="A0", font=("Segoe UI", 11, "bold"))
            cv.create_text(W - MR - 28, 14, anchor="e", fill=COLORS["A1"], text="A1",
                           font=("Segoe UI", 11, "bold"))
            root.after(33, draw)

        def key(e):
            k = e.keysym.lower()
            if k == "y":
                st["ymax"] = 1023.0 if st["ymax"] < 1000 else 700.0
            elif k == "p":
                st["paused"] = not st["paused"]

        root.bind("<Key>", key)
        root.protocol("WM_DELETE_WINDOW", root.destroy)   # closing the window leaves the run going
        root.after(33, draw)
        self._ready.set()
        root.mainloop()
