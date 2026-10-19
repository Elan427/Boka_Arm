"""Gripper touch feedback from two flex sensors read by an Arduino.

The sketch in firmware/gripper_touch streams "A0,A1" ADC counts per line (one flex
sensor per jaw, each in a divider with a 10k resistor). A background thread keeps
the latest filtered reading. Because the direction a flex sensor moves depends on
how it is wired and mounted, everything here works on the *absolute deviation from
a free-air baseline* taken by `tare()` -- bend either way and it counts as touch.

    touch = TouchSensors("COM3"); touch.start(); touch.tare()
    touch.contact()      # (left, right) bools, with hysteresis
    touch.levels()       # (left, right) counts away from baseline
    touch.force()        # (left, right) in units of the touch threshold
"""
import threading
import time

import numpy as np

# Detection thresholds, in ADC counts away from the tared baseline. The effective
# "on" level is max(TOUCH_ON_MIN, TOUCH_ON_SIGMAS * noise_std); "off" is half of it
# (hysteresis), so a reading hovering at the edge does not chatter.
TOUCH_ON_MIN = 15.0
TOUCH_ON_SIGMAS = 6.0
SMOOTHING = 0.3  # EMA weight of the newest sample (samples arrive at ~200 Hz)
STALE_AFTER = 0.5  # s without a sample -> reading considered dead


class TouchSensors:
    def __init__(self, port="COM3", baud=115200):
        self.port = port
        self.baud = baud
        self._ser = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._filt = None          # EMA-filtered [A0, A1] counts
        self._raw = None
        self._stamp = 0.0
        self._base = np.zeros(2)
        self._noise = np.full(2, 3.0)
        self._state = [False, False]
        self.available = False

    # --- lifecycle -------------------------------------------------------
    def start(self):
        import serial

        self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        time.sleep(2.0)  # the Arduino resets when the port opens
        self._ser.reset_input_buffer()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        t_end = time.time() + 3.0
        while time.time() < t_end and self._filt is None:
            time.sleep(0.02)
        if self._filt is None:
            self.stop()
            raise RuntimeError(f"no data from touch sensors on {self.port}")
        self.available = True

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self.available = False

    def _run(self):
        while not self._stop.is_set():
            try:
                line = self._ser.readline()
            except Exception:
                time.sleep(0.05)
                continue
            try:
                a, b = (float(v) for v in line.decode("ascii", "ignore").strip().split(","))
            except ValueError:
                continue
            v = np.array([a, b])
            with self._lock:
                self._raw = v
                self._filt = v.copy() if self._filt is None else self._filt + SMOOTHING * (v - self._filt)
                self._stamp = time.time()

    # --- calibration -------------------------------------------------------
    def tare(self, secs=1.5):
        """Record the free-air baseline and noise. Keep the jaws open and untouched.
        Returns (baseline, noise_std), each [A0, A1]."""
        samples = []
        t_end = time.time() + secs
        last = 0.0
        while time.time() < t_end:
            with self._lock:
                if self._raw is not None and self._stamp != last:
                    samples.append(self._raw.copy())
                    last = self._stamp
            time.sleep(0.002)
        if len(samples) < 20:
            raise RuntimeError("too few touch samples while taring")
        s = np.array(samples)
        self._base = s.mean(0)
        self._noise = np.maximum(s.std(0), 1.0)
        self._state = [False, False]
        return self._base.copy(), self._noise.copy()

    def thresholds(self):
        return np.maximum(TOUCH_ON_MIN, TOUCH_ON_SIGMAS * self._noise)

    # --- reading -------------------------------------------------------------
    @property
    def healthy(self):
        return self.available and (time.time() - self._stamp) < STALE_AFTER

    def levels(self):
        """(left, right) absolute counts away from baseline (filtered)."""
        with self._lock:
            f = None if self._filt is None else self._filt.copy()
        if f is None:
            return 0.0, 0.0
        d = np.abs(f - self._base)
        return float(d[0]), float(d[1])

    def contact(self):
        """(left, right) touch booleans, with hysteresis."""
        on = self.thresholds()
        lv = self.levels()
        for i in range(2):
            if self._state[i]:
                self._state[i] = lv[i] > 0.5 * on[i]
            else:
                self._state[i] = lv[i] > on[i]
        return bool(self._state[0]), bool(self._state[1])

    def both(self):
        c = self.contact()
        return c[0] and c[1]

    def any(self):
        c = self.contact()
        return c[0] or c[1]

    def force(self):
        """(left, right) in units of the 'on' threshold: 1.0 = just touching,
        3.0 = pressing three times harder. Handy for grip-force targets."""
        on = self.thresholds()
        lv = self.levels()
        return lv[0] / on[0], lv[1] / on[1]
