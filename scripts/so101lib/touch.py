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
from collections import deque

import numpy as np

# Detection thresholds, in ADC counts away from the tared baseline. The effective
# "on" level is max(TOUCH_ON_MIN, TOUCH_ON_SIGMAS * noise_std); "off" is half of it
# (hysteresis), so a reading hovering at the edge does not chatter.
TOUCH_ON_MIN = 15.0
TOUCH_ON_SIGMAS = 6.0
BOXCAR = 4  # samples averaged; the sketch runs 240 Hz = 4 per 60 Hz mains cycle,
            # so this cancels mains hum (use 5 with 200 Hz firmware on 50 Hz mains)
STALE_AFTER = 0.5  # s without a sample -> reading considered dead


class TouchSensors:
    def __init__(self, port="COM3", baud=115200, smooth=0.0):
        # smooth: extra exponential smoothing on top of the mains boxcar, as the weight
        # of the newest sample (0 or 1 = off, 0.1 = heavy). Can be changed at runtime.
        self.smooth = smooth
        self._ema = None
        self._boxed = None         # boxcar-only reading, before the extra smoothing
        self.port = port
        self.baud = baud
        self._ser = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._filt = None          # boxcar-filtered [A0, A1] counts
        self._hist = []
        self.log = deque(maxlen=400_000)   # (time, A0, A1, A0 smoothed, A1 smoothed) per sample
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
                self._hist.append(v)
                del self._hist[:-BOXCAR]
                self._boxed = np.mean(self._hist, axis=0)
                a = self.smooth
                if not (0.0 < a < 1.0) or self._ema is None:
                    self._ema = self._boxed.copy()
                else:
                    self._ema = self._ema + a * (self._boxed - self._ema)
                self._filt = self._ema.copy()
                self.log.append((time.time(), *self._boxed, *self._ema))
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
                if self._filt is not None and len(self._hist) == BOXCAR and self._stamp != last:
                    samples.append(self._filt.copy())
                    last = self._stamp
            time.sleep(0.002)
        if len(samples) < 20:
            raise RuntimeError("too few touch samples while taring")
        s = np.array(samples)
        self._base = s.mean(0)
        self._noise = np.maximum(s.std(0), 1.0)
        self._state = [False, False]
        return self._base.copy(), self._noise.copy()

    def problems(self):
        """Human-readable reasons the last tare() can't be trusted (empty = fine):
        a pinned or intermittent pin, or noise so large that the touch threshold
        would exceed what the ADC can even read."""
        out = []
        on = self.thresholds()
        for i, name in enumerate(("A0 (left)", "A1 (right)")):
            # A baseline of ~0 is NORMAL: these sensors are open-circuit until touched, so
            # the divider node sits at 0 V at rest. Use response_check() to prove a
            # sensor is actually connected.
            if self._base[i] > 1020:
                out.append(f"{name} is pinned at the ADC maximum -- shorted to the supply")
            if self._noise[i] > 20:
                out.append(f"{name} is unstable (noise {self._noise[i]:.0f} counts) -- loose or "
                           "intermittent connection, or the sensor was being moved during the tare")
            if on[i] > 300:
                out.append(f"{name} touch threshold {on[i]:.0f} counts is unusably large")
        return out

    def response_check(self, timeout=10.0, say=print):
        """Ask the user to press each sensor in turn and confirm it registers.
        Proves a sensor is wired, since an idle reading of 0 looks the same whether the
        sensor is connected or not. Returns (left_ok, right_ok)."""
        ok = [False, False]
        for i, name in enumerate(("LEFT (A0)", "RIGHT (A1)")):
            say(f"  Press and hold the {name} sensor now ({timeout:.0f} s)...")
            t_end = time.time() + timeout
            while time.time() < t_end:
                if self.levels()[i] > self.thresholds()[i]:
                    ok[i] = True
                    break
                time.sleep(0.02)
            if ok[i]:
                say("    registered -- release it")
                t_rel = time.time() + 4.0
                while time.time() < t_rel and self.levels()[i] > 0.5 * self.thresholds()[i]:
                    time.sleep(0.02)
            else:
                say("    NO RESPONSE")
        return tuple(ok)

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

    def unsmoothed(self):
        """(A0, A1) filtered by the mains boxcar only -- the trace before `smooth`."""
        with self._lock:
            f = None if self._boxed is None else self._boxed.copy()
        return (0.0, 0.0) if f is None else (float(f[0]), float(f[1]))

    def raw(self):
        """(A0, A1) absolute filtered ADC counts -- for wiring checks."""
        with self._lock:
            f = None if self._filt is None else self._filt.copy()
        return (0.0, 0.0) if f is None else (float(f[0]), float(f[1]))

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
