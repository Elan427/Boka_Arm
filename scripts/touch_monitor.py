"""Live view of the two gripper flex sensors -- wiring and threshold check.

Tares with the jaws open and untouched, then shows each sensor's deviation as a bar
and a TOUCH flag when it crosses the detection threshold. Press a sensor, bend it,
release it. No arm or sim needed.

Run: python scripts/touch_monitor.py [--port COM3]
"""
import argparse
import sys
import time

from so101lib.touch import TouchSensors


def bar(v, full, w=30):
    n = int(min(v / full, 1.0) * w)
    return "#" * n + "." * (w - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    args = ap.parse_args()
    t = TouchSensors(args.port)
    t.start()
    print("Taring -- keep the jaws open and untouched...")
    base, noise = t.tare()
    on = t.thresholds()
    print(f"baseline A0/A1 = {base.round(1).tolist()}  noise = {noise.round(1).tolist()}  "
          f"threshold = {on.round(1).tolist()} counts\nCtrl+C to quit.\n")
    try:
        while True:
            lv = t.levels()
            c = t.contact()
            full = max(on.max() * 4, 60)
            rw = t.raw()
            sys.stdout.write(
                f"\r raw A0 {rw[0]:5.0f} A1 {rw[1]:5.0f} | "
                f"L {bar(lv[0], full, 20)} {lv[0]:4.0f} {'TOUCH' if c[0] else '     '}"
                f" | R {bar(lv[1], full, 20)} {lv[1]:4.0f} {'TOUCH' if c[1] else '     '} ")
            sys.stdout.flush()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        t.stop()
        print()


if __name__ == "__main__":
    main()
