"""Validate TCP stability under a wrist_roll sweep.

Pure wrist_roll rotation should barely move the TCP: it is (near) the roll axis.
Large X/Y/Z travel means the tcp site is offset asymmetrically from the jaw
pinch line. Prints PASS/FAIL.

Run: python scripts/validate_tcp.py
"""
import mujoco
import numpy as np

from so101lib import scene

sim = scene.load()
m, d = sim.m, sim.d
tcp_sid = sim.site("tcp")
flange_sid = sim.site("gripperframe")

print("TCP STABILITY UNDER WRIST_ROLL SWEEP")
print("=" * 60)

tcp_positions = []
for wr in np.linspace(-2.5, 2.5, 11):
    d.ctrl[:] = 0
    d.ctrl[4] = wr  # wrist_roll is actuator 4
    for _ in range(2000):
        mujoco.mj_step(m, d)
    tcp = d.site_xpos[tcp_sid].copy()
    tcp_positions.append(tcp)
    print(f"roll={wr:+.1f}rad  tcp={np.round(tcp, 4)}  flange={np.round(d.site_xpos[flange_sid], 4)}")

tcp_array = np.array(tcp_positions)
rng = tcp_array.max(axis=0) - tcp_array.min(axis=0)

print("\n" + "=" * 60)
print(f"TCP travel over sweep:  x {rng[0] * 1000:.2f} mm   y {rng[1] * 1000:.2f} mm   z {rng[2] * 1000:.2f} mm")

ok_x = rng[0] < 0.003
ok_yz = rng[1] < 0.001 and rng[2] < 0.001
print(f"{'PASS' if ok_x else 'FAIL'}: X travel {rng[0] * 1000:.2f} mm (want < 3 mm)")
print(f"{'PASS' if ok_yz else 'FAIL'}: Y/Z travel negligible (want < 1 mm each)")

raise SystemExit(0 if (ok_x and ok_yz) else 1)
