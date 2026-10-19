# Degrees-of-freedom (Grubler-Kutzbach) — assignment

The tool for counting DOF by hand is the Grubler-Kutzbach criterion. For a **planar** mechanism:

```
M = 3(n - 1) - 2*j1 - j2

  n  = total number of links, INCLUDING the fixed ground/frame
  j1 = number of 1-DOF joints ("lower pairs": revolute pins, sliders)
  j2 = number of 2-DOF joints ("higher pairs": rolling/cam contact) -- 0 in every problem below
```

For a **spatial (3D)** mechanism, the general form is:

```
M = 6(n - 1 - j) + sum(f_i)

  j    = total number of joints
  f_i  = DOF of joint i  (revolute = 1, prismatic = 1, spherical = 3, ...)
```

Method every time: (1) count links including ground, (2) list every joint and its type, (3) plug into the formula, (4) sanity-check the answer against what the mechanism obviously does.

---

## Part 1 — Problems

**Q1 (four-bar linkage).** See the diagram above: ground pivots A and D are fixed to the frame, link 2 (crank) is the motor-driven input, link 3 (coupler) and link 4 (rocker) follow. All four joints A, B, C, D are revolute pins. Count links and joints, then find M.

**Q2 (slider-crank — piston engine).** Same idea as Q1 but the rocker is replaced by a piston sliding in a straight cylinder: ground, crank, connecting rod, piston/slider = 4 links. Three revolute joints (ground-crank, crank-rod, rod-slider) plus one prismatic joint (slider-ground). Find M.

**Q3 (five-bar linkage).** A closed loop of 5 links (including ground) and 5 revolute joints — the topology some low-cost tabletop arms use, with two motors at the two ground pivots instead of one. Find M and explain in one sentence why you need two actuators, not one.

**Q4 (sanity check — SO-101 itself).** The SO-101 is an *open* chain: base -> shoulder_pan -> shoulder_lift -> elbow_flex -> wrist_flex -> wrist_roll -> gripper, six revolute joints, no closed loop back to the base. Using the spatial formula, with n = 7 links (base + 6 moving links) and j = 6 joints, each f_i = 1: find M, and explain why an open chain always collapses to "just count the joints" regardless of which formula (planar or spatial) you use.

**Q5 (six-bar gripper linkage).** Real parallel-jaw grippers, including the kind on the SO-101, are usually not a single pin-jointed jaw driven directly — they're a six-bar linkage: one motor closes both jaws symmetrically. Topology: 6 links (including the gripper body as ground), 7 revolute joints, forming 2 independent closed loops. Find M and confirm it matches "one motor, fully determined motion."

---

## Part 2 — Solutions

**A1 — four-bar linkage**

Given: n = 4 (ground, crank, coupler, rocker). Joints: A (ground-crank), B (crank-coupler), C (coupler-rocker), D (rocker-ground) — all revolute, so j1 = 4, j2 = 0.

```
M = 3(n-1) - 2*j1 - j2
  = 3(4-1) - 2(4) - 0
  = 9 - 8
  = 1
```

**M = 1.** One actuator (turning the crank at A) fully determines the position of every other link. This is the baseline result every other problem gets compared to.

**A2 — slider-crank**

Given: n = 4 (ground, crank, rod, slider). Joints: 3 revolute (j1 contributes 3) + 1 prismatic (also 1-DOF, so it also counts toward j1) -> j1 = 4, j2 = 0.

```
M = 3(4-1) - 2(4) - 0 = 9 - 8 = 1
```

**M = 1.** Same formula, same answer as A1 — a prismatic joint is still a 1-DOF lower pair, so swapping the rocker for a slider doesn't change the DOF count at all. This is the mechanism inside an engine: one crank angle determines the piston position exactly.

**A3 — five-bar linkage**

Given: n = 5 (ground + 4 moving links). Joints: 5 revolute pins closing the loop, j1 = 5, j2 = 0.

```
M = 3(5-1) - 2(5) - 0 = 12 - 10 = 2
```

**M = 2.** Two independent inputs are needed — one link's position alone doesn't fix the rest of the mechanism, there's a whole family of configurations left over. That's why five-bar arms drive *two* of the ground-pivoted links independently (two motors at the base) rather than one: with only one motor, M=2 means one degree of motion would be left completely uncontrolled.

**A4 — SO-101 open chain (sanity check)**

Given: n = 7 (base + 6 moving links), j = 6 joints, all revolute so f_i = 1 for each.

```
M = 6(n - 1 - j) + sum(f_i)
  = 6(7 - 1 - 6) + 6(1)
  = 6(0) + 6
  = 6
```

**M = 6**, matching the six actuated joints exactly. The reason: for an *open* chain, every new link adds exactly one new joint connecting it to the chain, so `n - 1 = j` always — the `6(n-1-j)` term is identically zero no matter how many links you add. That's the whole reason "count the actuated joints" works as a shortcut for a serial arm, but it is a *special case* of Grubler's formula, not a separate rule — it only stops being that simple the moment you introduce a closed loop, which is exactly what happens inside the gripper (A5).

**A5 — six-bar gripper linkage**

Given: n = 6 links, j1 = 7 revolute joints, j2 = 0.

```
M = 3(n-1) - 2*j1 - j2
  = 3(6-1) - 2(7) - 0
  = 15 - 14
  = 1
```

**M = 1.** Even though this linkage has 6 links and 7 joints — visually a lot more complicated than the four-bar in A1 — it still reduces to exactly one degree of freedom. One motor input drives the whole mechanism, and the linkage geometry itself (not extra actuators) is what forces the two jaws to close symmetrically. This is the mechanical reason a gripper can look complex but still be controlled with a single `gripper` actuator command in your control code, same as in `grasp_test.py`.

**Cross-check as a loop count.** For any planar mechanism, independent closed loops `L = j - n + 1`. A1: L = 4-4+1 = 1 loop, M = 1. A5: L = 7-6+1 = 2 loops, but still M = 1 — more loops doesn't automatically mean more DOF, it depends on how the joints constrain each other. That distinction (loop count vs. DOF) is worth sitting with, since it's the thing people most often get backwards.
