# Planar arm kinematics — assignment

Based on the 3-link arm from the simulation: link lengths `L1 = 0.10 m`, `L2 = 0.10 m`, `L3 = 0.05 m`, joint angles `theta1, theta2, theta3` measured from the positive x-axis, each joint's angle adding onto the one before it (cumulative chain).

Governing equations you'll need throughout:

```
Forward kinematics (FK):
  a1 = t1
  a2 = t1 + t2
  a3 = t1 + t2 + t3

  x = L1*cos(a1) + L2*cos(a2) + L3*cos(a3)
  z = L1*sin(a1) + L2*sin(a2) + L3*sin(a3)

Jacobian (2x3), column j = derivative of [x,z] w.r.t. joint j:
  dx/dt1 = -L1*sin(a1) - L2*sin(a2) - L3*sin(a3)      dz/dt1 = L1*cos(a1) + L2*cos(a2) + L3*cos(a3)
  dx/dt2 =            - L2*sin(a2) - L3*sin(a3)       dz/dt2 =            L2*cos(a2) + L3*cos(a3)
  dx/dt3 =                         - L3*sin(a3)       dz/dt3 =                        L3*cos(a3)

Damped least squares IK solver:
  e  = target - FK(theta)
  dq = J^T (J J^T + lambda^2 I)^-1 e
  theta <- theta + dq   ... repeat until |e| small
```

---

## Part 1 — Problems

**Q1 (FK).** theta = (0.3, -0.5, 0.2) rad. Compute the end-effector position (x, z).

**Q2 (FK).** theta = (-0.2, -0.8, 0.4) rad. Compute the end-effector position (x, z).

**Q3 (workspace).** What is the maximum distance the end effector can reach from the base? What is the *minimum* distance — i.e. is there a "dead zone" near the base the arm can never reach? Justify using the triangle inequality on L1, L2, L3, not just the two-link formula `|L1-L2|`.

**Q4 (Jacobian, symbolic).** Derive the 2x3 Jacobian `d[x,z]/d[t1,t2,t3]` from the FK equations. Show the differentiation step, not just the final matrix.

**Q5 (Jacobian, numeric).** Evaluate the Jacobian from Q4 at the configuration from Q1. As a check on your work: the first column should equal `(-z, x)` from your Q1 answer. Explain in one sentence why that identity holds.

**Q6 (IK, one hand-computed step).** Start at theta0 = (0, 0, 0) (arm fully extended along +x). Target = (0.15, 0.05). Using damping `lambda = 0.05`, compute one damped-least-squares step by hand: the error `e`, the Jacobian `J0`, the 2x2 matrix `J0 J0^T + lambda^2 I`, its inverse, and the resulting `dq` and `theta1`.

**Q7 (underdetermined system).** Write the two FK equations as a system with x* and z* as knowns and t1, t2, t3 as unknowns. Two equations, three unknowns — what does that imply about the number of joint-angle solutions for a given reachable target?

**Q8 (reachability check).** Is the target (0.15, 0.05) from Q6 actually reachable? Compute the straight-line distance from the base and compare against your Q3 answer.

---

## Part 2 — Solutions

**A1.**
Cumulative angles: `a1 = 0.3`, `a2 = 0.3 - 0.5 = -0.2`, `a3 = -0.2 + 0.2 = 0.0`.

```
x = 0.10*cos(0.3) + 0.10*cos(-0.2) + 0.05*cos(0.0)
  = 0.10(0.95534) + 0.10(0.98007) + 0.05(1)
  = 0.24354 m

z = 0.10*sin(0.3) + 0.10*sin(-0.2) + 0.05*sin(0.0)
  = 0.10(0.29552) + 0.10(-0.19867) + 0
  = 0.00969 m
```
End effector ≈ **(0.2435, 0.0097) m**.

**A2.**
`a1 = -0.2`, `a2 = -1.0`, `a3 = -0.6`.

```
x = 0.10(0.98007) + 0.10(0.54030) + 0.05(0.82534) = 0.19330 m
z = 0.10(-0.19867) + 0.10(-0.84147) + 0.05(-0.56464) = -0.13225 m
```
End effector ≈ **(0.1933, -0.1322) m**.

**A3.**
Max reach: all three links point the same direction → `L1+L2+L3 = 0.25 m`.

Min reach: with three independently-rotating links (not two), the end effector can return to the base whenever the three lengths satisfy the triangle inequality — each length less than the sum of the other two. Here `0.05 < 0.10+0.10`, `0.10 < 0.10+0.05`, `0.10 < 0.10+0.05` — all hold, so the links can form a closed triangle and the effector can sit exactly on the base. **Min reach = 0 m**, no dead zone. (This is why `|L1-L2-L3|` — the two-link rule — doesn't apply once you add a third independently-rotating link.)

**A4.**
`x = L1 cos(t1) + L2 cos(t1+t2) + L3 cos(t1+t2+t3)`, differentiate each term by chain rule, e.g. `d/dt2 [L2 cos(t1+t2)] = -L2 sin(t1+t2)` since `t2` only appears from the second term onward:

```
dx/dt1 = -L1 sin(a1) - L2 sin(a2) - L3 sin(a3)        dz/dt1 =  L1 cos(a1) + L2 cos(a2) + L3 cos(a3)
dx/dt2 =              -L2 sin(a2) - L3 sin(a3)        dz/dt2 =               L2 cos(a2) + L3 cos(a3)
dx/dt3 =                          -L3 sin(a3)         dz/dt3 =                            L3 cos(a3)
```
Pattern: column `j` only sums links `j..3`, since earlier joints don't move once you're differentiating with respect to a later one.

**A5.**
At `a1=0.3, a2=-0.2, a3=0.0`:

```
J = [ -0.00969   0.01987   0.0  ]
    [  0.24354   0.14801   0.05 ]
```
Column 1 is `(-0.00969, 0.24354)`, which is exactly `(-z, x) = (-0.00969, 0.24354)` from A1. This holds because rotating `t1` spins the *entire* chain rigidly about the base — that's a pure rotation of the point `(x,z)` about the origin, and the velocity of a point under rotation about the origin is always `(-z, x)` per unit angular velocity.

**A6.**
At `theta0=(0,0,0)`: `a1=a2=a3=0`, so `p0 = (L1+L2+L3, 0) = (0.25, 0)`.

`e = target - p0 = (0.15-0.25, 0.05-0) = (-0.10, 0.05)`

```
J0 = [ 0      0      0    ]
     [ 0.25   0.15   0.05 ]
```
(all sines are zero at the fully-extended pose, so every column's x-entry is 0 — moving any joint from here only moves the tip in z, not x.)

```
J0 J0^T = [ 0     0    ]      +lambda^2 I (lambda=0.05, lambda^2=0.0025):
          [ 0     0.0875 ]

J0 J0^T + lambda^2 I = [ 0.0025   0    ]
                        [ 0        0.09 ]

inverse = [ 400      0     ]
          [ 0        11.111 ]

v = inverse * e = [ 400*(-0.10),  11.111*0.05 ] = [ -40,  0.5556 ]

dq = J0^T v:
  dq1 = 0*(-40) + 0.25*0.5556 = 0.1389
  dq2 = 0*(-40) + 0.15*0.5556 = 0.0833
  dq3 = 0*(-40) + 0.05*0.5556 = 0.0278

theta1 = theta0 + dq = (0.1389, 0.0833, 0.0278) rad
```
Notice the huge `-40` from the x-error contributes **nothing** to `dq`, because `J0`'s top row is all zeros — at full extension the arm is momentarily blind to x-direction corrections (a singular direction), so this first step fixes almost all of the z-error and barely touches x. Re-running FK at `theta1` gives `p1 ≈ (0.2450, 0.0483)` — z-error collapsed from 0.05 to 0.0017, x-error barely moved from -0.10 to -0.095. It takes more iterations, once the arm has rotated off full extension, for x-sensitivity to appear and the rest of the error to close.

**A7.**
```
x* = L1 cos(t1) + L2 cos(t1+t2) + L3 cos(t1+t2+t3)
z* = L1 sin(t1) + L2 sin(t1+t2) + L3 sin(t1+t2+t3)
```
Two equations, three unknowns: the system is underdetermined. For any reachable `(x*, z*)` there's a one-parameter family of joint solutions (a "self-motion" manifold) — the arm can wiggle one internal joint while keeping the tip fixed. This is exactly why the posture task exists in `ik_mink.py`: it picks one solution out of that family by pulling toward a preferred (home) posture in the null space.

**A8.**
`distance = sqrt(0.15^2 + 0.05^2) = sqrt(0.0225+0.0025) = sqrt(0.025) = 0.1581 m`

Compare to the A3 reachable range `[0, 0.25] m`: `0 <= 0.1581 <= 0.25`, so **yes, reachable**, with infinitely many joint solutions per A7.
