# Manipulability ellipse (eigenvalues of J J^T) — assignment

Same 3-link planar arm: `L1 = 0.10 m`, `L2 = 0.10 m`, `L3 = 0.05 m`. The Jacobian `J` (2x3) from the earlier assignment is rectangular, so it has no eigenvalues of its own — but `A = J J^T` (2x2, symmetric) always does. Its eigenvalues and eigenvectors define the **manipulability ellipse**: how fast the end effector can move in each direction for a fixed "budget" of joint speed.

```
Method:
  1. Build J at the given joint angles (same formulas as before).
  2. A = J J^T  (2x2 symmetric)
  3. Solve the characteristic equation: lambda^2 - trace(A)*lambda + det(A) = 0
  4. For each lambda_i, solve (A - lambda_i*I) v = 0 for the eigenvector v_i
  5. Semi-axis length along v_i is sqrt(lambda_i)
  6. Condition number kappa = sqrt(lambda_1/lambda_2) -- how lopsided the ellipse is.
     kappa near 1 = well-balanced pose. kappa -> infinity = singular.
  7. Achievable speed in an arbitrary unit direction u:
     r(u) = 1 / sqrt( (u.e1)^2/lambda_1 + (u.e2)^2/lambda_2 )
```

---

## Part 1 — Problems

**Q1 (build the ellipse).** theta = (1.3, -0.4, 0.6) rad — a bent, mid-reach pose. Compute `J`, then `A = J J^T`, then solve for the two eigenvalues and eigenvectors by hand. Report the semi-axis lengths and directions.

**Q2 (the singular case).** theta = (0, 0, 0) — the arm fully extended along +x (the same start pose used for the IK problem earlier). `J0 = [[0,0,0],[0.25,0.15,0.05]]`. Compute `A0 = J0 J0^T` and its eigenvalues/eigenvectors. What does an eigenvalue of exactly zero mean physically?

**Q3 (real-world scenario).** Your project plan's recommended first task is seating a lid onto a jar — a task that needs the arm to correct small **horizontal** misalignments (direction `u = (1, 0)`) as it lowers the lid, with roughly 3mm tolerance. Using the ellipse from Q1, find the achievable end-effector speed in the pure-horizontal direction, and compare it against the best-case (major axis) and worst-case (minor axis) directions. Is this pose good enough for that correction, or should the IK solver's posture task (the null-space term in `ik_mink.py`) be steering the arm toward a different configuration before it starts the insertion?

---

## Part 2 — Solutions

**A1.**
Cumulative angles: a1=1.3, a2=1.3-0.4=0.9, a3=0.9+0.6=1.5.

```
J = [ -0.2246  -0.1282  -0.0499 ]
    [  0.0924   0.0657   0.0035 ]
```

```
A = J J^T = [ 0.06935   -0.02936 ]
            [ -0.02936   0.01288 ]

trace(A) = 0.06935 + 0.01288 = 0.08223
det(A)   = (0.06935)(0.01288) - (-0.02936)^2 = 0.000893 - 0.000862 = 0.0000309

characteristic eq: lambda^2 - 0.08223*lambda + 0.0000309 = 0
lambda = [0.08223 +/- sqrt(0.08223^2 - 4*0.0000309)] / 2
       = [0.08223 +/- sqrt(0.006761 - 0.0001236)] / 2
       = [0.08223 +/- 0.08147] / 2

lambda1 = 0.08185     lambda2 = 0.000378
```

Eigenvector for lambda1: solve `(0.06935-0.08185)x - 0.02936 z = 0` -> `x = -2.349 z` -> normalize -> **e1 = (-0.920, 0.392)**

Eigenvector for lambda2: solve `(0.06935-0.000378)x - 0.02936 z = 0` -> `x = 0.426 z` -> normalize -> **e2 = (0.392, 0.920)**

(check: `e1 . e2 = (-0.920)(0.392) + (0.392)(0.920) = 0`, orthogonal as required for a symmetric matrix — good consistency check on the arithmetic.)

**Semi-axes: 0.286 m along (-0.920, 0.392), and 0.0194 m along (0.392, 0.920).** The ellipse is long and thin — about 14.7x longer on one axis than the other — and tilted roughly 23 degrees off horizontal.

**A2.**
`A0` is already diagonal:

```
A0 = [ 0      0     ]
     [ 0      0.0875 ]
```
Eigenvalues fall straight out: **lambda1 = 0.0875** with eigenvector `(0,1)` (pure z), **lambda2 = 0** with eigenvector `(1,0)` (pure x).

A zero eigenvalue means the semi-axis length in that direction is `sqrt(0) = 0` — the ellipse has completely collapsed to a line segment. Physically: no matter how fast you spin the joints, at this exact instant the end effector has **zero** velocity component in x. That's the mathematical definition of a singularity, not just "slow" — it's a hard wall.

**A3.**
`u = (1, 0)`. Project onto the Q1 eigenbasis:

```
u . e1 = (1)(-0.920) + (0)(0.392) = -0.920
u . e2 = (1)(0.392)  + (0)(0.920) =  0.392

r(u) = 1 / sqrt( (-0.920)^2/0.08185 + (0.392)^2/0.000378 )
     = 1 / sqrt( 10.35 + 406.6 )
     = 1 / sqrt(416.9)
     = 1 / 20.42
     = 0.049
```

Compare: best case (along e1) is `sqrt(lambda1) = 0.286`, worst case (along e2) is `sqrt(lambda2) = 0.0194`. Pure horizontal correction gives **0.049** — nearly 6x worse than the best direction, even though `u` is only about 23 degrees off the easy axis. That's the counterintuitive part: because `lambda2` is 200x smaller than `lambda1`, even a small component of `u` along `e2` (the `0.392` term) dominates the sum once you divide by `lambda2`. Being "mostly" aligned with the good direction isn't good enough near a badly-conditioned pose — the tiny remaining component toward the bad direction costs you almost everything.

`kappa = sqrt(lambda1/lambda2) = sqrt(216.6) = 14.7`. That's a real, checkable number an engineer would compute before committing to an approach pose — a common rule of thumb is to reject configurations above some threshold (say kappa > 10) for tasks that need fine correction. At kappa = 14.7, this pose is already borderline for a 3mm-tolerance insertion.

So: yes, the IK posture task should be doing real work here, not just picking *a* solution out of the redundant family from the earlier assignment's Q7 — it should be biased toward configurations with a lower condition number specifically along the correction directions the task needs, not just "closest to home." That's a concrete, measurable thing you could add to `ik_mink.py`: compute `kappa` at the candidate solution and reject or re-seed if it's above threshold, before ever touching the real hardware.
