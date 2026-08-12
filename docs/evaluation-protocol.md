# Evaluation Protocol

## Purpose

The evaluation protocol defines how policy performance is measured so that ACT, Diffusion Policy, and SmolVLA can be compared fairly.

## Frozen protocol

Before training any model, freeze these rules:

- 20 trials minimum per policy, ideally 30
- same held-out object positions for all policies
- same task definition and success condition for all policies
- same timeout and action limit for all policies
- same camera stream configuration
- same object reset logic using seeded RNG
- same logging and video recording pipeline

## Success definition

Success must be defined in writing before rollout begins. Recommended default for lid-seating tasks:

- object is placed into target fixture
- load-cell signal remains above threshold for N consecutive samples
- trial ends only if the object stays in place without falling

## Failure taxonomy

Log each failure under one category:

- missed grasp
- off-center grasp
- dropped in transit
- misplaced placement
- timeout
- collision

## Data collection requirements

- log all trials with timestamps
- log seed and random placement positions
- save video from every rollout
- validate 20% of trials with manual review for success-condition agreement

## Reporting

For each policy, report:

- success rate
- failure taxonomy counts
- training time
- parameter count
- inference latency
- control-loop frequency
- agreement rate between load-cell detector and sampled video review

## Reproducibility

- store the exact eval seed
- save full configs for all training runs
- keep the environment lockfile and package versions
- keep the same physical task setup and lighting
